import asyncio
import os
import logging
from typing import Annotated, TypedDict, List, Literal
from functools import partial

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

# --- Configuration & Logging ---
LOG_FILE = "agent_session.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8')]
)
logger = logging.getLogger("ResearchAgent")

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:1.7b") # Upgraded baseline recommendation
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# --- 1. State Definition ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: List[Document]
    iterations: int  # Track refinement loops
    plan: str        # Store the high-level tool execution plan

# --- 2. Resource Initialization ---
def get_retriever():
    documents = [Document(page_content="System initialized with NLP, Web, and File tools.", metadata={"source": "system"})]
    extensions = [".txt", ".md", ".py", ".json"]
    for file in os.listdir("."):
        if any(file.endswith(ext) for ext in extensions) and os.path.isfile(file):
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    documents.append(Document(page_content=f.read()[:2000], metadata={"source": file}))
            except Exception as e: 
                logger.warning(f"Could not read {file}: {e}")
                pass
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Retriever initialization failed: {e}")
        raise

MCP_CONFIG = {
    "nlp": {"command": "python", "args": [os.path.abspath("nlp_analyzer.py")], "transport": "stdio"},
    "web": {"command": "python", "args": [os.path.abspath("web_extractor.py")], "transport": "stdio"},
    "files": {"command": "python", "args": [os.path.abspath("file_loader.py")], "transport": "stdio"}
}

# --- 3. Graph Nodes ---
async def planner(state: State, llm):
    """Strategizes how to solve the user query using available tools."""
    user_query = state["messages"][0].content
    planner_prompt = f"""You are a Strategic Planning Agent. 
User Query: "{user_query}"

AVAILABLE TOOLS:
1. nlp_analyzer: Analyze text (summarize, sentiment, metrics, etc.).
2. web_extractor: Search Google or scrape specific URLs for content.
3. file_loader: Load and extract text from local files (PDF, DOCX, CSV, TXT).

TASK: Create a clear, step-by-step plan to answer the user's query using these tools.
- Identify which tool to use first.
- Explain what information is needed from each tool.
- Do NOT execute the tools yourself; just provide the plan.
- Be concise and focus only on the tool-based steps.

Format your response as a numbered list of steps.
"""
    response = await llm.ainvoke([HumanMessage(content=planner_prompt)])
    console.print(Panel(response.content, title="[bold cyan]STRATEGIC PLAN[/bold cyan]"))
    return {"plan": response.content}

async def retrieve(state: State, retriever):
    last_msg = state["messages"][-1].content
    try:
        docs = await retriever.ainvoke(last_msg)
        return {"context": docs}
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"context": []}

async def chatbot(state: State, llm_with_tools):
    # --- ROBUST SYSTEM PROMPT ---
    system_prompt = f"""You are an Advanced AI Research Agent. Your primary function is to provide highly accurate, comprehensive, and well-structured answers.

CURRENT PLAN:
{state.get('plan', 'No specific plan generated.')}

CORE DIRECTIVES:
1. FOLLOW THE PLAN: Use the Strategic Plan provided above to guide your tool usage.
2. TOOL CAPABILITY: You HAVE tools (nlp_analyzer, web_extractor, file_loader) that allow you to analyze text, scrape websites, and read files. NEVER claim that you cannot access external websites or files. Use your tools instead.
3. GROUNDING: Base your answers STRICTLY on the provided 'Knowledge' context and your tool outputs. Do not guess or hallucinate facts.
4. TOOL USAGE: If the context is insufficient, proactively use your available tools to gather the necessary data before answering. Do not provide generic disclaimers.
5. SYNTHESIS: Combine information from multiple sources logically. 
6. REFINEMENT: If your previous answer was critiqued or if you realize you haven't used tools yet, provide a corrected, superior response.
7. FORMATTING: Use Markdown formatting (bolding, bullet points, headers) for readability.
"""
    
    context_docs = state.get("context", [])
    context_str = "\n".join([f"--- Source: {d.metadata.get('source', 'Unknown')} ---\n{d.page_content[:500]}" for d in context_docs])
    
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"<KNOWLEDGE>\n{context_str}\n</KNOWLEDGE>")
    ] + state["messages"]
    
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

async def reflector(state: State, llm):
    """Observes the answer and decides if it needs modification."""
    iterations = state.get("iterations", 0)
    
    # Extract the original question and the AI's latest answer
    user_query = state["messages"][0].content
    last_ai_msg = state["messages"][-1].content
    
    critique_prompt = f"""You are a strict QA Evaluator. 

User Query: "{user_query}"
AI Response: "{last_ai_msg}"

TASK: Evaluate if the AI response fully, accurately, and directly answers the User Query without any hallucinations or fluff.
- If the AI is giving a generic disclaimer like 'I cannot access websites/files', 'I am an AI assistant', or 'I don't have access', you MUST reply with 'REFINE: Use your tools to perform the requested task. Do not make excuses.'
- If the response is perfect, complete, and needs no changes, reply ONLY with the exact word: CORRECT
- If the response is missing information, poorly formatted, or inaccurate, provide a concise directive on how to fix it, starting with 'REFINE: '
"""
    
    evaluation = await llm.ainvoke([HumanMessage(content=critique_prompt)])
    eval_content = evaluation.content.strip()
    
    if "CORRECT" in eval_content.upper() and "REFINE" not in eval_content.upper():
        return {"iterations": 4} # Signal to end the loop safely
    else:
        logger.info(f"Refining answer (Iteration {iterations + 1}): {eval_content}")
        return {
            "messages": [HumanMessage(content=f"CRITIQUE RECEIVED: {eval_content}\n\nPlease regenerate your response to fix these specific issues.")],
            "iterations": iterations + 1
        }

def should_continue(state: State) -> Literal["tools", "reflector", "__end__"]:
    """Determines the next step in the loop."""
    last_msg = state["messages"][-1]
    
    # 1. Did the AI decide to use a tool?
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    # 2. Have we hit our maximum reflection loops?
    if state.get("iterations", 0) >= 3:
        return "__end__"
        
    # 3. Otherwise, send the AI's answer to the reflector for QA
    return "reflector"

# --- 4. Execution Loop ---
async def run_agent():
    console.print(Panel("[bold green]Agent Initializing with Observation Loop (Max 3 Refinements)...[/bold green]", title="Research Agent V4"))
    
    retriever = get_retriever()
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    
    # PROPER MCP ASYNC CONTEXT MANAGEMENT
    client = MultiServerMCPClient(MCP_CONFIG)
    try:
        mcp_tools = await client.get_tools()
        llm_with_tools = llm.bind_tools(mcp_tools)
    except Exception as e:
        console.print(f"[bold red]Failed to load tools: {e}[/bold red]")
        return

    workflow = StateGraph(State)
    workflow.add_node("planner", partial(planner, llm=llm))
    workflow.add_node("retrieve", partial(retrieve, retriever=retriever))
    workflow.add_node("chatbot", partial(chatbot, llm_with_tools=llm_with_tools))
    workflow.add_node("tools", ToolNode(mcp_tools))
    workflow.add_node("reflector", partial(reflector, llm=llm))
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retrieve")
    workflow.add_edge("retrieve", "chatbot")
    workflow.add_conditional_edges("chatbot", should_continue)
    workflow.add_edge("tools", "chatbot")
    
    # If reflector sets iterations to 4 (CORRECT), we end. Otherwise, back to chatbot.
    workflow.add_conditional_edges("reflector", lambda s: "chatbot" if s["iterations"] < 4 else "__end__")

    app = workflow.compile(checkpointer=InMemorySaver())
    import uuid
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    console.print(Panel("Ready for multi-step research and self-correction.", title="SYSTEM ONLINE"))

    while True:
        try:
            user_input = console.input("\n[bold yellow]User ❯ [/bold yellow]").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            
            if user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                session_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": session_id}}
                console.print(Panel("Session history cleared. Ready for a new task.", title="SYSTEM RESET"))
                continue

            initial_state = {"messages": [HumanMessage(content=user_input)], "iterations": 0, "plan": ""}

            with console.status("[bold cyan]Agent is working and observing...", spinner="dots"):
                async for event in app.astream(initial_state, config, stream_mode="updates"):
                    for node, state_update in event.items():
                        if node == "chatbot":
                            msg = state_update["messages"][-1]
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    console.print(f"[bold magenta]THOUGHT:[/bold magenta] Calling [cyan]{tc['name']}[/cyan]")
                            elif msg.content and state_update.get("iterations", 0) == 0:
                                # We don't print intermediate drafts, only final or first attempts
                                pass 
                            
                        elif node == "reflector":
                            iters = state_update.get("iterations", 0)
                            if iters < 4:
                                critique_msg = state_update["messages"][-1].content
                                console.print(f"[bold yellow]OBSERVATION:[/bold yellow] Refining (Attempt {iters})...")
                                logger.debug(critique_msg)
                            else:
                                console.print("[bold green]OBSERVATION:[/bold green] Answer verified as correct.")
                                # Print the final validated message
                                final_state = app.get_state(config)
                                final_msg = final_state.values["messages"][-1].content
                                console.print(Panel(Markdown(final_msg), title="[bold green]AI Assistant[/bold green]"))
                        
                        elif node == "tools":
                            console.print("[dim italic]Tool execution finished.[/dim italic]")
                            
                        # Fallback if iterations max out without hitting 'CORRECT'
                        if node == "chatbot" and app.get_state(config).values.get("iterations", 0) >= 3:
                            final_msg = state_update["messages"][-1].content
                            console.print("[bold yellow]OBSERVATION:[/bold yellow] Max refinements reached.")
                            console.print(Panel(Markdown(final_msg), title="[bold green]AI Assistant (Final Attempt)[/bold green]"))

        except Exception as e:
            console.print(f"[bold red]Error during interaction:[/bold red] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        console.print("\n[bold red]Shutdown requested.[/bold red]")