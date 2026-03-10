import asyncio
import os
import logging
import shutil
from typing import Annotated, TypedDict, List, Literal
from functools import partial

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
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

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:1.7b") 

# --- 1. State Definition ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str     
    iterations: int  
    plan: str        

# --- 2. Resource Initialization ---
MCP_CONFIG = {
    "nlp": {"command": "python", "args": [os.path.abspath("nlp_analyzer.py")], "transport": "stdio"},
    "web": {"command": "python", "args": [os.path.abspath("web_extractor.py")], "transport": "stdio"},
    "files": {"command": "python", "args": [os.path.abspath("file_loader.py")], "transport": "stdio"},
    "rag": {"command": "python", "args": [os.path.abspath("rag_engine.py")], "transport": "stdio"}
}

def clear_rag_storage():
    """Deletes the persistent FAISS index to ensure a clean start."""
    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index")
            logger.info("Cleared RAG storage.")
        except Exception as e:
            logger.error(f"Failed to clear RAG storage: {e}")

# --- 3. Graph Nodes ---
async def planner(state: State, llm):
    """Strategizes how to solve the latest user query."""
    # Always look at the LATEST message from the user
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_query = user_messages[-1].content if user_messages else "No query found."
    
    planner_prompt = f"""You are a Strategic Planning Agent. 
LATEST USER QUERY: "{user_query}"

AVAILABLE TOOLS:
1. nlp_analyzer: Deep text analysis (sentiment, metrics, summarization).
2. web_extractor: Google search and web scraping.
3. file_loader: Read PDF, DOCX, CSV, TXT.
4. rag_engine: Search or inject into long-term knowledge base.

TASK:
1. If the query is a simple greeting or common knowledge you can answer directly, start with "DECISION: DIRECT_ANSWER" followed by the answer.
2. If the query requires research, web access, or file reading, start with "DECISION: TOOL_PLAN" and list the steps.
3. If it's something you cannot do, start with "DECISION: UNSUPPORTED" and explain why.

Be decisive. Do not use tools for simple conversation.
"""
    response = await llm.ainvoke([HumanMessage(content=planner_prompt)])
    console.print(Panel(response.content, title="[bold cyan]PLANNER DECISION[/bold cyan]"))
    return {"plan": response.content}

async def chatbot(state: State, llm_with_tools):
    plan = state.get('plan', '')
    
    if "DECISION: DIRECT_ANSWER" in plan or "DECISION: UNSUPPORTED" in plan:
        # Extract the actual content after the decision label
        content = plan.split(":", 1)[1].replace("DIRECT_ANSWER", "").replace("UNSUPPORTED", "").strip()
        # Use AIMessage to avoid 'tool_calls' attribute errors
        return {"messages": [AIMessage(content=content)]}

    system_prompt = f"""You are an Advanced AI Research Agent. 

STRICT PLAN TO FOLLOW:
{plan}

Use your tools to fulfill this plan. Once tools provide data, synthesize a final answer.
"""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

async def reflector(state: State, llm):
    plan = state.get('plan', '')
    if "DECISION: DIRECT_ANSWER" in plan or "DECISION: UNSUPPORTED" in plan:
        return {"iterations": 4}

    user_query = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
    last_ai_msg = state["messages"][-1].content
    
    critique_prompt = f"Does this answer '{user_query}'? Answer: '{last_ai_msg}'. Reply 'CORRECT' or 'REFINE: [reason]'."
    eval_res = await llm.ainvoke([HumanMessage(content=critique_prompt)])
    
    if "CORRECT" in eval_res.content.upper():
        return {"iterations": 4}
    return {"messages": [HumanMessage(content=f"REFINE: {eval_res.content}")], "iterations": state.get("iterations", 0) + 1}

def should_continue(state: State) -> Literal["tools", "reflector", "__end__"]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    if state.get("iterations", 0) >= 3:
        return "__end__"
    return "reflector"

# --- 4. Execution Loop ---
async def run_agent():
    console.print(Panel("[bold green]Initializing...[/bold green]", title="Research Agent V4"))
    clear_rag_storage()
    
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    client = MultiServerMCPClient(MCP_CONFIG)
    try:
        mcp_tools = await client.get_tools()
        llm_with_tools = llm.bind_tools(mcp_tools)
    except Exception as e:
        console.print(f"[bold red]Tool Loading Error: {e}[/bold red]")
        return

    workflow = StateGraph(State)
    workflow.add_node("planner", partial(planner, llm=llm))
    workflow.add_node("chatbot", partial(chatbot, llm_with_tools=llm_with_tools))
    workflow.add_node("tools", ToolNode(mcp_tools))
    workflow.add_node("reflector", partial(reflector, llm=llm))
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "chatbot")
    workflow.add_conditional_edges("chatbot", should_continue)
    workflow.add_edge("tools", "chatbot")
    workflow.add_conditional_edges("reflector", lambda s: "chatbot" if s["iterations"] < 4 else "__end__")

    app = workflow.compile(checkpointer=InMemorySaver())
    import uuid
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    console.print(Panel("System Online. Type 'exit' to quit or 'clear' to reset.", title="READY"))

    while True:
        try:
            user_input = console.input("\n[bold yellow]User ❯ [/bold yellow]").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() == "clear":
                clear_rag_storage()
                session_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": session_id}}
                console.print(Panel("Memory cleared.", title="RESET"))
                continue

            # We pass the input as a HumanMessage. LangGraph will append it to history via 'thread_id'.
            inputs = {"messages": [HumanMessage(content=user_input)], "iterations": 0}

            with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                async for event in app.astream(inputs, config, stream_mode="updates"):
                    for node, state_update in event.items():
                        if node == "chatbot":
                            msg = state_update["messages"][-1]
                            # SAFE CHECK: Only check tool_calls if msg is an AIMessage
                            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                                for tc in msg.tool_calls:
                                    console.print(f"[bold magenta]THOUGHT:[/bold magenta] Calling [cyan]{tc['name']}[/cyan]")
                        
                        elif node == "reflector":
                            if state_update.get("iterations", 0) >= 4:
                                final_state = app.get_state(config)
                                final_msg = final_state.values["messages"][-1].content
                                console.print(Panel(Markdown(final_msg), title="[bold green]AI Assistant[/bold green]"))
                        
                        if node == "chatbot" and app.get_state(config).values.get("iterations", 0) >= 3:
                            final_msg = state_update["messages"][-1].content
                            console.print(Panel(Markdown(final_msg), title="[bold green]AI Assistant[/bold green]"))

        except Exception as e:
            console.print(f"[bold red]Runtime Error:[/bold red] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        console.print("\n[bold red]Shutdown.[/bold red]")
