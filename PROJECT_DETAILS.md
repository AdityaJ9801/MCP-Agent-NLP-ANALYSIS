# NLP Research Agent - Project Details

## 🏗 Architecture Overview

The project is built as an **Autonomous Agentic System** using a Directed Cyclic Graph (DCG) workflow managed by **LangGraph**. The architecture is decoupled into a central orchestrator (the Agent) and multiple specialized microservices (MCP Servers).

### 1. Central Orchestrator (LangGraph)
The agent logic is divided into four main functional nodes:
*   **Planner**: Analyzes the user's intent. It decides whether to answer directly, use a sequence of tools, or reject the request if unsupported.
*   **Chatbot**: The primary reasoning engine. It takes the plan and any retrieved context to formulate tool calls or final responses.
*   **Tool Node**: An automated executor that routes requests to the appropriate MCP servers.
*   **Reflector (Self-Correction)**: A QA node that evaluates the chatbot's output. If the response is generic or incomplete, it sends a critique back to the chatbot for refinement (up to 3 iterations).

### 2. Model Context Protocol (MCP) Servers
The agent's capabilities are extended via independent MCP servers that communicate over `stdio`:
*   **NLP Analyzer**: Performs text summarization, sentiment analysis, and readability metrics using NLTK and TextBlob.
*   **Web Extractor**: Handles real-time information gathering via Google Search and BeautifulSoup4 scraping.
*   **File Loader**: Extracts text from local PDF, DOCX, CSV, and TXT files.
*   **RAG Engine**: A persistent vector database (FAISS) that stores and retrieves information using Ollama embeddings.
*   **Report Generator**: Generates formal Markdown reports based on structured analysis data.

---

## 🚀 Technical Features

*   **Autonomous Planning**: The system doesn't just react; it strategizes a step-by-step approach before executing tools.
*   **Self-Correction Loop**: Through the Reflector node, the agent critiques its own work, reducing hallucinations and generic "I don't know" answers.
*   **Multi-Server Tooling**: Uses the Model Context Protocol to maintain a clean separation between AI logic and tool implementation.
*   **Persistent RAG**: Uses a local FAISS index to "remember" information across a session, with an automatic cleanup mechanism at the start of new sessions.
*   **Hybrid Memory**: Combines LangGraph's short-term message history with the RAG engine's long-term vector storage.
*   **Structured Reporting**: Can output findings into professional `.md` files for external use.

---

## 📚 Main Libraries

### Core Frameworks
*   **LangGraph**: For managing the cyclic state machine and agent workflow.
*   **LangChain**: Provides the base abstractions for messages, LLM integration, and document handling.
*   **MCP (Model Context Protocol)**: For building and connecting the tool servers.

### AI & NLP
*   **LangChain Ollama**: Interface for running local models (Qwen, Nomic Embed).
*   **NLTK (Natural Language Toolkit)**: For tokenization, POS tagging, and frequency analysis.
*   **TextBlob**: For simplified sentiment analysis and processing.
*   **FAISS (Facebook AI Similarity Search)**: Efficient vector storage and similarity search.

### Utilities
*   **Rich**: For the advanced interactive CLI interface (Panels, Markdown rendering, Colors).
*   **BeautifulSoup4 & Requests**: For robust web scraping and HTTP management.
*   **PyPDF2 & python-docx**: For extracting content from complex document formats.
*   **Pandas**: For processing and summarizing CSV data.

---

## 🛠 Setup & Requirements
*   **Python**: 3.10+ (Optimized for 3.14 compatibility)
*   **Models**: `qwen3:1.7b` (Reasoning), `nomic-embed-text` (Embeddings) via Ollama.
*   **Environment**: Virtual environment (`.venv`) with dependencies installed via `pip`.
