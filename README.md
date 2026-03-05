# Advanced AI Research Agent

An autonomous agentic system built with **LangGraph**, **Ollama**, and **Model Context Protocol (MCP)** servers. This agent can plan its actions, retrieve local context, use specialized tools for NLP and Web research, and reflect on its own answers for maximum accuracy.
<img width="1807" height="1021" alt="Screenshot 2026-03-02 211934" src="https://github.com/user-attachments/assets/16c4694f-4711-45cb-b78f-566362f46651" />


## 🚀 Key Features

- **Autonomous Planning**: Every query is first analyzed by a Strategic Planner that selects the best tools for the job.
- **Multi-Server MCP Integration**: Uses three specialized servers:
  - **NLP Analyzer**: Sentiment analysis, readability metrics, and text summarization.
  - **Web Extractor**: Google search and deep web scraping.
  - **File Loader**: Text extraction from PDF, DOCX, CSV, and TXT files.
- **Self-Correction (Reflection)**: A dedicated Reflector node critiques answers and forces refinements if they are generic or incomplete.
- **RAG Capability**: Automatically indexes local project files for immediate context retrieval.
- **Rich UI**: Interactive CLI with colored panels and Markdown rendering.

## 🏗️ Project Architecture

The system follows a directed cyclic graph (DCG) workflow:

1.  **Planner**: Receives user query and generates a tool-based execution strategy.
2.  **Retrieve**: Searches the local vector database (FAISS) for relevant project files.
3.  **Chatbot**: Synthesizes the plan, retrieved context, and tool outputs to generate a response.
4.  **Tools (MCP)**: Executes requested tool calls (Web search, NLP analysis, File reading).
5.  **Reflector**: Evaluates the response quality. If it's a generic disclaimer or incomplete, it loops back to the Chatbot with a critique for refinement (Max 3 loops).

## 🛠️ Installation & Setup

### Prerequisites

1.  **Python 3.10+**: Ensure you have a recent version of Python installed.
2.  **Ollama Installation**:
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Follow the installation instructions for your operating system (Windows, macOS, or Linux).
    *   Ensure the Ollama server is running (you should see the Ollama icon in your system tray or be able to run `ollama list` in your terminal).
3.  **Required Models**:
    *   The agent uses `qwen3:1.7b` for reasoning and `nomic-embed-text` for embeddings.
    *   Open your terminal and pull the required models:
        ```bash
        ollama pull qwen3:1.7b
        ollama pull nomic-embed-text
        ```
4.  **System Dependencies**:
    *   (Optional) If you encounter issues with `lxml`, you may need to install system-level developer tools for C compilation, though the pre-built wheels usually suffice.

### Setup Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AdityaJ9801/MCP-Agent-NLP-ANALYSIS.git
   cd MCP-Agent-NLP-ANALYSIS
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install langchain-ollama langchain-community langchain-mcp-adapters langgraph faiss-cpu beautifulsoup4 requests googlesearch-python rich PyPDF2 python-docx pandas nltk textblob mcp
   ```

4. **Initialize NLTK Resources**:
   The `nlp_analyzer.py` will automatically download required NLTK data on the first run.

## 🏃 Running the Agent

Start the interactive CLI:
```bash
python agent.py
```

## 💡 Sample Prompts

- **Web Research**: `"Search for the latest features in Python 3.13 and summarize the key changes."`
- **NLP Analysis**: `"Scrape https://en.wikipedia.org/wiki/Artificial_intelligence and give me a full NLP analysis including sentiment and readability."`
- **File Handling**: `"Load the local file 'report.pdf' and tell me the main findings."`
- **Complex Task**: `"Find news about Podman's latest release, then analyze the sentiment of the top 3 articles found."`

## ⚙️ Configuration
You can change the default models by setting environment variables:
- `OLLAMA_MODEL`: Default is `qwen3:1.7b`
- `OLLAMA_EMBEDDING_MODEL`: Default is `nomic-embed-text`

---

