#!/usr/bin/env python3
"""
RAG Engine MCP Server
---------------------
Exposes Vector Search and Document Management tools to AI models.
Uses LangChain FAISS and Ollama Embeddings for persistence and memory.
"""

import os
import json
import logging
import warnings
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

# Suppression of heavy library warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-mcp-server")

# Initialize FastMCP
mcp = FastMCP("RAGEngine")

# --- Configuration ---
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
INDEX_PATH = "faiss_index"

# --- Resource Initialization ---
def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(INDEX_PATH):
        try:
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
    
    # Create an initial empty index if not exists
    initial_doc = Document(page_content="Knowledge base initialized.", metadata={"source": "system"})
    vectorstore = FAISS.from_documents([initial_doc], embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

vectorstore = get_vectorstore()

# --- MCP TOOLS ---

@mcp.tool()
def inject_documents(text: str, source: str = "user_input") -> str:
    """
    Injects text into the knowledge base for later retrieval.
    :param text: The text content to remember.
    :param source: The source of the information (e.g., 'web', 'file', 'conversation').
    """
    global vectorstore
    try:
        doc = Document(page_content=text, metadata={"source": source, "timestamp": str(datetime.now())})
        vectorstore.add_documents([doc])
        vectorstore.save_local(INDEX_PATH)
        return f"Successfully injected information from {source}. Knowledge base updated."
    except Exception as e:
        logger.error(f"Injection error: {e}")
        return f"Error during injection: {str(e)}"

@mcp.tool()
def search_knowledge(query: str, k: int = 3) -> str:
    """
    Searches the knowledge base for relevant information.
    :param query: The search query.
    :param k: Number of relevant results to return.
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        results = []
        for d in docs:
            results.append({
                "content": d.page_content,
                "metadata": d.metadata
            })
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error during search: {str(e)}"

@mcp.tool()
def clear_knowledge() -> str:
    """
    Clears the knowledge base and resets the index.
    """
    global vectorstore
    try:
        if os.path.exists(INDEX_PATH):
            import shutil
            shutil.rmtree(INDEX_PATH)
        
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        initial_doc = Document(page_content="Knowledge base reset.", metadata={"source": "system"})
        vectorstore = FAISS.from_documents([initial_doc], embeddings)
        vectorstore.save_local(INDEX_PATH)
        return "Knowledge base cleared successfully."
    except Exception as e:
        return f"Error clearing knowledge: {e}"

if __name__ == "__main__":
    mcp.run()
