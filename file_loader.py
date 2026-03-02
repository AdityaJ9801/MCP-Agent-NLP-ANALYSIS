#!/usr/bin/env python3
import os
import json
import logging
import pandas as pd
from typing import Any, Dict
from PyPDF2 import PdfReader
from docx import Document
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("file-loader-mcp")

mcp = FastMCP("FileLoader")

@mcp.tool()
def load_local_file(file_path: str) -> str:
    """
    Extracts text content from local files (PDF, DOCX, CSV, TXT).
    :param file_path: Absolute path to the file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return json.dumps({"text": text, "type": "pdf", "path": file_path})
            
        elif ext == ".docx":
            doc = Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
            return json.dumps({"text": text, "type": "docx", "path": file_path})
            
        elif ext == ".csv":
            df = pd.read_csv(file_path).head(20) # Limit to first 20 rows for context
            return json.dumps({"text": df.to_string(), "type": "csv", "path": file_path})
            
        else: # Default to text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return json.dumps({"text": f.read(), "type": "text", "path": file_path})
                
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return f"Error processing file: {str(e)}"

if __name__ == "__main__":
    mcp.run()
