#!/usr/bin/env python3
"""
Report Generator MCP Server
---------------------------
Provides tools for generating comprehensive analysis reports.
Optimized for structured Markdown output.
"""

import os
import logging
from typing import List, Dict
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report-generator-mcp")

# Initialize FastMCP
mcp = FastMCP("ReportGenerator")

@mcp.tool()
def create_analysis_report(report_title: str, sections: List[Dict[str, str]] = None, output_file: str = "analysis_report.md") -> str:
    """
    Generates a comprehensive Markdown report file from structured sections.
    :param report_title: The title of the report.
    :param sections: A list of sections, each with 'header' and 'content' keys.
    :param output_file: The name of the file to save (default is analysis_report.md).
    """
    try:
        report_content = f"# {report_title}\n\n"
        report_content += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n"

        if sections:
            for section in sections:
                header = section.get('header', 'Section')
                content = section.get('content', '')
                report_content += f"## {header}\n\n{content}\n\n"
        else:
            report_content += "## Summary\n\nNo structured sections provided.\n"

        # Ensure directory exists if path provided
        output_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return f"Report created successfully: {output_path}"
    except Exception as e:
        logger.error(f"Report creation error: {e}")
        return f"Error creating report: {str(e)}"

@mcp.tool()
def save_markdown_report(content: str, filename: str = "analysis_report.md") -> str:
    """
    Saves raw markdown content to a file. 
    Use this if you have already formatted the report as Markdown.
    :param content: The full markdown content to save.
    :param filename: The name of the file to save (default is analysis_report.md).
    """
    try:
        output_path = os.path.abspath(filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report created successfully: {output_path}"
    except Exception as e:
        logger.error(f"Error saving markdown: {e}")
        return f"Error saving report: {str(e)}"

if __name__ == "__main__":
    mcp.run()
