#!/usr/bin/env python3
"""
Web Extractor MCP Server
------------------------
Exposes web scraping and search capabilities as tools for AI models.
"""

import random
import time
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from googlesearch import search as gsearch
from mcp.server.fastmcp import FastMCP

# Configuration & Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-extractor-mcp")

# Initialize FastMCP
mcp = FastMCP("WebExtractor")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]

class ExtractorCore:
    def __init__(self, delay=1.0, verify_ssl=True):
        self.session = requests.Session()
        self.delay = delay
        self.verify_ssl = verify_ssl
        self.robot_parsers = {}

    def _get_headers(self):
        return {"User-Agent": random.choice(USER_AGENTS)}

    def can_fetch(self, url):
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            if base_url not in self.robot_parsers:
                rp = RobotFileParser()
                rp.set_url(urljoin(base_url, "/robots.txt"))
                rp.read()
                self.robot_parsers[base_url] = rp
            return self.robot_parsers.get(base_url).can_fetch("*", url)
        except Exception:
            return True # Fail-safe: allow if robots.txt check fails

    def scrape(self, url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        if not url.startswith(("http://", "https://")):
            return {"error": "Invalid URL protocol. Use http or https."}

        if not self.can_fetch(url):
            logger.warning(f"Robots.txt restricted access to {url}")
        
        try:
            time.sleep(self.delay)
            # verify_ssl is now configurable
            resp = self.session.get(url, headers=self._get_headers(), timeout=15, verify=self.verify_ssl)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, 'lxml')
            results = {
                "url": url,
                "title": soup.title.string if soup.title else "N/A",
                "timestamp": str(datetime.now())
            }

            if selector:
                results["custom_data"] = [el.get_text(strip=True) for el in soup.select(selector)]

            # Extract main text content
            for noise in soup(["script", "style", "nav", "footer", "header"]):
                noise.decompose()
            
            results["text_content"] = " ".join(soup.stripped_strings)[:5000] # Limit for LLM context
            return results
        except requests.exceptions.SSLError:
            return {"error": "SSL Verification failed. If this is a trusted internal site, set verify_ssl=False."}
        except Exception as e:
            logger.error(f"Scrape error for {url}: {e}")
            return {"error": str(e)}

# Set verify_ssl=False only if you are in a restricted network with self-signed certs
core = ExtractorCore(verify_ssl=True)

# --- MCP TOOLS ---

@mcp.tool()
def scrape_url(url: str, css_selector: Optional[str] = None) -> str:
    """
    Scrapes text content and metadata from a specific URL.
    :param url: The full web address to scrape.
    :param css_selector: Optional CSS selector to target specific elements.
    """
    logger.info(f"MCP Tool scrape_url called for: {url}")
    import json
    data = core.scrape(url, css_selector)
    return json.dumps(data, indent=2)

@mcp.tool()
def search_and_scrape(query: str, num_results: int = 3) -> str:
    """
    Performs a Google search and scrapes the top results for the query.
    :param query: The search term.
    :param num_results: Number of top websites to analyze (max 5).
    """
    logger.info(f"MCP Tool search_and_scrape called for: {query}")
    limit = min(num_results, 5)
    aggregated = []
    
    try:
        # Using a set to avoid duplicate URLs
        urls = list(dict.fromkeys(gsearch(query, num_results=limit)))
        if not urls:
            return "No results found for the query."

        for url in urls:
            logger.info(f"Scraping result: {url}")
            res = core.scrape(url)
            if "error" not in res:
                aggregated.append(res)
            
        if not aggregated:
            return "Search succeeded but all scraping attempts failed or were blocked."

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {str(e)}"

    import json
    return json.dumps(aggregated, indent=2)

if __name__ == "__main__":
    mcp.run()