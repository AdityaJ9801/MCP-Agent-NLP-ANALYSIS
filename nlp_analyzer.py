#!/usr/bin/env python3
"""
NLP Analyzer MCP Server
-----------------------
Exposes advanced Natural Language Processing tools to AI models.
"""

import os
import re
import json
import logging
import warnings
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import Counter

from mcp.server.fastmcp import FastMCP

# Suppression of heavy library warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nlp-mcp-server")

# Initialize FastMCP
mcp = FastMCP("NLPAnalyzer")

# --- Dependency Management ---
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.util import ngrams
import math

# Ensure basic NLTK resources
def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else f"corpora/{res}")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

setup_nltk()

class NLPProcessor:
    def clean_text(self, text: str) -> str:
        if not text: return ""
        return re.sub(r'\s+', ' ', text).strip()

    def get_stats(self, text: str) -> Dict[str, Any]:
        try:
            words = nltk.word_tokenize(text)
            sentences = nltk.sent_tokenize(text)
            
            def count_syllables(word):
                word = word.lower()
                vowels = "aeiouy"
                count = 0
                if not word: return 0
                if word[0] in vowels: count += 1
                for i in range(1, len(word)):
                    if word[i] in vowels and word[i-1] not in vowels: count += 1
                if word.endswith("e"): count -= 1
                return max(1, count)

            w_count = len(words)
            s_count = max(1, len(sentences))
            syll_count = sum(count_syllables(w) for w in words)
            
            # Flesch Reading Ease
            fre = 206.835 - 1.015 * (w_count / s_count) - 84.6 * (syll_count / w_count) if w_count > 0 else 0
            
            return {
                "word_count": w_count,
                "sentence_count": s_count,
                "lexical_diversity": round(len(set(w.lower() for w in words)) / w_count, 2) if w_count > 0 else 0,
                "readability_score": round(fre, 2)
            }
        except Exception as e:
            logger.error(f"Stats calculation error: {e}")
            return {"error": str(e)}
        
    def advanced_metrics(self, text: str) -> Dict[str, Any]:
        try:
            tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
            stop_words = set(stopwords.words("english"))

            total_words = len(tokens)
            unique_words = len(set(tokens))

            # Bag of Words
            bow = Counter(tokens).most_common(20)

            # Stopword statistics
            stopword_count = sum(1 for w in tokens if w in stop_words)

            # N-grams
            bigrams = Counter(ngrams(tokens, 2)).most_common(10)

            # Average word length
            avg_word_len = sum(len(w) for w in tokens) / total_words if total_words else 0

            # POS distribution
            pos_tags = nltk.pos_tag(tokens)
            pos_distribution = Counter(tag for _, tag in pos_tags)

            # Vocabulary richness
            ttr = unique_words / total_words if total_words else 0

            # Shannon entropy (text complexity)
            freq = Counter(tokens)
            entropy = -sum((count/total_words) * math.log2(count/total_words) for count in freq.values())

            return {
                "bag_of_words_top20": bow,
                "bigrams_top10": [" ".join(bg) for bg, _ in bigrams],
                "stopword_ratio": round(stopword_count / total_words, 3) if total_words else 0,
                "avg_word_length": round(avg_word_len, 2),
                "type_token_ratio": round(ttr, 3),
                "pos_distribution": dict(pos_distribution),
                "text_entropy": round(entropy, 3)
            }

        except Exception as e:
            return {"error": str(e)}
    def process(self, text: str) -> Dict[str, Any]:
        try:
            clean = self.clean_text(text)
            blob = TextBlob(clean)
            
            return {
                "statistics": self.get_stats(clean),
                "advanced_metrics": self.advanced_metrics(clean),
                "sentiment": {
                    "polarity": round(blob.sentiment.polarity, 2),
                    "subjectivity": round(blob.sentiment.subjectivity, 2),
                    "label": "Positive" if blob.sentiment.polarity > 0.1 else "Negative" if blob.sentiment.polarity < -0.1 else "Neutral"
                },
                "preview_pos": nltk.pos_tag(nltk.word_tokenize(clean)[:20])
            }
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {"error": str(e)}

processor = NLPProcessor()

# --- MCP TOOLS ---

def _extract_text(input_data: Any) -> str:
    """Helper to extract text from various formats (string, dict, JSON string)."""
    if not input_data:
        return ""
    
    # If it's already a string, check if it's actually a JSON string
    if isinstance(input_data, str):
        trimmed = input_data.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            try:
                data = json.loads(trimmed)
                if isinstance(data, dict):
                    input_data = data
                else:
                    return trimmed
            except:
                return trimmed # Not valid JSON, treat as raw text
        else:
            return trimmed

    # If it's a dictionary, look for common content keys
    if isinstance(input_data, dict):
        for key in ["text_content", "text", "content", "data", "body"]:
            if key in input_data and isinstance(input_data[key], str):
                return input_data[key]
        return " ".join([str(v) for v in input_data.values() if isinstance(v, str)])
    
    return str(input_data)

@mcp.tool()
def summarize_text(text: Any, max_sentences: int = 5) -> str:
    """
    Creates a concise summary of long text. 
    Use this before analysis if the text is very long.
    """
    processed_text = _extract_text(text)
    if not processed_text:
        return "Error: No text to summarize."
    
    try:
        sentences = nltk.sent_tokenize(processed_text)
        if len(sentences) <= max_sentences:
            return processed_text
        
        # Simple frequency-based summarization
        words = [w.lower() for w in nltk.word_tokenize(processed_text) if w.isalnum()]
        freq = Counter(words)
        sw = set(nltk.corpus.stopwords.words('english'))
        
        ranking = {}
        for i, sent in enumerate(sentences):
            for word in nltk.word_tokenize(sent.lower()):
                if word in freq and word not in sw:
                    ranking[i] = ranking.get(i, 0) + freq[word]
        
        top_indices = sorted(ranking, key=ranking.get, reverse=True)[:max_sentences]
        summary = " ".join([sentences[j] for j in sorted(top_indices)])
        return summary
    except Exception as e:
        return f"Summarization failed: {e}"

@mcp.tool()
def analyze_text(text: Any) -> str:
    """
    Performs full NLP analysis on text. 
    Accepts raw text, or a JSON object containing a 'text' or 'text_content' field.
    """
    processed_text = _extract_text(text)
    
    if not processed_text or len(processed_text.strip()) == 0:
        return "Error: No valid text found for analysis."
        
    logger.info(f"Processing text (length: {len(processed_text)})")
    try:
        results = processor.process(processed_text)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error during analysis: {str(e)}"

@mcp.tool()
def analyze_file(file_path: str) -> str:
    """
    Reads a local file (.txt supported) and performs NLP analysis.
    :param file_path: Absolute path to the file.
    """
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        results = processor.process(content)
        results["metadata"] = {"file": file_path, "timestamp": str(datetime.now())}
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def get_readability_metrics(text: Any) -> str:
    """
    Specifically calculates Flesch Reading Ease and lexical diversity.
    Accepts raw text or a JSON object with a 'text' field.
    """
    processed_text = _extract_text(text)
    if not processed_text:
        return "Error: No valid text found."
    stats = processor.get_stats(processed_text)
    return json.dumps(stats, indent=2)

# --- MCP RESOURCES ---

@mcp.resource("nlp://latest-analysis")
def get_latest_analysis() -> str:
    """Provides a log-style view of the last processed analysis."""
    return "This is a placeholder for the last run analysis results."

if __name__ == "__main__":
    mcp.run()