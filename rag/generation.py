"""
generation.py — Build prompts, call the LLM via OpenRouter, parse responses.
"""

import os
import logging
import time
from typing import List, Dict, Any, Tuple

import requests

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/free")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

SYSTEM_PROMPT = """You are a helpful HR and company-policy assistant.
You ONLY answer questions using the provided policy excerpts.
Rules you must follow:
1. If the answer is not found in the excerpts, respond: "I can only answer questions about our company policies. I could not find relevant information for your question."
2. Always cite the source document title(s) for every claim you make, using the format [Source: <title>].
3. Keep answers clear and concise (max ~300 words).
4. Do not speculate or add information beyond what the excerpts contain.
5. If multiple policies are relevant, synthesise them clearly."""

CONTEXT_TEMPLATE = """Policy Excerpts:
{context}

User Question: {question}

Answer (with citations):"""


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Title: {chunk['title']}\nSource: {chunk['source']}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(
    question: str,
    chunks: List[Dict[str, Any]],
    api_key: str,
    model: str = LLM_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> Tuple[str, float]:
    """
    Call the OpenRouter chat completion API.
    Returns (answer_text, latency_seconds).
    """
    context = build_context(chunks)
    user_message = CONTEXT_TEMPLATE.format(context=context, question=question)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://policy-rag-app.onrender.com",
        "X-Title": "Policy RAG Assistant",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,  # Low temperature for factual answers
    }

    t0 = time.perf_counter()
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    latency = time.perf_counter() - t0

    response.raise_for_status()
    data = response.json()

    answer = data["choices"][0]["message"]["content"].strip()
    logger.info("LLM latency: %.2fs, model: %s", latency, model)
    return answer, latency


def extract_citations(
    answer: str, chunks: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Parse [Source: <title>] references in the answer and match them
    back to the retrieved chunks, returning snippet previews.
    """
    citations = []
    seen_titles = set()
    for chunk in chunks:
        title = chunk["title"]
        # Check if LLM cited this document
        if title.lower() in answer.lower() or chunk["source"].lower() in answer.lower():
            if title not in seen_titles:
                seen_titles.add(title)
                snippet = chunk["text"][:200].replace("\n", " ") + "…"
                citations.append(
                    {
                        "title": title,
                        "source": chunk["source"],
                        "snippet": snippet,
                        "score": chunk["score"],
                    }
                )
    # If no explicit citations found, fall back to top-scored chunks
    if not citations:
        for chunk in chunks[:2]:
            title = chunk["title"]
            if title not in seen_titles:
                seen_titles.add(title)
                snippet = chunk["text"][:200].replace("\n", " ") + "…"
                citations.append(
                    {
                        "title": title,
                        "source": chunk["source"],
                        "snippet": snippet,
                        "score": chunk["score"],
                    }
                )
    return citations
