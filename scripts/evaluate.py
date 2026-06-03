#!/usr/bin/env python3
"""
evaluate.py — Evaluate the RAG pipeline on a fixed question set.
Metrics: Groundedness (LLM-as-judge), Citation Accuracy, Latency p50/p95.

Run: python scripts/evaluate.py
"""

import sys
import json
import time
import logging
import statistics
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rag.ingestion import build_index
from rag.retrieval import retrieve, rerank
from rag.generation import call_llm, extract_citations, build_context

logging.basicConfig(level=logging.WARNING)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ── Evaluation set ────────────────────────────────────────────────────────────
EVAL_SET = [
    # PTO
    {"id": "e01", "question": "How many PTO days do new employees get per year?",
     "gold": "15 days", "topic": "PTO"},
    {"id": "e02", "question": "Can I carry over unused PTO days to next year?",
     "gold": "Yes, up to 5 days", "topic": "PTO"},
    {"id": "e03", "question": "Will I be paid out for unused PTO if I resign?",
     "gold": "Yes, up to 10 days", "topic": "PTO"},

    # Remote Work
    {"id": "e04", "question": "How many days per week do I need to be in the office?",
     "gold": "Minimum 2 days", "topic": "Remote Work"},
    {"id": "e05", "question": "What is the home office equipment stipend?",
     "gold": "$500 one-time", "topic": "Remote Work"},
    {"id": "e06", "question": "What are core working hours?",
     "gold": "10 AM to 3 PM team primary time zone", "topic": "Remote Work"},

    # Expenses
    {"id": "e07", "question": "How long do I have to submit an expense claim?",
     "gold": "30 days", "topic": "Expenses"},
    {"id": "e08", "question": "What is the monthly internet reimbursement?",
     "gold": "Up to $50 per month", "topic": "Expenses"},
    {"id": "e09", "question": "Do I need a receipt for expenses over $25?",
     "gold": "Yes, an itemised receipt is required", "topic": "Expenses"},

    # Security
    {"id": "e10", "question": "How long should my password be?",
     "gold": "Minimum 14 characters", "topic": "Security"},
    {"id": "e11", "question": "How quickly must I report a security incident?",
     "gold": "Within 1 hour of discovery", "topic": "Security"},
    {"id": "e12", "question": "Is MFA required for company accounts?",
     "gold": "Yes, mandatory", "topic": "Security"},

    # Holidays
    {"id": "e13", "question": "How many floating holidays do employees get?",
     "gold": "2 floating holidays per year", "topic": "Holidays"},
    {"id": "e14", "question": "What happens if a holiday falls on a Sunday?",
     "gold": "The following Monday is observed", "topic": "Holidays"},

    # Parental Leave
    {"id": "e15", "question": "How much paid parental leave does the primary caregiver receive?",
     "gold": "16 weeks at 100% base salary", "topic": "Leave"},
    {"id": "e16", "question": "How many days of bereavement leave for an immediate family member?",
     "gold": "5 days paid", "topic": "Leave"},

    # Performance
    {"id": "e17", "question": "When are mid-year performance reviews conducted?",
     "gold": "July", "topic": "Performance"},
    {"id": "e18", "question": "What salary increase do I get with a rating of 5?",
     "gold": "6-10%", "topic": "Performance"},

    # L&D
    {"id": "e19", "question": "What is the annual L&D budget for individual contributors?",
     "gold": "$1,500", "topic": "L&D"},
    {"id": "e20", "question": "Do L&D budgets roll over to the next year?",
     "gold": "No, they do not roll over", "topic": "L&D"},

    # Out-of-scope (guardrail test)
    {"id": "e21", "question": "What is the stock price of Acme Corp?",
     "gold": "GUARDRAIL — should refuse", "topic": "Guardrail"},
    {"id": "e22", "question": "What is the capital of France?",
     "gold": "GUARDRAIL — should refuse", "topic": "Guardrail"},
]


GROUNDEDNESS_JUDGE_PROMPT = """You are evaluating whether an AI assistant's answer is grounded
in the provided context passages. Grounded means: every factual claim in the answer is
explicitly supported by the context, with no fabricated or contradicted information.

Context:
{context}

Answer:
{answer}

Is this answer fully grounded in the context? Reply with exactly one word: YES or NO, then a
brief reason on the same line. Example: "YES - all claims are present in the context."
"""


def judge_groundedness(answer: str, context: str, api_key: str) -> bool:
    """Use LLM-as-judge to evaluate groundedness."""
    prompt = GROUNDEDNESS_JUDGE_PROMPT.format(context=context[:3000], answer=answer)
    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return verdict.startswith("YES")
    except Exception as e:
        print(f"  Judge error: {e}")
        return False


def check_citation_accuracy(answer: str, citations: List[Dict], chunks: List[Dict]) -> bool:
    """
    Check that at least one citation points to a chunk that contains
    content relevant to the answer (heuristic: keyword overlap).
    """
    if not citations:
        return False
    answer_words = set(answer.lower().split())
    for cit in citations:
        cit_words = set(cit["snippet"].lower().split())
        overlap = len(answer_words & cit_words)
        if overlap > 5:
            return True
    return False


def partial_match(answer: str, gold: str) -> bool:
    """True if any key phrase from gold appears in answer (case-insensitive)."""
    if "GUARDRAIL" in gold:
        refuse_phrases = ["can only answer", "not find", "cannot find", "don't have", "outside"]
        return any(p in answer.lower() for p in refuse_phrases)
    gold_tokens = gold.lower().split()
    answer_lower = answer.lower()
    return sum(1 for t in gold_tokens if t in answer_lower) >= len(gold_tokens) // 2


def main():
    print("=" * 60)
    print("Policy RAG — Evaluation Report")
    print("=" * 60)

    # Build / load index
    collection = build_index(
        policies_dir=os.getenv("POLICIES_DIR", "./data/policies"),
        chroma_db_path=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )

    results = []
    latencies = []

    for item in EVAL_SET:
        q = item["question"]
        print(f"\n[{item['id']}] {q}")

        # Retrieve
        chunks = retrieve(q, collection, top_k=5)
        chunks = rerank(q, chunks)
        context = build_context(chunks)

        # Generate
        t0 = time.perf_counter()
        answer, llm_latency = call_llm(q, chunks, OPENROUTER_API_KEY)
        total_lat = (time.perf_counter() - t0) * 1000
        latencies.append(total_lat)

        citations = extract_citations(answer, chunks)

        # Evaluate
        grounded = judge_groundedness(answer, context, OPENROUTER_API_KEY)
        cit_accurate = check_citation_accuracy(answer, citations, chunks)
        partial = partial_match(answer, item["gold"])

        print(f"  Answer:      {answer[:120]}…")
        print(f"  Grounded:    {'✅' if grounded else '❌'}")
        print(f"  Cit. Accur.: {'✅' if cit_accurate else '❌'}")
        print(f"  Partial Mtch:{'✅' if partial else '❌'}")
        print(f"  Latency:     {total_lat:.0f}ms")

        results.append({
            **item,
            "answer": answer,
            "citations": citations,
            "grounded": grounded,
            "citation_accurate": cit_accurate,
            "partial_match": partial,
            "latency_ms": total_lat,
        })

    # Aggregate
    n = len(results)
    n_grounded = sum(r["grounded"] for r in results)
    n_cit = sum(r["citation_accurate"] for r in results)
    n_partial = sum(r["partial_match"] for r in results)
    sorted_lat = sorted(latencies)
    p50 = statistics.median(sorted_lat)
    p95 = sorted_lat[int(0.95 * len(sorted_lat))]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated : {n}")
    print(f"Groundedness        : {n_grounded}/{n} = {n_grounded/n*100:.1f}%")
    print(f"Citation Accuracy   : {n_cit}/{n} = {n_cit/n*100:.1f}%")
    print(f"Partial Match       : {n_partial}/{n} = {n_partial/n*100:.1f}%")
    print(f"Latency p50         : {p50:.0f}ms")
    print(f"Latency p95         : {p95:.0f}ms")
    print("=" * 60)

    # Save JSON report
    report = {
        "summary": {
            "n": n,
            "groundedness_pct": round(n_grounded / n * 100, 1),
            "citation_accuracy_pct": round(n_cit / n * 100, 1),
            "partial_match_pct": round(n_partial / n * 100, 1),
            "latency_p50_ms": round(p50),
            "latency_p95_ms": round(p95),
        },
        "results": results,
    }

    out_path = Path("data/eval_report.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to {out_path}")


if __name__ == "__main__":
    main()
