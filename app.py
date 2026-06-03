"""
app.py — Flask web application for the Policy RAG Assistant.
Endpoints:
  GET  /          — Chat UI
  POST /chat      — RAG Q&A API
  GET  /health    — Health check
"""

import os
import sys
import logging
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from rag.ingestion import build_index, get_collection
from rag.retrieval import retrieve, rerank
from rag.generation import call_llm, extract_citations

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
POLICIES_DIR = os.getenv("POLICIES_DIR", "./data/policies")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# ── App init ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

# Lazy-loaded collection (initialised on first request or at startup)
_collection = None


def get_or_build_collection():
    global _collection
    if _collection is None:
        _collection = build_index(
            policies_dir=POLICIES_DIR,
            chroma_db_path=CHROMA_DB_PATH,
            embedding_model_name=EMBEDDING_MODEL,
        )
    return _collection


# Pre-warm on startup (non-fatal if policies dir is empty during CI checks)
try:
    get_or_build_collection()
    logger.info("Vector store ready.")
except Exception as e:
    logger.warning("Could not pre-build index at startup: %s", e)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health check."""
    return jsonify({"status": "ok", "model": LLM_MODEL})


@app.route("/chat", methods=["POST"])
def chat():
    """
    RAG Q&A endpoint.
    Request JSON: { "question": "..." }
    Response JSON: { "answer": "...", "citations": [...], "latency_ms": ... }
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field."}), 400

    if len(question) > 1000:
        return jsonify({"error": "Question too long (max 1000 chars)."}), 400

    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OPENROUTER_API_KEY not configured."}), 500

    try:
        collection = get_or_build_collection()
    except Exception as e:
        logger.error("Collection unavailable: %s", e)
        return jsonify({"error": "Knowledge base not available."}), 500

    # Retrieve
    t0 = time.perf_counter()
    chunks = retrieve(
        query=question,
        collection=collection,
        top_k=TOP_K,
        embedding_model_name=EMBEDDING_MODEL,
    )
    chunks = rerank(question, chunks)

    # Generate
    try:
        answer, llm_latency = call_llm(
            question=question,
            chunks=chunks,
            api_key=OPENROUTER_API_KEY,
            model=LLM_MODEL,
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return jsonify({"error": f"LLM error: {str(e)}"}), 502

    total_latency_ms = round((time.perf_counter() - t0) * 1000)

    # Extract citations
    citations = extract_citations(answer, chunks)

    return jsonify(
        {
            "answer": answer,
            "citations": citations,
            "latency_ms": total_latency_ms,
            "model": LLM_MODEL,
            "chunks_retrieved": len(chunks),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
