"""
tests/test_app.py — Smoke tests for the Flask application and core modules.
"""

import sys
import os
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Import check ──────────────────────────────────────────────────────────────

def test_imports():
    """Verify all core modules can be imported without error."""
    import app as flask_app
    from rag.ingestion import load_document, chunk_documents
    from rag.retrieval import retrieve
    from rag.generation import build_context, extract_citations
    assert True


# ── Flask app tests ────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
    os.environ.setdefault("POLICIES_DIR", "./data/policies")
    os.environ.setdefault("CHROMA_DB_PATH", "./data/chroma_db")

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "model" in data


def test_index_route(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Policy Assistant" in resp.data


def test_chat_missing_question(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_chat_empty_question(client):
    resp = client.post("/chat", json={"question": "  "})
    assert resp.status_code == 400


def test_chat_question_too_long(client):
    resp = client.post("/chat", json={"question": "x" * 1001})
    assert resp.status_code == 400


# ── Ingestion unit tests ───────────────────────────────────────────────────────

def test_chunk_documents():
    from rag.ingestion import chunk_documents
    docs = [{"text": "Hello world. " * 100, "source": "test.txt", "title": "Test"}]
    chunks = chunk_documents(docs, chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert "source" in c
        assert "chunk_id" in c


def test_load_text_document(tmp_path):
    from rag.ingestion import load_document
    f = tmp_path / "test.txt"
    f.write_text("This is a test policy document.\n\nSection 1: Overview\nThis is the overview.")
    doc = load_document(str(f))
    assert "text" in doc
    assert len(doc["text"]) > 10
    assert doc["source"] == "test.txt"


# ── Generation unit tests ──────────────────────────────────────────────────────

def test_build_context():
    from rag.generation import build_context
    chunks = [
        {"text": "Employees get 15 PTO days.", "source": "pto.txt", "title": "PTO Policy", "score": 0.9},
        {"text": "Carry over up to 5 days.",   "source": "pto.txt", "title": "PTO Policy", "score": 0.8},
    ]
    ctx = build_context(chunks)
    assert "PTO Policy" in ctx
    assert "15 PTO days" in ctx


def test_extract_citations_fallback():
    from rag.generation import extract_citations
    chunks = [
        {"text": "Some text here.", "source": "policy.txt", "title": "HR Policy", "score": 0.9},
    ]
    # Answer doesn't explicitly cite, should fall back to top chunks
    cits = extract_citations("Some answer that doesn't mention the source.", chunks)
    assert len(cits) >= 1
