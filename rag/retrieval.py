"""
retrieval.py — Query the ChromaDB vector store and return ranked chunks.
"""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# Module-level cache so the model is only loaded once
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = 5,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve the top_k most similar chunks from ChromaDB.

    Returns a list of dicts with keys:
        text, source, title, score (cosine similarity, 0-1)
    """
    model = get_embedding_model(embedding_model_name)
    query_embedding = model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity
        similarity = 1.0 - dist
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "title": meta.get("title", "Unknown Policy"),
                "score": round(similarity, 4),
            }
        )

    logger.debug("Retrieved %d chunks for query: %s", len(chunks), query[:60])
    return chunks


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    model: Optional[SentenceTransformer] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Optional cross-encoder-style reranking using bi-encoder cosine similarity
    (lightweight; swap for a cross-encoder if latency budget allows).
    Already sorted by score from ChromaDB, so this is a no-op by default.
    Override with a cross-encoder model if desired.
    """
    return sorted(chunks, key=lambda c: c["score"], reverse=True)
