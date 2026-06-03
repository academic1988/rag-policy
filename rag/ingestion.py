"""
ingestion.py — Parse, chunk, embed, and index policy documents into ChromaDB.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Fixed seed for deterministic behaviour
RANDOM_SEED = 42


def load_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def load_text(path: str) -> str:
    """Load a plain text / markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(path: str) -> Dict[str, Any]:
    """Dispatch to the correct loader based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        text = load_pdf(path)
    elif ext in {".txt", ".md"}:
        text = load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Clean: collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return {
        "text": text.strip(),
        "source": Path(path).name,
        "title": Path(path).stem.replace("-", " ").replace("_", " ").title(),
    }


def chunk_documents(
    docs: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping chunks.
    Uses RecursiveCharacterTextSplitter (token-aware via tiktoken).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            chunks.append(
                {
                    "text": part,
                    "source": doc["source"],
                    "title": doc["title"],
                    "chunk_id": f"{doc['source']}::chunk{i}",
                }
            )
    logger.info("Created %d chunks from %d documents", len(chunks), len(docs))
    return chunks


def build_index(
    policies_dir: str,
    chroma_db_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Full ingestion pipeline:
    1. Load all PDFs/txt/md from policies_dir
    2. Chunk them
    3. Embed with SentenceTransformer
    4. Store in ChromaDB
    Returns the ChromaDB collection.
    """
    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = "policies"

    # Optionally wipe and rebuild
    if force_rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception:
            pass

    # Return existing collection if already built
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing and not force_rebuild:
        logger.info("Collection '%s' already exists — skipping rebuild.", collection_name)
        return client.get_collection(collection_name)

    # Load documents
    policy_path = Path(policies_dir)
    supported = {".pdf", ".txt", ".md"}
    files = [f for f in policy_path.iterdir() if f.suffix.lower() in supported]
    if not files:
        raise FileNotFoundError(f"No supported documents found in {policies_dir}")

    logger.info("Loading %d documents from %s", len(files), policies_dir)
    docs = []
    for f in files:
        try:
            docs.append(load_document(str(f)))
        except Exception as e:
            logger.warning("Skipping %s: %s", f.name, e)

    # Chunk
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Embed
    logger.info("Loading embedding model: %s", embedding_model_name)
    model = SentenceTransformer(embedding_model_name)
    texts = [c["text"] for c in chunks]

    logger.info("Embedding %d chunks…", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Index into ChromaDB
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[i : i + batch_size],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "title": c["title"]} for c in batch],
        )

    logger.info("Indexed %d chunks into collection '%s'", len(chunks), collection_name)
    return collection


def get_collection(chroma_db_path: str) -> chromadb.Collection:
    """Load an existing ChromaDB collection (no rebuild)."""
    client = chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("policies")
