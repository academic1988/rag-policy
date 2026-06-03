#!/usr/bin/env python3
"""
build_index.py — Rebuild the ChromaDB vector index from scratch.
Run: python scripts/build_index.py [--force]
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from rag.ingestion import build_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    policies_dir = os.getenv("POLICIES_DIR", "./data/policies")
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    print(f"Policies dir: {policies_dir}")
    print(f"ChromaDB path: {chroma_db_path}")
    print(f"Embedding model: {embedding_model}")
    print(f"Force rebuild: {args.force}")
    print()

    collection = build_index(
        policies_dir=policies_dir,
        chroma_db_path=chroma_db_path,
        embedding_model_name=embedding_model,
        force_rebuild=args.force,
    )

    count = collection.count()
    print(f"\n✅ Index ready — {count} chunks in collection.")

if __name__ == "__main__":
    main()
