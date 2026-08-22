"""Index the canonical knowledge contract into Qdrant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .chunker import KnowledgeChunker
from .retriever import VectorRetriever


DEFAULT_ENRICHMENT = Path("output/knowledge/knowledge_enrichment.json")


def load_canonical(enrichment_path: Path = DEFAULT_ENRICHMENT) -> Dict[str, Any]:
    document = json.loads(enrichment_path.read_text(encoding="utf-8"))
    canonical = document.get("canonical_metadata")
    if not isinstance(canonical, dict):
        raise ValueError("knowledge_enrichment.json does not contain canonical_metadata.")
    return canonical


def index_canonical(enrichment_path: Path = DEFAULT_ENRICHMENT) -> Dict[str, Any]:
    canonical = load_canonical(enrichment_path)
    chunks = KnowledgeChunker().build_chunks(canonical)
    retriever = VectorRetriever()
    retriever.connect()
    try:
        indexed = retriever.index(chunks)
        count = retriever.store.count()
    finally:
        retriever.close()
    return {
        "collection": retriever.store.collection_name,
        "chunks_built": len(chunks),
        "chunks_indexed": indexed,
        "collection_count": count,
        "embedding_model": retriever.embedder.model_name,
        "embedding_dimension": retriever.embedder.dimension,
    }


if __name__ == "__main__":
    print(json.dumps(index_canonical(), indent=2))
