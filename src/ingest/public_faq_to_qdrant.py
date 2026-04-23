from __future__ import annotations

import argparse
from dataclasses import dataclass

from datasets import load_dataset
from qdrant_client import QdrantClient, models

from src.config import load_config
from src.retrieval.embeddings import hash_embed


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    chunk_index: int


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        part = words[start : start + chunk_size]
        if not part:
            continue
        chunks.append(" ".join(part))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_public_chunks(dataset_name: str, split: str, limit: int, chunk_size: int, overlap: int) -> list[Chunk]:
    ds = load_dataset(dataset_name, split=split)
    rows = ds.select(range(min(limit, len(ds))))
    out: list[Chunk] = []
    for row in rows:
        context = str(row.get("context", "")).strip()
        title = str(row.get("title", "unknown")).strip() or "unknown"
        if not context:
            continue
        chunks = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            out.append(
                Chunk(
                    text=chunk,
                    source=f"{dataset_name}:{split}",
                    title=title,
                    chunk_index=idx,
                )
            )
    return out


def ingest(chunks: list[Chunk], qdrant_url: str, collection: str, vector_size: int, recreate: bool) -> int:
    client = QdrantClient(url=qdrant_url, timeout=10.0)
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection_name=collection)
    if recreate or not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    points: list[models.PointStruct] = []
    for idx, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=idx,
                vector=hash_embed(chunk.text, dim=vector_size),
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                },
            )
        )
    client.upsert(collection_name=collection, points=points)
    return len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest public QA docs into Qdrant with chunking.")
    parser.add_argument("--dataset", default="squad", help="HuggingFace public dataset name.")
    parser.add_argument("--split", default="train[:120]", help="Dataset split selector.")
    parser.add_argument("--limit", type=int, default=120, help="Max rows before chunking.")
    parser.add_argument("--chunk-size", type=int, default=120, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=30, help="Chunk overlap in words.")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection before ingestion.")
    args = parser.parse_args()

    cfg = load_config()
    chunks = load_public_chunks(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    total = ingest(
        chunks=chunks,
        qdrant_url=cfg.qdrant.url,
        collection=cfg.qdrant.collection,
        vector_size=cfg.qdrant.vector_size,
        recreate=args.recreate,
    )
    print(f"Ingested {total} chunks into '{cfg.qdrant.collection}' from {args.dataset}:{args.split}.")


if __name__ == "__main__":
    main()
