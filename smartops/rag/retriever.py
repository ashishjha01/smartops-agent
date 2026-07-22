"""RAG retrieval helpers."""

from __future__ import annotations

from smartops.rag.store import RetrievedChunk, VectorStore


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant documentation found."
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] source={c.source} score={c.score}\n{c.text}")
    return "\n\n".join(blocks)


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self.store.query(query, top_k=top_k)
