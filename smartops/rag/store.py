"""Vector store and document chunking for the knowledge base."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from smartops.config import Settings
from smartops.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    chunk_id: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Character-based sliding window chunker with paragraph preference.

    Overlap is applied only when splitting oversized paragraphs, avoiding a
    second pass that previously duplicated content across chunk boundaries.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    overlap = max(0, min(overlap, chunk_size - 1)) if chunk_size > 1 else 0

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n\n{para}".strip() if current else para
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                start = max(end - overlap, start + 1)
            current = ""

    if current:
        chunks.append(current)
    return chunks


def knowledge_base_fingerprint(
    directory: str | Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    embedding_model: str | None = None,
) -> str:
    """Stable hash of KB contents + chunk/embed settings for re-ingest detection."""
    path = Path(directory)
    digest = hashlib.sha1()
    digest.update(f"chunk={chunk_size}|overlap={chunk_overlap}|embed={embedding_model}".encode())
    digest.update(b"\0")
    for file_path in sorted(path.glob("**/*")):
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        # Relative path so renames/moves across folders invalidate the index
        rel = str(file_path.relative_to(path)).replace("\\", "/")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class VectorStore:
    """Chroma-backed persistent vector store with local sentence-transformer embeddings."""

    COLLECTION = "smartops_kb"

    def __init__(self, settings: Settings):
        self.settings = settings
        persist = Path(settings.rag_persist_dir)
        persist.mkdir(parents=True, exist_ok=True)

        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
        )
        self._client = chromadb.PersistentClient(
            path=str(persist),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def _fingerprint_path(self) -> Path:
        return Path(self.settings.rag_persist_dir) / "kb_fingerprint.txt"

    def ingest_directory(self, directory: str | Path, force: bool = False) -> int:
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {path}")

        fingerprint = knowledge_base_fingerprint(
            path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            embedding_model=self.settings.embedding_model,
        )
        fp_file = self._fingerprint_path()
        previous = fp_file.read_text(encoding="utf-8").strip() if fp_file.exists() else ""
        force = force or bool(self.settings.rag_force_reingest) or (
            self.count > 0 and previous and previous != fingerprint
        )

        if self.count > 0 and not force and previous == fingerprint:
            logger.info("vector_store_skip_ingest", count=self.count, fingerprint=fingerprint[:12])
            return self.count

        if self.count > 0:
            self._client.delete_collection(self.COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for file_path in sorted(path.glob("**/*")):
            if file_path.suffix.lower() not in {".md", ".txt"}:
                continue
            content = file_path.read_text(encoding="utf-8")
            parts = chunk_text(content, self.settings.chunk_size, self.settings.chunk_overlap)
            for idx, part in enumerate(parts):
                digest = hashlib.sha1(f"{file_path}:{idx}:{part[:64]}".encode()).hexdigest()[:16]
                ids.append(digest)
                documents.append(part)
                metadatas.append({"source": str(file_path.name), "chunk_index": idx})

        if not documents:
            raise RuntimeError(f"No documents found under {path}")

        # Chroma add in batches
        batch = 64
        for i in range(0, len(documents), batch):
            self._collection.add(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

        fp_file.parent.mkdir(parents=True, exist_ok=True)
        fp_file.write_text(fingerprint, encoding="utf-8")
        logger.info(
            "vector_store_ingested",
            chunks=len(documents),
            files=len(set(m["source"] for m in metadatas)),
            fingerprint=fingerprint[:12],
            forced=force,
        )
        return len(documents)

    def query(self, text: str, top_k: int = 3) -> list[RetrievedChunk]:
        top_k = max(1, min(top_k, 10))
        result = self._collection.query(
            query_texts=[text],
            n_results=min(top_k, max(self.count, 1)),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        for doc, meta, dist, cid in zip(docs, metas, dists, ids):
            # cosine distance → similarity-ish score
            score = 1.0 - float(dist) if dist is not None else 0.0
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    score=round(score, 4),
                    chunk_id=str(cid),
                )
            )
        return chunks
