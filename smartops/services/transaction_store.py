"""Transaction ledger with TTL + optional disk persistence (survives restart)."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal


@dataclass
class TransactionRecord:
    transaction_id: str
    query: str
    answer: str
    llm: str
    top_k: int
    context: str
    action_key: str
    latency_seconds: float
    created_at: float
    used_fallback_llm: bool = False
    agent_trace: dict[str, Any] = field(default_factory=dict)
    feedback_score: int | None = None
    reward: float | None = None
    feedback_reserved: bool = False


class TransactionStore:
    def __init__(self, ttl_seconds: int = 3600, state_path: str | None = None):
        self.ttl_seconds = ttl_seconds
        self.state_path = state_path
        self._lock = threading.RLock()
        self._items: dict[str, TransactionRecord] = {}
        self._load()

    def put(self, record: TransactionRecord) -> None:
        with self._lock:
            self._purge_locked()
            self._items[record.transaction_id] = record
            self._save_locked()

    def get(self, transaction_id: str) -> TransactionRecord | None:
        with self._lock:
            self._purge_locked()
            return self._items.get(transaction_id)

    def reserve_feedback(
        self, transaction_id: str
    ) -> Literal["ok", "missing", "already_scored", "expired"]:
        """Atomically claim a transaction for feedback to prevent double-apply races."""
        with self._lock:
            self._purge_locked()
            rec = self._items.get(transaction_id)
            if rec is None:
                return "missing"
            if rec.feedback_score is not None or rec.feedback_reserved:
                return "already_scored"
            if time.time() - rec.created_at > self.ttl_seconds:
                self._items.pop(transaction_id, None)
                self._save_locked()
                return "expired"
            rec.feedback_reserved = True
            self._save_locked()
            return "ok"

    def release_feedback_reservation(self, transaction_id: str) -> None:
        with self._lock:
            rec = self._items.get(transaction_id)
            if rec is not None and rec.feedback_score is None:
                rec.feedback_reserved = False
                self._save_locked()

    def mark_feedback(self, transaction_id: str, score: int, reward: float) -> TransactionRecord:
        with self._lock:
            rec = self._items.get(transaction_id)
            if rec is None:
                raise KeyError(transaction_id)
            if rec.feedback_score is not None:
                raise ValueError("Feedback already recorded for this transaction_id")
            rec.feedback_score = score
            rec.reward = reward
            rec.feedback_reserved = False
            self._save_locked()
            return rec

    def _purge_locked(self) -> None:
        """Expire both scored and unscored transactions past TTL."""
        now = time.time()
        expired = [
            tid
            for tid, rec in self._items.items()
            if now - rec.created_at > self.ttl_seconds
        ]
        if not expired:
            return
        for tid in expired:
            self._items.pop(tid, None)
        self._save_locked()

    def _load(self) -> None:
        if not self.state_path:
            return
        path = Path(self.state_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for tid, payload in (raw.get("items") or {}).items():
                known = {f.name for f in fields(TransactionRecord)}
                filtered = {k: v for k, v in payload.items() if k in known}
                self._items[tid] = TransactionRecord(**filtered)
            self._purge_locked()
        except Exception:  # noqa: BLE001
            self._items = {}

    def _save_locked(self) -> None:
        if not self.state_path:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": {tid: asdict(rec) for tid, rec in self._items.items()},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
