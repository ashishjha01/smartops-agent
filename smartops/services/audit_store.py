"""Optional durable audit log for queries/feedback (Postgres or JSONL file)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from smartops.core.logging import get_logger

logger = get_logger(__name__)


class AuditStore:
    """Append-only audit trail used for production feedback/query accountability."""

    def __init__(self, database_url: str = "", file_path: str = "./data/audit.jsonl"):
        self.database_url = (database_url or "").strip()
        self.file_path = file_path
        self._lock = threading.RLock()
        self._pg = None
        if self.database_url.startswith("postgres"):
            self._init_postgres()
        else:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_postgres(self) -> None:
        try:
            import psycopg

            self._pg = psycopg.connect(self.database_url, autocommit=True)
            with self._pg.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS smartops_audit (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        event_type TEXT NOT NULL,
                        transaction_id TEXT,
                        payload JSONB NOT NULL
                    )
                    """
                )
            logger.info("audit_postgres_ready")
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit_postgres_unavailable", error=str(exc))
            self._pg = None
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    def record(self, event_type: str, transaction_id: str | None, payload: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event_type": event_type,
            "transaction_id": transaction_id,
            "payload": payload,
        }
        with self._lock:
            if self._pg is not None:
                try:
                    with self._pg.cursor() as cur:
                        cur.execute(
                            "INSERT INTO smartops_audit (event_type, transaction_id, payload) "
                            "VALUES (%s, %s, %s::jsonb)",
                            (event_type, transaction_id, json.dumps(payload)),
                        )
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("audit_postgres_write_failed", error=str(exc))
            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    def close(self) -> None:
        if self._pg is not None:
            try:
                self._pg.close()
            except Exception:  # noqa: BLE001
                pass
