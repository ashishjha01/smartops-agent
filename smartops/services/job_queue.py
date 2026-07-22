"""In-process async job queue for long-running LLM queries."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from smartops.core.logging import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    query: str
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "query": self.query,
            "result": self.result,
            "error": self.error,
        }


class JobQueue:
    """Bounded asyncio worker pool backed by an in-memory job ledger."""

    def __init__(self, workers: int = 2, max_jobs: int = 1000):
        self.workers = max(1, int(workers))
        self.max_jobs = max(10, int(max_jobs))
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._worker_tasks: list[asyncio.Task] = []
        self._handler: Callable[[str], Awaitable[dict[str, Any]]] | None = None
        self._started = False

    def set_handler(self, handler: Callable[[str], Awaitable[dict[str, Any]]]) -> None:
        self._handler = handler

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for idx in range(self.workers):
            self._worker_tasks.append(asyncio.create_task(self._worker_loop(idx), name=f"job-worker-{idx}"))
        logger.info("job_queue_started", workers=self.workers)

    async def stop(self) -> None:
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._started = False
        logger.info("job_queue_stopped")

    async def enqueue(self, query: str) -> JobRecord:
        if len(self._jobs) >= self.max_jobs:
            # Drop oldest terminal jobs
            terminal = [
                jid
                for jid, job in self._jobs.items()
                if job.status in {JobStatus.completed, JobStatus.failed}
            ]
            terminal.sort(key=lambda jid: self._jobs[jid].updated_at)
            for jid in terminal[: max(1, len(terminal) // 4)]:
                self._jobs.pop(jid, None)
            if len(self._jobs) >= self.max_jobs:
                raise RuntimeError("Job queue is full; retry later")

        now = time.time()
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            created_at=now,
            updated_at=now,
            query=query,
        )
        self._jobs[job_id] = record
        await self._queue.put(job_id)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None or self._handler is None:
                self._queue.task_done()
                continue
            job.status = JobStatus.running
            job.updated_at = time.time()
            try:
                result = await self._handler(job.query)
                job.result = result
                job.status = JobStatus.completed
            except Exception as exc:  # noqa: BLE001
                logger.exception("job_failed", job_id=job_id, worker=worker_id)
                job.error = str(exc)
                job.status = JobStatus.failed
            job.updated_at = time.time()
            self._queue.task_done()
