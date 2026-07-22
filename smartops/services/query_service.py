"""Application service wiring query orchestration."""

from __future__ import annotations

import time
import uuid
from typing import Any

from smartops.agent.react import ReActAgent
from smartops.core.async_utils import run_sync
from smartops.core.logging import get_logger
from smartops.core.metrics import QUERY_LATENCY, QUERY_REQUESTS
from smartops.rl.bandit import BanditAction, ContextualBandit
from smartops.rl.features import build_context_key, categorize_query, estimate_complexity
from smartops.services.audit_store import AuditStore
from smartops.services.transaction_store import TransactionRecord, TransactionStore

logger = get_logger(__name__)


class QueryService:
    def __init__(
        self,
        agent: ReActAgent,
        bandit: ContextualBandit,
        transactions: TransactionStore,
        audit: AuditStore | None = None,
    ):
        self.agent = agent
        self.bandit = bandit
        self.transactions = transactions
        self.audit = audit

    async def handle_query(self, query: str) -> dict[str, Any]:
        started = time.perf_counter()
        context = build_context_key(query)
        action: BanditAction = await run_sync(self.bandit.select_action, context)

        try:
            result = await self.agent.run(query, model=action.llm, top_k=action.top_k)
            latency = time.perf_counter() - started
            txn_id = str(uuid.uuid4())
            created = time.time()

            record = TransactionRecord(
                transaction_id=txn_id,
                query=query,
                answer=result.answer,
                llm=action.llm,
                top_k=action.top_k,
                context=context,
                action_key=action.key,
                latency_seconds=latency,
                created_at=created,
                used_fallback_llm=result.trace.used_fallback_llm,
                agent_trace={
                    "thought": result.trace.thought,
                    "action": result.trace.action,
                    "action_input": result.trace.action_input,
                    "tool_results": result.trace.tool_results,
                    "steps": result.trace.steps,
                    "retrieved_sources": [
                        {"source": c.source, "score": c.score, "chunk_id": c.chunk_id}
                        for c in result.trace.retrieved
                    ],
                },
            )
            await run_sync(self.transactions.put, record)
            await run_sync(self.bandit.register_pending, txn_id, context, action, latency, created)

            QUERY_REQUESTS.labels(status="ok", llm=action.llm, action=action.key).inc()
            QUERY_LATENCY.labels(llm=action.llm, action=action.key).observe(latency)

            payload = {
                "transaction_id": txn_id,
                "answer": result.answer,
                "llm": action.llm,
                "latency_seconds": round(latency, 4),
                "routing": {
                    "context": context,
                    "category": categorize_query(query).value,
                    "complexity": estimate_complexity(query).value,
                    "action": action.key,
                    "top_k": action.top_k,
                    "epsilon": round(self.bandit.epsilon, 4),
                },
                "agent": record.agent_trace,
                "used_fallback_llm": result.trace.used_fallback_llm,
            }
            if self.audit is not None:
                await run_sync(
                    self.audit.record,
                    "query",
                    txn_id,
                    {
                        "query": query,
                        "llm": action.llm,
                        "action": action.key,
                        "latency_seconds": payload["latency_seconds"],
                    },
                )
            return payload
        except Exception:
            QUERY_REQUESTS.labels(status="error", llm=action.llm, action=action.key).inc()
            logger.exception("query_failed", context=context, action=action.key)
            raise

    async def handle_feedback(self, transaction_id: str, score: int) -> dict[str, Any]:
        # Mark first under txn lock so concurrent duplicates get 409 before bandit mutates.
        reserved = await run_sync(self.transactions.reserve_feedback, transaction_id)
        if reserved == "missing":
            raise KeyError(f"Unknown transaction_id: {transaction_id}")
        if reserved == "already_scored":
            raise ValueError("Feedback already recorded for this transaction_id")
        if reserved == "expired":
            raise KeyError(f"Feedback window expired for transaction_id: {transaction_id}")

        try:
            update = await run_sync(self.bandit.apply_feedback, transaction_id, score)
        except KeyError as exc:
            # Pending expired while transaction still present
            await run_sync(self.transactions.release_feedback_reservation, transaction_id)
            detail = str(exc)
            if "expired" in detail.lower() or "already-scored" in detail.lower() or "Unknown" in detail:
                raise KeyError(f"Feedback window expired for transaction_id: {transaction_id}") from exc
            raise

        await run_sync(self.transactions.mark_feedback, transaction_id, score, update["reward"])
        result = {
            "status": "updated",
            "transaction_id": transaction_id,
            "feedback_score": score,
            "reward": update["reward"],
            "rl": update,
        }
        if self.audit is not None:
            await run_sync(
                self.audit.record,
                "feedback",
                transaction_id,
                {"feedback_score": score, "reward": update["reward"], "action": update.get("action")},
            )
        return result
