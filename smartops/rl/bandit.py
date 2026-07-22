"""Epsilon-greedy contextual bandit for LLM + RAG hyperparameter routing.

State space:  query category × complexity (see features.build_context_key)
Action space: (llm_model, top_k) pairs
Update rule:  incremental sample-mean estimates per (context, action)
Reward:       (feedback * 10) - min(latency, latency_cap)
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from smartops.config import Settings
from smartops.core.logging import get_logger
from smartops.core.metrics import BANDIT_ARM_PULLS, BANDIT_EPSILON, BANDIT_REWARD
from smartops.rl.rewards import compute_reward

logger = get_logger(__name__)


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.total_reward += reward


@dataclass
class BanditAction:
    """One discrete action in the policy."""

    llm: str
    top_k: int

    @property
    def key(self) -> str:
        return f"{self.llm}::k{self.top_k}"

    @classmethod
    def from_key(cls, key: str) -> "BanditAction":
        llm, kpart = key.rsplit("::k", 1)
        return cls(llm=llm, top_k=int(kpart))


@dataclass
class PendingDecision:
    transaction_id: str
    context: str
    action_key: str
    latency_seconds: float
    created_at: float


class ContextualBandit:
    """Thread-safe epsilon-greedy contextual bandit with JSON (+ optional Redis) persistence."""

    REDIS_KEY = "smartops:bandit:state"

    def __init__(
        self,
        settings: Settings,
        actions: list[BanditAction] | None = None,
        redis_client: Any | None = None,
    ):
        self.settings = settings
        self.actions = actions or self._default_actions(settings)
        self.redis = redis_client
        self._lock = threading.RLock()
        self.epsilon = settings.rl_epsilon
        self.arms: dict[str, dict[str, ArmStats]] = {}
        self.total_updates = 0
        self._pending: dict[str, PendingDecision] = {}
        self._load()
        BANDIT_EPSILON.set(self.epsilon)

    @staticmethod
    def _default_actions(settings: Settings) -> list[BanditAction]:
        top_ks = (2, 5)
        return [BanditAction(llm=m, top_k=k) for m in settings.llm_models for k in top_ks]

    def _ensure_context(self, context: str) -> dict[str, ArmStats]:
        if context not in self.arms:
            self.arms[context] = {a.key: ArmStats() for a in self.actions}
        else:
            for a in self.actions:
                self.arms[context].setdefault(a.key, ArmStats())
        return self.arms[context]

    def _purge_expired_pending_locked(self) -> int:
        timeout = max(int(self.settings.rl_feedback_timeout_seconds), 1)
        now = time.time()
        expired = [
            tid
            for tid, pending in self._pending.items()
            if now - pending.created_at > timeout
        ]
        for tid in expired:
            self._pending.pop(tid, None)
        return len(expired)

    def select_action(self, context: str) -> BanditAction:
        with self._lock:
            self._sync_from_redis_locked()
            self._purge_expired_pending_locked()
            arms = self._ensure_context(context)
            explore = random.random() < self.epsilon
            if explore:
                action = random.choice(self.actions)
                policy = "explore"
            else:
                # Prefer higher mean; among equals prefer fewer pulls (explore under-sampled arms)
                best_key = max(
                    arms.keys(),
                    key=lambda k: (arms[k].mean_reward, -arms[k].pulls, random.random()),
                )
                action = BanditAction.from_key(best_key)
                policy = "exploit"

            BANDIT_ARM_PULLS.labels(context=context, action=action.key).inc()
            logger.info(
                "bandit_select",
                context=context,
                action=action.key,
                policy=policy,
                epsilon=round(self.epsilon, 4),
                mean=round(arms[action.key].mean_reward, 4),
            )
            return action

    def register_pending(
        self,
        transaction_id: str,
        context: str,
        action: BanditAction,
        latency_seconds: float,
        created_at: float,
    ) -> None:
        with self._lock:
            self._mutate_with_redis_retry(
                lambda: self._register_pending_unlocked(
                    transaction_id, context, action, latency_seconds, created_at
                )
            )

    def _register_pending_unlocked(
        self,
        transaction_id: str,
        context: str,
        action: BanditAction,
        latency_seconds: float,
        created_at: float,
    ) -> None:
        self._purge_expired_pending_locked()
        self._pending[transaction_id] = PendingDecision(
            transaction_id=transaction_id,
            context=context,
            action_key=action.key,
            latency_seconds=latency_seconds,
            created_at=created_at,
        )

    def apply_feedback(self, transaction_id: str, feedback_score: int) -> dict[str, Any]:
        with self._lock:
            result_box: dict[str, Any] = {}

            def _mutate() -> None:
                self._purge_expired_pending_locked()
                pending = self._pending.pop(transaction_id, None)
                if pending is None:
                    raise KeyError(
                        f"Unknown, expired, or already-scored transaction_id: {transaction_id}"
                    )

                reward = compute_reward(
                    feedback_score,
                    pending.latency_seconds,
                    latency_cap_seconds=self.settings.rl_latency_cap_seconds,
                )
                arms = self._ensure_context(pending.context)
                arms.setdefault(pending.action_key, ArmStats()).update(reward)
                self.total_updates += 1

                self.epsilon = max(
                    self.settings.rl_epsilon_min,
                    self.epsilon * self.settings.rl_epsilon_decay,
                )
                BANDIT_EPSILON.set(self.epsilon)
                BANDIT_REWARD.labels(context=pending.context, action=pending.action_key).observe(reward)

                result_box.update(
                    {
                        "transaction_id": transaction_id,
                        "context": pending.context,
                        "action": pending.action_key,
                        "latency_seconds": pending.latency_seconds,
                        "latency_cap_seconds": self.settings.rl_latency_cap_seconds,
                        "feedback_score": int(feedback_score),
                        "reward": reward,
                        "arm_mean_reward": arms[pending.action_key].mean_reward,
                        "arm_pulls": arms[pending.action_key].pulls,
                        "epsilon": self.epsilon,
                    }
                )

            self._mutate_with_redis_retry(_mutate)
            logger.info("bandit_update", **result_box)
            return result_box

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            purged = self._purge_expired_pending_locked()
            if purged:
                self._save()
            return {
                "epsilon": self.epsilon,
                "total_updates": self.total_updates,
                "actions": [a.key for a in self.actions],
                "latency_cap_seconds": self.settings.rl_latency_cap_seconds,
                "feedback_timeout_seconds": self.settings.rl_feedback_timeout_seconds,
                "arms": {
                    ctx: {
                        ak: {
                            "pulls": st.pulls,
                            "total_reward": st.total_reward,
                            "mean_reward": st.mean_reward,
                        }
                        for ak, st in arms.items()
                    }
                    for ctx, arms in self.arms.items()
                },
                "pending_transactions": len(self._pending),
            }

    def _sync_from_redis_locked(self) -> None:
        if self.redis is None:
            return
        try:
            blob = self.redis.get(self.REDIS_KEY)
            if blob:
                self._apply_payload(json.loads(blob))
        except Exception as exc:  # noqa: BLE001
            logger.warning("bandit_redis_sync_failed", error=str(exc))

    def _mutate_with_redis_retry(self, mutate_fn) -> None:
        """Apply local mutation after optional Redis WATCH reload (best-effort CAS)."""
        if self.redis is None:
            mutate_fn()
            self._save()
            return

        last_error: Exception | None = None
        for _ in range(5):
            try:
                self.redis.watch(self.REDIS_KEY)
                blob = self.redis.get(self.REDIS_KEY)
                if blob:
                    self._apply_payload(json.loads(blob))
                mutate_fn()
                payload_blob = json.dumps(self._payload(), indent=2)
                pipe = self.redis.pipeline()
                pipe.multi()
                pipe.set(self.REDIS_KEY, payload_blob)
                pipe.execute()
                # Keep file as durable backup
                path = Path(self.settings.rl_state_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(payload_blob, encoding="utf-8")
                tmp.replace(path)
                return
            except KeyError:
                try:
                    self.redis.unwatch()
                except Exception:  # noqa: BLE001
                    pass
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                try:
                    self.redis.unwatch()
                except Exception:  # noqa: BLE001
                    pass
                continue
        logger.warning("bandit_redis_cas_failed", error=str(last_error))
        mutate_fn()
        self._save()

    def _payload(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "arms": {
                ctx: {ak: asdict(st) for ak, st in arms.items()}
                for ctx, arms in self.arms.items()
            },
            "pending": {
                tid: asdict(pending) for tid, pending in self._pending.items()
            },
        }

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        self.epsilon = float(payload.get("epsilon", self.settings.rl_epsilon))
        self.total_updates = int(payload.get("total_updates", 0))
        raw_arms = payload.get("arms", {})
        self.arms = {
            ctx: {
                ak: ArmStats(**st) if isinstance(st, dict) else ArmStats()
                for ak, st in arms.items()
            }
            for ctx, arms in raw_arms.items()
        }
        raw_pending = payload.get("pending", {}) or {}
        self._pending = {
            tid: PendingDecision(**item)
            for tid, item in raw_pending.items()
            if isinstance(item, dict)
        }
        self._purge_expired_pending_locked()

    def _save(self) -> None:
        payload = self._payload()
        path = Path(self.settings.rl_state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        blob = json.dumps(payload, indent=2)
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(path)
        if self.redis is not None:
            try:
                self.redis.set(self.REDIS_KEY, blob)
            except Exception as exc:  # noqa: BLE001
                logger.warning("bandit_redis_save_failed", error=str(exc))

    def _load(self) -> None:
        # Prefer Redis shared state when available, else file.
        if self.redis is not None:
            try:
                blob = self.redis.get(self.REDIS_KEY)
                if blob:
                    self._apply_payload(json.loads(blob))
                    logger.info(
                        "bandit_state_loaded",
                        source="redis",
                        updates=self.total_updates,
                        pending=len(self._pending),
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning("bandit_redis_load_failed", error=str(exc))

        path = Path(self.settings.rl_state_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._apply_payload(payload)
            logger.info(
                "bandit_state_loaded",
                path=str(path),
                source="file",
                updates=self.total_updates,
                pending=len(self._pending),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("bandit_state_load_failed", error=str(exc))
