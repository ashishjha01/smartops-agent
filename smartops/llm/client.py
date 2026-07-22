"""LLM client: Ollama with deterministic fallback for CI / offline demos."""

from __future__ import annotations

import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from smartops.config import Settings
from smartops.core.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.ollama_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {
                "ok": True,
                "reachable": True,
                "models": models,
                "fallback": False,
                "fallback_mode": self.settings.llm_fallback_mode,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": bool(self.settings.llm_fallback_mode),
                "reachable": False,
                "error": str(exc),
                "fallback": bool(self.settings.llm_fallback_mode),
                "fallback_mode": self.settings.llm_fallback_mode,
                "models": [],
            }

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(2), reraise=True)
    async def _ollama_generate(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        resp = await self._client.post(
            "/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    async def generate(self, model: str, prompt: str, temperature: float = 0.2) -> tuple[str, bool]:
        """Return (text, used_fallback)."""
        try:
            text = await self._ollama_generate(model, prompt, temperature)
            return text, False
        except Exception as exc:  # noqa: BLE001
            logger.warning("ollama_generate_failed", model=model, error=str(exc))
            if not self.settings.llm_fallback_mode:
                raise
            return self._fallback_generate(prompt), True

    def _fallback_generate(self, prompt: str) -> str:
        """Lightweight rule-based responder for demos without Ollama."""
        lower = prompt.lower()

        # ReAct tool decision
        if "available tools" in lower or "strict decision policy" in lower or "thought:" in lower:
            # Prefer the user question section to avoid matching hostnames from docs
            q_part = lower
            if "user question:" in lower:
                q_part = lower.split("user question:")[-1]
                for marker in ("\nassistant:", "\nobservation:", "\nyou now have", "\nthought:"):
                    if marker in q_part:
                        q_part = q_part.split(marker)[0]
                        break
            host_match = re.search(
                r"\b([a-z0-9.-]+\.(?:com|local|internal|io|net|org))\b",
                q_part,
            )
            live_intent = any(
                k in q_part
                for k in ("status", "down", "outage", "ping", "reachable", "uptime", "check_server_status")
            )
            if live_intent and host_match:
                host = host_match.group(1)
                return (
                    f"Thought: The user asks about live host availability for {host}.\n"
                    f"Action: check_server_status\n"
                    f"Action Input: {host}"
                )
            # Explicit incident ask — avoid matching the word inside RAG context
            if re.search(r"\b(incidents?|recent outages|outage timeline)\b", q_part):
                return (
                    "Thought: The user asks about recent incidents.\n"
                    "Action: list_recent_incidents\n"
                    "Action Input: platform"
                )
            return (
                "Thought: I can answer from the retrieved documentation.\n"
                "Action: finish\n"
                "Action Input: none"
            )

        # Final answer synthesis
        context_match = re.search(r"context:\n(.*?)(?:\n\nuser question:|\n\nquestion:)", prompt, re.S | re.I)
        context = (context_match.group(1).strip() if context_match else "")[:1200]
        question_match = re.search(r"(?:user question|question):\s*(.*)$", prompt, re.S | re.I)
        question = (question_match.group(1).strip() if question_match else "the issue")[:300]

        if context:
            snippet = context.split("\n\n")[0][:500]
            return (
                f"Based on the SmartOps knowledge base, here is guidance for: {question}\n\n"
                f"{snippet}\n\n"
                "If the issue persists after these steps, escalate with logs and the transaction ID."
            )
        return (
            f"I could not find specific documentation for: {question}. "
            "Please provide more detail (service name, error message, timeframe)."
        )
