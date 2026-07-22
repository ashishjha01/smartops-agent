"""Unit tests for reward math, features, tools, and bandit updates."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from smartops.agent.react import ReActAgent
from smartops.agent.tools import check_server_status, run_tool
from smartops.config import Settings
from smartops.rag.store import chunk_text
from smartops.rl.bandit import BanditAction, ContextualBandit
from smartops.rl.features import QueryCategory, build_context_key, categorize_query, estimate_complexity
from smartops.rl.rewards import compute_reward
from smartops.services.transaction_store import TransactionRecord, TransactionStore


def test_reward_function():
    assert compute_reward(1, 2.5) == pytest.approx(7.5)
    assert compute_reward(0, 1.0) == pytest.approx(-1.0)
    # Latency above cap is clipped so helpful slow answers are not always negative
    assert compute_reward(1, 25.0, latency_cap_seconds=10.0) == pytest.approx(0.0)
    assert compute_reward(1, 25.0, latency_cap_seconds=10.0) > compute_reward(0, 2.0)


def test_categorize_networking():
    assert categorize_query("DNS resolution failures on VPN") == QueryCategory.NETWORKING


def test_complexity_and_context():
    assert estimate_complexity("What is DNS?").value == "simple"
    key = build_context_key("Is api.down.example.com unreachable right now?")
    assert "incident" in key or "networking" in key


def test_chunk_text():
    text = "\n\n".join([f"Paragraph number {i} with some unique content about topic {i}." for i in range(20)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    # Ensure chunks are distinct paragraph assemblies (no synthetic double-overlap pass)
    assert chunks[0] != chunks[1]


def test_check_server_status_deterministic():
    a = check_server_status("api.example.com")
    b = check_server_status("api.example.com")
    assert a == b
    down = check_server_status("db-offline.internal")
    assert down["reachable"] is False


def test_run_unknown_tool():
    result = run_tool("not_a_tool", "x")
    assert result.ok is False


def test_normalize_recovers_hostname_from_question():
    host = ReActAgent._normalize_action_input(
        "check_server_status",
        "none",
        "Is payments-down.internal up right now?",
    )
    assert host == "payments-down.internal"


def test_bandit_feedback_updates_mean():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            rl_state_path=str(Path(tmp) / "bandit.json"),
            rl_epsilon=0.0,
            llm_model_a="llama3.2:3b",
            llm_model_b="mistral:7b",
            rl_latency_cap_seconds=10.0,
        )
        bandit = ContextualBandit(
            settings,
            actions=[
                BanditAction("llama3.2:3b", 2),
                BanditAction("mistral:7b", 5),
            ],
        )
        action = bandit.select_action("networking::simple")
        bandit.register_pending("txn-1", "networking::simple", action, latency_seconds=1.0, created_at=time.time())
        update = bandit.apply_feedback("txn-1", 1)
        assert update["reward"] == pytest.approx(9.0)
        assert bandit.arms["networking::simple"][action.key].pulls == 1

        with pytest.raises(KeyError):
            bandit.apply_feedback("txn-1", 1)


def test_bandit_expires_pending_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "bandit.json"
        settings = Settings(
            rl_state_path=str(state_path),
            rl_epsilon=0.0,
            rl_feedback_timeout_seconds=1,
            llm_model_a="llama3.2:3b",
            llm_model_b="mistral:7b",
        )
        bandit = ContextualBandit(settings, actions=[BanditAction("llama3.2:3b", 2)])
        action = bandit.select_action("networking::simple")
        bandit.register_pending(
            "txn-old",
            "networking::simple",
            action,
            1.0,
            created_at=time.time() - 5,
        )
        snap = bandit.snapshot()
        assert snap["pending_transactions"] == 0
        with pytest.raises(KeyError):
            bandit.apply_feedback("txn-old", 1)

        # Fresh pending should persist across reload
        bandit.register_pending(
            "txn-new",
            "networking::simple",
            action,
            1.2,
            created_at=time.time(),
        )
        reloaded = ContextualBandit(settings, actions=[BanditAction("llama3.2:3b", 2)])
        assert "txn-new" in reloaded._pending


def test_transaction_store_expires_unscored():
    store = TransactionStore(ttl_seconds=1)
    store.put(
        TransactionRecord(
            transaction_id="t1",
            query="q",
            answer="a",
            llm="m",
            top_k=2,
            context="c",
            action_key="m::k2",
            latency_seconds=1.0,
            created_at=time.time() - 5,
        )
    )
    assert store.get("t1") is None


def test_transaction_store_persists_across_reload():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "txns.json")
        store = TransactionStore(ttl_seconds=3600, state_path=path)
        store.put(
            TransactionRecord(
                transaction_id="persist-1",
                query="q",
                answer="a",
                llm="m",
                top_k=2,
                context="c",
                action_key="m::k2",
                latency_seconds=1.0,
                created_at=time.time(),
            )
        )
        reloaded = TransactionStore(ttl_seconds=3600, state_path=path)
        assert reloaded.get("persist-1") is not None


def test_feedback_reservation_blocks_double_score():
    store = TransactionStore(ttl_seconds=3600)
    store.put(
        TransactionRecord(
            transaction_id="t-reserve",
            query="q",
            answer="a",
            llm="m",
            top_k=2,
            context="c",
            action_key="m::k2",
            latency_seconds=1.0,
            created_at=time.time(),
        )
    )
    assert store.reserve_feedback("t-reserve") == "ok"
    assert store.reserve_feedback("t-reserve") == "already_scored"


def test_parse_react_tolerates_markdown_noise():
    noisy = """```text
**Thought:** Need live status
- Action: check_server_status
- Action Input: `payments-down.internal`
```"""
    parsed = ReActAgent._parse_react(noisy)
    assert parsed is not None
    thought, action, action_input = parsed
    assert action.lower() == "check_server_status"
    assert "payments-down.internal" in action_input


@pytest.mark.asyncio
async def test_agent_tool_path_with_scripted_llm(tmp_path: Path):
    """Prove tool execution without relying on fallback heuristics alone."""
    from smartops.rag.store import RetrievedChunk

    class ScriptedLLM:
        def __init__(self):
            self.calls = 0

        async def generate(self, model: str, prompt: str, temperature: float = 0.2):
            self.calls += 1
            lower = prompt.lower()
            if "available tools" in lower or "strict decision policy" in lower:
                return (
                    "Thought: live check needed\n"
                    "Action: check_server_status\n"
                    "Action Input: payments-down.internal",
                    False,
                )
            return ("Host is unreachable based on tool output.", False)

    class StubRetriever:
        def retrieve(self, question: str, top_k: int = 2):
            return [
                RetrievedChunk(
                    chunk_id="c1",
                    text="Ops runbook stub",
                    source="stub.md",
                    score=0.9,
                )
            ]

    agent = ReActAgent(llm=ScriptedLLM(), retriever=StubRetriever(), max_steps=3)
    result = await agent.run(
        "Is payments-down.internal healthy?",
        model="scripted",
        top_k=2,
    )
    assert result.trace.tool_results
    assert result.trace.tool_results[0]["name"] == "check_server_status"
    assert result.trace.tool_results[0]["output"]["reachable"] is False
    assert result.trace.used_fallback_llm is False
    assert "unreachable" in result.answer.lower() or result.answer
