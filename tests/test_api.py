"""API integration tests (LLM fallback mode — no Ollama required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smartops.config import Settings
from smartops.main import create_app


@pytest.fixture()
def client(tmp_path: Path):
    kb = Path(__file__).resolve().parents[1] / "knowledge_base"
    settings = Settings(
        app_env="development",
        llm_fallback_mode=True,
        # Unreachable Ollama forces deterministic fallback in CI/local tests
        ollama_base_url="http://127.0.0.1:9",
        ollama_timeout_seconds=1.0,
        knowledge_base_dir=str(kb),
        rag_persist_dir=str(tmp_path / "chroma"),
        rl_state_path=str(tmp_path / "bandit.json"),
        transaction_state_path=str(tmp_path / "txns.json"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rate_limit_per_minute=1000,
        rl_latency_cap_seconds=10.0,
        api_key="",
        redis_url="",
        warmup_on_startup=False,
        agent_max_steps=3,
        otel_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def authed_client(tmp_path: Path):
    kb = Path(__file__).resolve().parents[1] / "knowledge_base"
    settings = Settings(
        app_env="development",
        llm_fallback_mode=True,
        ollama_base_url="http://127.0.0.1:9",
        ollama_timeout_seconds=1.0,
        knowledge_base_dir=str(kb),
        rag_persist_dir=str(tmp_path / "chroma-auth"),
        rl_state_path=str(tmp_path / "bandit-auth.json"),
        transaction_state_path=str(tmp_path / "txns-auth.json"),
        audit_log_path=str(tmp_path / "audit-auth.jsonl"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rate_limit_per_minute=1000,
        api_key="test-secret",
        warmup_on_startup=False,
        otel_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_ok(client: TestClient):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["vector_store_chunks"] > 0


def test_root_redirects_to_docs(client: TestClient):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in {307, 302}
    assert "/docs" in r.headers.get("location", "")


def test_query_and_feedback_loop(client: TestClient):
    q = client.post("/query", json={"query": "How do I fix DNS resolution failures?"})
    assert q.status_code == 200
    body = q.json()
    assert body["transaction_id"]
    assert body["answer"]
    assert body["llm"]
    assert isinstance(body["latency_seconds"], float)

    fb = client.post(
        "/feedback",
        json={"transaction_id": body["transaction_id"], "feedback_score": 1},
    )
    assert fb.status_code == 200
    # With latency cap, penalty cannot exceed the cap
    latency = body["latency_seconds"]
    expected = 10 - min(latency, 10.0)
    assert fb.json()["reward"] == pytest.approx(expected, rel=1e-3)

    fb2 = client.post(
        "/feedback",
        json={"transaction_id": body["transaction_id"], "feedback_score": 0},
    )
    assert fb2.status_code == 409


def test_blank_query_rejected(client: TestClient):
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 422


def test_agent_must_call_check_server_status(client: TestClient):
    q = client.post(
        "/query",
        json={"query": "Is payments-down.internal server status healthy right now?"},
    )
    assert q.status_code == 200
    agent = q.json().get("agent") or {}
    tools = agent.get("tool_results") or []
    assert tools, "Expected check_server_status tool invocation for live host query"
    assert tools[0]["name"] == "check_server_status"
    assert "payments-down.internal" in (tools[0].get("input") or "")
    assert tools[0]["output"]["reachable"] is False


def test_agent_finishes_faq_without_tools(client: TestClient):
    q = client.post(
        "/query",
        json={"query": "How do I fix DNS resolution failures on VPN?"},
    )
    assert q.status_code == 200
    agent = q.json().get("agent") or {}
    tools = agent.get("tool_results") or []
    assert tools == []
    assert (agent.get("action") or "finish").lower() in {"finish", "final", "answer", "none", None}


def test_rl_state(client: TestClient):
    r = client.get("/rl/state")
    assert r.status_code == 200
    body = r.json()
    assert "epsilon" in body
    assert "latency_cap_seconds" in body


def test_auth_required_when_api_key_set(authed_client: TestClient):
    denied = authed_client.post("/query", json={"query": "How do I fix DNS?"})
    assert denied.status_code == 401

    ok = authed_client.post(
        "/query",
        json={"query": "How do I fix DNS?"},
        headers={"X-API-Key": "test-secret"},
    )
    assert ok.status_code == 200

    bearer = authed_client.post(
        "/query",
        json={"query": "How do I fix DNS resolution?"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert bearer.status_code == 200

    # Health remains open
    assert authed_client.get("/health").status_code == 200


def test_production_rejects_weak_api_key():
    from smartops.main import create_app

    with pytest.raises(RuntimeError, match="API_KEY"):
        create_app(
            Settings(
                app_env="production",
                api_key="smartops-demo-key",
                llm_fallback_mode=True,
                warmup_on_startup=False,
            )
        )


def test_xff_ignored_unless_trusted(tmp_path: Path):
    kb = Path(__file__).resolve().parents[1] / "knowledge_base"
    settings = Settings(
        app_env="development",
        llm_fallback_mode=True,
        ollama_base_url="http://127.0.0.1:9",
        ollama_timeout_seconds=1.0,
        knowledge_base_dir=str(kb),
        rag_persist_dir=str(tmp_path / "chroma-xff"),
        rl_state_path=str(tmp_path / "bandit-xff.json"),
        transaction_state_path=str(tmp_path / "txns-xff.json"),
        audit_log_path=str(tmp_path / "audit-xff.jsonl"),
        rate_limit_per_minute=2,
        trust_proxy_headers=False,
        warmup_on_startup=False,
        otel_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        # Same socket client; forged XFF must NOT create a fresh bucket
        h = {"X-Forwarded-For": "203.0.113.10"}
        assert c.get("/rl/state", headers=h).status_code == 200
        assert c.get("/rl/state", headers={"X-Forwarded-For": "203.0.113.11"}).status_code == 200
        limited = c.get("/rl/state", headers={"X-Forwarded-For": "203.0.113.12"})
        assert limited.status_code == 429
        assert limited.headers.get("Retry-After") == "60"


def test_async_query_job(client: TestClient):
    import time

    accepted = client.post("/query/async", json={"query": "How do I fix DNS resolution failures?"})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    body = None
    for _ in range(100):
        poll = client.get(f"/jobs/{job_id}")
        assert poll.status_code == 200
        body = poll.json()
        if body["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert body is not None
    assert body["status"] == "completed"
    assert body["result"]["transaction_id"]


def test_admin_rbac_blocks_user_key(tmp_path: Path):
    kb = Path(__file__).resolve().parents[1] / "knowledge_base"
    settings = Settings(
        app_env="development",
        llm_fallback_mode=True,
        ollama_base_url="http://127.0.0.1:9",
        ollama_timeout_seconds=1.0,
        knowledge_base_dir=str(kb),
        rag_persist_dir=str(tmp_path / "chroma-rbac"),
        rl_state_path=str(tmp_path / "bandit-rbac.json"),
        transaction_state_path=str(tmp_path / "txns-rbac.json"),
        audit_log_path=str(tmp_path / "audit-rbac.jsonl"),
        api_key="user-secret",
        admin_api_key="admin-secret",
        rate_limit_per_minute=1000,
        warmup_on_startup=False,
        otel_enabled=False,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        user_h = {"X-API-Key": "user-secret"}
        admin_h = {"X-API-Key": "admin-secret"}
        assert c.post("/query", json={"query": "How do I fix DNS?"}, headers=user_h).status_code == 200
        assert c.get("/rl/state", headers=user_h).status_code == 403
        assert c.get("/metrics", headers=user_h).status_code == 403
        assert c.get("/rl/state", headers=admin_h).status_code == 200
        assert c.get("/metrics", headers=admin_h).status_code == 200
