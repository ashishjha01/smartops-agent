"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse

from smartops import __version__
from smartops.agent.react import ReActAgent
from smartops.api.deps import AppState
from smartops.api.health import router as health_router
from smartops.api.routes import router as api_router
from smartops.config import Settings, get_settings
from smartops.core.async_utils import run_sync
from smartops.core.logging import configure_logging, get_logger
from smartops.core.middleware import RateLimitMiddleware, RequestContextMiddleware
from smartops.core.redis_client import create_redis_client
from smartops.core.telemetry import setup_telemetry
from smartops.llm.client import LLMClient
from smartops.rag.retriever import Retriever
from smartops.rag.store import VectorStore
from smartops.rl.bandit import ContextualBandit
from smartops.services.audit_store import AuditStore
from smartops.services.job_queue import JobQueue
from smartops.services.query_service import QueryService
from smartops.services.transaction_store import TransactionStore

logger = get_logger(__name__)

_WEAK_API_KEYS = {
    "",
    "changeme",
    "password",
    "secret",
    "test",
    "smartops-demo-key",
}


def _validate_production_settings(settings: Settings) -> None:
    if settings.app_env != "production":
        return
    key = (settings.api_key or "").strip()
    if not key or key.lower() in _WEAK_API_KEYS:
        raise RuntimeError(
            "API_KEY is required in production and must not be a weak/default value. "
            "Set a strong API_KEY before starting."
        )


async def _warmup(state: AppState) -> None:
    """Prime embeddings + LLM connectivity to reduce first-query cold start."""
    try:
        await run_sync(state.retriever.retrieve, "warmup health check", 1)
        logger.info("warmup_embeddings_ok")
    except Exception as exc:  # noqa: BLE001
        logger.warning("warmup_embeddings_failed", error=str(exc))
    try:
        health = await state.llm.health()
        logger.info(
            "warmup_llm_ok",
            ok=health.get("ok"),
            reachable=health.get("reachable"),
            fallback=health.get("fallback"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("warmup_llm_failed", error=str(exc))


def build_state(settings: Settings, redis_client=None) -> AppState:
    store = VectorStore(settings)
    ingested = store.ingest_directory(settings.knowledge_base_dir, force=False)
    logger.info("startup_ingest", chunks=ingested)

    llm = LLMClient(settings)
    retriever = Retriever(store)
    agent = ReActAgent(llm=llm, retriever=retriever, max_steps=settings.agent_max_steps)
    bandit = ContextualBandit(settings, redis_client=redis_client)
    ttl = max(int(settings.rl_feedback_timeout_seconds), 60)
    transactions = TransactionStore(
        ttl_seconds=ttl,
        state_path=settings.transaction_state_path,
    )
    audit = AuditStore(
        database_url=settings.database_url,
        file_path=settings.audit_log_path,
    )
    jobs = JobQueue(workers=settings.job_workers, max_jobs=settings.job_queue_max)
    query_service = QueryService(
        agent=agent,
        bandit=bandit,
        transactions=transactions,
        audit=audit,
    )
    jobs.set_handler(query_service.handle_query)

    return AppState(
        settings=settings,
        vector_store=store,
        retriever=retriever,
        llm=llm,
        agent=agent,
        bandit=bandit,
        transactions=transactions,
        query_service=query_service,
        audit=audit,
        jobs=jobs,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _validate_production_settings(settings)
    configure_logging(settings.log_level)
    redis_client = create_redis_client(settings.redis_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = build_state(settings, redis_client=redis_client)
        app.state.smartops = state
        await state.jobs.start()
        if settings.warmup_on_startup:
            await _warmup(state)
        logger.info(
            "app_started",
            version=__version__,
            env=settings.app_env,
            models=settings.llm_models,
            auth_enabled=settings.auth_enabled,
            rate_limit_backend="redis" if redis_client is not None else "memory",
            trust_proxy_headers=settings.trust_proxy_headers,
            agent_max_steps=settings.agent_max_steps,
            otel=settings.otel_enabled,
            audit_backend="postgres" if settings.database_url.startswith("postgres") else "jsonl",
            job_workers=settings.job_workers,
        )
        yield
        await state.jobs.stop()
        await state.llm.aclose()
        state.audit.close()
        if redis_client is not None:
            try:
                redis_client.close()
            except Exception:  # noqa: BLE001
                pass
        logger.info("app_stopped")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Self-optimizing technical support agent with RAG, ReAct tooling, "
            "and an epsilon-greedy contextual bandit router."
        ),
        lifespan=lifespan,
    )

    setup_telemetry(app, settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
        redis_client=redis_client,
        trust_proxy_headers=settings.trust_proxy_headers,
    )
    app.add_middleware(
        RequestContextMiddleware,
        request_id_header=settings.request_id_header,
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    app.include_router(health_router)
    app.include_router(api_router)

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        if settings.auth_enabled:
            schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            }
            schema["security"] = [{"ApiKeyAuth": []}]
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
    return app


app = create_app()
