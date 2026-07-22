from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from smartops import __version__
from smartops.api.auth import require_admin
from smartops.api.deps import AppState, get_state
from smartops.api.schemas import HealthResponse, ReadyResponse
from smartops.core.async_utils import run_sync

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=state.settings.app_env,
    )


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
async def ready(state: AppState = Depends(get_state)):
    """Readiness probe.

    Ready when the vector store is populated and either:
    - Ollama is reachable, or
    - explicit demo fallback mode is enabled (CI/offline demos only).
    """
    llm_health = await state.llm.health()
    chunks = await run_sync(lambda: state.vector_store.count)
    snap = await run_sync(state.bandit.snapshot)
    llm_ok = bool(llm_health.get("reachable")) or (
        bool(state.settings.llm_fallback_mode) and bool(llm_health.get("ok"))
    )
    is_ready = chunks > 0 and llm_ok
    payload = ReadyResponse(
        status="ready" if is_ready else "degraded",
        vector_store_chunks=chunks,
        llm=llm_health,
        bandit={
            "epsilon": snap["epsilon"],
            "total_updates": snap["total_updates"],
            "pending": snap["pending_transactions"],
        },
    )
    if not is_ready:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@router.get("/metrics", dependencies=[Depends(require_admin)])
async def metrics() -> Response:
    """Prometheus metrics (admin-protected when auth is enabled)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
