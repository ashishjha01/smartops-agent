from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from smartops.api.auth import require_admin, require_user
from smartops.api.deps import AppState, get_state
from smartops.api.schemas import (
    AsyncQueryAccepted,
    FeedbackRequest,
    FeedbackResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
)
from smartops.core.async_utils import run_sync
from smartops.core.metrics import FEEDBACK_REQUESTS

router = APIRouter(tags=["smartops"])


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_user)])
async def query_endpoint(payload: QueryRequest, state: AppState = Depends(get_state)) -> QueryResponse:
    result = await state.query_service.handle_query(payload.query)
    return QueryResponse(**result)


@router.post(
    "/query/async",
    response_model=AsyncQueryAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_user)],
)
async def query_async_endpoint(
    payload: QueryRequest,
    state: AppState = Depends(get_state),
) -> AsyncQueryAccepted:
    """Enqueue a slow LLM query and return a job id for polling (single-process workers)."""
    try:
        job = await state.jobs.enqueue(payload.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return AsyncQueryAccepted(
        job_id=job.job_id,
        status=job.status.value,
        poll_url=f"/jobs/{job.job_id}",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_user)])
async def job_status(job_id: str, state: AppState = Depends(get_state)) -> JobStatusResponse:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return JobStatusResponse(**job.to_dict())


@router.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_user)])
async def feedback_endpoint(
    payload: FeedbackRequest,
    state: AppState = Depends(get_state),
) -> FeedbackResponse:
    try:
        result = await state.query_service.handle_feedback(
            payload.transaction_id, int(payload.feedback_score)
        )
    except KeyError as exc:
        FEEDBACK_REQUESTS.labels(score=str(payload.feedback_score)).inc()
        detail = str(exc)
        code = 410 if "expired" in detail.lower() else 404
        raise HTTPException(status_code=code, detail=detail) from exc
    except ValueError as exc:
        FEEDBACK_REQUESTS.labels(score=str(payload.feedback_score)).inc()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    FEEDBACK_REQUESTS.labels(score=str(payload.feedback_score)).inc()
    return FeedbackResponse(**result)


@router.get("/rl/state", dependencies=[Depends(require_admin)])
async def rl_state(state: AppState = Depends(get_state)) -> dict:
    """Inspect bandit arm statistics (admin)."""
    return await run_sync(state.bandit.snapshot)
