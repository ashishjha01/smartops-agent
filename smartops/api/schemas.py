"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "Why am I getting DNS resolution failures?"},
                {"query": "Is payments-down.internal up right now?"},
            ]
        }
    )

    query: str = Field(..., min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def query_must_be_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class QueryResponse(BaseModel):
    transaction_id: str
    answer: str
    llm: str
    latency_seconds: float
    routing: dict[str, Any] | None = None
    agent: dict[str, Any] | None = None
    used_fallback_llm: bool | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "transaction_id": "PASTE_TRANSACTION_ID_FROM_QUERY_RESPONSE",
                    "feedback_score": 1,
                }
            ]
        }
    )

    transaction_id: str = Field(
        ...,
        min_length=8,
        description="Copy transaction_id from the /query response (do not use the placeholder text).",
    )
    feedback_score: Literal[0, 1] = Field(
        ...,
        description="1 = helpful, 0 = unhelpful",
    )


class FeedbackResponse(BaseModel):
    status: str
    transaction_id: str
    feedback_score: int
    reward: float
    rl: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    status: str
    vector_store_chunks: int
    llm: dict[str, Any]
    bandit: dict[str, Any]


class AsyncQueryAccepted(BaseModel):
    job_id: str
    status: str
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    updated_at: float
    query: str
    result: dict[str, Any] | None = None
    error: str | None = None
