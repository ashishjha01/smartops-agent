"""Shared FastAPI dependencies / app state accessors."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from smartops.agent.react import ReActAgent
from smartops.config import Settings
from smartops.llm.client import LLMClient
from smartops.rag.retriever import Retriever
from smartops.rag.store import VectorStore
from smartops.rl.bandit import ContextualBandit
from smartops.services.audit_store import AuditStore
from smartops.services.job_queue import JobQueue
from smartops.services.query_service import QueryService
from smartops.services.transaction_store import TransactionStore


@dataclass
class AppState:
    settings: Settings
    vector_store: VectorStore
    retriever: Retriever
    llm: LLMClient
    agent: ReActAgent
    bandit: ContextualBandit
    transactions: TransactionStore
    query_service: QueryService
    audit: AuditStore
    jobs: JobQueue


def get_state(request: Request) -> AppState:
    return request.app.state.smartops
