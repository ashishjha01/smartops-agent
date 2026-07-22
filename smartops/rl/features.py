"""Query feature extraction → discrete RL contexts (state space)."""

from __future__ import annotations

import re
from enum import Enum


class QueryCategory(str, Enum):
    NETWORKING = "networking"
    RUNTIME = "runtime"
    DATABASE = "database"
    SECURITY = "security"
    LLM_RAG = "llm_rag"
    INCIDENT = "incident"
    GENERAL = "general"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


CATEGORY_KEYWORDS: dict[QueryCategory, tuple[str, ...]] = {
    QueryCategory.NETWORKING: (
        "dns", "tcp", "timeout", "502", "504", "gateway", "firewall", "mtu", "vpn", "port", "network",
    ),
    QueryCategory.RUNTIME: (
        "crash", "oom", "memory", "python", "container", "restart", "health", "latency", "cpu",
    ),
    QueryCategory.DATABASE: (
        "database", "postgres", "sql", "connection pool", "wal", "index", "query slow", "disk",
    ),
    QueryCategory.SECURITY: (
        "jwt", "token", "auth", "cors", "401", "403", "secret", "rbac", "certificate",
    ),
    QueryCategory.LLM_RAG: (
        "rag", "embedding", "top_k", "ollama", "llm", "bandit", "retrieval", "vector",
    ),
    QueryCategory.INCIDENT: (
        "down", "outage", "status", "uptime", "reachable", "incident", "offline", "unavailable",
    ),
}


def categorize_query(query: str) -> QueryCategory:
    q = query.lower()
    scores: dict[QueryCategory, int] = {c: 0 for c in QueryCategory}
    for cat, words in CATEGORY_KEYWORDS.items():
        for w in words:
            if w in q:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else QueryCategory.GENERAL


def estimate_complexity(query: str) -> QueryComplexity:
    q = query.strip()
    tokens = re.findall(r"\w+", q.lower())
    n = len(tokens)
    multi_part = len(re.findall(r"\b(and|also|then|plus|as well)\b", q.lower()))
    has_code = bool(re.search(r"[`{}]|traceback|exception|error code", q.lower()))

    score = 0
    if n > 40:
        score += 2
    elif n > 18:
        score += 1
    score += multi_part
    if has_code:
        score += 1

    if score >= 3:
        return QueryComplexity.COMPLEX
    if score >= 1:
        return QueryComplexity.MEDIUM
    return QueryComplexity.SIMPLE


def build_context_key(query: str) -> str:
    """Discrete state for the contextual bandit: category::complexity."""
    return f"{categorize_query(query).value}::{estimate_complexity(query).value}"
