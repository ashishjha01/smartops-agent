"""Mock operational tools the agent can invoke."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    name: str
    input: str
    output: dict[str, Any]
    ok: bool


def check_server_status(hostname: str) -> dict[str, Any]:
    """Deterministic mock health probe for a hostname.

    Uses a hash of the hostname so demos are reproducible while still
    looking like a real status payload.
    """
    host = (hostname or "").strip().lower()
    if not host:
        return {
            "hostname": host,
            "status": "unknown",
            "reachable": False,
            "latency_ms": None,
            "message": "Hostname is required.",
        }

    digest = hashlib.sha256(host.encode()).hexdigest()
    # Stable pseudo-random based on hostname
    rng = random.Random(int(digest[:8], 16))
    # Most hosts healthy; a few known "bad" patterns for demos
    forced_down = any(token in host for token in ("down", "fail", "broken", "offline"))
    reachable = False if forced_down else rng.random() > 0.18
    latency = None if not reachable else rng.randint(12, 220)
    cpu = rng.randint(5, 95) if reachable else None
    mem = rng.randint(20, 92) if reachable else None

    status = "healthy" if reachable and (cpu or 0) < 85 else ("degraded" if reachable else "down")
    return {
        "hostname": host,
        "status": status,
        "reachable": reachable,
        "latency_ms": latency,
        "cpu_percent": cpu,
        "memory_percent": mem,
        "checked_via": "mock_probe_v1",
        "message": (
            f"Host {host} is {status}."
            + (f" Probe latency {latency}ms." if latency is not None else " No response to probe.")
        ),
    }


def list_recent_incidents(service: str = "platform") -> dict[str, Any]:
    """Mock incident timeline for a service name."""
    svc = (service or "platform").strip().lower() or "platform"
    return {
        "service": svc,
        "incidents": [
            {
                "id": "INC-1042",
                "severity": "minor",
                "summary": f"Elevated 5xx on {svc} edge proxy",
                "status": "resolved",
                "window": "last 24h",
            },
            {
                "id": "INC-1038",
                "severity": "major",
                "summary": f"DNS cache poisoning suspicion affecting {svc} clients",
                "status": "monitoring",
                "window": "last 7d",
            },
        ],
    }


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "check_server_status": check_server_status,
    "list_recent_incidents": list_recent_incidents,
}


def tool_descriptions() -> str:
    return (
        "- check_server_status(hostname: str): ONLY for live host up/down checks. "
        "Action Input must be a hostname like payments-down.internal.\n"
        "- list_recent_incidents(service: str): ONLY when user asks about recent incidents. "
        "Action Input is a service name (default platform)."
    )


def run_tool(name: str, raw_input: str) -> ToolResult:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return ToolResult(name=name, input=raw_input, output={"error": f"Unknown tool: {name}"}, ok=False)

    arg = (raw_input or "").strip()
    if arg.lower() in {"none", "null", "n/a", ""}:
        arg = ""

    try:
        # Single-string tools for this assignment
        if name == "check_server_status":
            out = fn(arg)
        elif name == "list_recent_incidents":
            out = fn(arg or "platform")
        else:
            out = fn(arg)
        return ToolResult(name=name, input=arg, output=out, ok=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(name=name, input=arg, output={"error": str(exc)}, ok=False)
