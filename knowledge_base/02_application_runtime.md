# SmartOps Knowledge Base — Application Runtime

## Python Application Crashes

Typical traceback patterns:
- `ModuleNotFoundError`: missing dependency — recreate venv and `pip install -r requirements.txt`.
- `Address already in use`: another process holds the port — `lsof -i :<port>` then kill or change `PORT`.
- `PermissionError` writing logs: ensure the service user owns `./logs` and `./data`.

## Memory Pressure / OOMKilled

Containers killed with exit code 137 usually indicate Out-Of-Memory.

Mitigations:
1. Raise container memory limit.
2. Reduce embedding model / LLM context size.
3. Cap concurrent requests with a worker queue.
4. Enable swap only as a temporary bridge (not production-preferred).

## Health Checks

A healthy SmartOps instance should expose:
- `GET /health` → process alive
- `GET /ready` → vector store + LLM reachable
- `GET /metrics` → Prometheus scrapes

If `/ready` fails but `/health` succeeds, dependencies are degraded; traffic should be drained.

## Rolling Restarts

1. Scale new replicas.
2. Wait for readiness probes.
3. Drain old pods.
4. Confirm error rate and p95 latency in dashboards.
