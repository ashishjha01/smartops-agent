# SmartOps Self-Optimizing Agent

Production-minded FastAPI backend for an intelligent technical support agent that combines:

- **RAG** over a local markdown knowledge base (ChromaDB + sentence-transformers)
- **ReAct agentic reasoning** with mock ops tools (`check_server_status`, `list_recent_incidents`)
- **Online Reinforcement Learning** via an **epsilon-greedy contextual bandit** that routes each query to the best *(LLM × RAG top-k)* action using user feedback and latency

```text
Client ──► POST /query ──► Feature extract (state)
                         │
                         ▼
                   Contextual Bandit ── selects (llm, top_k)
                         │
                         ▼
                   ReAct Agent ── RAG retrieve ── optional tools
                         │
                         ▼
                   Answer + transaction_id + latency
                         │
Client ──► POST /feedback (0|1) ──► Reward update ──► bandit policy improves
```

---

## Quick start

### Prerequisites

- Python **3.11+**
- Optional but recommended: [Ollama](https://ollama.com) with two models pulled
- Git

### 1) Clone & install

```bash
git clone https://github.com/ashishjha01/smartops-agent.git
cd smartops-agent

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

For development/tests:

```bash
pip install -r requirements-dev.txt
```

Optional OpenTelemetry packages:

```bash
pip install -r requirements-otel.txt
```

### 2) (Recommended) Pull Ollama models

```bash
ollama pull llama3.2:3b
ollama pull mistral:7b
```

If Ollama is offline, keep `LLM_FALLBACK_MODE=true` (default). The API still runs with a deterministic fallback LLM suitable for demos and CI.

### 3) Run the API

```bash
uvicorn smartops.main:app --host 0.0.0.0 --port 8000 --reload
# or: python -m smartops
```

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Local default leaves `API_KEY` empty → auth disabled. Set `API_KEY` (and optionally `ADMIN_API_KEY`) when you want protection.

### 4) Docker Compose (API + Ollama + Redis + Postgres)

```bash
export API_KEY="$(openssl rand -hex 24)"   # required; weak defaults are rejected
# optional: export ADMIN_API_KEY="$(openssl rand -hex 24)"
docker compose up --build
```

Pass the same key as `X-API-Key` on `/query` and `/feedback`. Redis and Postgres are internal-only (not published to the host).

Pull models inside the Ollama container once (required when `LLM_FALLBACK_MODE=false`):

```bash
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull mistral:7b
```

Compose healthcheck probes **`/ready`** (not `/health`).

---

## API reference

### `POST /query`

```bash
# Local (auth off):
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"How do I fix DNS resolution failures?\"}"

# When API_KEY is set:
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"query\": \"How do I fix DNS resolution failures?\"}"
```

Example response (trimmed):

```json
{
  "transaction_id": "9f3c2e5a-....",
  "answer": "Based on the SmartOps knowledge base...",
  "llm": "llama3.2:3b",
  "latency_seconds": 1.84,
  "routing": {
    "context": "networking::simple",
    "action": "llama3.2:3b::k2",
    "top_k": 2,
    "epsilon": 0.15
  },
  "agent": {
    "thought": "...",
    "action": "finish",
    "tool_results": [],
    "retrieved_sources": [{"source": "01_networking.md", "score": 0.72}]
  }
}
```

### `POST /feedback`

Use the **real** `transaction_id` from `/query` (not Swagger placeholders).

```bash
curl -s -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"transaction_id\": \"9f3c2e5a-....\", \"feedback_score\": 1}"
```

Statuses: `200` updated · `404` unknown · `409` already scored · `410` feedback window expired.

Reward (latency capped so slow local LLMs do not drown helpfulness):

\[
R = (feedback\_score \times 10) - \min(latency\_seconds,\ latency\_cap)
\]

Default `latency_cap` is `RL_LATENCY_CAP_SECONDS=10`.

### Async query (single-process workers)

```bash
curl -s -X POST http://127.0.0.1:8000/query/async \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"query\": \"How do I fix DNS resolution failures?\"}"
# → { "job_id": "...", "status": "queued", "poll_url": "/jobs/..." }

curl -s http://127.0.0.1:8000/jobs/<job_id> -H "X-API-Key: $API_KEY"
```

### Ops endpoints

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /health` | Liveness | open |
| `GET /ready` | Readiness (vector store + LLM); **503** if degraded | open |
| `GET /metrics` | Prometheus metrics | admin |
| `GET /rl/state` | Bandit arm statistics | admin |
| `POST /query/async` | Enqueue slow query | user |
| `GET /jobs/{job_id}` | Poll async job | user |

### Tool-triggering example

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Is payments-down.internal up right now?\"}"
```

The ReAct loop should call `check_server_status` before answering.

---

## Architecture

| Layer | Implementation |
|-------|----------------|
| API | FastAPI + Pydantic v2 + Uvicorn |
| RAG | 5 markdown docs → chunk → ChromaDB persistent store → MiniLM embeddings |
| LLMs | Ollama `llama3.2:3b` & `mistral:7b` (+ offline fallback) |
| Agent | ReAct parse loop; tools in `smartops/agent/tools.py` |
| RL | Epsilon-greedy **contextual bandit** (`smartops/rl/bandit.py`) |
| Jobs | In-process async worker pool (`POST /query/async`) |
| Audit | JSONL file or Postgres (`DATABASE_URL`) |
| Observability | structlog JSON logs, Prometheus metrics, optional OpenTelemetry |
| Hardening | API keys + RBAC, rate limit, `/ready` 503, Docker non-root |

Knowledge base files live in [`knowledge_base/`](knowledge_base/).

---

## RL strategy (graded section)

### Why a contextual bandit?

Routing quality depends on **query type**. A small model + low `top_k` may win on simple FAQs; a larger model + higher `top_k` may win on complex questions. A contextual bandit learns this online without training a deep network.

### State space

`context = category::complexity` from keyword / length heuristics:

- Categories: `networking`, `runtime`, `database`, `security`, `llm_rag`, `incident`, `general`
- Complexity: `simple`, `medium`, `complex`

### Action space

Cartesian product of configured LLMs and RAG hyperparameters:

- Models: `LLM_MODEL_A`, `LLM_MODEL_B`
- `top_k ∈ {2, 5}`

### Policy

Epsilon-greedy with sample-mean updates and ε decay. Arm statistics **and pending decisions** persist to `RL_STATE_PATH` (and Redis CAS when `REDIS_URL` is set). Pending feedback older than `RL_FEEDBACK_TIMEOUT_SECONDS` expires.

### Reward

```text
reward = (feedback_score * 10) - min(latency_seconds, latency_cap)
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

Expect **27** tests. Suite forces offline LLM fallback (no Ollama required). First run downloads the embedding model (~80MB).

GitHub Actions CI runs the same suite on every push/PR.

---

## Project layout

```text
smartops/
  api/           # routes, schemas, deps, auth (RBAC)
  agent/         # ReAct loop + tools
  rag/           # chunking, Chroma store, retriever
  llm/           # Ollama client + prompts
  rl/            # features, rewards, contextual bandit
  services/      # query, transactions, jobs, audit
  core/          # logging, metrics, middleware, redis, telemetry
  main.py        # app factory
knowledge_base/  # mock technical docs
tests/
Dockerfile
docker-compose.yml
requirements.txt
requirements-dev.txt
requirements-otel.txt
```

---

## Production readiness

1. Env-based config (`.env.example`, no secrets in git)  
2. LLM retries + deterministic fallback for demos/CI  
3. Persistent bandit arms + pending feedback; Redis CAS when configured  
4. Persistent Chroma index; auto re-ingest on KB/chunk/embed fingerprint change  
5. Structured logs, Prometheus, request IDs; optional OTEL  
6. `/health` vs `/ready` (**503** when degraded); Docker healthcheck uses `/ready`  
7. Validation, 404/409/410 feedback semantics  
8. Redis or in-memory rate limit; XFF only if `TRUST_PROXY_HEADERS=true`  
9. Production requires strong `API_KEY`; optional `ADMIN_API_KEY` for admin routes  
10. Non-root Docker image, `.dockerignore`, Compose Redis + Postgres  
11. CI + 27 automated tests  
12. Feedback reservation (no double-score races)  
13. Aligned TTL for transactions + pending bandit decisions  
14. RL latency cap for stable local-LLM learning  
15. Startup warmup for embeddings/LLM  
16. Configurable ReAct `AGENT_MAX_STEPS` + tolerant parsing  
17. Async query enqueue for slow calls (single-process workers)  
18. Audit trail (JSONL / Postgres)  
19. Blocking I/O offloaded with `asyncio.to_thread`  

### Honest next steps

- Horizontal async workers (Redis/RQ or Celery)  
- Hybrid BM25 + vector retrieval  
- Alerting on p95 / error rate  
- Managed secrets (Vault/KMS)  

---

## Configuration reference

See [`.env.example`](.env.example).

| Variable | Meaning |
|----------|---------|
| `LLM_MODEL_A` / `LLM_MODEL_B` | Ollama model tags |
| `LLM_FALLBACK_MODE` | Allow offline demo mode |
| `RL_EPSILON` | Initial exploration rate |
| `RL_STATE_PATH` | Bandit persistence file |
| `TRANSACTION_STATE_PATH` | Query/feedback ledger file |
| `AUDIT_LOG_PATH` | JSONL audit path |
| `RL_LATENCY_CAP_SECONDS` | Max latency penalty in reward |
| `RL_FEEDBACK_TIMEOUT_SECONDS` | Pending feedback expiry |
| `RAG_PERSIST_DIR` | Chroma path |
| `RAG_FORCE_REINGEST` | Force rebuild vector index |
| `RATE_LIMIT_PER_MINUTE` | API rate limit |
| `TRUST_PROXY_HEADERS` | Trust `X-Forwarded-For` only behind known proxy |
| `REDIS_URL` | Shared rate limit + bandit CAS |
| `API_KEY` | User auth (required in production) |
| `ADMIN_API_KEY` | Admin auth for `/rl/state` `/metrics` |
| `DATABASE_URL` | Postgres audit (else JSONL) |
| `JOB_WORKERS` | In-process async workers |
| `OTEL_ENABLED` | Enable OpenTelemetry (needs `requirements-otel.txt`) |
| `AGENT_MAX_STEPS` | ReAct tool hops (default 3) |
| `WARMUP_ON_STARTUP` | Prime embeddings/LLM on boot |

---

## License

MIT — assignment / portfolio use.
