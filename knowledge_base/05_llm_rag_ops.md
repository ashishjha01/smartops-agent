# SmartOps Knowledge Base — LLM, RAG & Agent Operations

## RAG Retrieval Quality

If answers cite irrelevant context:
- Increase `top_k` for broad queries; decrease for precise lookups.
- Re-chunk documents (smaller chunks for FAQ-style content).
- Re-embed after documentation updates.
- Inspect retrieved chunks in debug logs (`retrieved_docs`).

## LLM Latency & Cost Trade-offs

| Model profile | Typical use | Latency | Quality |
|---------------|-------------|---------|---------|
| Small (e.g. llama3.2:3b) | Simple FAQ / routing | Lower | Adequate |
| Medium (e.g. mistral:7b) | Multi-step reasoning | Higher | Stronger |

The SmartOps contextual bandit learns which model + `top_k` combination maximizes:
`Reward = (feedback * 10) - latency_seconds`.

## Agentic Tool Use

The agent may call `check_server_status(hostname)` when the user asks about live availability.
Do not invent tool results — only report what the tool returns.

## Ollama Offline Mode

If Ollama is unreachable, SmartOps can operate in deterministic fallback mode for demos and CI.
Fallback answers still use RAG context but skip remote model inference.

## Feedback Loop

Always submit `POST /feedback` with the `transaction_id` from `/query`.
Without feedback, the bandit only observes latency and cannot optimize helpfulness.
