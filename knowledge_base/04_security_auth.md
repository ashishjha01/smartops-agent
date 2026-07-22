# SmartOps Knowledge Base — Authentication & Security

## JWT / API Token Failures

- `401 Unauthorized`: missing or expired token — re-authenticate and refresh.
- `403 Forbidden`: token valid but RBAC denies the action — check role bindings.
- Clock skew: ensure NTP sync (`chronyc tracking` / Windows Time service).

## CORS Issues in Browsers

Browsers block cross-origin calls when `Access-Control-Allow-Origin` is missing.
Configure `CORS_ORIGINS` to explicit allowlists in production; avoid `*` with credentials.

## Rate Limiting

SmartOps returns `429` when a client exceeds the per-minute quota.
Backoff with jitter: wait `Retry-After` / 60s before retrying.

## Secrets Management

Never commit `.env` files. Inject secrets via environment variables or a vault.
Rotate Ollama/API keys and database passwords on a fixed schedule.

## Incident Response Quick Steps

1. Confirm blast radius (which endpoints / tenants).
2. Check auth service health and certificate expiry.
3. Revoke compromised tokens.
4. Publish a status update and open a post-incident review.
