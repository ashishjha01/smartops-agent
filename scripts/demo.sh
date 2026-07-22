#!/usr/bin/env bash
# Demo script: query → feedback → inspect bandit state
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
AUTH_HDR=()
ADMIN_HDR=()
if [[ -n "${API_KEY:-}" ]]; then
  AUTH_HDR=(-H "X-API-Key: ${API_KEY}")
  ADMIN_HDR=(-H "X-API-Key: ${API_KEY}")
fi
if [[ -n "${ADMIN_API_KEY:-}" ]]; then
  ADMIN_HDR=(-H "X-API-Key: ${ADMIN_API_KEY}")
fi

echo "== health =="
curl -s "$BASE/health" | python -m json.tool

echo "== query =="
RESP=$(curl -s -X POST "$BASE/query" \
  -H "Content-Type: application/json" \
  "${AUTH_HDR[@]}" \
  -d '{"query":"How do I fix DNS resolution failures on VPN?"}')
echo "$RESP" | python -m json.tool
TXN=$(echo "$RESP" | python -c "import sys,json; print(json.load(sys.stdin)['transaction_id'])")

echo "== feedback =="
curl -s -X POST "$BASE/feedback" \
  -H "Content-Type: application/json" \
  "${AUTH_HDR[@]}" \
  -d "{\"transaction_id\":\"$TXN\",\"feedback_score\":1}" | python -m json.tool

echo "== rl state =="
curl -s "$BASE/rl/state" "${ADMIN_HDR[@]}" | python -m json.tool
