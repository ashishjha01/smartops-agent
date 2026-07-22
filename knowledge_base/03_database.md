# SmartOps Knowledge Base — Database & Persistence

## Connection Pool Exhaustion

Symptoms: intermittent `too many connections` or request latency spikes.

Actions:
1. Check active connections: `SELECT count(*) FROM pg_stat_activity;`
2. Lower application `pool_size` / raise DB `max_connections` carefully.
3. Ensure connections are returned on request completion (no leaked sessions).
4. Prefer connection pooling middleware (PgBouncer) in production.

## Slow Queries

1. Enable slow-query logging (threshold 200–500ms).
2. Run `EXPLAIN ANALYZE` on offending SQL.
3. Add indexes for high-cardinality filters; avoid over-indexing writes.
4. Cache hot read paths with Redis when eventual consistency is acceptable.

## Disk Full / WAL Growth

- Free space under 10% can stall writes.
- Rotate and compress logs.
- Archive WAL segments and verify backup retention policies.
- For Chroma/FAISS local stores, monitor `./data` growth and prune stale collections.

## Backup Verification

Weekly restore drills: restore a backup into a staging instance and run smoke tests against `/query` and `/feedback`.
