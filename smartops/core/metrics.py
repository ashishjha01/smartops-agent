"""Prometheus metrics for production observability."""

from prometheus_client import Counter, Histogram, Gauge

QUERY_REQUESTS = Counter(
    "smartops_query_requests_total",
    "Total /query requests",
    ["status", "llm", "action"],
)

FEEDBACK_REQUESTS = Counter(
    "smartops_feedback_requests_total",
    "Total /feedback requests",
    ["score"],
)

QUERY_LATENCY = Histogram(
    "smartops_query_latency_seconds",
    "End-to-end query latency",
    ["llm", "action"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

BANDIT_EPSILON = Gauge(
    "smartops_bandit_epsilon",
    "Current epsilon exploration rate",
)

BANDIT_ARM_PULLS = Counter(
    "smartops_bandit_arm_pulls_total",
    "Contextual bandit arm selections",
    ["context", "action"],
)

BANDIT_REWARD = Histogram(
    "smartops_bandit_reward",
    "Observed rewards from feedback loop",
    ["context", "action"],
    buckets=(-50, -20, -10, -5, 0, 2, 5, 8, 10, 15),
)
