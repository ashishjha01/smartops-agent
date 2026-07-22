"""Reward math for the online learning loop."""


def compute_reward(
    feedback_score: int,
    latency_seconds: float,
    *,
    latency_cap_seconds: float = 10.0,
) -> float:
    """Assignment-aligned reward with bounded latency penalty.

    Base idea from the brief:
        reward = (feedback * 10) - latency_seconds

    Real local LLM latency often exceeds 10–40s, which makes even helpful
    answers strongly negative and biases the bandit toward unseen arms.
    We therefore clip the latency term:

        reward = (feedback * 10) - min(latency_seconds, latency_cap_seconds)

    Default cap=10 keeps the original scale for fast answers while preventing
    slow-but-helpful responses from always losing to cold-start zeros.
    """
    score = 1 if int(feedback_score) == 1 else 0
    cap = max(float(latency_cap_seconds), 0.0)
    latency_penalty = min(max(float(latency_seconds), 0.0), cap)
    return float(score * 10) - latency_penalty
