"""Application settings loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SmartOps Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 120.0
    llm_model_a: str = "llama3.2:3b"
    llm_model_b: str = "mistral:7b"
    llm_fallback_mode: bool = True

    rag_persist_dir: str = "./data/chroma"
    knowledge_base_dir: str = "./knowledge_base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_force_reingest: bool = False

    rl_epsilon: float = 0.15
    rl_epsilon_decay: float = 0.995
    rl_epsilon_min: float = 0.05
    rl_state_path: str = "./data/rl_bandit_state.json"
    rl_feedback_timeout_seconds: int = 300
    rl_latency_cap_seconds: float = 10.0

    transaction_state_path: str = "./data/transactions.json"
    audit_log_path: str = "./data/audit.jsonl"
    database_url: str = ""

    cors_origins: str = "*"
    rate_limit_per_minute: int = 60
    request_id_header: str = "X-Request-ID"
    # Only trust X-Forwarded-For when behind a known reverse proxy
    trust_proxy_headers: bool = False

    # Empty API_KEY disables auth (local/CI). Required when APP_ENV=production.
    api_key: str = ""
    # Optional distinct admin key for /rl/state and /metrics
    admin_api_key: str = ""
    redis_url: str = ""
    agent_max_steps: int = 3
    warmup_on_startup: bool = True

    # Async job queue for slow LLM calls
    job_workers: int = 2
    job_queue_max: int = 1000

    # OpenTelemetry (optional)
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "smartops-agent"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_models(self) -> list[str]:
        return [self.llm_model_a, self.llm_model_b]

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key.strip() or self.admin_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
