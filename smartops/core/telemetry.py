"""Optional OpenTelemetry bootstrap (no-op when disabled / packages missing)."""

from __future__ import annotations

from typing import Any

from smartops.config import Settings
from smartops.core.logging import get_logger

logger = get_logger(__name__)


def setup_telemetry(app: Any, settings: Settings) -> bool:
    """Instrument FastAPI when OTEL is enabled. Returns True if active."""
    if not settings.otel_enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.warning("otel_packages_missing")
        return False

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name or settings.app_name,
            "deployment.environment": settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)
    endpoint = (settings.otel_exporter_otlp_endpoint or "").strip()
    if endpoint:
        exporter: Any = OTLPSpanExporter(endpoint=endpoint)
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("otel_enabled", endpoint=endpoint or "console")
    return True
