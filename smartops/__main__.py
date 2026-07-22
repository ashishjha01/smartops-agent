"""CLI entrypoint: `python -m smartops` or `uvicorn smartops.main:app`."""

import uvicorn

from smartops.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "smartops.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
