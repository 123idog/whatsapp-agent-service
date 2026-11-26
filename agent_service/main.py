"""Entry point: `uvicorn agent_service.main:app`."""

from .app import configure_logging, create_app

configure_logging()
app = create_app()
