"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from server.persistence.db import init_db
from server.core.config import settings
from server.core.workspace import get_workspace_path, prompt_workspace_path, set_workspace_path

logger = logging.getLogger(__name__)

_FRONTEND_ROUTE: str | None = None


def _log_workspace_status() -> None:
    """Log workspace status on startup (but don't require it)."""
    try:
        workspace_path = get_workspace_path()
        logger.info("Workspace path: %s", workspace_path)
    except RuntimeError:
        logger.warning(
            "Workspace path not configured. "
            "Configure via API endpoint /api/v1/workspace/path or set PROJECTS_ROOT environment variable."
        )


# Log workspace status but allow server to start without it
# Workspace will be configured via frontend API
_log_workspace_status()


def _normalize_route(route: str) -> str:
    route = route.strip()
    if not route:
        return "/"
    if not route.startswith("/"):
        route = f"/{route}"
    if len(route) > 1 and route.endswith("/"):
        route = route.rstrip("/")
    return route


def _mount_frontend(app: FastAPI) -> str | None:
    """Mount the compiled frontend SPA using StaticFiles."""
    if not settings.SERVE_FRONTEND:
        return None

    frontend_path = Path(settings.FRONTEND_DIST_PATH).resolve()
    if not frontend_path.is_dir():
        raise RuntimeError(
            f"Configured frontend dist path '{frontend_path}' does not exist. "
            "Build the frontend (npm run build) or disable SERVE_FRONTEND."
        )

    index_file = frontend_path / "index.html"
    if not index_file.is_file():
        raise RuntimeError(
            f"Frontend dist path '{frontend_path}' is missing index.html. "
            "Ensure the SPA build completed successfully."
        )

    route = _normalize_route(settings.FRONTEND_ROUTE)
    app.mount(route, StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info("Mounted frontend assets from %s at route '%s'", frontend_path, route)
    return route


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting Flowlib Server...")
    # Workspace is already validated at module import time
    # Initialize database schema
    init_db()
    if settings.SERVE_FRONTEND and _FRONTEND_ROUTE:
        ui_url = f"{settings.APP_BASE_URL.rstrip('/')}{_FRONTEND_ROUTE}"
        logger.info("Flowlib UI available at %s", ui_url)
    # Startup: initialize database, load configurations, etc.
    yield
    # Shutdown: cleanup resources, close connections, etc.
    logger.info("Shutting down Flowlib Server...")


app = FastAPI(
    title="Flowlib Server",
    description="Backend API for Flowlib project and agent management",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers after workspace is validated (they depend on services that need workspace)
from server.api import agents, auth, configs, diff, health, knowledge, projects, workspace

# Register routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(workspace.router, prefix="/api/v1/workspace", tags=["workspace"])
app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
app.include_router(configs.router, prefix="/api/v1/configs", tags=["configs"])
app.include_router(diff.router, prefix="/api/v1/diff", tags=["diff"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])

_FRONTEND_ROUTE = _mount_frontend(app)


@app.get("/", include_in_schema=False, response_model=None)
async def root() -> RedirectResponse | dict[str, str]:
    """Root endpoint.

    Returns either a redirect to the frontend or API info.
    Uses response_model=None to allow union of Response and dict.
    """
    if settings.SERVE_FRONTEND and _FRONTEND_ROUTE:
        return RedirectResponse(url=_FRONTEND_ROUTE)
    return {"message": "Flowlib Server API", "version": "0.1.0"}

