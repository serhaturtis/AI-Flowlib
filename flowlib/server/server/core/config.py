"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Flowlib Server"
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=True, description="Debug mode (defaults to True for development)")

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description=(
            "Allowed CORS origins. "
            "IMPORTANT: Override this in production via environment variables to restrict to actual production URLs. "
            "Example: CORS_ORIGINS='[\"https://app.example.com\"]'"
        ),
    )

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./flowlib.db",
        description="Database connection URL",
    )

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Frontend / SPA hosting
    SERVE_FRONTEND: bool = Field(
        default=True,
        description="Serve the compiled React SPA via StaticFiles",
    )
    FRONTEND_DIST_PATH: str = Field(
        default=str(Path(__file__).resolve().parents[2] / "ui" / "dist"),
        description="Absolute path to the compiled frontend dist directory",
    )
    FRONTEND_ROUTE: str = Field(
        default="/app",
        description="Route prefix where the SPA is mounted (e.g., '/', '/app')",
    )
    APP_BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Public base URL used when logging UI location",
    )

    # Projects / filesystem
    PROJECTS_ROOT: str | None = Field(
        default=None,
        description="Root directory for Flowlib projects workspace. If not set, will be prompted on first start.",
    )


settings = Settings()

