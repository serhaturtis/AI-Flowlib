"""Workspace discovery models with strict Pydantic validation.

Single source of truth for workspace manifest and discovery configuration.
No fallbacks, no defaults that bypass validation.
"""

from typing import Any

from pydantic import Field, field_validator

from flowlib.core.models import StrictBaseModel


class DomainArtifact(StrictBaseModel):
    """Lightweight representation of a discovered workspace artifact.

    Strict contract - all fields required except metadata.
    No assumptions about artifact structure.
    """

    name: str = Field(..., description="Artifact name (e.g., 'Hellstorm', 'MyProject')")
    path: str = Field(..., description="Absolute path to artifact")
    artifact_type: str = Field(
        ..., description="Type identifier (e.g., 'music_session', 'code_project')"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Domain-specific metadata (optional)"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not empty."""
        if not v or not v.strip():
            raise ValueError("Artifact name cannot be empty")
        return v.strip()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path is not empty."""
        if not v or not v.strip():
            raise ValueError("Artifact path cannot be empty")
        return v

    @field_validator("artifact_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure artifact_type is not empty."""
        if not v or not v.strip():
            raise ValueError("Artifact type cannot be empty")
        return v.strip()


class WorkspaceManifest(StrictBaseModel):
    """Lightweight manifest of workspace artifacts.

    Single source of truth for discovered workspace state.
    Immutable once created (frozen=True prevents modification).
    """

    working_directory: str = Field(..., description="Root directory that was scanned")
    domains: dict[str, list[DomainArtifact]] = Field(
        default_factory=dict, description="Artifacts grouped by domain (e.g., 'music', 'projects')"
    )
    scan_timestamp: float = Field(..., description="Unix timestamp of scan completion")

    model_config = {"frozen": True}  # Immutable after creation

    @field_validator("working_directory")
    @classmethod
    def validate_directory(cls, v: str) -> str:
        """Ensure working_directory is not empty."""
        if not v or not v.strip():
            raise ValueError("Working directory cannot be empty")
        return v

    @field_validator("scan_timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        """Ensure timestamp is positive."""
        if v <= 0:
            raise ValueError("Scan timestamp must be positive")
        return v


class WorkspaceDiscoveryConfig(StrictBaseModel):
    """Configuration for workspace discovery component.

    Strict validation - no silent fallbacks.
    All behavior explicitly configured.
    """

    enabled: bool = Field(default=True, description="Enable workspace discovery")
    cache_ttl_seconds: int = Field(
        default=60, ge=0, description="Cache TTL in seconds (0 = no cache)"
    )
    excluded_domains: list[str] = Field(
        default_factory=list, description="Domains to exclude from scanning"
    )
    max_scan_depth: int = Field(
        default=3, ge=1, le=10, description="Maximum directory depth for recursive scanning"
    )
    fail_on_scanner_error: bool = Field(
        default=False,
        description="If True, fail discovery on any scanner error. If False, log and continue.",
    )

    def should_scan_domain(self, domain: str) -> bool:
        """Check if domain should be scanned.

        Args:
            domain: Domain name to check

        Returns:
            True if domain should be scanned, False if excluded
        """
        if not domain:
            raise ValueError("Domain name cannot be empty")
        return domain not in self.excluded_domains
