"""Workspace discovery component package.

Provides workspace artifact discovery for context-aware planning.
"""

from .component import WorkspaceDiscoveryComponent
from .models import (
    DomainArtifact,
    WorkspaceDiscoveryConfig,
    WorkspaceManifest,
)
from .scanners import BaseDomainScanner, ScannerRegistry

__all__ = [
    "WorkspaceDiscoveryComponent",
    "DomainArtifact",
    "WorkspaceDiscoveryConfig",
    "WorkspaceManifest",
    "BaseDomainScanner",
    "ScannerRegistry",
]
