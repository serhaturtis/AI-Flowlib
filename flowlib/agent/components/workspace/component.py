"""Workspace discovery component for context-aware planning.

Discovers domain artifacts in workspace to provide context for planning.
Follows flowlib AgentComponent patterns with strict validation.
"""

import logging
import time

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError

from .models import DomainArtifact, WorkspaceDiscoveryConfig, WorkspaceManifest
from .scanners import BaseDomainScanner, ScannerRegistry

logger = logging.getLogger(__name__)


class WorkspaceDiscoveryComponent(AgentComponent):
    """Discovers domain artifacts in workspace for context-aware planning.

    This component scans the workspace for domain-specific artifacts
    (music sessions, code projects, documents, etc.) and provides a
    lightweight manifest for planning and execution components.

    Follows flowlib principles:
    - Single source of truth for workspace state
    - Domain-agnostic core with extensible scanners
    - Fast, lightweight manifest generation (metadata only)
    - No assumptions about domain structure
    - Fail fast - no silent fallbacks
    - Strict Pydantic validation
    """

    def __init__(
        self, config: WorkspaceDiscoveryConfig | None = None, name: str = "workspace_discovery"
    ):
        """Initialize workspace discovery component.

        Args:
            config: Discovery configuration (creates default if None)
            name: Component name
        """
        super().__init__(name)
        self._config = config if config is not None else WorkspaceDiscoveryConfig()
        self._scanner_registry = ScannerRegistry()
        self._manifest_cache: WorkspaceManifest | None = None
        self._cache_timestamp: float | None = None

    async def _initialize_impl(self) -> None:
        """Initialize scanners and registry.

        Raises:
            ExecutionError: If scanner discovery fails critically
        """
        logger.info("Initializing workspace discovery component")

        if not self._config.enabled:
            logger.info("Workspace discovery disabled by configuration")
            return

        # Auto-discover and register domain scanners
        try:
            await self._scanner_registry.discover_scanners()
        except Exception as e:
            raise ExecutionError(f"Failed to discover scanners: {e}") from e

        logger.info(
            f"Workspace discovery initialized: {len(self._scanner_registry)} "
            f"scanner(s) registered: {list(self._scanner_registry.scanners.keys())}"
        )

    async def _shutdown_impl(self) -> None:
        """Cleanup resources."""
        self._manifest_cache = None
        self._cache_timestamp = None
        self._scanner_registry.clear()
        logger.info("Workspace discovery component shutdown complete")

    async def scan_workspace(
        self, working_directory: str, force_refresh: bool = False
    ) -> WorkspaceManifest:
        """Scan workspace for domain artifacts.

        Args:
            working_directory: Absolute path to workspace root
            force_refresh: If True, bypass cache and rescan

        Returns:
            WorkspaceManifest with discovered artifacts

        Raises:
            ValueError: If working_directory is invalid
            RuntimeError: If component not initialized or disabled
            ExecutionError: If scanning fails and fail_on_scanner_error=True
        """
        self._check_initialized()

        if not self._config.enabled:
            raise RuntimeError("Workspace discovery is disabled by configuration")

        # Validate input
        if not working_directory or not working_directory.strip():
            raise ValueError("Working directory cannot be empty")

        working_directory = working_directory.strip()

        # Check cache
        if not force_refresh and self._is_cache_valid(working_directory):
            logger.debug("Returning cached workspace manifest")
            assert (
                self._manifest_cache is not None
            )  # Type guard - cache valid means manifest exists
            return self._manifest_cache

        logger.debug(f"Scanning workspace: {working_directory}")
        scan_start = time.time()

        # Run all registered scanners
        domain_artifacts = {}
        scan_errors = []

        for domain_name, scanner in self._scanner_registry.scanners.items():
            # Check if domain should be scanned
            if not self._config.should_scan_domain(domain_name):
                logger.debug(f"Skipping excluded domain: {domain_name}")
                continue

            try:
                logger.debug(f"Running scanner: {domain_name}")
                artifacts = await scanner.scan(working_directory)

                # Validate result type
                if not isinstance(artifacts, list):
                    raise TypeError(
                        f"Scanner {domain_name} returned {type(artifacts)}, expected List[DomainArtifact]"
                    )

                # Validate all artifacts
                for artifact in artifacts:
                    if not isinstance(artifact, DomainArtifact):
                        raise TypeError(
                            f"Scanner {domain_name} returned invalid artifact type: {type(artifact)}"
                        )

                if artifacts:
                    domain_artifacts[domain_name] = artifacts
                    logger.debug(f"Scanner '{domain_name}' found {len(artifacts)} artifact(s)")
                else:
                    logger.debug(f"Scanner '{domain_name}' found no artifacts")

            except Exception as e:
                error_msg = f"Scanner '{domain_name}' failed: {e}"
                scan_errors.append(error_msg)

                if self._config.fail_on_scanner_error:
                    raise ExecutionError(error_msg) from e
                else:
                    logger.warning(error_msg)

        # Build manifest (always succeeds, even if all scanners failed)
        scan_duration = time.time() - scan_start
        manifest = WorkspaceManifest(
            working_directory=working_directory,
            domains=domain_artifacts,
            scan_timestamp=time.time(),
        )

        # Update cache
        self._manifest_cache = manifest
        self._cache_timestamp = time.time()

        logger.info(
            f"Workspace scan complete: {len(domain_artifacts)} domain(s) "
            f"with {sum(len(arts) for arts in domain_artifacts.values())} total artifact(s) "
            f"in {scan_duration:.2f}s"
        )

        if scan_errors:
            logger.warning(f"Scan completed with {len(scan_errors)} error(s)")

        return manifest

    def _is_cache_valid(self, working_directory: str) -> bool:
        """Check if cached manifest is still valid.

        Args:
            working_directory: Directory to check cache for

        Returns:
            True if cache is valid for this directory
        """
        if not self._manifest_cache or not self._cache_timestamp:
            return False

        # Check if directory matches
        if self._manifest_cache.working_directory != working_directory:
            return False

        # Check TTL (0 = no cache)
        if self._config.cache_ttl_seconds == 0:
            return False

        age = time.time() - self._cache_timestamp
        return age < self._config.cache_ttl_seconds

    def register_scanner(self, domain: str, scanner: BaseDomainScanner) -> None:
        """Register a domain scanner.

        Args:
            domain: Domain name
            scanner: Scanner implementation

        Raises:
            ValueError: If domain or scanner is invalid
            RuntimeError: If domain already registered
        """
        self._scanner_registry.register(domain, scanner)
        logger.debug(f"Registered scanner for domain: {domain}")

        # Invalidate cache when new scanner added
        self._manifest_cache = None
        self._cache_timestamp = None

    def clear_cache(self) -> None:
        """Clear cached manifest."""
        self._manifest_cache = None
        self._cache_timestamp = None
        logger.debug("Workspace manifest cache cleared")

    @property
    def config(self) -> WorkspaceDiscoveryConfig:
        """Get component configuration (read-only)."""
        return self._config

    @property
    def registered_domains(self) -> list[str]:
        """Get list of registered domain names."""
        return sorted(self._scanner_registry.scanners.keys())
