"""Base scanner interface for workspace discovery.

Defines strict contract for domain scanners.
No optional methods, no fallbacks - fail fast if contract not met.
"""

from abc import ABC, abstractmethod

from ..models import DomainArtifact


class BaseDomainScanner(ABC):
    """Base interface for domain-specific workspace scanners.

    Scanners MUST:
    1. Implement scan() to discover artifacts
    2. Provide domain_name property
    3. Return valid DomainArtifact list (can be empty)
    4. Raise exceptions on errors (no silent failures)

    Scanners MUST NOT:
    1. Modify workspace
    2. Load full artifacts (metadata only)
    3. Make assumptions about other domains
    4. Cache results (handled by component)
    """

    @abstractmethod
    async def scan(self, working_directory: str) -> list[DomainArtifact]:
        """Scan working directory for domain artifacts.

        MUST return list of DomainArtifact (can be empty list).
        MUST raise exception on errors (no silent failures).
        MUST NOT modify workspace.

        Args:
            working_directory: Absolute path to scan

        Returns:
            List of discovered artifacts (empty if none found)

        Raises:
            ValueError: If working_directory is invalid
            OSError: If directory access fails
            Any other exception on scanner-specific errors
        """
        pass

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Get domain name for this scanner.

        MUST return non-empty string.
        MUST be unique across all scanners.
        MUST be lowercase, alphanumeric + underscores only.

        Returns:
            Domain name (e.g., 'music', 'code_projects', 'documents')

        Raises:
            NotImplementedError: If not implemented in subclass
        """
        pass

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(domain='{self.domain_name}')"
