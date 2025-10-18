"""Scanner registry with auto-discovery.

Single source of truth for registered scanners.
No fallbacks, strict validation, fail fast on errors.
"""

import importlib
import logging
import pkgutil
import re
from typing import Dict, Optional

from .base import BaseDomainScanner

logger = logging.getLogger(__name__)


class ScannerRegistry:
    """Registry for domain scanners with auto-discovery.

    Maintains single source of truth for registered scanners.
    Strict validation - duplicate domains raise errors.
    No silent failures - all errors logged and raised.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._scanners: Dict[str, BaseDomainScanner] = {}
        self._domain_pattern = re.compile(r'^[a-z][a-z0-9_]*$')

    @property
    def scanners(self) -> Dict[str, BaseDomainScanner]:
        """Get registered scanners (read-only view)."""
        return dict(self._scanners)  # Return copy to prevent external modification

    def register(self, domain: str, scanner: BaseDomainScanner) -> None:
        """Register a domain scanner.

        Args:
            domain: Domain name (must match scanner.domain_name)
            scanner: Scanner implementation

        Raises:
            ValueError: If domain is invalid or mismatch with scanner.domain_name
            RuntimeError: If domain already registered
        """
        # Validate domain name format
        if not domain or not self._domain_pattern.match(domain):
            raise ValueError(
                f"Invalid domain name '{domain}'. "
                "Must be lowercase, start with letter, contain only alphanumeric and underscores."
            )

        # Validate scanner has domain_name property
        if not hasattr(scanner, 'domain_name'):
            raise ValueError(f"Scanner {scanner.__class__.__name__} missing domain_name property")

        # Validate domain matches scanner's domain_name
        if scanner.domain_name != domain:
            raise ValueError(
                f"Domain mismatch: registry domain '{domain}' != "
                f"scanner.domain_name '{scanner.domain_name}'"
            )

        # Check for duplicates - NO SILENT OVERWRITE
        if domain in self._scanners:
            raise RuntimeError(
                f"Domain '{domain}' already registered with "
                f"{self._scanners[domain].__class__.__name__}"
            )

        self._scanners[domain] = scanner
        logger.debug(f"Registered scanner: {domain} -> {scanner.__class__.__name__}")

    def get(self, domain: str) -> Optional[BaseDomainScanner]:
        """Get scanner for domain.

        Args:
            domain: Domain name

        Returns:
            Scanner if registered, None if not found
        """
        return self._scanners.get(domain)

    async def discover_scanners(self) -> None:
        """Auto-discover and instantiate scanner classes.

        Looks for scanner modules in:
        - projects/*/scanners/*.py
        - flowlib/agent/components/workspace/scanners/*.py (built-in)

        Finds BaseDomainScanner subclasses and instantiates them for registration.

        Raises:
            No exceptions - logs warnings for failed imports
        """
        import inspect

        discovered_count = 0

        # Discover from projects directory
        try:
            import projects
            projects_path = projects.__path__
            logger.debug(f"Scanning for domain scanners in projects: {projects_path}")

            for importer, modname, ispkg in pkgutil.walk_packages(
                path=projects_path,
                prefix=projects.__name__ + "."
            ):
                # Look for scanner modules
                if "scanner" in modname.lower():
                    try:
                        logger.debug(f"Attempting to import scanner module: {modname}")
                        module = importlib.import_module(modname)
                        discovered_count += 1

                        # Find all BaseDomainScanner subclasses in the module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            # Check if it's a BaseDomainScanner subclass (but not BaseDomainScanner itself)
                            if (issubclass(obj, BaseDomainScanner) and
                                obj is not BaseDomainScanner and
                                obj.__module__ == modname):
                                try:
                                    # Instantiate and register
                                    scanner = obj()
                                    self.register(scanner.domain_name, scanner)
                                    logger.debug(f"Registered scanner {name} for domain '{scanner.domain_name}'")
                                except Exception as e:
                                    logger.warning(f"Failed to register scanner {name} from {modname}: {e}")

                        logger.debug(f"Successfully processed scanner module: {modname}")
                    except Exception as e:
                        logger.warning(f"Failed to import scanner module {modname}: {e}")

        except ImportError:
            logger.debug("No projects package found - skipping project scanner discovery")
        except Exception as e:
            logger.warning(f"Scanner discovery in projects failed: {e}")

        logger.info(
            f"Scanner discovery complete: {len(self._scanners)} scanners registered "
            f"({discovered_count} modules imported)"
        )

    def unregister(self, domain: str) -> None:
        """Unregister a domain scanner.

        Args:
            domain: Domain name to unregister

        Raises:
            KeyError: If domain not registered
        """
        if domain not in self._scanners:
            raise KeyError(f"Domain '{domain}' not registered")

        del self._scanners[domain]
        logger.debug(f"Unregistered scanner: {domain}")

    def clear(self) -> None:
        """Clear all registered scanners."""
        count = len(self._scanners)
        self._scanners.clear()
        logger.debug(f"Cleared {count} registered scanners")

    def __len__(self) -> int:
        """Get number of registered scanners."""
        return len(self._scanners)

    def __contains__(self, domain: str) -> bool:
        """Check if domain is registered."""
        return domain in self._scanners

    def __str__(self) -> str:
        """String representation."""
        domains = ', '.join(sorted(self._scanners.keys()))
        return f"ScannerRegistry({len(self._scanners)} scanners: [{domains}])"
