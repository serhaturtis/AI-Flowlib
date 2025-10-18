"""ArangoDB graph database provider module."""

from typing import Any, Optional, Type

# Type variables for components that can be None or actual types
ArangoProvider: Optional[Type[Any]]
ArangoProviderSettings: Optional[Type[Any]]

# Import availability and components
try:
    from .provider import ArangoProvider, ArangoProviderSettings
    ARANGO_AVAILABLE = True
except ImportError:
    ArangoProvider = None
    ArangoProviderSettings = None
    ARANGO_AVAILABLE = False

__all__ = [
    "ArangoProvider",
    "ArangoProviderSettings",
    "ARANGO_AVAILABLE",
]
