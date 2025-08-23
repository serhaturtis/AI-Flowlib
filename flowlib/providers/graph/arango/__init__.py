"""ArangoDB graph database provider module."""

# Import availability and components
try:
    from .provider import ArangoProvider
    from .models import ArangoProviderSettings
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