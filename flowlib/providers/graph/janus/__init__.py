"""JanusGraph database provider module."""

from typing import Any, Optional

# Import availability and components
try:
    from .provider import JanusGraphProvider, JanusProviderSettings

    JANUS_AVAILABLE = True

    # Re-export gremlin components for test compatibility
    try:
        from gremlin_python.driver.client import Client  # type: ignore[import-untyped]
        from gremlin_python.driver.driver_remote_connection import (
            DriverRemoteConnection,  # type: ignore[import-untyped]
        )
        from gremlin_python.driver.serializer import (
            GraphBinaryMessageSerializer,  # type: ignore[import-untyped]
        )
        from gremlin_python.process.anonymous_traversal import (
            traversal,  # type: ignore[import-untyped]
        )
    except ImportError:
        Client = None
        DriverRemoteConnection = None
        traversal = None
        GraphBinaryMessageSerializer = None

except ImportError:
    JanusGraphProvider: type[Any] | None = None  # type: ignore[no-redef]
    JanusProviderSettings: type[Any] | None = None  # type: ignore[no-redef]
    JANUS_AVAILABLE = False
    Client = None
    DriverRemoteConnection = None
    traversal = None
    GraphBinaryMessageSerializer = None

__all__ = [
    "JanusGraphProvider",
    "JanusProviderSettings",
    "JANUS_AVAILABLE",
    "Client",
    "DriverRemoteConnection",
    "traversal",
    "GraphBinaryMessageSerializer",
]
