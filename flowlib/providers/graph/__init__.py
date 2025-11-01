"""Graph database provider interface.

This package provides providers for graph databases with entity and relationship operations,
supporting knowledge graph capabilities for the entity-centric memory system.
"""

import importlib.util

from flowlib.providers.graph.base import GraphDBProvider, GraphDBProviderSettings
from flowlib.providers.graph.memory_graph import MemoryGraphProvider

# Check for optional dependencies
NEO4J_AVAILABLE = importlib.util.find_spec("neo4j") is not None
ARANGO_AVAILABLE = importlib.util.find_spec("python-arango") is not None
JANUS_AVAILABLE = importlib.util.find_spec("gremlinpython") is not None

# Import providers only if dependencies are available
if NEO4J_AVAILABLE:
    from flowlib.providers.graph.neo4j.provider import (  # noqa: F401
        Neo4jProvider,
        Neo4jProviderSettings,
    )

if ARANGO_AVAILABLE:
    from flowlib.providers.graph.arango.provider import (  # noqa: F401
        ArangoProvider,
        ArangoProviderSettings,
    )

if JANUS_AVAILABLE:
    from flowlib.providers.graph.janus.provider import (  # noqa: F401
        JanusGraphProvider,
        JanusProviderSettings,
    )

# Build exports dynamically
__all__ = ["GraphDBProvider", "GraphDBProviderSettings", "MemoryGraphProvider"]

# Add new providers to exports if available
if NEO4J_AVAILABLE:
    __all__.extend(["Neo4jProvider", "Neo4jProviderSettings"])

if ARANGO_AVAILABLE:
    __all__.extend(["ArangoProvider", "ArangoProviderSettings"])

if JANUS_AVAILABLE:
    __all__.extend(["JanusProvider", "JanusProviderSettings"])

# Export availability flags
__all__.extend(["NEO4J_AVAILABLE", "ARANGO_AVAILABLE", "JANUS_AVAILABLE"])
