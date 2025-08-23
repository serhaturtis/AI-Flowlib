"""Graph database provider interface.

This package provides providers for graph databases with entity and relationship operations,
supporting knowledge graph capabilities for the entity-centric memory system.
"""

from .base import GraphDBProvider, GraphDBProviderSettings
from .memory_graph import MemoryGraphProvider

# Import new providers
try:
    from .neo4j.provider import Neo4jProvider
    from .neo4j.models import Neo4jProviderSettings
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from .arango.provider import ArangoProvider
    from .arango.models import ArangoProviderSettings
    ARANGO_AVAILABLE = True
except ImportError:
    ARANGO_AVAILABLE = False

try:
    from .janus.provider import JanusProvider
    from .janus.models import JanusProviderSettings
    JANUS_AVAILABLE = True
except ImportError:
    JANUS_AVAILABLE = False

# Build exports dynamically
__all__ = [
    "GraphDBProvider",
    "GraphDBProviderSettings",
    "MemoryGraphProvider"
]

# Add new providers to exports if available
if NEO4J_AVAILABLE:
    __all__.extend(["Neo4jProvider", "Neo4jProviderSettings"])

if ARANGO_AVAILABLE:
    __all__.extend(["ArangoProvider", "ArangoProviderSettings"])

if JANUS_AVAILABLE:
    __all__.extend(["JanusProvider", "JanusProviderSettings"])

# Export availability flags
__all__.extend(["NEO4J_AVAILABLE", "ARANGO_AVAILABLE", "JANUS_AVAILABLE"])

# Add module aliases for test compatibility - always set these attributes
try:
    from . import neo4j
    neo4j_provider = neo4j
except ImportError:
    # Create a dummy module-like object with NEO4J_AVAILABLE
    import types
    neo4j_provider = types.ModuleType('neo4j_provider')
    neo4j_provider.NEO4J_AVAILABLE = NEO4J_AVAILABLE

try:
    from . import arango
    arango_provider = arango
except ImportError:
    import types
    arango_provider = types.ModuleType('arango_provider')
    arango_provider.ARANGO_AVAILABLE = ARANGO_AVAILABLE

try:
    from . import janus
    janus_provider = janus
except ImportError:
    import types
    janus_provider = types.ModuleType('janus_provider')
    janus_provider.JANUS_AVAILABLE = JANUS_AVAILABLE

