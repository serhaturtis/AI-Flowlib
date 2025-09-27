"""Neo4j graph database provider module."""

from typing import Any, Type

# Define default classes first (single source of truth)
class _DefaultNeo4jProvider:
    """Default Neo4j provider stub."""
    pass

class _DefaultNeo4jProviderSettings:
    """Default Neo4j provider settings stub."""
    pass

class _DefaultGraphDatabase:
    """Default GraphDatabase stub."""
    pass

class _DefaultEntity:
    """Default Entity stub."""
    pass

# Type variables for components that can have multiple types
Neo4jProvider: Type[Any]
Neo4jProviderSettings: Type[Any]
GraphDatabase: Type[Any]
Entity: Type[Any]

# Import availability and components
try:
    from .provider import Neo4jProvider as _Neo4jProvider, Neo4jProviderSettings as _Neo4jProviderSettings
    NEO4J_AVAILABLE = True

    # Use actual imports when available
    Neo4jProvider = _Neo4jProvider
    Neo4jProviderSettings = _Neo4jProviderSettings

    # Re-export neo4j components for test compatibility
    try:
        from neo4j import GraphDatabase as _GraphDatabase
        GraphDatabase = _GraphDatabase
    except ImportError:
        GraphDatabase = _DefaultGraphDatabase

    # Import Entity from core models for test compatibility
    try:
        from flowlib.providers.graph.base import Entity as _Entity
        Entity = _Entity
    except ImportError:
        Entity = _DefaultEntity

except ImportError:
    Neo4jProvider = _DefaultNeo4jProvider
    Neo4jProviderSettings = _DefaultNeo4jProviderSettings
    NEO4J_AVAILABLE = False
    GraphDatabase = _DefaultGraphDatabase
    Entity = _DefaultEntity

__all__ = [
    "Neo4jProvider",
    "Neo4jProviderSettings", 
    "NEO4J_AVAILABLE",
    "GraphDatabase",
    "Entity",
]