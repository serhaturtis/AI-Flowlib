"""Neo4j graph database provider module."""

# Import availability and components
try:
    from .provider import Neo4jProvider, Neo4jProviderSettings
    NEO4J_AVAILABLE = True
    
    # Re-export neo4j components for test compatibility
    try:
        from neo4j import GraphDatabase
    except ImportError:
        GraphDatabase = None
        
    # Import Entity from core models for test compatibility
    try:
        from flowlib.providers.graph.base import Entity
    except ImportError:
        Entity = None
        
except ImportError:
    Neo4jProvider = None
    Neo4jProviderSettings = None
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    Entity = None

__all__ = [
    "Neo4jProvider",
    "Neo4jProviderSettings", 
    "NEO4J_AVAILABLE",
    "GraphDatabase",
    "Entity",
]