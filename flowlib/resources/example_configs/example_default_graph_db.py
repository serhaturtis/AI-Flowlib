"""Example configuration for default-graph-db role.

This file shows how to configure a graph database provider.  
Copy this file to ~/.flowlib/active_configs/default_graph_db.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import graph_db_config
from flowlib.resources.models.base import ResourceBase


@graph_db_config("default-graph-db")
class DefaultGraphDBConfig(ResourceBase):
    """Example configuration for the default graph database provider.
    
    Used for storing knowledge graphs, entity relationships, etc.
    Choose one of the supported providers below.
    """
    
    # === NEO4J (Recommended) ===
    provider_type: str = "neo4j"
    
    # Connection settings
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "your-neo4j-password"
    
    # Database settings
    database: str = "neo4j"  # Default database name
    
    # Connection pool settings
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60
    
    # Performance settings
    encrypted: bool = False  # Use encryption (set True for production)
    trust: str = "TRUST_ALL_CERTIFICATES"
    
    # === Alternative: ARANGODB ===
    # provider_type: str = "arango"
    # host: str = "localhost"
    # port: int = 8529
    # username: str = "root"
    # password: str = "your-arango-password" 
    # database: str = "flowlib_graph"
    # 
    # # Collection settings
    # vertex_collection: str = "entities"
    # edge_collection: str = "relationships"
    
    # === Alternative: AMAZON NEPTUNE ===
    # provider_type: str = "neptune"
    # endpoint: str = "your-neptune-cluster.cluster-xyz.us-east-1.neptune.amazonaws.com"
    # port: int = 8182
    # use_iam: bool = True
    # region: str = "us-east-1"
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(name=name, type=type)