"""Example graph database provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Alias bindings are handled separately in ~/.flowlib/configs/aliases.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import graph_db_config


@graph_db_config("example-graph-db-provider")
class ExampleGraphDBProviderConfig:
    """Example configuration for the default graph database provider.

    Used for storing knowledge graphs, entity relationships, etc.
    Choose one of the supported providers below.
    """

    # === NEO4J (Recommended) ===
    provider_type = "neo4j"
    settings = {
        # Connection settings
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "flowlib123",
        # Database settings
        "database": "neo4j",  # Default database name
        # Connection pool settings
        "max_connection_lifetime": 3600,
        "max_connection_pool_size": 50,
        "connection_acquisition_timeout": 60,
        # Performance settings
        "encryption": False,  # Use encryption (set True for production)
        "trust": "TRUST_ALL_CERTIFICATES",
    }

    # === Alternative: ARANGODB ===
    # provider_type = "arango"
    # settings = {
    #     "host": "localhost",
    #     "port": 8529,
    #     "username": "root",
    #     "password": "your-arango-password",
    #     "database": "flowlib_graph",
    #     "vertex_collection": "entities",
    #     "edge_collection": "relationships"
    # }

    # === Alternative: AMAZON NEPTUNE ===
    # provider_type = "neptune"
    # settings = {
    #     "endpoint": "your-neptune-cluster.cluster-xyz.us-east-1.neptune.amazonaws.com",
    #     "port": 8182,
    #     "use_iam": True,
    #     "region": "us-east-1"
    # }
