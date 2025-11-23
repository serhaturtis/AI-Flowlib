"""Example vector database provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Alias bindings are handled separately in ~/.flowlib/configs/aliases.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import vector_db_config


@vector_db_config("example-vector-db-provider")
class ExampleVectorDBProviderConfig:
    """Example configuration for the default vector database provider.

    Used for storing and searching document embeddings, memory vectors, etc.
    Choose one of the supported providers below.
    """

    # === CHROMA (Local/Simple) ===
    provider_type = "chroma"
    settings = {
        # Chroma settings
        "host": "localhost",
        "port": 8000,
        # Collection settings
        "collection_name": "flowlib_vectors",
        "distance_metric": "cosine",  # cosine, euclidean, manhattan
        # Storage settings
        "persist_directory": "~/.flowlib/chroma_data",
    }

    # === Alternative: QDRANT ===
    # provider_type = "qdrant"
    # settings = {
    #     "host": "localhost",
    #     "port": 6333,
    #     "prefer_grpc": False,
    #     "collection_name": "flowlib_vectors",
    #     "distance_metric": "Cosine"
    # }

    # === Alternative: PINECONE ===
    # provider_type = "pinecone"
    # settings = {
    #     "api_key": "your-pinecone-api-key",
    #     "environment": "us-west1-gcp",
    #     "index_name": "flowlib-vectors",
    #     "dimension": 384  # Must match your embedding model
    # }

    # === Alternative: WEAVIATE ===
    # provider_type = "weaviate"
    # settings = {
    #     "url": "http://localhost:8080",
    #     "api_key": "optional-api-key",
    #     "class_name": "FlowlibDocument"
    # }
