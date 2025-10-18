"""Example vector database provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from typing import Any

from flowlib.resources.decorators.decorators import vector_db_config
from flowlib.resources.models.config_resource import VectorDBConfigResource


@vector_db_config("example-vector-db-provider")
class ExampleVectorDBProviderConfig(VectorDBConfigResource):
    """Example configuration for the default vector database provider.
    
    Used for storing and searching document embeddings, memory vectors, etc.
    Choose one of the supported providers below.
    """

    # === CHROMA (Local/Simple) ===
    provider_type: str = "chroma"

    # Chroma settings
    host: str = "localhost"
    port: int = 8000

    # Collection settings
    collection_name: str = "flowlib_vectors"
    distance_metric: str = "cosine"  # cosine, euclidean, manhattan

    # Storage settings
    persist_directory: str = "~/.flowlib/chroma_data"

    # === Alternative: QDRANT ===
    # provider_type: str = "qdrant"
    # host: str = "localhost"
    # port: int = 6333
    # prefer_grpc: bool = False
    # collection_name: str = "flowlib_vectors"
    # distance_metric: str = "Cosine"

    # === Alternative: PINECONE ===
    # provider_type: str = "pinecone"
    # api_key: str = "your-pinecone-api-key"
    # environment: str = "us-west1-gcp"
    # index_name: str = "flowlib-vectors"
    # dimension: int = 384  # Must match your embedding model

    # === Alternative: WEAVIATE ===
    # provider_type: str = "weaviate"
    # url: str = "http://localhost:8080"
    # api_key: str = "optional-api-key"
    # class_name: str = "FlowlibDocument"

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(name=name, type=type, provider_type=self.provider_type)
