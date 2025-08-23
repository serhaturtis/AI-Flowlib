"""Example configuration for default-vector-db role.

This file shows how to configure a vector database provider.
Copy this file to ~/.flowlib/active_configs/default_vector_db.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import vector_db_config
from flowlib.resources.models.base import ResourceBase


@vector_db_config("default-vector-db")
class DefaultVectorDBConfig(ResourceBase):
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
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(name=name, type=type)