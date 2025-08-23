"""Example embedding model configuration.

This file shows how to configure a specific embedding model using @model_config.
Copy this file to ~/.flowlib/active_configs/my_embedding_model.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import model_config
from flowlib.core.models import StrictBaseModel
from pydantic import Field


class BGEEmbeddingModelConfig(StrictBaseModel):
    """Configuration for BGE embedding models."""
    
    path: str = Field(..., description="Path to the embedding model file")
    model_name: str = Field(..., description="Name identifier for the model")
    dimensions: int = Field(..., description="Embedding vector dimensions")
    max_length: int = Field(..., description="Maximum input text length")
    normalize: bool = Field(..., description="Whether to normalize embeddings")


@model_config("bge-m3-embedding", provider_type="llamacpp_embedding")  
class BGEM3EmbeddingModelConfig(BGEEmbeddingModelConfig):
    """Example configuration for BGE-M3 embedding model.
    
    This creates a named embedding model configuration that uses the "llamacpp_embedding"
    provider type to load and run this specific embedding model file.
    Model configs specify the provider_type, provider configs handle infrastructure.
    """
    
    def __init__(self):
        super().__init__(
            # REQUIRED: Model file and identification
            path="/path/to/models/bge-m3-f16.gguf",
            model_name="bge-m3",
            
            # REQUIRED: Model capabilities
            dimensions=1024,        # BGE-M3 produces 1024-dimensional embeddings
            max_length=8192,        # BGE-M3 supports long context (8K tokens)
            normalize=True,         # Always normalize BGE embeddings
        )