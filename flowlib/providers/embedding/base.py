"""
Base class for embedding providers.
"""


from typing import List, Union, Any, Generic, TypeVar
from pydantic import Field

from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.core.errors.models import ProviderErrorContext

# Type variable for embedding settings
T = TypeVar('T', bound='EmbeddingProviderSettings')


class EmbeddingProviderSettings(ProviderSettings):
    """Settings for embedding providers."""
    
    # Model configuration
    model_name: str = Field("default", description="Name of the embedding model")
    model_path: str = Field("", description="Path to the model file (for local models)")
    
    # Performance settings
    batch_size: int = Field(32, ge=1, description="Batch size for processing multiple texts")
    max_length: int = Field(512, ge=1, description="Maximum sequence length")
    normalize: bool = Field(True, description="Whether to normalize embeddings")
    
    # Embedding dimensions
    embedding_dim: int = Field(384, ge=1, description="Dimensionality of embeddings")


class EmbeddingProvider(Provider[T], Generic[T]):
    """Base class for embedding providers.
    
    Subclasses must implement the 'embed' method.
    """
    
    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for the given text(s).
        
        Args:
            text: A single string or a list of strings to embed.
            
        Returns:
            A list of embeddings (each embedding is a list of floats).
            If input was a single string, returns a list containing one embedding.
            
        Raises:
            ProviderError: If embedding generation fails.
        """
        raise NotImplementedError("Subclasses must implement 'embed'.")

    # Optional: Add methods for specific embedding tasks if needed
    # async def embed_query(self, query: str) -> List[float]: ...
    # async def embed_documents(self, documents: List[str]) -> List[List[float]]: ...

    # Default initialize/shutdown can be inherited from Provider
    # if no specific logic is needed for the base class. 