"""
Base class for embedding providers.
"""


import logging
from typing import Dict, Generic, List, TypeVar, Union

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)

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

    async def get_model_config(self, model_name: str) -> Dict[str, object]:
        """Get configuration for a model from the resource registry.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ProviderError: If model is not found or invalid
        """
        try:
            # Use legitimate registry method - this is NOT a fallback pattern
            resource = resource_registry.get(model_name)

            # Log the model config for debugging
            logger.info(f"Retrieved embedding model resource for '{model_name}': {resource}")

            # Convert ResourceBase to dict for backward compatibility with existing code
            model_config = resource.model_dump()

            return model_config
        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="model_config_retrieval",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.get_model_config",
                component=self.name,
                operation="get_model_config"
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="embedding",
                operation="get_model_config",
                retry_count=0
            )

            raise ProviderError(
                message=f"Failed to get embedding model config '{model_name}': {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e
            ) from e

    # Optional: Add methods for specific embedding tasks if needed
    # async def embed_query(self, query: str) -> List[float]: ...
    # async def embed_documents(self, documents: List[str]) -> List[List[float]]: ...
