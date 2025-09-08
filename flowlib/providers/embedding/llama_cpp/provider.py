"""
Embedding provider using llama-cpp-python.
"""

import logging
from typing import List, Union, Any, Dict, Optional, Generic, TypeVar
import asyncio
from pydantic import Field
from abc import ABC, abstractmethod

from flowlib.providers.embedding.base import EmbeddingProvider
from flowlib.providers.core.base import ProviderSettings
from flowlib.core.errors.errors import ProviderError, ConfigurationError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext, ConfigurationErrorContext

# Lazy import llama_cpp
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)

# --- Settings Model ---
class LlamaCppEmbeddingProviderSettings(ProviderSettings):
    """LlamaCpp embedding provider settings - direct inheritance, only LlamaCpp embedding fields.
    
    LlamaCpp embedding requires:
    1. Model file path
    2. Context and threading configuration
    3. Batch processing settings
    
    No host/port needed - uses local model files.
    """
    
    # LlamaCpp embedding infrastructure settings
    n_ctx: int = Field(default=512, description="Context size for embedding model")
    n_threads: Optional[int] = Field(default=None, description="Number of threads for inference")
    n_batch: int = Field(default=512, description="Batch size for embedding processing")
    
    # LlamaCpp embedding specific settings
    use_gpu: bool = Field(default=False, description="Whether to use GPU acceleration")
    n_gpu_layers: int = Field(default=0, description="Number of layers to offload to GPU")
    verbose: bool = Field(default=False, description="Enable verbose logging from LlamaCpp")
    use_mlock: bool = Field(default=False, description="Use mlock to keep model in memory")
    
    # Embedding-specific settings
    embedding_dim: Optional[int] = Field(default=None, description="Dimension of embedding vectors")
    normalize: bool = Field(default=True, description="Whether to normalize embedding vectors")
    batch_size: int = Field(default=32, description="Batch size for embedding processing")

# Type variable for settings
SettingsType = TypeVar('SettingsType', bound=LlamaCppEmbeddingProviderSettings)

# --- Provider Implementation ---
from flowlib.providers.core.decorators import provider

@provider(
    provider_type="embedding", 
    name="llamacpp_embedding", 
    settings_class=LlamaCppEmbeddingProviderSettings
)
class LlamaCppEmbeddingProvider(EmbeddingProvider[LlamaCppEmbeddingProviderSettings]):
    """Provider for local embeddings using llama-cpp-python.
    
    Supports GGUF models compatible with llama.cpp for generating embeddings.
    """
    
    def __init__(self, name: str, provider_type: str, settings: Optional[LlamaCppEmbeddingProviderSettings] = None, **kwargs: Any):
        """Initialize LlamaCppEmbeddingProvider.
        
        Args:
            name: Unique provider name.
            provider_type: The type of the provider (e.g., 'embedding')
            settings: Provider settings.
            **kwargs: Additional keyword arguments for the base EmbeddingProvider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        object.__setattr__(self, '_model', None)
        
        # --- Post-initialization validation and setup ---
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please install it with embedding support: "
                "pip install llama-cpp-python[server] or similar."
            )
        
        self._model_path = None
        self._model_config = None
        self._lock = asyncio.Lock()
        
        logger.info(f"LlamaCppEmbeddingProvider '{name}' configured")

    async def _initialize(self) -> None:
        """Initialize the embedding provider."""
        logger.info(f"LlamaCppEmbeddingProvider '{self.name}' initialized successfully")
        
    async def _initialize_model(self, model_name: str):
        """Initialize a specific embedding model.
        
        Args:
            model_name: Name of the model to initialize
            
        Raises:
            ProviderError: If initialization fails
        """
        if self._model and self._model_config and hasattr(self._model_config, 'path'):
            return  # Already loaded
            
        try:
            # Get model configuration from registry - similar to LlamaCppProvider
            model_config_raw = await self.get_model_config(model_name)
            
            # Convert to model config with path attribute
            if isinstance(model_config_raw, dict):
                # Extract path from dict
                if 'path' not in model_config_raw:
                    raise ValueError(f"Model config for '{model_name}' missing required 'path' field")
                model_path = model_config_raw['path']
                self._model_config = model_config_raw
            else:
                # Handle ModelResource format - access config dictionary
                if hasattr(model_config_raw, 'config') and isinstance(model_config_raw.config, dict):
                    if 'path' not in model_config_raw.config:
                        raise ValueError(f"Model config for '{model_name}' missing required 'path' field")
                    model_path = model_config_raw.config['path']
                    self._model_config = model_config_raw.config
                elif hasattr(model_config_raw, 'path'):
                    # Direct access to path attribute
                    model_path = model_config_raw.path
                    self._model_config = model_config_raw
                else:
                    raise ValueError(f"Model config for '{model_name}' has no accessible 'path' field")
                    
            # Load the model with the path from config
            await self._load_model(model_path)
            
            logger.info(f"Initialized embedding model '{model_name}' from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{model_name}': {e}")
            raise ProviderError(
                f"Embedding model initialization failed for '{model_name}': {str(e)}",
                context=ErrorContext.create(
                    flow_name="embedding_model_initialization",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}._initialize_model",
                    component=self.name,
                    operation="model_initialization"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="embedding",
                    operation="model_initialization",
                    retry_count=0
                ),
                cause=e
            ) from e

    async def _load_model(self, model_path: str) -> None:
        """Load a specific embedding model."""
        async with self._lock:
            if self._model and self._model_path == model_path:
                return  # Already loaded
            
            logger.info(f"Loading embedding model: {model_path}...")
            try:
                # Prepare arguments for Llama constructor using self.settings
                llama_args = {
                    "model_path": model_path,
                    "embedding": True,  # Crucial for embedding models
                    "n_ctx": self.settings.n_ctx,
                    "n_threads": self.settings.n_threads,
                    "n_batch": self.settings.n_batch,
                    "use_mlock": self.settings.use_mlock,
                    "n_gpu_layers": self.settings.n_gpu_layers,
                    "verbose": self.settings.verbose
                }
                # Remove None values
                llama_args = {k: v for k, v in llama_args.items() if v is not None}

                object.__setattr__(self, '_model', Llama(**llama_args))
                self._model_path = model_path
                logger.info(f"Embedding model loaded successfully: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{model_path}': {e}", exc_info=True)
                object.__setattr__(self, '_model', None)
                self._model_path = None
                raise ProviderError(
                    message=f"Failed to load embedding model '{model_path}': {e}",
                    context=ErrorContext.create(
                        flow_name="llama_cpp_embedding_provider",
                        error_type="ModelLoadError",
                        error_location="_load_model",
                        component=self.name,
                        operation="model_loading"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="embedding",
                        operation="model_loading",
                        retry_count=0
                    ),
                    cause=e
                )

    async def _shutdown(self) -> None:
        """Release the Llama model."""
        async with self._lock:
            if self._model:
                # llama-cpp-python doesn't have an explicit close/shutdown
                # relies on garbage collection - set to None for frozen models
                object.__setattr__(self, '_model', None)
                logger.info(f"Embedding model resources released for: {self._model_path}")
            
    async def embed(self, text: Union[str, List[str]], model_name: str = None) -> List[List[float]]:
        """Generate embeddings for the given text(s).
        
        Args:
            text: Text(s) to generate embeddings for
            model_name: Name of the model to use (optional, will use default if not provided)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ProviderError: If embedding generation fails
        """
        if not self.initialized:
            raise ProviderError(
                message="Embedding provider is not initialized.",
                context=ErrorContext.create(
                    flow_name="llama_cpp_embedding_provider",
                    error_type="StateError",
                    error_location="embed",
                    component=self.name,
                    operation="embedding_generation"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="embedding",
                    operation="embedding_generation",
                    retry_count=0
                )
            )
            
        # Initialize model if not already loaded
        if not self._model:
            if model_name:
                await self._initialize_model(model_name)
            else:
                # Use default embedding model role
                await self._initialize_model("default-embedding-model")
            
        async with self._lock:
            try:
                # llama-cpp expects a list, even for a single string
                input_texts = [text] if isinstance(text, str) else text
                if not input_texts:
                    return []
                
                # Get embeddings
                # Note: Llama.embed() might be synchronous depending on version/setup
                # If it blocks significantly, consider running in a thread pool executor
                embeddings = self._model.embed(input_texts)
                
                # Ensure the output is List[List[float]]
                if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
                     raise ProviderError(
                         message="Llama model did not return expected embedding format.",
                         context=ErrorContext.create(
                             flow_name="llama_cpp_embedding_provider",
                             error_type="ValidationError",
                             error_location="embed",
                             component=self.name,
                             operation="embedding_validation"
                         ),
                         provider_context=ProviderErrorContext(
                             provider_name=self.name,
                             provider_type="embedding",
                             operation="embedding_validation",
                             retry_count=0
                         )
                     )
                 
                return embeddings
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
                raise ProviderError(
                    message=f"Failed to generate embeddings: {e}",
                    context=ErrorContext.create(
                        flow_name="llama_cpp_embedding_provider",
                        error_type="EmbeddingError",
                        error_location="embed",
                        component=self.name,
                        operation="embedding_generation"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="embedding",
                        operation="embedding_generation",
                        retry_count=0
                    ),
                    cause=e
                ) 