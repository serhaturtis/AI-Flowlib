"""
Embedding provider using llama-cpp-python.
"""

import asyncio
import logging
from typing import Any, TypeVar

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.embedding.base import (
    EmbeddingProvider,
    EmbeddingProviderSettings,
)

# Lazy import llama_cpp
try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger(__name__)


# --- Settings Model ---
class LlamaCppEmbeddingProviderSettings(EmbeddingProviderSettings):
    """LlamaCpp embedding provider settings - direct inheritance, only LlamaCpp embedding fields.

    LlamaCpp embedding requires:
    1. Model file path
    2. Context and threading configuration
    3. Batch processing settings

    No host/port needed - uses local model files.
    """

    # LlamaCpp embedding infrastructure settings
    n_ctx: int = Field(default=512, description="Context size for embedding model")
    n_threads: int | None = Field(default=None, description="Number of threads for inference")
    n_batch: int = Field(default=512, description="Batch size for embedding processing")

    # LlamaCpp embedding specific settings
    use_gpu: bool = Field(default=False, description="Whether to use GPU acceleration")
    n_gpu_layers: int = Field(default=0, description="Number of layers to offload to GPU")
    verbose: bool = Field(default=False, description="Enable verbose logging from LlamaCpp")
    use_mlock: bool = Field(default=False, description="Use mlock to keep model in memory")

    # Embedding-specific settings
    embedding_dim: int = Field(default=384, description="Dimension of embedding vectors")
    normalize: bool = Field(default=True, description="Whether to normalize embedding vectors")
    batch_size: int = Field(default=32, description="Batch size for embedding processing")


# Type variable for settings
SettingsType = TypeVar("SettingsType", bound=LlamaCppEmbeddingProviderSettings)

# --- Provider Implementation ---


@provider(
    provider_type="embedding",
    name="llamacpp_embedding",
    settings_class=LlamaCppEmbeddingProviderSettings,
)
class LlamaCppEmbeddingProvider(EmbeddingProvider[LlamaCppEmbeddingProviderSettings]):
    """Provider for local embeddings using llama-cpp-python.

    Supports GGUF models compatible with llama.cpp for generating embeddings.
    """

    def __init__(
        self,
        name: str,
        provider_type: str,
        settings: LlamaCppEmbeddingProviderSettings,
        **kwargs: object,
    ):
        """Initialize LlamaCppEmbeddingProvider.

        Args:
            name: Unique provider name.
            provider_type: The type of the provider (e.g., 'embedding')
            settings: Provider settings.
            **kwargs: Additional keyword arguments for the base EmbeddingProvider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)

        # Initialize model-related attributes
        self._model: Any | None = None

        # --- Post-initialization validation and setup ---
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please install it with embedding support: "
                "pip install llama-cpp-python[server] or similar."
            )

        self._model_path: str | None = None
        self._model_config: dict[str, object] | None = None
        self._lock = asyncio.Lock()

        logger.info(f"LlamaCppEmbeddingProvider '{name}' configured")

    async def _initialize(self) -> None:
        """Initialize the embedding provider."""
        logger.info(f"LlamaCppEmbeddingProvider '{self.name}' initialized successfully")

    async def _initialize_model(self, model_name: str) -> None:
        """Initialize a specific embedding model.

        Args:
            model_name: Name of the model to initialize

        Raises:
            ProviderError: If initialization fails
        """
        if self._model and self._model_config and hasattr(self._model_config, "path"):
            return  # Already loaded

        try:
            # Get model configuration from registry - similar to LlamaCppProvider
            model_config_raw = await self.get_model_config(model_name)

            # Convert to model config with path attribute - get_model_config always returns Dict[str, object]
            # Path is always in the nested 'config' field - single source of truth
            if "config" not in model_config_raw or not isinstance(model_config_raw["config"], dict):
                raise ValueError(
                    f"Model config for '{model_name}' has invalid structure - missing 'config' field"
                )

            if "path" not in model_config_raw["config"]:
                raise ValueError(
                    f"Model config for '{model_name}' missing required 'path' field in config"
                )

            path_value = model_config_raw["config"]["path"]
            if not isinstance(path_value, str):
                raise TypeError(f"Model path must be a string, got {type(path_value)}")
            model_path = path_value
            self._model_config = model_config_raw

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
                    operation="model_initialization",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="embedding",
                    operation="model_initialization",
                    retry_count=0,
                ),
                cause=e,
            ) from e

    async def _load_model(self, model_path: str) -> None:
        """Load a specific embedding model."""
        async with self._lock:
            if self._model and self._model_path == model_path:
                return  # Already loaded

            logger.info(f"Loading embedding model: {model_path}...")
            try:
                # Prepare arguments for Llama constructor with explicit types
                llama_kwargs: dict[str, Any] = {}
                llama_kwargs["model_path"] = model_path  # str
                llama_kwargs["embedding"] = True  # bool
                llama_kwargs["n_ctx"] = int(self.settings.n_ctx)  # int
                if self.settings.n_threads is not None:
                    llama_kwargs["n_threads"] = int(self.settings.n_threads)  # int
                llama_kwargs["n_batch"] = int(self.settings.n_batch)  # int
                llama_kwargs["use_mlock"] = bool(self.settings.use_mlock)  # bool
                llama_kwargs["n_gpu_layers"] = int(self.settings.n_gpu_layers)  # int
                llama_kwargs["verbose"] = bool(self.settings.verbose)  # bool

                object.__setattr__(self, "_model", Llama(**llama_kwargs))
                self._model_path = model_path
                logger.info(f"Embedding model loaded successfully: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{model_path}': {e}", exc_info=True)
                object.__setattr__(self, "_model", None)
                self._model_path = None
                raise ProviderError(
                    message=f"Failed to load embedding model '{model_path}': {e}",
                    context=ErrorContext.create(
                        flow_name="llama_cpp_embedding_provider",
                        error_type="ModelLoadError",
                        error_location="_load_model",
                        component=self.name,
                        operation="model_loading",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="embedding",
                        operation="model_loading",
                        retry_count=0,
                    ),
                    cause=e,
) from e

    async def _shutdown(self) -> None:
        """Release the Llama model."""
        async with self._lock:
            if self._model:
                # llama-cpp-python doesn't have an explicit close/shutdown
                # relies on garbage collection - set to None for frozen models
                object.__setattr__(self, "_model", None)
                logger.info(f"Embedding model resources released for: {self._model_path}")

    async def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[list[float]]:
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
                    operation="embedding_generation",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="embedding",
                    operation="embedding_generation",
                    retry_count=0,
                ),
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

                # Ensure model is loaded
                if self._model is None:
                    raise ProviderError(
                        message="Model failed to initialize",
                        context=ErrorContext.create(
                            flow_name="LlamaCppEmbeddingProvider",
                            error_type="StateError",
                            error_location="generate_embeddings",
                            component=self.name,
                            operation="embedding_generation",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="embedding",
                            operation="embedding_generation",
                            retry_count=0,
                        ),
                    )

                # Get embeddings
                # Note: Llama.embed() might be synchronous depending on version/setup
                # If it blocks significantly, consider running in a thread pool executor
                embeddings = self._model.embed(input_texts)

                # Ensure the output is List[List[float]]
                if not isinstance(embeddings, list) or not all(
                    isinstance(e, list) for e in embeddings
                ):
                    raise ProviderError(
                        message="Llama model did not return expected embedding format.",
                        context=ErrorContext.create(
                            flow_name="llama_cpp_embedding_provider",
                            error_type="ValidationError",
                            error_location="embed",
                            component=self.name,
                            operation="embedding_validation",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="embedding",
                            operation="embedding_validation",
                            retry_count=0,
                        ),
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
                        operation="embedding_generation",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="embedding",
                        operation="embedding_generation",
                        retry_count=0,
                    ),
                    cause=e,
                ) from e

    async def embed_batch(
        self, texts: list[str], model_name: str | None = None
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        This is an alias for embed() that accepts a list of strings.
        Provided for API compatibility with code that expects embed_batch.

        Args:
            texts: List of texts to generate embeddings for
            model_name: Name of the model to use (optional, will use default if not provided)

        Returns:
            List of embedding vectors
        """
        return await self.embed(texts, model_name)
