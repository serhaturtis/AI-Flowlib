"""Multimodal LLM provider base class for vision-language models.

This module provides the base class for multimodal LLM providers that can process
both text and image inputs, such as LLaVA, Moondream, and other vision-language models.
"""

import logging
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, PrivateAttr

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)


class PromptTemplate(Protocol):
    """Protocol for prompt templates."""

    template: str


class ImageInput(BaseModel):
    """Represents an image input for multimodal generation.

    Supports multiple input formats:
    - URL (http/https)
    - Local file path
    - Base64 encoded image data
    """

    model_config = {"extra": "forbid"}

    type: Literal["url", "path", "base64"] = Field(
        ..., description="Type of image input (url, path, or base64)"
    )
    data: str = Field(..., description="Image data: URL, file path, or base64 string")


class MultimodalLLMProviderSettings(ProviderSettings):
    """Settings for multimodal LLM providers.

    This class provides:
    1. Generation parameters
    2. Context management
    3. Image processing settings
    """

    # Generation parameters
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature for generation (0.0 = deterministic, 2.0 = very random)",
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate (None = no limit)"
    )
    top_p: float = Field(default=1.0, description="Top-p (nucleus) sampling threshold (0.0-1.0)")

    # Context management
    max_input_tokens: int | None = Field(
        default=None, description="Maximum input context tokens (None = model default)"
    )
    max_output_tokens: int | None = Field(
        default=None, description="Maximum output tokens (None = no limit)"
    )

    # Streaming settings
    stream: bool = Field(default=False, description="Enable streaming response generation")


SettingsT = TypeVar("SettingsT", bound=MultimodalLLMProviderSettings)
ModelType = TypeVar("ModelType", bound=BaseModel)


class MultimodalLLMProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for multimodal LLM providers.

    This class provides the interface for:
    1. Text generation with image inputs
    2. Structured generation with vision capabilities
    3. Type-safe multimodal response handling
    """

    # Declare private attributes for runtime state
    _initialized: bool = PrivateAttr(default=False)
    _models: dict[str, Any] = PrivateAttr(default_factory=dict)
    _system_prompt: str | None = PrivateAttr(default=None)

    def __init__(
        self, name: str, provider_type: str, settings: SettingsT | None = None, **kwargs: object
    ):
        """Initialize multimodal LLM provider.

        Args:
            name: Unique provider name
            provider_type: The type of the provider (e.g., 'multimodal_llm')
            settings: Optional provider settings
            **kwargs: Additional keyword arguments for the base Provider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)

    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized

    @property
    def system_prompt(self) -> str | None:
        """Get the current system prompt."""
        return self._system_prompt

    async def initialize(self) -> None:
        """Initialize the provider.

        This method should be implemented by subclasses.
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources.

        This method should be implemented by subclasses.
        """
        self._initialized = False

    def set_system_prompt(self, prompt: str | None) -> None:
        """Set the system prompt to prepend to all LLM calls.

        Args:
            prompt: System prompt to use, or None to clear
        """
        self._system_prompt = prompt
        logger.info(f"System prompt set for {self.name} ({len(prompt) if prompt else 0} chars)")

    async def get_model_config(self, model_name: str) -> dict[str, object]:
        """Get configuration for a model from the resource registry.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            Model configuration dictionary

        Raises:
            ProviderError: If model is not found or invalid
        """
        try:
            # Use resource registry to get model configuration
            resource = resource_registry.get(model_name)

            # Log the model config for debugging
            logger.info(f"Retrieved model resource for '{model_name}': {resource}")

            # Convert ResourceBase to dict for backward compatibility with existing code
            model_config = resource.model_dump()

            return model_config
        except Exception as e:
            logger.error(f"Error retrieving model '{model_name}': {str(e)}")

            raise ProviderError(
                message=f"Error retrieving model configuration for '{model_name}': {str(e)}",
                context=ErrorContext.create(
                    flow_name="multimodal_llm_provider",
                    error_type="ModelConfigError",
                    error_location="get_model_config",
                    component=self.name,
                    operation="retrieve_model_config",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="retrieve_model_config",
                    retry_count=0,
                ),
                cause=e,
            ) from e

    async def generate(
        self,
        prompt: str | PromptTemplate,
        model_name: str,
        images: list[ImageInput] | None = None,
        prompt_variables: dict[str, object] | None = None,
    ) -> str:
        """Generate a response from the multimodal LLM with optional image inputs.

        Args:
            prompt: The prompt template or string to generate from
            model_name: Name of the model to use
            images: Optional list of images to include in the generation
            prompt_variables: Dictionary of variables to format the prompt template

        Returns:
            Generated text response

        Raises:
            ProviderError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate()")

    async def generate_structured(
        self,
        prompt: str | PromptTemplate,
        output_type: type[ModelType],
        model_name: str,
        images: list[ImageInput] | None = None,
        prompt_variables: dict[str, object] | None = None,
    ) -> ModelType:
        """Generate a structured response from the multimodal LLM with optional image inputs.

        Args:
            prompt: The prompt template or string to generate from
            output_type: Pydantic model to parse the response into
            model_name: Name of the model to use
            images: Optional list of images to include in the generation
            prompt_variables: Dictionary of variables to format the prompt template

        Returns:
            Pydantic model instance parsed from response

        Raises:
            ProviderError: If generation or parsing fails
        """
        raise NotImplementedError("Subclasses must implement generate_structured()")
