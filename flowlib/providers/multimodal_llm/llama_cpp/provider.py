"""LlamaCpp multimodal provider implementation for vision-language models.

This module implements a provider for multimodal language models using llama-cpp-python
with support for vision models like LLaVA, Moondream, and Llama 3.2 Vision.
"""

import json
import logging
from typing import Any

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.multimodal_llm.base import (
    ImageInput,
    ModelType,
    MultimodalLLMProvider,
    MultimodalLLMProviderSettings,
    PromptTemplate,
)
from flowlib.utils.pydantic.schema import model_to_simple_json_schema

logger = logging.getLogger(__name__)


class LlamaCppMultimodalSettings(MultimodalLLMProviderSettings):
    """Settings for LlamaCpp multimodal provider.

    Provider-level settings that control how the LlamaCpp multimodal provider operates.
    Model-specific settings (clip_model_path, model_path, n_ctx, etc.) are defined
    in model configurations for clean separation of concerns.
    """

    # Infrastructure settings
    max_concurrent_models: int = Field(
        default=3, description="Maximum models loaded simultaneously (prevents OOM)"
    )


@provider(
    provider_type="multimodal_llm", name="llamacpp", settings_class=LlamaCppMultimodalSettings
)
class LlamaCppMultimodalProvider(MultimodalLLMProvider[LlamaCppMultimodalSettings]):
    """Provider for local multimodal inference using llama-cpp-python.

    This provider supports:
    1. Vision-language models (LLaVA, Moondream, Llama 3.2 Vision, etc.)
    2. Text generation with image inputs
    3. Structured output generation with vision capabilities
    4. Optional GPU acceleration with Metal or CUDA
    """

    def __init__(
        self,
        name: str,
        provider_type: str,
        settings: LlamaCppMultimodalSettings | None = None,
        **kwargs: object,
    ):
        """Initialize LlamaCpp multimodal provider.

        Args:
            name: Unique provider name
            provider_type: The type of the provider (e.g., 'multimodal_llm')
            settings: Provider settings
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        if not isinstance(self.settings, LlamaCppMultimodalSettings):
            raise TypeError(
                f"settings must be a LlamaCppMultimodalSettings instance, got {type(self.settings)}"
            )

        self._models = {}
        self._settings = self.settings

    async def _initialize_model(self, model_name: str) -> None:
        """Initialize a specific multimodal model.

        Args:
            model_name: Name of the model to initialize

        Raises:
            ProviderError: If initialization fails
        """
        if model_name in self._models:
            return

        try:
            # Import here to avoid requiring llama-cpp-python for all users
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            # Get model configuration from registry
            model_config_raw = await self.get_model_config(model_name)

            # Extract model configuration
            if "config" not in model_config_raw:
                raise ValueError(
                    f"Model config for '{model_name}' missing 'config' field. "
                    "Multimodal models require clip_model_path in config."
                )

            config = model_config_raw["config"]

            # Validate required multimodal fields
            if "clip_model_path" not in config:
                raise ValueError(
                    f"Model config for '{model_name}' must include 'clip_model_path' "
                    "for multimodal vision models"
                )

            # Extract configuration
            model_path = config.get("path")
            if not model_path:
                raise ValueError(f"Model config for '{model_name}' missing 'path' field")

            clip_model_path = config["clip_model_path"]
            chat_format = config.get("chat_format", "llava-1-5")
            n_ctx = config.get("n_ctx", 2048)
            use_gpu = config.get("use_gpu", False)
            n_gpu_layers = config.get("n_gpu_layers", 0)
            n_threads = config.get("n_threads", 4)
            n_batch = config.get("n_batch", 512)
            verbose = config.get("verbose", self._settings.verbose)

            logger.info(f"Initializing multimodal model '{model_name}' with:")
            logger.info(f"  Model path: {model_path}")
            logger.info(f"  CLIP model path: {clip_model_path}")
            logger.info(f"  Chat format: {chat_format}")
            logger.info(f"  Context size: {n_ctx}")
            logger.info(f"  GPU: {use_gpu} (layers: {n_gpu_layers})")

            # Create chat handler with clip model
            chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path, verbose=verbose)

            # Initialize the model with chat handler
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                chat_format=chat_format,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers if use_gpu else 0,
                n_threads=n_threads,
                n_batch=n_batch,
                verbose=verbose,
            )

            # Store model with its config
            self._models[model_name] = {
                "llm": llm,
                "config": config,
                "chat_format": chat_format,
            }

            logger.info(f"Successfully initialized multimodal model '{model_name}'")

        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="multimodal_model_initialization",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}._initialize_model",
                component=self.name,
                operation="initialize_model",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type=self.provider_type,
                operation="initialize_model",
                retry_count=0,
            )

            raise ProviderError(
                message=f"Failed to initialize multimodal model '{model_name}': {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e,
            ) from e

    def _build_messages(
        self,
        prompt_text: str,
        images: list[ImageInput] | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-style messages with text and image content.

        Args:
            prompt_text: The text prompt
            images: Optional list of images to include

        Returns:
            List of message dictionaries in OpenAI format
        """
        # Build content array with text and images
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        # Add images if provided
        if images:
            for img in images:
                if img.type == "url":
                    content.append({"type": "image_url", "image_url": {"url": img.data}})
                elif img.type == "path":
                    # For local files, use file:// URL scheme
                    content.append({"type": "image_url", "image_url": {"url": f"file://{img.data}"}})
                elif img.type == "base64":
                    # Base64 data should include data URI scheme
                    if not img.data.startswith("data:"):
                        data_uri = f"data:image/jpeg;base64,{img.data}"
                    else:
                        data_uri = img.data
                    content.append({"type": "image_url", "image_url": {"url": data_uri}})

        # Build messages
        messages = []

        # Add system prompt if set
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Add user message with content
        messages.append({"role": "user", "content": content})

        return messages

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
        if not self._initialized:
            await self.initialize()

        # Initialize model if needed
        await self._initialize_model(model_name)

        try:
            # Extract prompt text
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                # Format template
                template = prompt.template
                if prompt_variables:
                    # Simple variable replacement
                    for key, value in prompt_variables.items():
                        template = template.replace(f"{{{{{key}}}}}", str(value))
                prompt_text = template

            # Get model
            model_data = self._models[model_name]
            llm = model_data["llm"]
            config = model_data["config"]

            # Build messages
            messages = self._build_messages(prompt_text, images)

            # Extract generation parameters
            temperature = config.get("temperature", self.settings.temperature)
            max_tokens = config.get("max_tokens", self.settings.max_tokens)
            top_p = config.get("top_p", self.settings.top_p)

            # Generate response
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else -1,
                top_p=top_p,
            )

            # Extract response text
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]

            raise ValueError("Invalid response format from model")

        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="multimodal_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate",
                component=self.name,
                operation="generate",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type=self.provider_type,
                operation="generate",
                retry_count=0,
            )

            raise ProviderError(
                message=f"Multimodal generation failed: {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e,
            ) from e

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
        try:
            # Extract prompt text and add JSON schema instructions
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                template = prompt.template
                if prompt_variables:
                    for key, value in prompt_variables.items():
                        template = template.replace(f"{{{{{key}}}}}", str(value))
                prompt_text = template

            # Add JSON schema instructions
            schema_json = model_to_simple_json_schema(output_type)
            enhanced_prompt = f"""{prompt_text}

IMPORTANT: Format your response as a JSON object matching this schema exactly:

{schema_json}

Generate appropriate field values based on the content provided above.
"""

            # Generate response
            response_text = await self.generate(
                prompt=enhanced_prompt, model_name=model_name, images=images
            )

            # Parse JSON response
            # Remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            response_data = json.loads(response_text)

            # Validate with Pydantic
            return output_type.model_validate(response_data)

        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="multimodal_structured_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate_structured",
                component=self.name,
                operation="generate_structured",
            )

            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type=self.provider_type,
                operation="generate_structured",
                retry_count=0,
            )

            raise ProviderError(
                message=f"Multimodal structured generation failed: {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e,
            ) from e

    async def _initialize(self) -> None:
        """Initialize the provider (called by base class)."""
        logger.info(f"Initializing {self.__class__.__name__}")
        self._initialized = True

    async def _shutdown(self) -> None:
        """Clean up resources (called by base class)."""
        logger.info(f"Shutting down {self.__class__.__name__}")
        # Unload all models
        self._models.clear()
        self._initialized = False
