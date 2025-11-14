"""LlamaCpp multimodal provider implementation for vision-language models.

This module implements a provider for multimodal language models using llama-cpp-python
with support for various vision models. Supported architectures:
- LLaVA 1.5, 1.6 (chat_format: 'llava-1-5', 'llava-1-6')
- Gemma 3 Vision (chat_format: 'gemma-3')
- Moondream (chat_format: 'moondream')
- Llama 3 Vision (chat_format: 'llama-3-vision')

The provider dynamically selects the appropriate chat handler based on the
'chat_format' parameter in the model configuration.
"""

import importlib
import json
import logging
import re
from typing import Any

from pydantic import Field

# Lazy import llama_cpp
try:
    from llama_cpp import Llama, LlamaGrammar

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    LlamaGrammar = None  # type: ignore
    LLAMA_CPP_AVAILABLE = False

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
    provider_type="multimodal_llm", name="llamacpp_multimodal", settings_class=LlamaCppMultimodalSettings
)
class LlamaCppMultimodalProvider(MultimodalLLMProvider[LlamaCppMultimodalSettings]):
    """Provider for local multimodal inference using llama-cpp-python.

    This provider supports:
    1. Vision-language models (LLaVA 1.5/1.6, Gemma 3, Moondream, Llama 3 Vision)
    2. Text generation with image inputs
    3. Structured output generation with vision capabilities
    4. Optional GPU acceleration with Metal or CUDA
    5. Dynamic chat handler selection based on model architecture

    The appropriate chat handler is automatically selected based on the
    'chat_format' parameter in the model configuration.
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

    def _get_chat_handler(self, chat_format: str, clip_model_path: str, verbose: bool) -> Any:
        """Get appropriate chat handler for the specified format.

        Args:
            chat_format: Chat format identifier (e.g., 'llava-1-5', 'gemma-3', 'moondream')
            clip_model_path: Path to CLIP/vision projection model file
            verbose: Enable verbose logging

        Returns:
            Chat handler instance (dynamically imported based on chat_format)

        Raises:
            ValueError: If chat format is unsupported or handler unavailable
        """
        # Map chat formats to their handler classes
        # Format: chat_format -> (module_path, handler_class_name)
        handler_mapping = {
            "llava-1-5": ("llama_cpp.llama_chat_format", "Llava15ChatHandler"),
            "llava-1-6": ("llama_cpp.llama_chat_format", "Llava16ChatHandler"),
            "moondream": ("llama_cpp.llama_chat_format", "MoondreamChatHandler"),
            "gemma-3": ("llama_cpp.llama_chat_format", "Gemma3ChatHandler"),
            "llama-3-vision": ("llama_cpp.llama_chat_format", "Llama3VisionAlphaChatHandler"),
            "qwen2.5-vl": ("llama_cpp.llama_chat_format", "Qwen25VLChatHandler"),
        }

        if chat_format not in handler_mapping:
            raise ValueError(
                f"Unsupported chat format: '{chat_format}'. "
                f"Supported formats: {', '.join(handler_mapping.keys())}"
            )

        module_path, handler_class_name = handler_mapping[chat_format]

        # Try to import the handler
        try:
            module = importlib.import_module(module_path)
            handler_class = getattr(module, handler_class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Chat handler '{handler_class_name}' for format '{chat_format}' is not available. "
                f"This may require a newer version of llama-cpp-python or a custom build. "
                f"Error: {str(e)}"
            ) from e

        # Create handler instance
        return handler_class(clip_model_path=clip_model_path, verbose=verbose)

    async def _initialize_model(self, model_name: str) -> None:
        """Initialize a specific multimodal model.

        Args:
            model_name: Name of the model to initialize

        Raises:
            ProviderError: If initialization fails
        """
        if model_name in self._models:
            return

        # Check if llama-cpp-python is available
        if not LLAMA_CPP_AVAILABLE or Llama is None:
            raise ProviderError(
                message="llama-cpp-python is not installed. Install it with: pip install llama-cpp-python",
                context=ErrorContext.create(
                    flow_name="multimodal_model_initialization",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}._initialize_model",
                    component=self.name,
                    operation="initialize_model",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="initialize_model",
                    retry_count=0,
                ),
            )

        try:
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

            # Get appropriate chat handler for the format
            chat_handler = self._get_chat_handler(chat_format, clip_model_path, verbose)

            # Initialize the model with chat handler
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
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
            # Priority: prompt config > model config > provider settings
            prompt_config = getattr(prompt, "config", None) if not isinstance(prompt, str) else None

            if prompt_config:
                temperature = prompt_config.get("temperature", config.get("temperature", self.settings.temperature))
                max_tokens = prompt_config.get("max_tokens", config.get("max_tokens", self.settings.max_tokens))
                top_p = prompt_config.get("top_p", config.get("top_p", self.settings.top_p))
            else:
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
            # Extract prompt text
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                template = prompt.template
                if prompt_variables:
                    for key, value in prompt_variables.items():
                        template = template.replace(f"{{{{{key}}}}}", str(value))
                prompt_text = template

            # Add system prompt if set (agent persona)
            if self.system_prompt:
                prompt_text = f"{self.system_prompt}\n\n{prompt_text}"
                logger.info(f"System prompt prepended ({len(self.system_prompt)} chars)")

            # Get simplified schema for prompt and full schema for grammar
            try:
                # Get simplified schema to include in prompt (user-friendly format)
                example_json = model_to_simple_json_schema(output_type)

                # Get full Pydantic schema for grammar generation
                schema = output_type.model_json_schema()
            except AttributeError as e:
                raise ProviderError(
                    message=f"Output type must be a Pydantic model with model_json_schema(): {str(e)}",
                    context=ErrorContext.create(
                        flow_name="multimodal_structured_generation",
                        error_type="ProviderError",
                        error_location=f"{self.__class__.__name__}.generate_structured",
                        component=self.name,
                        operation="schema_extraction",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="schema_extraction",
                        retry_count=0,
                    ),
                    cause=e,
                ) from e

            # Extract constraint warnings from schema
            try:
                schema_dict = json.loads(example_json)
                constraint_warnings = self._extract_constraint_warnings(schema_dict)
            except Exception:
                constraint_warnings = ""

            # Append JSON instructions and schema to prompt
            json_instructions = f"""

IMPORTANT: Format your response as a JSON object matching this schema.
Do NOT use schema metadata as values - extract actual values from the task content.

Schema:
{example_json}
{constraint_warnings}
Remember: Generate appropriate field values based on the context provided above.
"""
            prompt_text = prompt_text + json_instructions

            # Create grammar from full schema for enforcement
            schema_str = json.dumps(schema)
            try:
                grammar = LlamaGrammar.from_json_schema(schema_str)
            except Exception as e:
                raise ProviderError(
                    message=f"Failed to create grammar from schema: {str(e)}",
                    context=ErrorContext.create(
                        flow_name="multimodal_structured_generation",
                        error_type="ProviderError",
                        error_location=f"{self.__class__.__name__}.generate_structured",
                        component=self.name,
                        operation="grammar_creation",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="grammar_creation",
                        retry_count=0,
                    ),
                    cause=e,
                ) from e

            # Initialize model if needed
            await self._initialize_model(model_name)

            # Get model and config
            model_data = self._models[model_name]
            llm = model_data["llm"]
            config = model_data["config"]

            # Build messages with formatted prompt (includes schema instructions)
            messages = self._build_messages(prompt_text, images)

            # Log the formatted prompt
            logger.info("=============== FORMATTED PROMPT ===============")
            logger.info(prompt_text)
            logger.info("================================================")

            # DEBUG: Show final formatted prompt
            print("\n📤 FINAL PROMPT SENT TO MODEL:")
            print("=" * 80)
            print(prompt_text)
            print("=" * 80)

            # Extract generation parameters
            # Priority: prompt config > model config > provider settings
            prompt_config = getattr(prompt, "config", None) if not isinstance(prompt, str) else None

            if prompt_config:
                temperature = prompt_config.get("temperature", config.get("temperature", self.settings.temperature))
                max_tokens = prompt_config.get("max_tokens", config.get("max_tokens", self.settings.max_tokens))
                top_p = prompt_config.get("top_p", config.get("top_p", self.settings.top_p))
            else:
                temperature = config.get("temperature", self.settings.temperature)
                max_tokens = config.get("max_tokens", self.settings.max_tokens)
                top_p = config.get("top_p", self.settings.top_p)

            # Generate with grammar
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else -1,
                top_p=top_p,
                grammar=grammar,
            )

            # Extract response text
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    response_text = choice["message"]["content"]
                else:
                    raise ValueError("Invalid response format from model")
            else:
                raise ValueError("No choices in response from model")

            # DEBUG: Show raw response from model
            print("\n📥 RAW RESPONSE FROM MODEL:")
            print("=" * 80)
            print(response_text)
            print("=" * 80)

            # Parse JSON response (grammar ensures proper formatting)
            response_text = response_text.strip()

            # Fix common JSON issues from LLM output
            # LLMs often generate raw newlines inside string values, which is invalid JSON
            # We need to escape them properly
            try:
                # Try direct parse first (fast path)
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                # JSON parse failed - likely due to unescaped control characters
                logger.warning(f"Initial JSON parse failed: {e}. Attempting repair...")

                # Escape unescaped newlines and other control characters in string values
                # This is a simplified repair - replace actual newlines with \n
                response_text = response_text.replace('\r\n', '\\n')  # Windows newlines
                response_text = response_text.replace('\n', '\\n')     # Unix newlines
                response_text = response_text.replace('\r', '\\n')     # Mac newlines
                response_text = response_text.replace('\t', '\\t')     # Tabs

                # Try parsing again
                try:
                    response_data = json.loads(response_text)
                    logger.info("JSON repair successful")
                except json.JSONDecodeError as e2:
                    # Still failed - log the problematic area
                    logger.error(f"JSON repair failed: {e2}")
                    logger.error(f"Problematic area: {response_text[max(0, e2.pos-50):e2.pos+50]}")
                    raise

            # DEBUG: Show parsed JSON data
            print("\n📊 PARSED JSON DATA:")
            print("=" * 80)
            print(json.dumps(response_data, indent=2))
            print("=" * 80)

            # Validate with Pydantic
            validated_output = output_type.model_validate(response_data)

            # DEBUG: Show validated Pydantic model
            print("\n✅ VALIDATED PYDANTIC MODEL:")
            print("=" * 80)
            print(validated_output.model_dump_json(indent=2))
            print("=" * 80)

            return validated_output

        except Exception as e:
            # Add debug info to error message
            debug_info = ""
            try:
                if 'response_text' in locals():
                    debug_info = f"\n\nDEBUG - Raw response (first 500 chars): {response_text[:500]}"
                if 'response_data' in locals():
                    debug_info += f"\n\nDEBUG - Parsed JSON: {response_data}"
            except:
                pass

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
                message=f"Multimodal structured generation failed: {str(e)}{debug_info}",
                context=error_context,
                provider_context=provider_context,
                cause=e,
            ) from e

    def _extract_constraint_warnings(self, schema_dict: dict[str, Any]) -> str:
        """Extract numeric and length constraints from schema and format as explicit warnings.

        This helps ensure LLMs respect constraints like le=10, ge=1, etc. by making them
        very explicit at the end of the prompt.

        Args:
            schema_dict: Simplified schema dictionary (from model_to_simple_json_schema)

        Returns:
            Formatted constraint warnings string, or empty string if no constraints found
        """
        warnings = []

        # Check main schema properties
        if "properties" in schema_dict:
            for field_name, field_type in schema_dict["properties"].items():
                # field_type may be like "integer (≥1, ≤10)" - extract constraints
                if isinstance(field_type, str) and "(" in field_type and ")" in field_type:
                    # Extract the constraint part
                    type_part, constraint_part = field_type.split("(", 1)
                    constraint_part = constraint_part.rstrip(")")
                    type_name = type_part.strip()

                    # Format warning based on type
                    if type_name in ["integer", "number"]:
                        warnings.append(
                            f"- {field_name}: Must be {type_name} with constraints: {constraint_part}"
                        )
                    elif type_name == "string":
                        warnings.append(
                            f"- {field_name}: Must be {type_name} with constraints: {constraint_part}"
                        )
                    elif type_name.startswith("array"):
                        warnings.append(
                            f"- {field_name}: Must be {type_name} with constraints: {constraint_part}"
                        )

        # Check nested schemas (sub-models)
        for key, value in schema_dict.items():
            if key not in [
                "title",
                "description",
                "properties",
                "field_descriptions",
                "required",
            ] and isinstance(value, dict):
                # This is a nested schema
                if "properties" in value:
                    for field_name, field_type in value["properties"].items():
                        if isinstance(field_type, str) and "(" in field_type and ")" in field_type:
                            type_part, constraint_part = field_type.split("(", 1)
                            constraint_part = constraint_part.rstrip(")")
                            type_name = type_part.strip()

                            nested_name = value.get("title", key)
                            if type_name in ["integer", "number"]:
                                warnings.append(
                                    f"- {nested_name}.{field_name}: Must be {type_name} with constraints: {constraint_part}"
                                )
                            elif type_name == "string":
                                warnings.append(
                                    f"- {nested_name}.{field_name}: Must be {type_name} with constraints: {constraint_part}"
                                )
                            elif type_name.startswith("array"):
                                warnings.append(
                                    f"- {nested_name}.{field_name}: Must be {type_name} with constraints: {constraint_part}"
                                )

        if warnings:
            return "\n\nCRITICAL CONSTRAINTS - MUST BE RESPECTED:\n" + "\n".join(warnings) + "\n"
        return ""

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
