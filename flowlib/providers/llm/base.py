"""LLM provider base class and related functionality.

This module provides the base class for implementing local LLM providers 
such as LlamaCpp, which share common functionality for generating responses 
and extracting structured data.
"""

import logging
from typing import Any, Dict, Generic, List, Optional, Protocol, Type, TypeVar, cast

from pydantic import BaseModel, Field, field_validator

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.base import Provider, ProviderSettings
from flowlib.resources.registry.registry import resource_registry
from flowlib.utils.pydantic.schema import model_to_simple_json_schema


class PromptTemplate(Protocol):
    template: str

logger = logging.getLogger(__name__)


class PromptConfigOverride(BaseModel):
    """Pydantic model for prompt-specific generation parameter overrides.
    
    All parameters are optional - only specified parameters will override
    the model defaults from LLMProviderSettings.
    """
    model_config = {"extra": "forbid"}

    # Generation parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Streaming settings
    stream: Optional[bool] = None

    # Advanced settings
    stop_sequences: Optional[List[str]] = None

    @field_validator("temperature")
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        """Validate temperature."""
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    def validate_top_p(cls, v: Optional[float]) -> Optional[float]:
        """Validate top_p."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v

    @field_validator("top_k")
    def validate_top_k(cls, v: Optional[int]) -> Optional[int]:
        """Validate top_k."""
        if v is not None and v < 1:
            raise ValueError("Top_k must be at least 1")
        return v


class LLMProviderSettings(ProviderSettings):
    """Settings for LLM providers.
    
    This class provides:
    1. Model configuration
    2. Generation parameters
    3. Token management
    """

    # Generation parameters
    temperature: float = Field(default=0.7, description="Sampling temperature for generation (0.0 = deterministic, 2.0 = very random)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate (None = no limit)")
    top_p: float = Field(default=1.0, description="Top-p (nucleus) sampling threshold (0.0-1.0)")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty to reduce repetition (-2.0 to 2.0)")
    presence_penalty: float = Field(default=0.0, description="Presence penalty to encourage topic diversity (-2.0 to 2.0)")

    # Token management
    max_input_tokens: Optional[int] = Field(default=None, description="Maximum input context tokens (None = model default)")
    max_output_tokens: Optional[int] = Field(default=None, description="Maximum output tokens (None = no limit)")

    # Streaming settings
    stream: bool = Field(default=False, description="Enable streaming response generation")

    # Advanced settings
    stop_sequences: List[str] = Field(default_factory=list, description="List of stop sequences to end generation")

    @field_validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature."""
        if v < 0 or v > 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    def validate_top_p(cls, v: float) -> float:
        """Validate top_p."""
        if v < 0 or v > 1:
            raise ValueError("Top_p must be between 0 and 1")
        return v


SettingsT = TypeVar('SettingsT', bound=LLMProviderSettings)
ModelType = TypeVar('ModelType', bound=BaseModel)


class LLMProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for local LLM backends.
    
    This class provides the interface for:
    1. Structured generation with Pydantic models
    2. Grammar-based parsing and validation
    3. Type-safe response handling
    """

    def __init__(self, name: str, provider_type: str, settings: Optional[SettingsT] = None, **kwargs: object):
        """Initialize LLM provider.
        
        Args:
            name: Unique provider name
            provider_type: The type of the provider (e.g., 'llm')
            settings: Optional provider settings
            **kwargs: Additional keyword arguments for the base Provider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._initialized = False
        self._models: Dict[str, Any] = {}

    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized

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
            logger.info(f"Retrieved model resource for '{model_name}': {resource}")

            # Convert ResourceBase to dict for backward compatibility with existing code
            model_config = resource.model_dump()

            return model_config
        except Exception as e:
            logger.error(f"Error retrieving model '{model_name}': {str(e)}")

            raise ProviderError(
                message=f"Error retrieving model configuration for '{model_name}': {str(e)}",
                context=ErrorContext.create(
                    flow_name="llm_provider",
                    error_type="ModelConfigError",
                    error_location="get_model_config",
                    component=self.name,
                    operation="retrieve_model_config"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="retrieve_model_config",
                    retry_count=0
                ),
                cause=e
            )

    def _extract_prompt_config(self, prompt: PromptTemplate) -> Optional[PromptConfigOverride]:
        """Extract prompt config overrides.
        
        Args:
            prompt: The prompt template to extract config from
            
        Returns:
            PromptConfigOverride instance if config exists, None otherwise
            
        Raises:
            TypeError: If config exists but is not a PromptConfigOverride instance
        """
        if not hasattr(prompt, 'config'):
            return None

        config_attr = getattr(prompt, 'config', None)
        if config_attr is None:
            return None

        if not isinstance(config_attr, PromptConfigOverride):
            raise TypeError(f"Prompt config must be PromptConfigOverride instance, got {type(config_attr)}. No fallbacks or dict conversion allowed.")

        return config_attr

    def _merge_generation_config(self, model_config: Dict[str, object], prompt_config: Optional[PromptConfigOverride] = None) -> Dict[str, object]:
        """Merge model defaults with prompt config overrides.
        
        Args:
            model_config: Model configuration from resource registry
            prompt_config: Optional prompt-specific overrides
            
        Returns:
            Merged configuration dictionary
        """
        # Start with empty merged config - no defaults from provider settings
        # Generation parameters come from model config, not provider settings
        merged_config = {}

        # Override with model config
        for key, value in model_config.items():
            if value is not None:
                merged_config[key] = value

        # Override with prompt config
        if prompt_config:
            prompt_dict = prompt_config.model_dump(exclude_none=True)
            for key, value in prompt_dict.items():
                merged_config[key] = value

        # Filter out None values
        return {k: v for k, v in merged_config.items() if v is not None}

    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, object]] = None) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt template to generate from
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Generated text response
            
        Raises:
            ProviderError: If generation fails
        """
        raise NotImplementedError("Subclasses must implement generate()")

    async def generate_structured(self, prompt: PromptTemplate, output_type: Type[ModelType], model_name: str, prompt_variables: Optional[Dict[str, object]] = None) -> ModelType:
        """Generate a structured response from the LLM.
        
        Args:
            prompt: The prompt template to generate from
            output_type: Pydantic model to parse the response into
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Pydantic model instance parsed from response
            
        Raises:
            ProviderError: If generation or parsing fails
        """
        raise NotImplementedError("Subclasses must implement generate_structured()")

    def format_template(self, template: str, kwargs: Dict[str, object]) -> str:
        """Format a template with variables.
        
        Replaces {{variable}} placeholders in the template with corresponding values.
        Uses double curly braces to avoid conflicts with JSON formatting.
        
        Args:
            template: Template string with {{variable}} placeholders
            kwargs: Dict containing variables and their values
            
        Returns:
            Formatted template string
        """
        # Debug: Log what variables we're receiving - strict access
        if 'variables' not in kwargs:
            raise ValueError("Required 'variables' key missing from kwargs")
        variables_value = kwargs['variables']
        if not isinstance(variables_value, dict):
            raise TypeError(f"Variables must be a dict, got {type(variables_value).__name__}")

        available_vars = list(variables_value.keys())
        logger.info("=" * 80)
        logger.info("FORMAT_TEMPLATE CALLED:")
        logger.info(f"Template length: {len(template)}")
        logger.info(f"Variables available: {available_vars}")
        logger.info("=" * 80)

        # Variables already validated - proceed with formatting
        variables = cast(Dict[str, object], kwargs["variables"])
        result = template

        # Replace {{variable}} with corresponding values
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in result:
                logger.info(f"✓ Replacing placeholder: {placeholder} (value length: {len(str(value))})")
                result = result.replace(placeholder, str(value))
            else:
                logger.warning(f"✗ Placeholder {placeholder} NOT FOUND in template")

        # Debug: Log if any {{...}} remain in the template
        import re
        remaining = re.findall(r'\{\{([^}]+)\}\}', result)
        if remaining:
            logger.error("=" * 80)
            logger.error("🚨 UNREPLACED PLACEHOLDERS REMAIN:")
            logger.error(f"Remaining: {remaining}")
            logger.error("=" * 80)

        return result

    def _format_prompt(self, prompt: str, chat_format: str = "default", output_type: Optional[Type[ModelType]] = None) -> str:
        """Format a prompt according to model-specific requirements.

        This is a base implementation that provides common formatting patterns.
        If output_type is provided, automatically appends JSON structure instructions.

        Args:
            prompt: The main prompt text
            chat_format: The chat format of the model
            output_type: Optional Pydantic model type for structured output

        Returns:
            Formatted prompt string
        """
        # If output_type is provided, append JSON structure information
        # Strict contract: output_type must be a proper Pydantic model if provided
        if output_type is not None:
            # Enforce strict contract - no hasattr fallbacks
            if not hasattr(output_type, 'model_json_schema'):
                raise TypeError(
                    f"output_type must be a Pydantic model with model_json_schema method. "
                    f"Got {type(output_type).__name__} which does not have this method."
                )

            # No broad exception handling - let schema generation errors propagate
            example_json = model_to_simple_json_schema(output_type)

            # Append the structure information to the prompt
            # Explicitly clarify that field values must come from extraction, not schema metadata
            json_instructions = f"""

IMPORTANT: Format your response as a JSON object matching this schema.
Do NOT use schema metadata as values - extract actual values from the task content.

Schema:
{example_json}

Remember: Generate appropriate field values based on the context provided above.
"""
            return prompt + json_instructions

        # Default formatting does nothing
        return prompt


    def _get_model_templates(self) -> Dict[str, Dict[str, str]]:
        """Get model-specific prompt templates.
        
        Override this in subclasses to provide templates for different model types.
        
        Returns:
            Dictionary mapping model_type to pre/post prompt templates
        """
        return {
            "default": {
                "pre_prompt": "",
                "post_prompt": ""
            }
        }
