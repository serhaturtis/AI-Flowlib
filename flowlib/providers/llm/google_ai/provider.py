"""GoogleAI (Gemini) provider implementation for cloud-based language models.

This module implements a provider for Google's Gemini models using the
google-genai library.
"""

import logging
import inspect
import json
import time
import asyncio
from typing import Any, Dict, Optional, Type, cast
from pydantic import Field

# Attempt to import Google AI library with correct API pattern
try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None # type: ignore
    types = None # type: ignore


from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.llm.base import LLMProvider, ModelType
from flowlib.providers.core.base import ProviderSettings
from flowlib.resources.decorators.decorators import PromptTemplate
from flowlib.agent.core.resilience import ResilienceManager

logger = logging.getLogger(__name__)


class GoogleAISettings(ProviderSettings):
    """Settings for the GoogleAI (Gemini) provider - direct inheritance, only Google AI fields.
    
    Google AI is a cloud API service that requires:
    1. API key for authentication
    2. Safety settings for content filtering
    3. Rate limiting and timeout configuration
    
    No host/port needed - it's cloud-managed API.
    """
    
    # Google AI authentication (API-specific fields)
    api_key: str = Field(default="", description="Google AI API key (e.g., 'AIzaSyBJDV...', get from Google AI Studio)")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL (optional)")
    
    # Google AI safety controls
    safety_settings: Optional[Dict[Any, Any]] = Field(
        default=None, 
        description="Safety settings for content generation filtering"
    )
    
    # Google AI API rate limiting (inherits timeout from ProviderSettings)
    max_concurrent_requests: int = Field(default=10, description="API rate limiting")
    
    # Rate limiting configuration (conservative defaults to avoid 429 errors)
    requests_per_minute: int = Field(default=5, description="Max requests per minute (conservative default)")
    enable_rate_limiting: bool = Field(default=True, description="Enable built-in rate limiting")
    min_request_interval: float = Field(default=12.0, description="Minimum seconds between requests (conservative)")
    
    # 429 retry configuration
    rate_limit_retry_attempts: int = Field(default=8, description="Extra retries for 429 errors")
    rate_limit_backoff_factor: float = Field(default=1.5, description="Backoff multiplier for 429 retries")
    rate_limit_max_backoff: float = Field(default=120.0, description="Maximum backoff delay in seconds")


@provider(provider_type="llm", name="googleai", settings_class=GoogleAISettings)
class GoogleAIProvider(LLMProvider):
    """Provider for Google AI (Gemini) models.
    
    This provider supports:
    1. Text generation with various Gemini models.
    2. Structured output generation using Gemini's tool/function calling.
    """
    
    def __init__(self, name: str, provider_type: str, settings: Optional[GoogleAISettings] = None, **kwargs: Any):
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        if not isinstance(self.settings, GoogleAISettings):
            raise TypeError(f"settings must be a GoogleAISettings instance, got {type(self.settings)}")
        
        self._settings = cast(GoogleAISettings, self.settings) # For type hinting
        self._client = None  # Will store genai.Client instance
        self._models: Dict[str, Dict[str, Any]] = {} # Stores model configs
        
        # Rate limiting infrastructure
        self._resilience_manager = ResilienceManager()
        self._rate_limiter = None
        self._last_request_time: Optional[float] = None

        if not GOOGLE_AI_AVAILABLE:
            logger.warning(
                "google-genai package not found. GoogleAIProvider will not function."
                "Install with: pip install google-genai"
            )

    async def initialize(self):
        """Initialize the Google AI provider by configuring the API key."""
        if not GOOGLE_AI_AVAILABLE:
            error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="DependencyError",
                    error_location="initialize",
                    component="GoogleAIProvider",
                    operation="package_check"
                )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="package_check",
                retry_count=0
            )
            
            raise ProviderError(
                message="google-genai package not installed.",
                context=error_context,
                provider_context=provider_context
            )
        try:
            if not self._settings.api_key:
                error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="ConfigurationError",
                    error_location="initialize",
                    component="GoogleAIProvider",
                    operation="api_key_check"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="api_key_check",
                    retry_count=0
                )
                
                raise ProviderError(
                    message="Google AI API key not configured.",
                    context=error_context,
                    provider_context=provider_context
                )
            # Create Google AI client with the API key  
            self._client = genai.Client(api_key=self._settings.api_key)
            
            # Initialize rate limiting if enabled
            if self._settings.enable_rate_limiting:
                self._rate_limiter = self._resilience_manager.get_rate_limiter(
                    name=f"googleai_{self.name}",
                    max_calls=self._settings.requests_per_minute,
                    time_window=60.0  # 1 minute
                )
                logger.info(f"Rate limiting enabled: {self._settings.requests_per_minute} requests/minute, {self._settings.min_request_interval}s minimum interval")
            
            self._initialized = True
            logger.info("GoogleAIProvider initialized with client.")
        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="GoogleAIProvider",
                error_type="InitializationError",
                error_location="initialize",
                component="GoogleAIProvider",
                operation="configure_api"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="configure_api",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Failed to initialize Google AI provider: {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e
            )

    async def _initialize(self):
        """Internal initialization, called by base class if needed.
        Actual initialization is done in `initialize()` which should be called explicitly.
        """
        if not self._initialized:
            # This typically shouldn't be hit if initialize() is called externally as intended
            logger.warning("GoogleAIProvider._initialize() called but provider not yet initialized. Call initialize() first.")
            await self.initialize() 


    def _get_or_cache_model_config(self, model_name: str, model_config: Dict[str, Any]) -> str:
        """Get or cache model configuration and return the model ID."""
        if model_name not in self._models:
            # Model ID must come from model config (no provider default)
            if 'model_id' not in model_config:
                raise ValueError(f"Model configuration for '{model_name}' must specify 'model_id' (e.g., 'gemini-2.0-flash-001')")
            
            self._models[model_name] = {
                "model_id": model_config['model_id'],
                "config": model_config
            }
        
        return self._models[model_name]["model_id"]

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limiting error."""
        error_str = str(error).lower()
        return any(indicator in error_str for indicator in [
            '429', 'rate limit', 'quota exceeded', 'resource exhausted',
            'rate_limit_exceeded', 'quota_exceeded', 'too many requests'
        ])

    async def _handle_rate_limit_retry(self, operation_name: str, operation, original_error: Exception, *args, **kwargs):
        """Handle rate limit errors with exponential backoff."""
        for attempt in range(self._settings.rate_limit_retry_attempts):
            backoff_delay = min(
                self._settings.rate_limit_backoff_factor ** attempt,
                self._settings.rate_limit_max_backoff
            )
            
            logger.warning(
                f"Rate limit hit for {operation_name} (attempt {attempt + 1}/{self._settings.rate_limit_retry_attempts}), "
                f"backing off for {backoff_delay:.1f}s"
            )
            
            await asyncio.sleep(backoff_delay)
            
            try:
                result = await operation(*args, **kwargs)
                self._last_request_time = time.time()
                logger.info(f"Rate limit retry succeeded for {operation_name} after {backoff_delay:.1f}s backoff")
                return result
            except Exception as e:
                if not self._is_rate_limit_error(e):
                    # Different error, don't continue rate limit retry
                    logger.error(f"Non-rate-limit error during retry for {operation_name}: {e}")
                    raise e
                if attempt == self._settings.rate_limit_retry_attempts - 1:
                    # Final attempt failed
                    logger.error(f"Rate limit retry exhausted for {operation_name} after {self._settings.rate_limit_retry_attempts} attempts")
                    raise e
        
        raise original_error

    async def _execute_with_rate_limiting(self, operation_name: str, operation, *args, **kwargs):
        """Execute operation with rate limiting and 429-specific retry logic."""
        # Apply rate limiter if enabled
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        # Add minimum interval between requests
        if self._last_request_time is not None:
            time_since_last = time.time() - self._last_request_time
            if time_since_last < self._settings.min_request_interval:
                sleep_time = self._settings.min_request_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s before {operation_name}")
                await asyncio.sleep(sleep_time)
        
        try:
            result = await operation(*args, **kwargs)
            self._last_request_time = time.time()
            return result
        except Exception as e:
            # Check for 429 errors
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limit error detected for {operation_name}: {e}")
                return await self._handle_rate_limit_retry(operation_name, operation, e, *args, **kwargs)
            raise

    async def shutdown(self):
        """Release model resources (clears cache)."""
        self._models = {}
        self._initialized = False
        logger.info("GoogleAIProvider shutdown, model cache cleared.")
        
    def _create_generation_config(self, model_config: Dict[str, Any]) -> 'types.GenerateContentConfig':
        """Create generation config for Google AI from model config."""
        # All generation parameters must come from model config (no provider defaults)
        params = {}
        
        # Map our config to Google AI API parameters
        if 'temperature' in model_config:
            params['temperature'] = model_config['temperature']
        else:
            params['temperature'] = 0.7  # Default
            
        if 'max_tokens' in model_config:
            params['max_output_tokens'] = model_config['max_tokens']
        else:
            params['max_output_tokens'] = 2048  # Default
            
        if 'top_p' in model_config and model_config['top_p'] is not None:
            params['top_p'] = model_config['top_p']
            
        if 'top_k' in model_config and model_config['top_k'] is not None:
            params['top_k'] = model_config['top_k']
        
        return types.GenerateContentConfig(**params)

    def _clean_schema_for_google_ai(self, schema: Dict[str, Any]) -> None:
        """Remove additionalProperties from schema since Google AI API doesn't support it."""
        if isinstance(schema, dict):
            # Remove additionalProperties at this level
            if 'additionalProperties' in schema:
                del schema['additionalProperties']
            
            # Recursively clean nested objects
            for key, value in schema.items():
                if isinstance(value, dict):
                    self._clean_schema_for_google_ai(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._clean_schema_for_google_ai(item)

    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> str:
        if not self._initialized or not self._client:
            error_context = ErrorContext.create(
                flow_name="GoogleAIProvider",
                error_type="InitializationError",
                error_location="generate",
                component="GoogleAIProvider",
                operation="check_initialization"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="check_initialization",
                retry_count=0
            )
            
            raise ProviderError(
                message="Provider not initialized",
                context=error_context,
                provider_context=provider_context
            )

        # Get model config from resource registry first
        try:
            model_registry_config = await self.get_model_config(model_name)
            if hasattr(model_registry_config, 'model_dump'):
                model_registry_config = model_registry_config.model_dump()
            elif not isinstance(model_registry_config, dict):
                model_registry_config = {}
        except Exception as e:
            logger.warning(f"Could not retrieve model config for '{model_name}': {e}. Using defaults.")
            model_registry_config = {}
        
        # Get model ID from config
        gemini_model_id = self._get_or_cache_model_config(model_name, model_registry_config)
        
        # Extract prompt config overrides
        prompt_config = self._extract_prompt_config(prompt)
        
        # Merge model defaults with prompt overrides
        merged_config = self._merge_generation_config(model_registry_config, prompt_config)

        if not hasattr(prompt, 'template'):
            raise TypeError(f"prompt must be a template object with 'template' attribute, got {type(prompt).__name__}")
        
        template_str = prompt.template
        formatted_prompt_text = template_str
        if prompt_variables:
            formatted_prompt_text = self.format_template(template_str, {"variables": prompt_variables})
        
        final_prompt_content = formatted_prompt_text
        generation_config = self._create_generation_config(merged_config)

        logger.info(f"Generating text with Gemini model '{gemini_model_id}' (registry: '{model_name}')")
        logger.debug(f"Generation Config: {generation_config}")
        logger.debug(f"Prompt: {final_prompt_content[:200]}...")

        try:
            # Use the correct Google AI API with rate limiting
            async def _generation_operation():
                return self._client.models.generate_content(
                    model=gemini_model_id,
                    contents=final_prompt_content,
                    config=generation_config
                )
            
            response = await self._execute_with_rate_limiting(
                f"generate_{model_name}",
                _generation_operation
            )
            # Safety ratings check (optional, for logging or specific handling)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Prompt blocked for model '{gemini_model_id}'. Reason: {response.prompt_feedback.block_reason_message}")
                error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="BlockedPromptError",
                    error_location="generate",
                    component="GoogleAIProvider",
                    operation="safety_check"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="safety_check",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Prompt blocked by Google AI safety filters: {response.prompt_feedback.block_reason_message}",
                    context=error_context,
                    provider_context=provider_context
                )

            if not response.candidates or not response.candidates[0].content.parts:
                 logger.warning(f"No content generated by model '{gemini_model_id}'. Response: {response}")
                 return "" # Or raise error

            return response.text # Simplest way to get text
        # Note: BlockedPromptException and StopCandidateException are from old API
        # The new API handles these through response objects
        except Exception as e:
            error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="GenerationError",
                    error_location="generate",
                    component="GoogleAIProvider",
                    operation=f"generate_with_{model_name}"
                )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation=f"generate_with_{model_name}",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Google AI generation failed: {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e
            )
            
    async def generate_structured(self, prompt: PromptTemplate, output_type: Type[ModelType], model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> ModelType:
        if not self._initialized or not self._client:
            error_context = ErrorContext.create(
                flow_name="GoogleAIProvider",
                error_type="InitializationError",
                error_location="generate_structured",
                component="GoogleAIProvider",
                operation="check_initialization"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="check_initialization",
                retry_count=0
            )
            
            raise ProviderError(
                message="Provider not initialized",
                context=error_context,
                provider_context=provider_context
            )
        if not inspect.isclass(output_type):
            raise TypeError(f"output_type must be a class, not an instance of {type(output_type)}")

        # Get model config from resource registry first
        try:
            model_registry_config = await self.get_model_config(model_name)
            if hasattr(model_registry_config, 'model_dump'):
                model_registry_config = model_registry_config.model_dump()
            elif not isinstance(model_registry_config, dict):
                model_registry_config = {}
        except Exception as e:
            logger.warning(f"Could not retrieve model config for '{model_name}': {e}. Using defaults.")
            model_registry_config = {}
        
        # Get model ID from config
        gemini_model_id = self._get_or_cache_model_config(model_name, model_registry_config)
        
        # Extract prompt config overrides
        prompt_config = self._extract_prompt_config(prompt)
        
        # Merge model defaults with prompt overrides
        merged_config = self._merge_generation_config(model_registry_config, prompt_config)

        if not hasattr(prompt, 'template'):
            raise TypeError(f"prompt must be a template object with 'template' attribute, got {type(prompt).__name__}")

        template_str = prompt.template
        formatted_prompt_text = template_str
        if prompt_variables:
            formatted_prompt_text = self.format_template(template_str, {"variables": prompt_variables})

        final_prompt_content = formatted_prompt_text
        
        # Create generation config with structured output
        generation_config = self._create_generation_config(merged_config)
        
        # Configure for structured output using the new API
        generation_config.response_mime_type = 'application/json'
        
        # Get JSON schema and clean it for Google AI compatibility
        schema = output_type.model_json_schema()
        self._clean_schema_for_google_ai(schema)
        generation_config.response_schema = schema
        
        logger.info(f"Generating structured output with Gemini model '{gemini_model_id}' (registry: '{model_name}')")
        logger.info(f"  Response Model: {output_type.__name__}")
        logger.debug(f"Generation Config: {generation_config}")
        logger.debug(f"Prompt: {final_prompt_content[:200]}...")

        try:
            # Use the correct Google AI API for structured output with rate limiting
            async def _structured_generation_operation():
                return self._client.models.generate_content(
                    model=gemini_model_id,
                    contents=final_prompt_content,
                    config=generation_config
                )
            
            response = await self._execute_with_rate_limiting(
                f"generate_structured_{model_name}",
                _structured_generation_operation
            )

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"Prompt blocked for model. Reason: {response.prompt_feedback.block_reason_message}")
                error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="BlockedPromptError",
                    error_location="generate_structured",
                    component="GoogleAIProvider",
                    operation="safety_check"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="safety_check",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Prompt blocked by Google AI safety filters: {response.prompt_feedback.block_reason_message}",
                    context=error_context,
                    provider_context=provider_context
                )

            if not response.candidates or not response.candidates[0].content.parts:
                error_context = ErrorContext.create(
                                        flow_name="GoogleAIProvider",
                                        error_type="NoContentError",
                                        error_location="generate_structured",
                                        component="GoogleAIProvider",
                                        operation=f"structured_generation_with_{model_name}"
                                    )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation=f"structured_generation_with_{model_name}",
                    retry_count=0
                )
                
                raise ProviderError(
                    message="No content generated by Gemini model for structured output.",
                    context=error_context,
                    provider_context=provider_context
                )

            # With the new API, structured output is returned as JSON text
            # when response_mime_type is 'application/json' and response_schema is set
            generated_text = response.text
            if not generated_text:
                error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="EmptyResponseError",
                    error_location="generate_structured",
                    component="GoogleAIProvider",
                    operation="parse_response_text"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="parse_response_text",
                    retry_count=0
                )
                
                raise ProviderError(
                    message="Empty response text from Gemini, cannot parse for structured output.",
                    context=error_context,
                    provider_context=provider_context
                )
            
            # Clean the text if it's wrapped in markdown ```json ... ```
            if generated_text.strip().startswith("```json"):
                generated_text = generated_text.strip()[7:-3].strip()
            elif generated_text.strip().startswith("```"):
                generated_text = generated_text.strip()[3:-3].strip()

            try:
                parsed_data = json.loads(generated_text)
                return output_type.model_validate(parsed_data)
            except (json.JSONDecodeError, AttributeError, IndexError) as parse_err:
                logger.error(f"JSON parsing failed for structured output: {parse_err}. Response text: {response.text[:200]}")
                error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="JSONParseError",
                    error_location="generate_structured",
                    component="GoogleAIProvider",
                    operation=f"parse_json_for_{model_name}"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation=f"parse_json_for_{model_name}",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Failed to parse structured response: {parse_err}",
                    context=error_context,
                    provider_context=provider_context,
                    cause=parse_err
                )

        # Note: Old API exceptions (BlockedPromptException, StopCandidateException) not used in new API
        # Safety filtering is handled through response.prompt_feedback
        except Exception as e:
            # Catch Pydantic validation errors specifically if needed
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                 error_context = ErrorContext.create(
                        flow_name="GoogleAIProvider",
                        error_type="ValidationError",
                        error_location="generate_structured",
                        component="GoogleAIProvider",
                        operation=f"validate_response_for_{output_type.__name__}"
                    )
                 
                 provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation=f"validate_response_for_{output_type.__name__}",
                    retry_count=0
                )
                 
                 raise ProviderError(
                    message=f"Failed to validate Google AI response against model {output_type.__name__}: {str(e)}",
                    context=error_context,
                    provider_context=provider_context,
                    cause=e
                )
            error_context = ErrorContext.create(
                    flow_name="GoogleAIProvider",
                    error_type="StructuredGenerationError",
                    error_location="generate_structured",
                    component="GoogleAIProvider",
                    operation=f"structured_generation_with_{model_name}"
                )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation=f"structured_generation_with_{model_name}",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Google AI structured generation failed: {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e
            )

    def _format_prompt(self, prompt: str, model_type: str = "gemini", output_type: Optional[Type[ModelType]] = None) -> str:
        """Format a prompt for Gemini.
        
        Gemini models generally don't require strict special tokens like Llama.
        This method mainly relies on the base class to add JSON schema instructions if `output_type` is provided.
        
        Args:
            prompt: The main prompt text.
            model_type: A string indicating "gemini", "gemini_structured", etc. (currently not strictly used for different templates).
            output_type: Optional Pydantic model for structured output guidance.
            
        Returns:
            Formatted prompt string.
        """
        # Let the base class add JSON structure information if output_type is provided.
        # This can serve as a textual hint to the model.
        prompt_with_json_guidance = super()._format_prompt(prompt, model_type, output_type)
        
        # For Gemini, specific pre/post prompts like Llama2's [INST] are not typically needed.
        # If certain Gemini models or use cases benefit from specific instructions,
        # they could be added here based on `model_type`.
        # For now, the base class formatting (which adds schema description) is sufficient.
        
        # Example: if model_type == "gemini_chat_optimized"
        # return f"USER: {prompt_with_json_guidance}\nMODEL:\n"
        
        return prompt_with_json_guidance

    # _get_model_templates is not needed as Gemini doesn't rely on these like Llama.cpp models.
    # _extract_json is also less critical as Gemini's structured output handles JSON parsing.
    # _sanitize_strings might be useful for input prompts, but Gemini's output should be clean JSON
    # if structured output is used correctly.
