"""LlamaCpp provider implementation for local language models.

This module implements a provider for local language models using the llama-cpp-python
library, which provides efficient inference on consumer hardware.
"""

import logging
import inspect
import os
import json
from typing import Any, Dict, Optional, Type
from llama_cpp import LlamaGrammar
from pydantic import BaseModel, Field, ConfigDict

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from ...core.decorators import provider
# Removed ProviderType import - using config-driven provider access
from ..base import LLMProvider, ModelType
from flowlib.providers.core.base import ProviderSettings
from ..models import LlamaModelConfig
from flowlib.resources.decorators.decorators import PromptTemplate

logger = logging.getLogger(__name__)


class GenerationParams(BaseModel):
    """Generation parameters model for LlamaCpp."""
    model_config = ConfigDict(extra="forbid")
    
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=40, description="Top-k sampling")
    repeat_penalty: float = Field(default=1.1, description="Repetition penalty")
    
    @classmethod
    def from_model_config(cls, model_config: Any) -> 'GenerationParams':
        """Extract generation parameters from model config.
        
        Args:
            model_config: Model configuration object
            
        Returns:
            Validated generation parameters
            
        Raises:
            ValueError: If model_config has invalid parameter types
        """
        params = {}
        
        # Extract parameters with type validation
        for field_name, field_info in cls.model_fields.items():
            if hasattr(model_config, field_name):
                value = getattr(model_config, field_name)
                # Validate type matches expected
                expected_type = field_info.annotation
                if expected_type == int and not isinstance(value, int):
                    raise ValueError(f"Parameter '{field_name}' must be int, got {type(value).__name__}")
                elif expected_type == float and not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{field_name}' must be float, got {type(value).__name__}")
                params[field_name] = value
        
        return cls(**params)


class StructuredGenerationParams(GenerationParams):
    """Generation parameters optimized for structured output."""
    # Override defaults for structured generation
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.2, description="Lower temperature for structured output") 
    top_p: float = Field(default=0.95, description="Higher top_p for structured output")


class LlamaCppSettings(ProviderSettings):
    """Settings for the LlamaCpp provider - Infrastructure/Provider concerns only.
    
    Provider-level settings that control how the LlamaCpp provider operates,
    independent of specific models loaded through it. Inherits common fields
    from ProviderSettings (timeout, max_retries, verbose, etc.) and adds
    LlamaCpp-specific fields.
    
    LlamaCpp-specific attributes:
        n_threads: Number of CPU threads to use for inference
        n_batch: Batch size for processing (infrastructure optimization)
        use_gpu: Whether to enable GPU acceleration capability
        n_gpu_layers: Number of layers to offload to GPU (can be overridden per model)
        max_concurrent_models: Maximum number of models to keep loaded simultaneously
    """
    
    # LlamaCpp-specific settings (infrastructure-level only)
    n_threads: int = Field(default=4, description="Number of CPU threads for inference (e.g., 4 for quad-core)")
    n_batch: int = Field(default=512, description="Batch size for processing optimization (512 for balanced performance)")
    use_gpu: bool = Field(default=False, description="Enable GPU acceleration (requires CUDA/Metal)")
    n_gpu_layers: int = Field(default=0, description="GPU layers to offload (0=CPU only, -1=all layers)")
    max_concurrent_models: int = Field(default=3, description="Maximum models loaded simultaneously (prevents OOM)")


@provider(provider_type="llm", name="llamacpp", settings_class=LlamaCppSettings)
class LlamaCppProvider(LLMProvider):
    """Provider for local inference using llama-cpp-python.
    
    This provider supports:
    1. Text generation with various LLM architectures (llama, phi, mistral, etc.)
    2. Structured output generation with format guidance
    3. Optional GPU acceleration with Metal or CUDA
    """
    
    def __init__(self, name: str, provider_type: str, settings: Optional[LlamaCppSettings] = None, **kwargs: Any):
        """Initialize LlamaCpp provider.
        
        Args:
            name: Unique provider name
            provider_type: The type of the provider (e.g., 'llm')
            settings: Provider settings
            **kwargs: Additional keyword arguments for the base LLMProvider
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        if not isinstance(self.settings, LlamaCppSettings):
            raise TypeError(f"settings must be a LlamaCppSettings instance, got {type(self.settings)}")
        
        # Store settings for local use
        self._models = {}
        self._settings = settings
            
    async def _initialize_model(self, model_name: str):
        """Initialize a specific model.
        
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
            
            # Get model configuration from registry - must be LlamaModelConfig
            model_config_raw = await self.get_model_config(model_name)
            
            # Convert to strict Pydantic model
            if isinstance(model_config_raw, dict):
                model_config = LlamaModelConfig(**model_config_raw)
            else:
                # Handle ModelResource format - access config dictionary
                if hasattr(model_config_raw, 'config') and isinstance(model_config_raw.config, dict):
                    model_config = LlamaModelConfig(**model_config_raw.config)
                else:
                    # Assume it's already a LlamaModelConfig or convert from object attributes
                    model_config = LlamaModelConfig(
                        path=model_config_raw.path,
                        model_type=model_config_raw.model_type,
                        n_ctx=model_config_raw.n_ctx,
                        n_threads=model_config_raw.n_threads,
                        n_batch=model_config_raw.n_batch,
                        use_gpu=model_config_raw.use_gpu,
                        n_gpu_layers=model_config_raw.n_gpu_layers,
                        verbose=model_config_raw.verbose,
                        temperature=model_config_raw.temperature,
                        max_tokens=model_config_raw.max_tokens
                    )
            
            # Extract model-specific values
            model_path = model_config.path
            model_type = model_config.model_type
            n_ctx = model_config.n_ctx
            use_gpu = model_config.use_gpu
            n_gpu_layers = model_config.n_gpu_layers
            
            # Use provider settings for infrastructure concerns, with model overrides
            n_threads = model_config.n_threads if model_config.n_threads is not None else self._settings.n_threads
            n_batch = model_config.n_batch if model_config.n_batch is not None else self._settings.n_batch
            verbose = model_config.verbose if model_config.verbose is not None else self._settings.verbose
            
            # Check if model path exists
            if not os.path.exists(model_path):
                error_context = ErrorContext.create(
                    flow_name="model_initialization",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}.load_model",
                    component=self.name,
                    operation="load_model"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="load_model",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Model path does not exist: {model_path}",
                    context=error_context,
                    provider_context=provider_context
                )
            
            # Check if we already have a model loaded with the same path to avoid OOM
            existing_model_entry = None
            for existing_name, existing_entry in self._models.items():
                existing_config = existing_entry["config"]
                # Strict contract: all model configs must have path attribute
                if not hasattr(existing_config, 'path'):
                    error_context = ErrorContext.create(
                        flow_name="model_initialization",
                        error_type="ProviderError",
                        error_location=f"{self.__class__.__name__}._initialize_model",
                        component=self.name,
                        operation="config_validation"
                    )
                    
                    provider_context = ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="llm",
                        operation="config_validation",
                        retry_count=0
                    )
                    
                    raise ProviderError(
                        message=f"Model config for '{existing_name}' missing required 'path' attribute",
                        context=error_context,
                        provider_context=provider_context
                    )
                existing_path = existing_config.path
                
                if existing_path == model_path:
                    logger.info(f"Reusing existing model instance for '{model_name}' (same path as '{existing_name}'): {model_path}")
                    existing_model_entry = existing_entry
                    break
            
            if existing_model_entry:
                # Reuse the existing model instance
                self._models[model_name] = {
                    "model": existing_model_entry["model"],
                    "config": model_config,
                    "type": model_type
                }
            else:
                # Load model with specified settings
                logger.info(f"Loading LlamaCpp model '{model_name}' from: {model_path}")
                model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    n_gpu_layers=n_gpu_layers if use_gpu else 0,
                    verbose=verbose
                )
                
                # Store model and its configuration
                self._models[model_name] = {
                    "model": model,
                    "config": model_config,
                    "type": model_type
                }
            
            logger.info(f"Loaded LlamaCpp model: {model_name} ({os.path.basename(model_path)})")
            
        except ImportError as e:
            error_context = ErrorContext.create(
                flow_name="model_initialization",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.load_model",
                component=self.name,
                operation="import_llama_cpp"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="import_llama_cpp",
                retry_count=0
            )
            
            raise ProviderError(
                message="llama-cpp-python package not installed. Install with: pip install llama-cpp-python",
                context=error_context,
                provider_context=provider_context,
                cause=e
            )
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for specific error types to provide better diagnostics
            if "failed to load model from file" in error_str:
                # This could be either a missing file or OOM/resource error
                if os.path.exists(model_path):
                    # File exists but loading failed - likely OOM or resource exhaustion
                    error_context = ErrorContext.create(
                        flow_name="model_loading",
                        error_type="ProviderError",
                        error_location=f"{self.__class__.__name__}._initialize_model",
                        component=self.name,
                        operation="load_model"
                    )
                    
                    provider_context = ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="llm",
                        operation="load_model",
                        retry_count=0
                    )
                    
                    raise ProviderError(
                        message=f"Failed to load LlamaCpp model '{model_name}': Model file exists but loading failed. This is typically caused by insufficient memory (OOM) or the model already being loaded by another process.",
                        context=error_context,
                        provider_context=provider_context,
                        cause=e
                    )
                else:
                    # File doesn't exist
                    error_context = ErrorContext.create(
                        flow_name="model_loading",
                        error_type="ProviderError",
                        error_location=f"{self.__class__.__name__}._initialize_model",
                        component=self.name,
                        operation="load_model"
                    )
                    
                    provider_context = ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="llm",
                        operation="load_model",
                        retry_count=0
                    )
                    
                    raise ProviderError(
                        message=f"Failed to load LlamaCpp model '{model_name}': Model file not found at path: {model_path}",
                        context=error_context,
                        provider_context=provider_context,
                        cause=e
                    )
            elif "out of memory" in error_str or "oom" in error_str or "memory" in error_str:
                # Explicit memory errors
                error_context = ErrorContext.create(
                    flow_name="model_loading",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}._initialize_model",
                    component=self.name,
                    operation="load_model"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="load_model",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Failed to initialize LlamaCpp model '{model_name}': Insufficient memory to load model",
                    context=error_context,
                    provider_context=provider_context,
                    cause=e
                )
            elif "device" in error_str or "cuda" in error_str or "gpu" in error_str:
                # GPU/device related errors
                error_context = ErrorContext.create(
                    flow_name="model_loading",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}._initialize_model",
                    component=self.name,
                    operation="load_model"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="load_model",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Failed to initialize LlamaCpp model '{model_name}': GPU/device error - {str(e)}",
                    context=error_context,
                    provider_context=provider_context,
                    cause=e
                )
            else:
                # Generic fallback with original error
                error_context = ErrorContext.create(
                    flow_name="model_loading",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}._initialize_model",
                    component=self.name,
                    operation="load_model"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="load_model",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"Failed to initialize LlamaCpp model '{model_name}': {str(e)}",
                    context=error_context,
                    provider_context=provider_context,
                    cause=e
                )
            
    async def initialize(self):
        """Initialize the provider."""
        self._initialized = True
        
    async def _initialize(self):
        """Initialize the provider.
        
        This implements the required abstract method from the Provider base class.
        The actual model initialization is done lazily in _initialize_model when needed.
        """
        # The LlamaCpp provider uses lazy initialization of models
        # when they are first requested, so there's no work to do here
        pass
        
    async def shutdown(self):
        """Release model resources."""
        for model_name, model_data in self._models.items():
            logger.info(f"Released LlamaCpp model: {model_name}")
        
        self._models = {}
        self._initialized = False
        
    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> str:
        """Generate text completion.
        
        Args:
            prompt: Prompt template object with template and config attributes
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Generated text
            
        Raises:
            ProviderError: If generation fails
            TypeError: If prompt is not a valid template object
        """
        # Make sure model is initialized
        if model_name not in self._models:
            await self._initialize_model(model_name)
            
        model_data = self._models[model_name]
        model = model_data["model"]
        model_config = model_data["config"]
            
        try:
            # Validate prompt is a template object with template attribute
            if not hasattr(prompt, 'template'):
                raise TypeError(f"prompt must be a template object with 'template' attribute, got {type(prompt).__name__}")
            
            # Get template string from prompt
            template_str = prompt.template
            
            # Start with default generation parameters from model config
            gen_params = GenerationParams.from_model_config(model_config)
            
            # Override with prompt-specific parameters if provided
            if hasattr(prompt, 'config') and prompt.config:
                # Extract generation parameters from prompt config and override
                overrides = {}
                for param_name in ['max_tokens', 'temperature', 'top_p', 'top_k', 'repeat_penalty']:
                    if param_name in prompt.config:
                        overrides[param_name] = prompt.config[param_name]
                
                if overrides:
                    # Create new generation params with overrides
                    base_params = gen_params.model_dump()
                    base_params.update(overrides)
                    gen_params = GenerationParams(**base_params)
            
            # Format prompt with variables if provided
            formatted_prompt = template_str
            if prompt_variables:
                # Format the template with variables
                formatted_prompt = self.format_template(template_str, {"variables": prompt_variables})
            
            # Run generation using validated parameters
            result = model(
                formatted_prompt,
                max_tokens=gen_params.max_tokens,
                temperature=gen_params.temperature,
                top_p=gen_params.top_p,
                top_k=gen_params.top_k,
                repeat_penalty=gen_params.repeat_penalty,
                stop=[]
            )
            
            # Extract generated text
            if isinstance(result, dict) and "choices" in result:
                # Extract from completion format
                return result["choices"][0]["text"]
            elif isinstance(result, list) and len(result) > 0:
                # Extract from list format
                return result[0]["text"] if "text" in result[0] else ""
            else:
                # Handle unexpected response format
                return str(result)
                
        except Exception as e:
            if isinstance(e, TypeError) and ("must be a template object" in str(e) or "output_type must be a class" in str(e)):
                # Re-raise TypeError for invalid prompt or output_type
                raise
            error_context = ErrorContext.create(
                flow_name="text_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate",
                component=self.name,
                operation="generate"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="generate",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Generation failed: {str(e)}",
                context=error_context,
                provider_context=provider_context,
                cause=e
            )
            
    async def generate_structured(self, prompt: PromptTemplate, output_type: Type[ModelType], model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> ModelType:
        """Generate structured output using a JSON grammar.
        
        Args:
            prompt: Prompt template object with template and config attributes
            output_type: Pydantic model for output validation
            model_name: Name of the model to use
            prompt_variables: Dictionary of variables to format the prompt template
            
        Returns:
            Validated response model instance
            
        Raises:
            ProviderError: If generation fails
            TypeError: If prompt is not a valid template object or if output_type is not a class
        """
        # Ensure output_type is a class, not an instance
        if not inspect.isclass(output_type):
            raise TypeError(f"output_type must be a class, not an instance of {type(output_type)}")
        
        # Make sure model is initialized
        if model_name not in self._models:
            await self._initialize_model(model_name)
        
        model_data = self._models[model_name]
        model = model_data["model"]
        model_config = model_data["config"]
        model_type = model_data["type"]
        
        # Validate prompt is a template object with template attribute
        if not hasattr(prompt, 'template'):
            raise TypeError(f"prompt must be a template object with 'template' attribute, got {type(prompt).__name__}")
        
        # Get template string from prompt
        template_str = prompt.template
        
        # Format prompt with variables if provided
        formatted_prompt_text = template_str
        if prompt_variables:
            # Format the template with variables
            formatted_prompt_text = self.format_template(template_str, {"variables": prompt_variables})
        
        # Format the prompt according to model type
        formatted_prompt = self._format_prompt(formatted_prompt_text, model_type, output_type)
        
        # Start with default generation parameters from model config  
        base_gen_params = GenerationParams.from_model_config(model_config)
        
        # Use structured generation defaults but inherit from model config
        gen_params = StructuredGenerationParams(
            max_tokens=base_gen_params.max_tokens,
            temperature=base_gen_params.temperature,
            top_p=base_gen_params.top_p,
            top_k=base_gen_params.top_k,
            repeat_penalty=base_gen_params.repeat_penalty
        )
        
        # Override with prompt-specific parameters if provided
        if hasattr(prompt, 'config') and prompt.config:
            # Extract generation parameters from prompt config and override
            overrides = {}
            for param_name in ['max_tokens', 'temperature', 'top_p', 'top_k', 'repeat_penalty']:
                if param_name in prompt.config:
                    overrides[param_name] = prompt.config[param_name]
            
            if overrides:
                # Create new generation params with overrides
                base_params = gen_params.model_dump()
                base_params.update(overrides)
                gen_params = StructuredGenerationParams(**base_params)
        
        # Log generation parameters
        logger.info("Starting LLM structured generation with parameters:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Model Type: {model_type}")
        logger.info(f"  Response Model: {output_type.__name__}")
        logger.info("  Generation Parameters:")
        for name, value in {
            "max_tokens": gen_params.max_tokens,
            "temperature": gen_params.temperature,
            "top_p": gen_params.top_p,
            "top_k": gen_params.top_k,
            "repeat_penalty": gen_params.repeat_penalty
        }.items():
            logger.info(f"    {name}: {value}")

        # Get schema from model
        try:
            schema = output_type.model_json_schema()
        except AttributeError as e:
            error_context = ErrorContext.create(
                flow_name="structured_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate_structured",
                component=self.name,
                operation="schema_generation"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="schema_generation",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Cannot generate structured output: {str(e)}. Model type does not support schema generation.",
                context=error_context,
                provider_context=provider_context
            ) from e
        
        # Create grammar from schema
        schema_str = json.dumps(schema)
        try:
            grammar = LlamaGrammar.from_json_schema(schema_str)
        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="structured_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate_structured",
                component=self.name,
                operation="grammar_creation"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="grammar_creation",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Failed to create grammar from schema: {str(e)}",
                context=error_context,
                provider_context=provider_context
            ) from e
        
        # Log the formatted prompt
        logger.info("=============== FORMATTED PROMPT ===============")
        logger.info(formatted_prompt)
        logger.info("================================================")
        
        # Generate with grammar
        try:
            result = model(
                formatted_prompt,
                max_tokens=gen_params.max_tokens,
                temperature=gen_params.temperature,
                top_p=gen_params.top_p,
                top_k=gen_params.top_k,
                repeat_penalty=gen_params.repeat_penalty,
                grammar=grammar
            )
        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="structured_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate_structured",
                component=self.name,
                operation="grammar_generation"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="grammar_generation",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Failed to generate with grammar: {str(e)}",
                context=error_context,
                provider_context=provider_context
            ) from e
        
        # Extract generated text
        if isinstance(result, dict) and "choices" in result:
            generated_text = result["choices"][0]["text"]
        elif isinstance(result, list) and len(result) > 0:
            generated_text = result[0]["text"] if "text" in result[0] else str(result[0])
        else:
            generated_text = str(result)
        
        logger.info(f"Generated text: {generated_text[:200]}...")
        
        # Attempt to parse the JSON
        # === ADDED SANITIZATION ===
        # Try to remove common problematic control chars before parsing
        # This specifically targets replacing literal newline/tab characters within the JSON string content,
        # which often cause issues, while preserving escaped ones (\n, \t).
        # It also removes other low-ASCII control characters except tab.
        import re
        sanitized_text = re.sub(r'(?<!\\)[\x00-\x08\x0B\x0C\x0E-\x1F]', '', generated_text)
        # Replace unescaped newlines and tabs within strings specifically
        # This regex is complex: it finds "key": "...content..." and replaces 
        # unescaped newlines/tabs within the content part.
        def replace_control_chars(match):
            key = match.group(1)
            content = match.group(2)
            # Replace unescaped newlines/tabs within the content
            cleaned_content = content.replace('\n', '\\n').replace('\t', '\\t')
            return f'"{key}": "{cleaned_content}"'
            
        # Apply the regex substitution - might need refinement based on JSON structure variance
        # This is a simplified attempt, complex nested structures might need more robust parsing
        try:
            sanitized_text = re.sub(r'"(\w+)":\s*"((?:[^"\\]|\\.)*)"' , replace_control_chars, sanitized_text)
        except Exception as regex_err:
             logger.warning(f"Regex sanitization failed: {regex_err}. Proceeding with original text.")
             # Fallback or just use the simpler char removal if regex fails
             sanitized_text = re.sub(r'(?<!\\)[\x00-\x08\x0B\x0C\x0E-\x1F]', '', generated_text)
        # ==========================
        
        parsed_data = json.loads(sanitized_text)
        
        # Validate with Pydantic model if provided
        try:
            # Create an instance of the output_type class using the parsed data
            validated_response = output_type.model_validate(parsed_data)
            return validated_response
        except Exception as e:
            error_context = ErrorContext.create(
                flow_name="structured_generation",
                error_type="ProviderError",
                error_location=f"{self.__class__.__name__}.generate_structured",
                component=self.name,
                operation="response_validation"
            )
            
            provider_context = ProviderErrorContext(
                provider_name=self.name,
                provider_type="llm",
                operation="response_validation",
                retry_count=0
            )
            
            raise ProviderError(
                message=f"Failed to validate response against model {output_type.__name__}: {str(e)}",
                context=error_context,
                provider_context=provider_context
            ) from e
            
    def _format_prompt(self, prompt: str, model_type: str = "default", output_type: Optional[Type[ModelType]] = None) -> str:
        """Format a prompt according to model-specific requirements.
        
        Applies model-specific formatting templates.
        
        Args:
            prompt: The main prompt text
            model_type: The type/name of the model
            output_type: Optional Pydantic model type for structured output
            
        Returns:
            Formatted prompt string
        """
        # First, let the base class potentially add JSON structure information
        prompt_with_json = super()._format_prompt(prompt, model_type, output_type)
        
        # Get template for model type with strict lookup
        templates = self._get_model_templates()
        model_type_key = model_type.lower()
        
        if model_type_key not in templates:
            if "default" not in templates:
                error_context = ErrorContext.create(
                    flow_name="prompt_formatting",
                    error_type="ProviderError",
                    error_location=f"{self.__class__.__name__}._format_prompt",
                    component=self.name,
                    operation="template_lookup"
                )
                
                provider_context = ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="llm",
                    operation="template_lookup",
                    retry_count=0
                )
                
                raise ProviderError(
                    message=f"No template found for model type '{model_type}' and no default template available",
                    context=error_context,
                    provider_context=provider_context
                )
            template = templates["default"]
        else:
            template = templates[model_type_key]
        
        # Apply template formatting
        formatted = template["pre_prompt"] + prompt_with_json + template["post_prompt"]
        
        return formatted
        
    def _get_model_templates(self) -> Dict[str, Dict[str, str]]:
        """Get model-specific prompt templates for different LLM architectures.
        
        Returns:
            Dictionary mapping model_type to pre/post prompt templates
        """
        return {
            "default": {
                "pre_prompt": "",
                "post_prompt": ""
            },
            "llama2": {
                "pre_prompt": "<s>[INST] ",
                "post_prompt": " [/INST]"
            },
            "phi2": {
                "pre_prompt": "Instruct: ",
                "post_prompt": "\nOutput: "
            },
            "phi4": {
                "pre_prompt": "assistant<|im_sep|>",
                "post_prompt": "assistant<|im_sep|>"
            },
            "mistral": {
                "pre_prompt": "",
                "post_prompt": "\n</s><|assistant|>\n"
            },
            "chatml": {
                "pre_prompt": "",
                "post_prompt": "\nassistant\n"
            }
        }
        
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string or empty string if none found
        """
        # Look for JSON object in the response
        import re
        
        # Try to find a JSON object using regex
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            potential_json = json_match.group(0)
            try:
                # Validate it's actually JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON object found, try the whole text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
            
        return ""

    def _sanitize_strings(self, obj):
        """Sanitize strings in parsed data to ensure consistent formatting.
        
        Args:
            obj: Object to sanitize (dict, list, string, or primitive)
            
        Returns:
            Sanitized object with special characters in strings properly escaped
        """
        if isinstance(obj, str):
            # Remove any special formatting characters that might cause issues
            return obj.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        elif isinstance(obj, dict):
            return {k: self._sanitize_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_strings(item) for item in obj]
        else:
            # Return other types unchanged
            return obj 