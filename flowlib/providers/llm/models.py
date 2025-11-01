"""Strict Pydantic models for LLM providers.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from typing import Any

from pydantic import Field

from flowlib.core.models import StrictBaseModel


class LlamaModelConfig(StrictBaseModel):
    """Strict configuration for Llama models - Model-specific settings only.

    Contains all settings that are specific to a particular model file,
    including the model's capabilities, default generation parameters,
    and model-specific hardware requirements.

    All fields are required except where explicitly made optional.
    """

    # Inherits strict configuration from StrictBaseModel
    # Model file and identity
    path: str = Field(..., description="Path to the model file")

    # Model-specific capabilities and limits
    n_ctx: int = Field(..., description="Context window size (model-specific)")

    # Model-specific hardware requirements (no provider defaults - all required)
    use_gpu: bool = Field(..., description="Whether this model should use GPU acceleration")
    n_gpu_layers: int = Field(..., description="Number of layers to offload to GPU for this model")
    n_threads: int = Field(..., description="Number of CPU threads for inference with this model")
    n_batch: int = Field(..., description="Batch size for processing optimization with this model")

    # Model-specific generation parameters
    temperature: float = Field(..., description="Default sampling temperature for this model")
    max_tokens: int = Field(..., description="Default maximum tokens to generate for this model")
    top_p: float = Field(default=0.9, description="Default top-p sampling parameter")
    top_k: int = Field(default=40, description="Default top-k sampling parameter")
    repeat_penalty: float = Field(default=1.1, description="Default repetition penalty")

    # Model-specific chat formatting
    chat_format: str = Field(
        ..., description="Chat format template for this model (e.g., 'chatml', 'alpaca', 'vicuna')"
    )

    # Optional provider-level overrides
    verbose: bool | None = Field(
        default=None, description="Override provider verbose setting for this model"
    )

    # Note: All strict settings inherited from StrictBaseModel


class LLMGenerationConfig(StrictBaseModel):
    """Strict configuration for LLM text generation."""

    # Inherits strict configuration from StrictBaseModel

    temperature: float = Field(..., description="Sampling temperature")
    max_tokens: int = Field(..., description="Maximum tokens to generate")
    top_p: float = Field(..., description="Top-p sampling parameter")
    top_k: int = Field(..., description="Top-k sampling parameter")
    frequency_penalty: float = Field(..., description="Frequency penalty")
    presence_penalty: float = Field(..., description="Presence penalty")


class GoogleAIModelConfig(StrictBaseModel):
    """Strict configuration for Google AI models."""

    # Inherits strict configuration from StrictBaseModel

    model_name: str = Field(..., description="Name of the Google AI model")
    api_key: str = Field(..., description="Google AI API key")
    safety_settings: dict[str, Any] = Field(..., description="Safety filter settings")
    generation_config: LLMGenerationConfig = Field(..., description="Generation parameters")


class BaseModelConfig(StrictBaseModel):
    """Base configuration for all LLM models."""

    # Inherits strict configuration from StrictBaseModel

    model_id: str = Field(..., description="Unique identifier for the model")
    provider_type: str = Field(..., description="Type of provider (llama_cpp, google_ai, etc)")
    model_name: str = Field(..., description="Human-readable model name")
