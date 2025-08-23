"""
Dynamic Settings Discovery for Providers

This module provides functionality to discover provider settings classes
dynamically from the registry, avoiding any hardcoding of provider types
or settings structures.
"""

import logging
from typing import Dict, Type, Optional, List, Any, Tuple
from flowlib.core.models import StrictBaseModel

from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from flowlib.providers.core.base import ProviderSettings

logger = logging.getLogger(__name__)


class SettingsDiscovery:
    """Discovers provider settings dynamically from registries."""
    
    @staticmethod
    def get_available_provider_types() -> Dict[str, List[str]]:
        """Get all available provider types grouped by category.
        
        Returns:
            Dict mapping provider categories to list of provider types
            Example: {'llm': ['llamacpp', 'openai'], 'vector_db': ['chroma', 'qdrant']}
        """
        # Access the internal factory metadata to discover registered providers
        provider_types = {}
        
        try:
            # Access the registry's internal storage (this is a bit hacky but necessary)
            if hasattr(provider_registry, '_factory_metadata'):
                for (category, provider_type), metadata in provider_registry._factory_metadata.items():
                    if category not in provider_types:
                        provider_types[category] = []
                    if provider_type not in provider_types[category]:
                        provider_types[category].append(provider_type)
            
            logger.debug(f"Discovered provider types: {provider_types}")
            return provider_types
            
        except Exception as e:
            logger.error(f"Failed to discover provider types: {e}")
            return {}
    
    @staticmethod
    def get_provider_settings_class(category: str, provider_type: str) -> Optional[Type[ProviderSettings]]:
        """Get the settings class for a specific provider.
        
        Args:
            category: Provider category (e.g., 'llm', 'vector_db')
            provider_type: Provider type (e.g., 'llamacpp', 'chroma')
            
        Returns:
            The settings class if found, None otherwise
        """
        try:
            key = (category, provider_type)
            
            # Check if provider is registered - fail-fast approach
            if hasattr(provider_registry, '_factory_metadata'):
                if key not in provider_registry._factory_metadata:
                    raise ValueError(f"Provider key {key} not found in factory metadata")
                metadata = provider_registry._factory_metadata[key]
                
                if 'settings_class' not in metadata:
                    raise ValueError(f"Settings class not found for provider {key}")
                settings_class = metadata['settings_class']
                
                if settings_class and issubclass(settings_class, ProviderSettings):
                    logger.debug(f"Found settings class for {category}/{provider_type}: {settings_class.__name__}")
                    return settings_class
                else:
                    logger.warning(f"No valid settings class found for {category}/{provider_type}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get settings class for {category}/{provider_type}: {e}")
            return None
    
    @staticmethod
    def get_provider_factory_metadata(category: str, provider_type: str) -> Dict[str, Any]:
        """Get all metadata for a provider factory.
        
        Args:
            category: Provider category
            provider_type: Provider type
            
        Returns:
            Dictionary of metadata
        """
        try:
            key = (category, provider_type)
            
            if hasattr(provider_registry, '_factory_metadata'):
                if key not in provider_registry._factory_metadata:
                    raise ValueError(f"Provider key {key} not found in factory metadata")
                return provider_registry._factory_metadata[key].copy()
            
            raise ValueError("Provider registry does not have factory metadata")
            
        except Exception as e:
            logger.error(f"Failed to get factory metadata: {e}")
            return {}
    
    @staticmethod
    def get_model_config_schema(provider_type: str) -> Dict[str, Dict[str, Any]]:
        """Get model configuration schema for a specific provider type.
        
        Args:
            provider_type: Provider type (e.g., 'llamacpp', 'openai')
            
        Returns:
            Dict mapping field names to field metadata for model-specific settings
        """
        # Define model config schemas based on provider capabilities
        # This is the single source of truth for what model fields each provider supports
        
        model_schemas = {
            'llamacpp': {
                'path': {
                    'type': 'str',
                    'required': True,
                    'description': 'Path to the GGUF model file'
                },
                'model_type': {
                    'type': 'str',
                    'required': False,
                    'description': 'Model architecture type (e.g., phi4, llama3)'
                },
                'n_ctx': {
                    'type': 'int',
                    'required': False,
                    'default': 4096,
                    'description': 'Context window size in tokens'
                },
                'temperature': {
                    'type': 'float',
                    'required': False,
                    'default': 0.7,
                    'description': 'Sampling temperature (0.0 = deterministic, 2.0 = very random)'
                },
                'max_tokens': {
                    'type': 'int',
                    'required': False,
                    'default': 2048,
                    'description': 'Maximum tokens to generate'
                },
                'use_gpu': {
                    'type': 'bool',
                    'required': False,
                    'default': False,
                    'description': 'Enable GPU acceleration'
                },
                'n_gpu_layers': {
                    'type': 'int',
                    'required': False,
                    'default': 0,
                    'description': 'Number of layers to offload to GPU (-1 for all)'
                },
                'top_p': {
                    'type': 'float',
                    'required': False,
                    'default': 1.0,
                    'description': 'Top-p (nucleus) sampling threshold'
                },
                'frequency_penalty': {
                    'type': 'float',
                    'required': False,
                    'default': 0.0,
                    'description': 'Frequency penalty to reduce repetition'
                },
                'presence_penalty': {
                    'type': 'float',
                    'required': False,
                    'default': 0.0,
                    'description': 'Presence penalty to encourage topic diversity'
                }
            },
            'openai': {
                'model': {
                    'type': 'str',
                    'required': True,
                    'description': 'OpenAI model name (e.g., gpt-4, gpt-3.5-turbo)'
                },
                'temperature': {
                    'type': 'float',
                    'required': False,
                    'default': 0.7,
                    'description': 'Sampling temperature (0.0 = deterministic, 2.0 = very random)'
                },
                'max_tokens': {
                    'type': 'int',
                    'required': False,
                    'description': 'Maximum tokens to generate'
                },
                'top_p': {
                    'type': 'float',
                    'required': False,
                    'default': 1.0,
                    'description': 'Top-p (nucleus) sampling threshold'
                },
                'frequency_penalty': {
                    'type': 'float',
                    'required': False,
                    'default': 0.0,
                    'description': 'Frequency penalty to reduce repetition'
                },
                'presence_penalty': {
                    'type': 'float',
                    'required': False,
                    'default': 0.0,
                    'description': 'Presence penalty to encourage topic diversity'
                }
            }
        }
        
        return model_schemas.get(provider_type, {})
    
    @staticmethod
    def get_model_config_fields_from_registry(provider_type: str = None) -> Dict[str, Any]:
        """Get model config fields dynamically from the registry.
        
        DEPRECATED: Use get_model_config_schema() instead for single source of truth.
        """
        logger.warning("get_model_config_fields_from_registry is deprecated, use get_model_config_schema instead")
        return {}
    
    @staticmethod
    def get_settings_fields(settings_class: Type[ProviderSettings]) -> Dict[str, Dict[str, Any]]:
        """Extract field information from a settings class.
        
        Args:
            settings_class: Pydantic settings class
            
        Returns:
            Dict mapping field names to field metadata
        """
        fields = {}
        
        try:
            # Get Pydantic model fields
            for field_name, field_info in settings_class.model_fields.items():
                field_data = {
                    "type": str(field_info.annotation),
                    "required": field_info.is_required(),
                    "default": field_info.default,
                    "description": field_info.description or "",
                }
                
                # Extract additional validation info if available
                if hasattr(field_info, 'constraints'):
                    field_data["constraints"] = field_info.constraints
                
                fields[field_name] = field_data
            
            return fields
            
        except Exception as e:
            logger.error(f"Failed to extract fields from settings class: {e}")
            return {}
    
    @staticmethod
    def discover_all_provider_settings() -> Dict[Tuple[str, str], Type[ProviderSettings]]:
        """Discover all provider settings classes from the registry.
        
        Returns:
            Dict mapping (category, provider_type) to settings class
        """
        all_settings = {}
        
        provider_types = SettingsDiscovery.get_available_provider_types()
        
        for category, types in provider_types.items():
            for provider_type in types:
                settings_class = SettingsDiscovery.get_provider_settings_class(category, provider_type)
                if settings_class:
                    all_settings[(category, provider_type)] = settings_class
        
        return all_settings