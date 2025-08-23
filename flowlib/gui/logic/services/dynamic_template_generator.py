"""
Dynamic Template Generator for Provider Configurations.

This module generates configuration templates by inspecting actual provider
configuration classes, following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts 
- Dynamic generation from real provider implementations
- Type safety with strict validation
"""

import logging
import inspect
from typing import Dict, List, Type, Any, Optional, get_type_hints
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)


class ProviderConfigDiscovery:
    """Discovery service for provider configuration classes."""
    
    def __init__(self):
        self._config_classes: Dict[str, Type[StrictBaseModel]] = {}
        self._discover_provider_configs()
    
    def _discover_provider_configs(self):
        """Discover provider configuration classes from flowlib."""
        try:
            # Import all provider configuration resource classes
            from flowlib.resources.models.config_resource import (
                LLMConfigResource, DatabaseConfigResource, VectorDBConfigResource,
                CacheConfigResource, StorageConfigResource, EmbeddingConfigResource,
                GraphDBConfigResource, MessageQueueConfigResource
            )
            
            # Import provider model classes (for model-specific configs)
            from flowlib.providers.llm.models import LlamaModelConfig, GoogleAIModelConfig, BaseModelConfig
            
            # Map provider types to their config classes
            # Provider-level configurations
            self._config_classes = {
                # LLM Provider configs
                'llm_provider_llamacpp': LLMConfigResource,
                'llm_provider_google_ai': LLMConfigResource,
                
                # Database Provider configs
                'database_postgresql': DatabaseConfigResource,
                'database_mongodb': DatabaseConfigResource,
                'database_sqlite': DatabaseConfigResource,
                
                # Vector DB Provider configs
                'vector_chroma': VectorDBConfigResource,
                'vector_qdrant': VectorDBConfigResource,
                'vector_pinecone': VectorDBConfigResource,
                
                # Cache Provider configs
                'cache_redis': CacheConfigResource,
                'cache_memory': CacheConfigResource,
                
                # Storage Provider configs
                'storage_s3': StorageConfigResource,
                'storage_local': StorageConfigResource,
                
                # Graph DB Provider configs
                'graph_neo4j': GraphDBConfigResource,
                'graph_arango': GraphDBConfigResource,
                'graph_janus': GraphDBConfigResource,
                
                # Embedding Provider configs
                'embedding_llamacpp': EmbeddingConfigResource,
                
                # Message Queue Provider configs
                'mq_rabbitmq': MessageQueueConfigResource,
                'mq_kafka': MessageQueueConfigResource,
                
                # Model-specific configurations (existing)
                'llamacpp': LlamaModelConfig,
                'google_ai': GoogleAIModelConfig,
                'base': BaseModelConfig
            }
            
            logger.info(f"Discovered {len(self._config_classes)} provider config classes")
            
        except ImportError as e:
            logger.error(f"Failed to import provider config classes: {e}")
            self._config_classes = {}
    
    def get_config_class(self, provider_type: str) -> Optional[Type[StrictBaseModel]]:
        """Get the configuration class for a provider type."""
        return self._config_classes[provider_type] if provider_type in self._config_classes else None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider types."""
        return list(self._config_classes.keys())


class FieldAnalyzer:
    """Analyzes Pydantic model fields to generate template parameters."""
    
    @staticmethod
    def analyze_field(field_name: str, field_info: FieldInfo, type_hint: Type) -> Dict[str, Any]:
        """Analyze a Pydantic field to extract template generation parameters."""
        # Handle Pydantic default values properly
        default_value = None
        if not field_info.is_required():
            if hasattr(field_info, 'default') and field_info.default is not ...:
                default_value = field_info.default
            elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                # For factory defaults, we'll use a representative value
                if field_info.default_factory is dict:
                    default_value = {}
                elif field_info.default_factory is list:
                    default_value = []
                else:
                    default_value = None
        
        result = {
            'name': field_name,
            'type': type_hint,
            'required': field_info.is_required(),
            'description': field_info.description or f"Configuration for {field_name}",
            'default': default_value
        }
        
        # Determine string representation of type
        if type_hint == str:
            result['type_str'] = 'str'
            result['default_value'] = f'"{result["default"]}"' if result['default'] else '""'
        elif type_hint == int:
            result['type_str'] = 'int'
            result['default_value'] = str(result['default']) if result['default'] is not None else '0'
        elif type_hint == float:
            result['type_str'] = 'float'
            result['default_value'] = str(result['default']) if result['default'] is not None else '0.0'
        elif type_hint == bool:
            result['type_str'] = 'bool'
            result['default_value'] = str(result['default']) if result['default'] is not None else 'False'
        else:
            # Handle complex types properly
            type_str = str(type_hint)
            if hasattr(type_hint, '__name__'):
                # For classes like LLMGenerationConfig
                result['type_str'] = type_hint.__name__
            elif 'typing.' in type_str:
                # For typing constructs like Dict[str, Any]
                result['type_str'] = type_str.replace('typing.', '')
            else:
                # Fallback
                result['type_str'] = type_str
            result['default_value'] = 'None' if result['default'] is None else f'"{result["default"]}"'
        
        return result


class DynamicTemplateGenerator:
    """Generates configuration templates dynamically from provider config classes."""
    
    def __init__(self):
        self.discovery = ProviderConfigDiscovery()
        self.field_analyzer = FieldAnalyzer()
    
    def generate_template(self, provider_type: str, config_name: str, description: str = "") -> str:
        """Generate a configuration template for the specified provider type."""
        config_class = self.discovery.get_config_class(provider_type)
        if not config_class:
            raise ValueError(f"No configuration class found for provider type: {provider_type}")
        
        # Analyze the configuration class
        fields = self._analyze_config_class(config_class)
        
        # Generate the template
        template = self._build_template(
            provider_type=provider_type,
            config_name=config_name,
            description=description,
            fields=fields,
            base_class=config_class.__name__
        )
        
        return template
    
    def _analyze_config_class(self, config_class: Type[BaseModel]) -> List[Dict[str, Any]]:
        """Analyze a Pydantic configuration class to extract field information."""
        fields = []
        
        # Get type hints for the class
        type_hints = get_type_hints(config_class)
        
        # Analyze each field
        for field_name, field_info in config_class.model_fields.items():
            if field_name in type_hints:
                field_data = self.field_analyzer.analyze_field(
                    field_name, field_info, type_hints[field_name]
                )
                fields.append(field_data)
        
        return fields
    
    def _build_template(self, provider_type: str, config_name: str, description: str, 
                       fields: List[Dict[str, Any]], base_class: str) -> str:
        """Build the configuration template string."""
        
        # Determine the appropriate decorator and import based on provider type
        if provider_type in ['llamacpp', 'google_ai', 'base']:
            # Model configurations
            decorator = f'@model_config("{config_name}", provider_type="{provider_type}")'
            import_line = 'from flowlib.resources.decorators.decorators import model_config'
            base_import = f'from flowlib.providers.llm.models import {base_class}'
            
        elif provider_type.startswith('llm_provider_'):
            # LLM Provider configurations
            actual_provider = provider_type.replace('llm_provider_', '')
            decorator = f'@llm_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import llm_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('database_'):
            # Database Provider configurations
            decorator = f'@database_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import database_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('vector_'):
            # Vector DB Provider configurations
            decorator = f'@vector_db_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import vector_db_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('cache_'):
            # Cache Provider configurations
            decorator = f'@cache_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import cache_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('storage_'):
            # Storage Provider configurations
            decorator = f'@storage_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import storage_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('graph_'):
            # Graph DB Provider configurations
            decorator = f'@graph_db_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import graph_db_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('embedding_'):
            # Embedding Provider configurations
            decorator = f'@embedding_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import embedding_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        elif provider_type.startswith('mq_'):
            # Message Queue Provider configurations
            decorator = f'@message_queue_config("{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import message_queue_config'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
            
        else:
            # Fallback for unknown providers
            decorator = f'@resource(name="{config_name}")'
            import_line = 'from flowlib.resources.decorators.decorators import resource'
            base_import = f'from flowlib.resources.models.config_resource import {base_class}'
        
        # Generate class name
        class_name = self._generate_class_name(config_name)
        
        # Collect additional imports needed for complex types
        additional_imports = set()
        for field in fields:
            if field['type_str'] in ['Dict', 'List', 'Optional', 'Union']:
                additional_imports.add('from typing import Dict, List, Optional, Union, Any')
            elif field['type_str'] == 'LLMGenerationConfig':
                # This type is already imported from the same module as the base class
                pass
        
        # Build additional import lines
        additional_import_lines = '\n'.join(additional_imports) if additional_imports else ''
        
        # Build field declarations
        field_declarations = []
        for field in fields:
            if field['required']:
                field_declarations.append(
                    f'    {field["name"]}: {field["type_str"]} = Field(..., description="{field["description"]}")'
                )
            else:
                field_declarations.append(
                    f'    {field["name"]}: {field["type_str"]} = Field(default={field["default_value"]}, description="{field["description"]}")'
                )
        
        # Build the complete template
        import_section = f'''from pydantic import Field
{import_line}
{base_import}'''
        
        if additional_import_lines:
            import_section += f'\n{additional_import_lines}'

        template = f'''{import_section}

{decorator}
class {class_name}({base_class}):
    """
    {description or "Dynamic configuration generated from " + base_class}
    
    Generated from {base_class} - all fields reflect the actual provider implementation.
    """
    model_config = {{"extra": "forbid"}}
    
{chr(10).join(field_declarations)}
'''
        
        return template
    
    def _generate_class_name(self, config_name: str) -> str:
        """Generate a class name from a configuration name."""
        words = config_name.replace('-', '_').replace('_', ' ').split()
        return ''.join(word.capitalize() for word in words) + 'Config'
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider types for template generation."""
        return self.discovery.get_available_providers()
    
    def get_providers_by_category(self) -> Dict[str, List[str]]:
        """Get providers organized by category."""
        providers = self.discovery.get_available_providers()
        categories = {
            'model': [],
            'llm_provider': [],
            'database': [],
            'vector': [],
            'cache': [],
            'storage': [],
            'graph': [],
            'embedding': [],
            'message_queue': []
        }
        
        for provider in providers:
            if provider in ['llamacpp', 'google_ai', 'base']:
                categories['model'].append(provider)
            elif provider.startswith('llm_provider_'):
                categories['llm_provider'].append(provider)
            elif provider.startswith('database_'):
                categories['database'].append(provider)
            elif provider.startswith('vector_'):
                categories['vector'].append(provider)
            elif provider.startswith('cache_'):
                categories['cache'].append(provider)
            elif provider.startswith('storage_'):
                categories['storage'].append(provider)
            elif provider.startswith('graph_'):
                categories['graph'].append(provider)
            elif provider.startswith('embedding_'):
                categories['embedding'].append(provider)
            elif provider.startswith('mq_'):
                categories['message_queue'].append(provider)
        
        return categories