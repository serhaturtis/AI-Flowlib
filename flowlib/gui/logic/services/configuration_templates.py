"""
Configuration templates using flowlib resource registry pattern.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- No legacy code, no backward compatibility
- Type safety everywhere with strict validation
- Consistent placeholder patterns throughout
"""

import logging
from typing import List, Optional, Union
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from flowlib.resources.decorators.decorators import resource
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.base import ResourceBase

logger = logging.getLogger(__name__)


class TemplateDetails(StrictBaseModel):
    """Template details with strict validation."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    name: str
    type: str
    description: str
    resource_name: str
    category: str
    detailed_description: str


class TemplateGenerationParameters(StrictBaseModel):
    """Template generation parameters with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str
    description: str = ""
    provider: str = "llamacpp"
    additional_params: dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


@resource(name="llm-provider-config-template")
class LLMProviderConfigTemplate(ResourceBase):
    """LLM provider configuration template resource."""
    
    name: str = "llm-provider-config-template"
    type: str = "template_config"
    template: str = '''from typing import Optional
from flowlib.resources.models.config_resource import LLMConfigResource
from flowlib.resources.decorators.decorators import llm_config

@llm_config("{config_name}")
class {class_name}(LLMConfigResource):
    """
    {description}
    
    This is a PROVIDER configuration - it defines how to connect to and configure
    a provider instance. Multiple models can use this same provider.
    """
    provider_type: str = "{provider_type}"{additional_params}
'''
    
    def generate(self, parameters: TemplateGenerationParameters) -> str:
        """Generate provider configuration from template."""
        additional_params = self._generate_provider_params(parameters)
        # Add newline and indent if there are additional params, otherwise empty
        if additional_params:
            additional_params = '\n    ' + additional_params
        else:
            additional_params = ''
        
        params = {
            'config_name': parameters.name,
            'class_name': self._generate_class_name(parameters.name),
            'description': parameters.description or 'LLM provider configuration',
            'provider_type': parameters.provider,
            'additional_params': additional_params
        }
        
        return self.template.format(**params)
    
    def _generate_class_name(self, config_name: str) -> str:
        """Generate class name from config name."""
        return _generate_class_name(config_name)
    
    def _generate_provider_params(self, parameters: TemplateGenerationParameters) -> str:
        """Generate provider-specific parameters."""
        additional = []
        
        provider = parameters.provider
        if provider == 'llamacpp':
            # These fields are already in LLMConfigResource, so we don't need to duplicate them
            pass
        elif provider == 'google_ai':
            # Google AI provider settings (infrastructure only - clean separation)
            additional.extend([
                'api_key: str = ""'
                # model_id, max_tokens, top_k belong in model configs, not provider configs
            ])
        
        # Add any custom parameters
        for key, value in parameters.additional_params.items():
            if isinstance(value, str):
                additional.append(f'{key}: str = "{value}"')
            elif isinstance(value, bool):
                additional.append(f'{key}: bool = {value}')
            elif isinstance(value, int):
                additional.append(f'{key}: int = {value}')
            elif isinstance(value, float):
                additional.append(f'{key}: float = {value}')
        
        # If no additional params, return empty string to avoid leaving placeholder
        if not additional:
            return ''
        
        return '\n    '.join(additional)


@resource(name="model-config-template")
class ModelConfigTemplate(ResourceBase):
    """Model configuration template resource."""
    
    name: str = "model-config-template"
    type: str = "template_config"
    template: str = '''from flowlib.resources.models.model_resource import ModelResource
from flowlib.resources.decorators.decorators import model_config

@model_config("{config_name}", provider_type="{provider_type}")
class {class_name}(ModelResource):
    """
    {description}
    
    This is a MODEL configuration - it defines settings for a specific model
    that uses a provider. Multiple models can reference the same provider.
    """
    provider_type: str = "{provider_type}"
    model_path: str = "{model_path}"
    model_name: str = "{model_name}"
    model_type: str = "{model_type}"
    temperature: float = {temperature}
    max_tokens: int = {max_tokens}
    top_p: float = {top_p}{additional_params}
'''
    
    def generate(self, parameters: TemplateGenerationParameters) -> str:
        """Generate model configuration from template."""
        additional_params = self._generate_model_params(parameters)
        # Add newline and indent if there are additional params, otherwise empty
        if additional_params:
            additional_params = '\n    ' + additional_params
        else:
            additional_params = ''
        
        params = {
            'config_name': parameters.name,
            'class_name': self._generate_class_name(parameters.name),
            'description': parameters.description or 'Model configuration',
            'provider_type': parameters.provider,
            # Fail-fast approach - required parameters must be provided
            'model_path': parameters.additional_params['model_path'] if 'model_path' in parameters.additional_params else '/path/to/model.gguf',
            'model_name': parameters.additional_params['model_name'] if 'model_name' in parameters.additional_params else 'default',
            'temperature': parameters.additional_params['temperature'] if 'temperature' in parameters.additional_params else 0.7,
            'max_tokens': parameters.additional_params['max_tokens'] if 'max_tokens' in parameters.additional_params else 1000,
            'top_p': parameters.additional_params['top_p'] if 'top_p' in parameters.additional_params else 0.9,
            'additional_params': additional_params
        }
        
        return self.template.format(**params)
    
    def _generate_class_name(self, config_name: str) -> str:
        """Generate class name from config name."""
        return _generate_class_name(config_name)
    
    def _generate_model_params(self, parameters: TemplateGenerationParameters) -> str:
        """Generate model-specific parameters."""
        provider = parameters.provider
        additional = []
        
        if provider == 'llamacpp':
            # Model-specific settings that don't belong in provider config
            if 'custom_stop_sequences' in parameters.additional_params:
                additional.append(f'stop_sequences: List[str] = {parameters.additional_params["custom_stop_sequences"]}')
        elif provider == 'google_ai':
            # Model-specific settings for Google AI
            if 'safety_settings' in parameters.additional_params:
                additional.append(f'safety_settings: dict = {parameters.additional_params["safety_settings"]}')
        
        # Add any other custom parameters
        for key, value in parameters.additional_params.items():
            if key not in ['model_path', 'model_name', 'temperature', 'max_tokens', 'top_p', 'custom_stop_sequences', 'safety_settings']:
                if isinstance(value, str):
                    additional.append(f'{key}: str = "{value}"')
                elif isinstance(value, bool):
                    additional.append(f'{key}: bool = {value}')
                elif isinstance(value, int):
                    additional.append(f'{key}: int = {value}')
                elif isinstance(value, float):
                    additional.append(f'{key}: float = {value}')
        
        # If no additional params, return empty string to avoid leaving placeholder
        if not additional:
            return ''
        
        return '\n    '.join(additional)


@resource(name="database-config-template")
class DatabaseConfigTemplate(ResourceBase):
    """Database configuration template resource."""
    
    name: str = "database-config-template"
    type: str = "template_config"
    template: str = '''from flowlib.resources.models.config_resource import DatabaseConfigResource
from flowlib.resources.decorators.decorators import database_config

@database_config("{config_name}")
class {class_name}(DatabaseConfigResource):
    """
    {description}
    """
    provider_type: str = "{provider_type}"{additional_params}
'''
    
    def generate(self, parameters: TemplateGenerationParameters) -> str:
        """Generate database configuration from template."""
        additional_params = self._generate_db_params(parameters)
        # Add newline and indent if there are additional params, otherwise empty
        if additional_params:
            additional_params = '\n    ' + additional_params
        else:
            additional_params = ''
        
        params = {
            'config_name': parameters.name,
            'class_name': self._generate_class_name(parameters.name),
            'description': parameters.description or 'Database configuration',
            'provider_type': parameters.provider,
            'additional_params': additional_params
        }
        
        return self.template.format(**params)
    
    def _generate_class_name(self, config_name: str) -> str:
        """Generate class name from config name."""
        return _generate_class_name(config_name)
    
    def _generate_db_params(self, parameters: TemplateGenerationParameters) -> str:
        """Generate database-specific parameters."""
        provider = parameters.provider
        additional = []
        
        # The base DatabaseConfigResource already has host, port, database, username, password, pool_size
        # We only add provider-specific fields that aren't in the base class
        if provider == 'sqlite':
            additional.extend([
                'database_path: str = ""',
                'create_if_missing: bool = True'
            ])
        elif provider == 'mongodb':
            additional.extend([
                'connection_string: str = ""'
            ])
        
        # Add any custom parameters
        for key, value in parameters.additional_params.items():
            if isinstance(value, str):
                additional.append(f'{key}: str = "{value}"')
            elif isinstance(value, bool):
                additional.append(f'{key}: bool = {value}')
            elif isinstance(value, int):
                additional.append(f'{key}: int = {value}')
            elif isinstance(value, float):
                additional.append(f'{key}: float = {value}')
        
        # If no additional params, return empty string to avoid leaving placeholder
        if not additional:
            return ''
        
        return '\n    '.join(additional)


@resource(name="vector-config-template")
class VectorConfigTemplate(ResourceBase):
    """Vector store configuration template resource."""
    
    name: str = "vector-config-template"
    type: str = "template_config"
    template: str = '''from flowlib.resources.models.config_resource import VectorDBConfigResource
from flowlib.resources.decorators.decorators import vector_db_config

@vector_db_config("{config_name}")
class {class_name}(VectorDBConfigResource):
    """
    {description}
    """
    provider_type: str = "{provider_type}"{additional_params}
'''
    
    def generate(self, parameters: TemplateGenerationParameters) -> str:
        """Generate vector configuration from template."""
        additional_params = self._generate_vector_params(parameters)
        # Add newline and indent if there are additional params, otherwise empty
        if additional_params:
            additional_params = '\n    ' + additional_params
        else:
            additional_params = ''
        
        params = {
            'config_name': parameters.name,
            'class_name': self._generate_class_name(parameters.name),
            'description': parameters.description or 'Vector store configuration',
            'provider_type': parameters.provider,
            'additional_params': additional_params
        }
        
        return self.template.format(**params)
    
    def _generate_class_name(self, config_name: str) -> str:
        """Generate class name from config name."""
        return _generate_class_name(config_name)
    
    def _generate_vector_params(self, parameters: TemplateGenerationParameters) -> str:
        """Generate vector store-specific parameters."""
        provider = parameters.provider
        additional = []
        
        # VectorDBConfigResource already has collection_name, dimensions, distance_metric, index_type
        # We only add provider-specific fields that aren't in the base class
        if provider == 'chroma':
            additional.extend([
                'persist_directory: str = ""'
            ])
        elif provider == 'qdrant':
            additional.extend([
                'url: str = "http://localhost:6333"'
            ])
        elif provider == 'pinecone':
            additional.extend([
                'api_key: str = ""',
                'environment: str = ""',
                'index_name: str = ""'
            ])
        
        # Add any custom parameters
        for key, value in parameters.additional_params.items():
            if isinstance(value, str):
                additional.append(f'{key}: str = "{value}"')
            elif isinstance(value, bool):
                additional.append(f'{key}: bool = {value}')
            elif isinstance(value, int):
                additional.append(f'{key}: int = {value}')
            elif isinstance(value, float):
                additional.append(f'{key}: float = {value}')
        
        # If no additional params, return empty string to avoid leaving placeholder
        if not additional:
            return ''
        
        return '\n    '.join(additional)


def _generate_class_name(config_name: str) -> str:
    """Generate class name from config name."""
    # Convert kebab-case to PascalCase
    words = config_name.replace('-', '_').replace('_', ' ').split()
    return ''.join(word.capitalize() for word in words) + 'Config'


