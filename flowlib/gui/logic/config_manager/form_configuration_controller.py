"""
Form-Based Configuration Controller

Enhanced configuration controller that integrates form-based configuration creation
with the existing configuration management system. Bridges form dialogs with the 
text-based configuration service backend.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from PySide6.QtCore import Signal, QObject
from PySide6.QtWidgets import QWidget, QMessageBox

from .configuration_controller import ConfigurationController
from ..services.models import ConfigurationCreateData, ConfigurationType
from flowlib.gui.ui.dialogs.provider_form_factory import (
    ProviderFormFactory, 
    create_provider_form_dialog,
    show_provider_form_wizard
)

logger = logging.getLogger(__name__)


class FormConfigurationData(StrictBaseModel):
    """Form-based configuration data with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    config_name: str = Field(..., min_length=1, description="Configuration name")
    provider_type: str = Field(..., description="Provider type")
    provider_specific_type: str = Field(..., description="Specific provider (e.g., 'llamacpp', 'openai')")
    config_type: str = Field(default="provider", description="Configuration type - 'provider' or 'model'")
    form_data: Dict[str, Any] = Field(..., description="Form field data")
    description: str = Field(default="", description="Configuration description")


class ConfigurationCodeGenerator:
    """Generates configuration code from form data following flowlib patterns."""
    
    @classmethod
    def _get_provider_config_templates(cls) -> Dict[str, str]:
        """Get provider configuration templates (infrastructure-level only)."""
        return {
        'llm': '''from flowlib.resources.models.config_resource import LLMConfigResource
from flowlib.resources.decorators.decorators import llm_config

@llm_config("{config_name}")
class {class_name}(LLMConfigResource):
    """Generated LLM provider configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'vector_db': '''from flowlib.resources.models.config_resource import VectorDBConfigResource
from flowlib.resources.decorators.decorators import vector_config

@vector_config("{config_name}")
class {class_name}(VectorDBConfigResource):
    """Generated Vector DB configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'database': '''from flowlib.resources.models.config_resource import DatabaseConfigResource
from flowlib.resources.decorators.decorators import database_config

@database_config("{config_name}")
class {class_name}(DatabaseConfigResource):
    """Generated Database configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'cache': '''from flowlib.resources.models.config_resource import CacheConfigResource
from flowlib.resources.decorators.decorators import cache_config

@cache_config("{config_name}")
class {class_name}(CacheConfigResource):
    """Generated Cache configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'storage': '''from flowlib.resources.models.config_resource import StorageConfigResource
from flowlib.resources.decorators.decorators import storage_config

@storage_config("{config_name}")
class {class_name}(StorageConfigResource):
    """Generated Storage configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'embedding': '''from flowlib.resources.models.config_resource import EmbeddingConfigResource
from flowlib.resources.decorators.decorators import embedding_config

@embedding_config("{config_name}")
class {class_name}(EmbeddingConfigResource):
    """Generated Embedding configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'graph_db': '''from flowlib.resources.models.config_resource import GraphDBConfigResource
from flowlib.resources.decorators.decorators import graph_config

@graph_config("{config_name}")
class {class_name}(GraphDBConfigResource):
    """Generated Graph DB configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'message_queue': '''from flowlib.resources.models.config_resource import MessageQueueConfigResource
from flowlib.resources.decorators.decorators import mq_config

@mq_config("{config_name}")
class {class_name}(MessageQueueConfigResource):
    """Generated Message Queue configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'db': '''from flowlib.resources.models.config_resource import DatabaseConfigResource
from flowlib.resources.decorators.decorators import database_config

@database_config("{config_name}")
class {class_name}(DatabaseConfigResource):
    """Generated Database configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'mcp_client': '''from flowlib.resources.models.config_resource import MCPClientConfigResource
from flowlib.resources.decorators.decorators import mcp_client_config

@mcp_client_config("{config_name}")
class {class_name}(MCPClientConfigResource):
    """Generated MCP Client configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
''',
        'mcp_server': '''from flowlib.resources.models.config_resource import MCPServerConfigResource
from flowlib.resources.decorators.decorators import mcp_server_config

@mcp_server_config("{config_name}")
class {class_name}(MCPServerConfigResource):
    """Generated MCP Server configuration from form data."""
    
    def __init__(self, **data):
        super().__init__(
            provider_type="{provider_specific_type}",
{config_fields}
            **data
        )
'''
        }
    
    @classmethod
    def _get_model_config_templates(cls) -> Dict[str, str]:
        """Get model configuration templates (model-specific settings only).
        
        Only LLM and embedding providers have meaningful model configurations.
        Other providers (vector DB, cache, storage, etc.) are infrastructure components.
        """
        return {
        'llm': '''from flowlib.resources.decorators.decorators import model_config

@model_config("{config_name}", provider_type="{provider_specific_type}", config={{
{model_config_fields}
}})
class {class_name}:
    """Generated LLM model configuration from form data."""
    pass
''',
        'embedding': '''from flowlib.resources.decorators.decorators import model_config

@model_config("{config_name}", provider_type="{provider_specific_type}", config={{
{model_config_fields}
}})
class {class_name}:
    """Generated embedding model configuration from form data."""
    pass
'''
        }
    
    @classmethod
    def _validate_form_config(cls, form_config: FormConfigurationData, config_type: str) -> None:
        """Validate form configuration data has all required fields."""
        # Validate basic required fields
        if not form_config.config_name.strip():
            raise ValueError("Configuration name cannot be empty")
        
        if not form_config.provider_type.strip():
            raise ValueError("Provider type cannot be empty")
        
        if not form_config.provider_specific_type.strip():
            raise ValueError("Provider specific type cannot be empty")
        
        if not form_config.form_data:
            raise ValueError("Form data cannot be empty")
        
        # Additional validation based on config type
        if config_type == "provider":
            # For provider configs, ensure we have infrastructure-level fields
            # This is provider-specific, but we can do basic checks
            pass
        elif config_type == "model":
            # For model configs, ensure we have model-specific fields
            # This is also provider-specific, but we can do basic checks
            pass
        else:
            raise ValueError(f"Invalid config_type: {config_type}. Must be 'provider' or 'model'")
    
    @classmethod
    def generate_configuration_code(cls, form_config: FormConfigurationData, config_type: str = "provider") -> str:
        """
        Generate Python configuration code from form data.
        
        Args:
            form_config: Validated form configuration data
            config_type: Type of config to generate - "provider" or "model"
            
        Returns:
            Generated Python configuration code
        """
        try:
            # Validate form configuration data first
            cls._validate_form_config(form_config, config_type)
            
            # Get appropriate template based on config type
            if config_type == "provider":
                templates = cls._get_provider_config_templates()
            else:  # model
                templates = cls._get_model_config_templates()
            
            if form_config.provider_type not in templates:
                # No fallbacks - strict contract enforcement
                available_templates = list(templates.keys())
                raise ValueError(
                    f"Configuration template not found for provider type '{form_config.provider_type}'. "
                    f"Available templates: {', '.join(available_templates)}"
                )
            template = templates[form_config.provider_type]
            
            # Generate class name from config name
            class_name = cls._generate_class_name(form_config.config_name)
            
            # Generate safe config name for variable names
            config_name_safe = form_config.config_name.replace('-', '_').replace(' ', '_')
            
            # Generate configuration fields based on config type
            if config_type == "provider":
                # Provider configs use infrastructure-level settings
                config_fields = cls._generate_config_fields(form_config.form_data)
                model_config_fields = ""
            else:  # model
                # Model configs use model-specific settings
                model_config_fields = cls._generate_model_config_fields(form_config.form_data)
                config_fields = ""
            
            # Format template with all possible variables
            format_vars = {
                'config_name': form_config.config_name,
                'class_name': class_name,
                'config_fields': config_fields,
                'model_config_fields': model_config_fields,
                'provider_specific_type': form_config.provider_specific_type,
                'config_name_safe': config_name_safe,
            }
            
            try:
                generated_code = template.format(**format_vars)
            except KeyError as e:
                logger.error(f"Template formatting failed, missing variable: {e}")
                raise ValueError(f"Template error: missing variable {e}")
            
            logger.info(f"Generated {config_type} configuration code for {form_config.config_name}")
            return generated_code
            
        except Exception as e:
            logger.error(f"Failed to generate configuration code: {e}")
            raise
    
    @classmethod
    def _generate_class_name(cls, config_name: str) -> str:
        """Generate valid Python class name from config name."""
        import re
        import keyword
        
        # Remove invalid characters and convert to valid identifier
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', config_name)
        parts = clean_name.split('_')
        class_name = ''.join(word.capitalize() for word in parts if word and word.isalnum())
        
        # Ensure it starts with a letter
        if not class_name or not class_name[0].isalpha():
            class_name = f"Config{class_name}"
        
        # Handle Python keywords
        if keyword.iskeyword(class_name.lower()):
            class_name = f"{class_name}Config"
        
        # Fallback if empty
        if not class_name:
            class_name = "GeneratedConfig"
        
        return class_name
    
    @classmethod
    def _generate_model_config_fields(cls, form_data: Dict[str, Any]) -> str:
        """Generate model configuration fields for @model_config decorator.
        
        Dynamically processes all fields in form_data instead of using hardcoded list.
        Follows CLAUDE.md single source of truth principle.
        """
        fields = []
        
        # Process all fields from form_data - no hardcoded field list
        # Skip internal form metadata fields  
        for field_name, value in form_data.items():
            if field_name.startswith('_') or value is None:
                continue
                
            # Skip provider-level fields that don't belong in model config
            # Get these dynamically from ProviderSettings base class
            if cls._is_provider_level_field(field_name):
                continue
            
            # Generate field entry based on value type
            if isinstance(value, str):
                fields.append(f'    "{field_name}": "{value}"')
            elif isinstance(value, bool):
                fields.append(f'    "{field_name}": {str(value)}')
            elif isinstance(value, (int, float)):
                fields.append(f'    "{field_name}": {value}')
            elif isinstance(value, (list, dict)):
                # Use Python repr for complex types
                fields.append(f'    "{field_name}": {repr(value)}')
        
        return ',\n'.join(fields)
    
    @classmethod
    def _is_provider_level_field(cls, field_name: str) -> bool:
        """Check if field is a provider-level infrastructure setting.
        
        Dynamically discovers from ProviderSettings base class instead of hardcoding.
        """
        try:
            from flowlib.providers.core.base import ProviderSettings
            
            # Get field names from ProviderSettings base class
            provider_fields = set(ProviderSettings.model_fields.keys())
            return field_name in provider_fields
            
        except Exception:
            # Fallback to minimal known provider fields if import fails
            known_provider_fields = {'timeout', 'max_retries', 'verbose', 'retry_delay_seconds', 'custom_settings'}
            return field_name in known_provider_fields
    
    @classmethod
    def _generate_config_fields(cls, form_data: Dict[str, Any]) -> str:
        """Generate configuration field parameters for __init__ method from form data."""
        fields = []
        
        for field_name, value in form_data.items():
            if value is None:
                continue
                
            # Skip internal form fields
            if field_name.startswith('_'):
                continue
            
            # Generate parameter for __init__ method
            param_def = cls._generate_init_parameter(field_name, value)
            if param_def:
                fields.append(param_def)
        
        # Indent all fields consistently for __init__ method
        indented_fields = [f"            {field}" for field in fields]
        return '\n'.join(indented_fields)
    
    @classmethod
    def _generate_init_parameter(cls, field_name: str, value: Any) -> Optional[str]:
        """Generate a parameter assignment for __init__ method."""
        try:
            if isinstance(value, str):
                # Escape quotes in strings
                escaped_value = value.replace('"', '\\"')
                return f'{field_name}="{escaped_value}",'
            elif isinstance(value, bool):
                return f'{field_name}={str(value)},'
            elif isinstance(value, (int, float)):
                return f'{field_name}={value},'
            elif isinstance(value, list):
                # Convert list to string representation
                return f'{field_name}={repr(value)},'
            elif isinstance(value, dict):
                # Convert dict to string representation
                return f'{field_name}={repr(value)},'
            else:
                # Default string representation
                return f'{field_name}="{str(value)}",'
        except Exception as e:
            logger.warning(f"Failed to generate init parameter for {field_name}: {e}")
            return None
    
    @classmethod  
    def _generate_field_definition(cls, field_name: str, value: Any) -> Optional[str]:
        """Generate a single field definition."""
        try:
            if isinstance(value, str):
                # Proper string escaping for Python code generation
                escaped_value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                return f'{field_name}: str = "{escaped_value}"'
            elif isinstance(value, bool):
                return f'{field_name}: bool = {str(value)}'
            elif isinstance(value, int):
                return f'{field_name}: int = {value}'
            elif isinstance(value, float):
                return f'{field_name}: float = {value}'
            elif isinstance(value, list):
                # Handle list values
                if all(isinstance(item, str) for item in value):
                    # Proper escaping for string items in lists
                    escaped_items = [item.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') for item in value]
                    list_repr = '[' + ', '.join(f'"{item}"' for item in escaped_items) + ']'
                    return f'{field_name}: List[str] = {list_repr}'
                else:
                    list_repr = repr(value)  # Use repr for proper escaping
                    return f'{field_name}: List = {list_repr}'
            elif isinstance(value, dict):
                # Handle dictionary values
                dict_repr = str(value).replace("'", '"')
                return f'{field_name}: Dict[str, Any] = {dict_repr}'
            else:
                # Fallback for other types
                return f'{field_name}: Any = {repr(value)}'
                
        except Exception as e:
            logger.warning(f"Failed to generate field definition for {field_name}: {e}")
            return None


class FormConfigurationController(ConfigurationController):
    """
    Enhanced configuration controller with form-based creation support.
    
    Extends the base ConfigurationController to provide form-based UI for
    configuration creation while maintaining backward compatibility.
    """
    
    # Additional signals for form-based operations
    form_dialog_requested = Signal(str)  # provider_type
    form_configuration_created = Signal(str, str)  # config_name, provider_type
    
    def show_create_configuration_form(self, provider_type: str = None, parent: QWidget = None) -> None:
        """
        Show form-based configuration creation dialog.
        
        Args:
            provider_type: Specific provider type to configure, or None to show selector
            parent: Parent widget for the dialog
        """
        try:
            if provider_type:
                # Show form for specific provider type
                self._show_provider_configuration_form(provider_type, parent)
            else:
                # Show provider selection wizard
                self._show_provider_selection_wizard(parent)
                
        except Exception as e:
            logger.error(f"Failed to show configuration form: {e}")
            if parent:
                QMessageBox.critical(
                    parent,
                    "Form Error",
                    f"Failed to show configuration form:\n\n{str(e)}"
                )
    
    def _show_provider_configuration_form(self, provider_type: str, parent: QWidget = None) -> None:
        """Show configuration form for specific provider type."""
        try:
            # Create provider-specific form dialog
            dialog = create_provider_form_dialog(
                provider_type=provider_type,
                parent=parent
            )
            
            if not dialog:
                return
            
            # Connect form completion signal
            dialog.configuration_saved.connect(self._on_form_configuration_saved)
            
            # Show dialog
            if dialog.exec() == dialog.DialogCode.Accepted:
                logger.info(f"Configuration form completed for {provider_type}")
            
        except Exception as e:
            logger.error(f"Failed to show provider configuration form: {e}")
            if parent:
                QMessageBox.critical(
                    parent,
                    "Configuration Form Error", 
                    f"Failed to show configuration form for {provider_type}:\n\n{str(e)}"
                )
    
    def _show_provider_selection_wizard(self, parent: QWidget = None) -> None:
        """Show provider selection and configuration wizard."""
        try:
            # Show provider form wizard
            result = show_provider_form_wizard(parent)
            
            if result:
                # Process wizard result
                config_name = result['config_name']
                provider_type = result['provider_type']
                config_data = result['config_data']
                
                # Convert to form configuration data
                if '_config_type' not in config_data:
                    raise ValueError(f"Configuration {config_name} missing required '_config_type' field")
                config_type = config_data['_config_type']
                
                if 'provider_type' not in config_data:
                    raise ValueError(f"Configuration {config_name} missing required 'provider_type' field")
                provider_specific_type = config_data['provider_type']
                
                form_config = FormConfigurationData(
                    config_name=config_name,
                    provider_type=provider_type,
                    provider_specific_type=provider_specific_type,
                    config_type=config_type,
                    form_data=config_data,
                    description=f"Generated {provider_type} {config_type} configuration"
                )
                
                # Create configuration from form data
                self._create_configuration_from_form(form_config)
                
        except Exception as e:
            logger.error(f"Provider selection wizard failed: {e}")
            if parent:
                QMessageBox.critical(
                    parent,
                    "Configuration Wizard Error",
                    f"The configuration wizard failed:\n\n{str(e)}"
                )
    
    def _on_form_configuration_saved(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """Handle form configuration save event."""
        try:
            # Determine provider type from the dialog
            sender = self.sender()
            if hasattr(sender, 'provider_type'):
                provider_type = sender.provider_type.lower().replace(' ', '_')
            else:
                # Must determine from config data - no fallbacks allowed
                if 'provider_type' not in config_data:
                    raise ValueError(f"Cannot determine provider type for configuration {config_name} - missing provider_type field")
                provider_type = config_data['provider_type']
            
            # Convert to form configuration data
            if '_config_type' not in config_data:
                raise ValueError(f"Configuration {config_name} missing required '_config_type' field")
            config_type = config_data['_config_type']
            
            if 'provider_type' not in config_data:
                raise ValueError(f"Configuration {config_name} missing required 'provider_type' field")
            provider_specific_type = config_data['provider_type']
            
            form_config = FormConfigurationData(
                config_name=config_name,
                provider_type=provider_type,
                provider_specific_type=provider_specific_type,
                config_type=config_type,
                form_data=config_data,
                description=f"Form-generated {provider_type} {config_type} configuration"
            )
            
            # Create configuration from form data
            self._create_configuration_from_form(form_config)
            
        except Exception as e:
            logger.error(f"Failed to process form configuration save: {e}")
    
    def _create_configuration_from_form(self, form_config: FormConfigurationData) -> None:
        """Create configuration from validated form data."""
        try:
            # Generate configuration code from form data with config type
            generated_code = ConfigurationCodeGenerator.generate_configuration_code(
                form_config, form_config.config_type
            )
            
            # Get configuration type dynamically from provider registry
            config_type = cls._get_configuration_type_for_provider(form_config.provider_type)
            
            # Create configuration data for the service
            config_data = ConfigurationCreateData(
                name=form_config.config_name,
                content=generated_code,
                type=config_type,
                description=form_config.description
            )
            
            # Use parent controller's create method
            super().create_configuration(config_data.model_dump())
            
            # Emit form-specific signal
            self.form_configuration_created.emit(form_config.config_name, form_config.provider_type)
            
            logger.info(f"Created configuration from form: {form_config.config_name}")
            
        except Exception as e:
            logger.error(f"Failed to create configuration from form: {e}")
            raise
    
    @classmethod
    def _get_configuration_type_for_provider(cls, provider_type: str) -> 'ConfigurationType':
        """Get ConfigurationType dynamically from provider capabilities.
        
        No hardcoded mappings - discovers from provider registry metadata.
        """
        from flowlib.gui.logic.settings_discovery import SettingsDiscovery
        from ..services.models import ConfigurationType
        
        # Discover from provider registry
        discovery = SettingsDiscovery()
        available_types = discovery.get_available_provider_types()
        
        # Find the provider category dynamically  
        for category, providers in available_types.items():
            if provider_type in [category] or any(provider_type in [prov] for prov in providers):
                # Map category to ConfigurationType using enum names
                category_mapping = {
                    'llm': ConfigurationType.LLM,
                    'vector_db': ConfigurationType.VECTOR, 
                    'database': ConfigurationType.DATABASE,
                    'db': ConfigurationType.DATABASE,
                    'cache': ConfigurationType.CACHE,
                    'storage': ConfigurationType.STORAGE,
                    'embedding': ConfigurationType.EMBEDDING,
                    'graph_db': ConfigurationType.GRAPH,
                    'message_queue': ConfigurationType.MESSAGE_QUEUE,
                    'mcp_client': ConfigurationType.MCP,
                    'mcp_server': ConfigurationType.MCP,
                }
                
                if category in category_mapping:
                    return category_mapping[category]
        
        # If not found, fail fast - no default fallbacks
        raise ValueError(
            f"Provider type '{provider_type}' not found in registry or no mapping to ConfigurationType exists. "
            f"Available provider categories: {list(available_types.keys())}"
        )
    
    def edit_configuration_with_form(self, config_name: str, parent: QWidget = None) -> None:
        """
        Edit existing configuration using form-based UI.
        
        Args:
            config_name: Name of configuration to edit
            parent: Parent widget for dialogs
        """
        try:
            # This would need to:
            # 1. Load existing configuration 
            # 2. Parse it to extract form data
            # 3. Show form dialog with existing data loaded
            # 4. Save updated configuration
            
            # For now, show message that this is not yet implemented
            if parent:
                QMessageBox.information(
                    parent,
                    "Feature Not Yet Available",
                    f"Form-based editing of existing configuration '{config_name}' "
                    "is not yet implemented.\n\n"
                    "Please use the text editor for now."
                )
            
            logger.info(f"Form-based editing requested for {config_name} (not yet implemented)")
            
        except Exception as e:
            logger.error(f"Failed to edit configuration with form: {e}")
            if parent:
                QMessageBox.critical(
                    parent,
                    "Edit Error",
                    f"Failed to edit configuration with form:\n\n{str(e)}"
                )
    
    @classmethod
    def generate_configuration_code(cls, form_config: FormConfigurationData, config_type: str = "provider") -> str:
        """
        Generate Python configuration code from form data.
        
        Delegates to ConfigurationCodeGenerator for actual code generation.
        
        Args:
            form_config: Validated form configuration data
            config_type: Type of config to generate - "provider" or "model"
            
        Returns:
            Generated Python configuration code
            
        Raises:
            ValueError: If form_config is invalid or code generation fails
        """
        return ConfigurationCodeGenerator.generate_configuration_code(form_config, config_type)
    
    def validate_all_configurations(self) -> None:
        """Validate all configurations using the form-enhanced approach."""
        # Delegate to base class method which handles the operation properly
        super().validate_all_configurations()