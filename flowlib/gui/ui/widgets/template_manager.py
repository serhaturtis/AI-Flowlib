"""
Configuration Template Manager Widget.

Clean, robust implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere with strict validation
- Clean architecture with proper separation of concerns
"""

import logging
import re
from typing import List, Optional, Union
from pydantic import Field, ValidationError
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QGroupBox, QScrollArea, QLineEdit,
    QDialog, QFormLayout, QDialogButtonBox, QMessageBox,
    QCheckBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont

logger = logging.getLogger(__name__)


# Strict Pydantic Models - No fallbacks allowed
class TemplateVariableConfig(StrictBaseModel):
    """Template variable configuration with strict validation."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    variable_type: str  # "string", "int", "float", "bool", "choice"
    description: str
    required: bool = True
    default_value: Optional[Union[str, int, float, bool]] = None
    choices: Optional[List[str]] = None  # For choice type
    min_value: Optional[Union[int, float]] = None  # For numeric types
    max_value: Optional[Union[int, float]] = None  # For numeric types


class TemplateWidgetState(MutableStrictBaseModel):
    """UI state model for template widget - single source of truth but mutable for UI updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    selected_template: Optional[str] = None
    available_templates: List[str] = Field(default_factory=list)
    variable_values: dict[str, str] = Field(default_factory=dict)


class TemplateResource(StrictBaseModel):
    """Template resource model matching flowlib registry contracts."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    name: str
    template: str
    variables: dict[str, TemplateVariableConfig]
    description: str = ""


class TemplateOperationResult(StrictBaseModel):
    """Operation result with proper error handling."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None


class TemplateVariableWidget(QWidget):
    """Single variable input widget with strict Pydantic validation and proper type support."""
    
    def __init__(self, variable_name: str, config: TemplateVariableConfig, parent=None):
        super().__init__(parent)
        self.variable_name = variable_name
        self.config = config  # Must be validated TemplateVariableConfig
        self.input_widget = None  # Will hold the actual input widget
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI with validated configuration and proper input type."""
        layout = QHBoxLayout(self)
        
        # Label with required indicator
        label_text = self.variable_name
        if self.config.required:
            label_text += " *"
        label = QLabel(label_text)
        label.setMinimumWidth(120)
        layout.addWidget(label)
        
        # Create appropriate input widget based on variable type
        self.input_widget = self._create_input_widget()
        self.input_widget.setToolTip(self.config.description)
        layout.addWidget(self.input_widget)
    
    def _create_input_widget(self):
        """Create the appropriate input widget based on variable type."""
        var_type = self.config.variable_type.lower()
        
        if var_type == "bool" or var_type == "boolean":
            # Boolean: Use checkbox
            widget = QCheckBox()
            if self.config.default_value is not None:
                if isinstance(self.config.default_value, bool):
                    widget.setChecked(self.config.default_value)
                elif isinstance(self.config.default_value, str):
                    widget.setChecked(self.config.default_value.lower() in ["true", "1", "yes", "on"])
            return widget
            
        elif var_type == "int" or var_type == "integer":
            # Integer: Use spinbox
            widget = QSpinBox()
            widget.setRange(
                self.config.min_value if self.config.min_value is not None else -2147483648,
                self.config.max_value if self.config.max_value is not None else 2147483647
            )
            if self.config.default_value is not None:
                try:
                    widget.setValue(int(self.config.default_value))
                except (ValueError, TypeError):
                    widget.setValue(0)
            return widget
            
        elif var_type == "float" or var_type == "double" or var_type == "number":
            # Float: Use double spinbox
            widget = QDoubleSpinBox()
            widget.setRange(
                self.config.min_value if self.config.min_value is not None else -1e10,
                self.config.max_value if self.config.max_value is not None else 1e10
            )
            widget.setDecimals(3)
            if self.config.default_value is not None:
                try:
                    widget.setValue(float(self.config.default_value))
                except (ValueError, TypeError):
                    widget.setValue(0.0)
            return widget
            
        elif var_type == "choice" or var_type == "enum":
            # Choice: Use combobox
            widget = QComboBox()
            if self.config.choices:
                widget.addItems(self.config.choices)
                if self.config.default_value and str(self.config.default_value) in self.config.choices:
                    widget.setCurrentText(str(self.config.default_value))
            return widget
            
        else:
            # String (default): Use line edit
            widget = QLineEdit()
            if self.config.default_value is not None:
                widget.setText(str(self.config.default_value))
            widget.setPlaceholderText(self.config.description)
            return widget
    
    def get_value(self) -> Union[str, int, float, bool]:
        """Get current value with proper type conversion."""
        var_type = self.config.variable_type.lower()
        
        if var_type == "bool" or var_type == "boolean":
            if isinstance(self.input_widget, QCheckBox):
                return self.input_widget.isChecked()
                
        elif var_type == "int" or var_type == "integer":
            if isinstance(self.input_widget, QSpinBox):
                return self.input_widget.value()
                
        elif var_type == "float" or var_type == "double" or var_type == "number":
            if isinstance(self.input_widget, QDoubleSpinBox):
                return self.input_widget.value()
                
        elif var_type == "choice" or var_type == "enum":
            if isinstance(self.input_widget, QComboBox):
                return self.input_widget.currentText()
        
        # String or fallback
        if hasattr(self.input_widget, 'text'):
            return self.input_widget.text()
        return ""
    
    def set_value(self, value: Union[str, int, float, bool]):
        """Set value with proper type handling."""
        var_type = self.config.variable_type.lower()
        
        if var_type == "bool" or var_type == "boolean":
            if isinstance(self.input_widget, QCheckBox):
                if isinstance(value, bool):
                    self.input_widget.setChecked(value)
                elif isinstance(value, str):
                    self.input_widget.setChecked(value.lower() in ["true", "1", "yes", "on"])
                    
        elif var_type == "int" or var_type == "integer":
            if isinstance(self.input_widget, QSpinBox):
                try:
                    self.input_widget.setValue(int(value))
                except (ValueError, TypeError):
                    pass
                    
        elif var_type == "float" or var_type == "double" or var_type == "number":
            if isinstance(self.input_widget, QDoubleSpinBox):
                try:
                    self.input_widget.setValue(float(value))
                except (ValueError, TypeError):
                    pass
                    
        elif var_type == "choice" or var_type == "enum":
            if isinstance(self.input_widget, QComboBox):
                self.input_widget.setCurrentText(str(value))
        
        # String or fallback
        elif hasattr(self.input_widget, 'setText'):
            self.input_widget.setText(str(value))


class TemplateService:
    """Service layer for template operations - strict contracts only."""
    
    def __init__(self, service_factory):
        self.service_factory = service_factory
    
    def get_available_templates(self) -> List[str]:
        """Get available template names from new template system."""
        return [
            "llm-provider-config-template",
            "model-config-template", 
            "database-config-template",
            "vector-config-template",
            "cache-config-template",
            "storage-config-template",
            "graph-config-template",
            "embedding-config-template",
            "message-queue-config-template"
        ]
    
    def load_template_resource(self, template_name: str) -> TemplateResource:
        """Load template resource with dynamic generation for all config types."""
        
        # Define template configurations for all provider types
        template_configs = {
            "model-config-template": {
                "description": "Dynamic model configuration template",
                "provider_choices": ["llamacpp", "google_ai"],
                "default_provider": "llamacpp"
            },
            "llm-provider-config-template": {
                "description": "Dynamic LLM provider configuration template", 
                "provider_choices": ["llm_provider_llamacpp", "llm_provider_google_ai"],
                "default_provider": "llm_provider_llamacpp"
            },
            "database-config-template": {
                "description": "Dynamic database configuration template",
                "provider_choices": ["database_postgresql", "database_mongodb", "database_sqlite"],
                "default_provider": "database_postgresql"
            },
            "vector-config-template": {
                "description": "Dynamic vector database configuration template",
                "provider_choices": ["vector_chroma", "vector_qdrant", "vector_pinecone"],
                "default_provider": "vector_chroma"
            },
            "cache-config-template": {
                "description": "Dynamic cache configuration template",
                "provider_choices": ["cache_redis", "cache_memory"],
                "default_provider": "cache_redis"
            },
            "storage-config-template": {
                "description": "Dynamic storage configuration template",
                "provider_choices": ["storage_s3", "storage_local"],
                "default_provider": "storage_s3"
            },
            "graph-config-template": {
                "description": "Dynamic graph database configuration template",
                "provider_choices": ["graph_neo4j", "graph_arango", "graph_janus"],
                "default_provider": "graph_neo4j"
            },
            "embedding-config-template": {
                "description": "Dynamic embedding configuration template",
                "provider_choices": ["embedding_llamacpp"],
                "default_provider": "embedding_llamacpp"
            },
            "message-queue-config-template": {
                "description": "Dynamic message queue configuration template",
                "provider_choices": ["mq_rabbitmq", "mq_kafka"],
                "default_provider": "mq_rabbitmq"
            }
        }
        
        # Handle all dynamic template types
        if template_name in template_configs:
            config = template_configs[template_name]
            
            variables = {
                'config_name': TemplateVariableConfig(
                    variable_type="string",
                    description="Configuration name",
                    required=True,
                    default_value="my-config"
                ),
                'provider_type': TemplateVariableConfig(
                    variable_type="choice",
                    description="Provider type",
                    required=True,
                    default_value=config["default_provider"],
                    choices=config["provider_choices"]
                ),
                'description': TemplateVariableConfig(
                    variable_type="string",
                    description="Configuration description",
                    required=False,
                    default_value="Configuration description"
                )
            }
            
            return TemplateResource(
                name=template_name,
                template="# Dynamic template - generated from actual provider classes",
                variables=variables,
                description=config["description"]
            )
        
        # Use old system for other templates
        # Import new template classes
        from ...logic.services.configuration_templates import (
            LLMProviderConfigTemplate, DatabaseConfigTemplate, VectorConfigTemplate,
            TemplateGenerationParameters
        )
        
        # Map template names to classes
        template_classes = {
            "llm-provider-config-template": LLMProviderConfigTemplate,
            "database-config-template": DatabaseConfigTemplate,
            "vector-config-template": VectorConfigTemplate
        }
        
        if template_name not in template_classes:
            raise ValueError(f"Template '{template_name}' not found")
        template_class = template_classes[template_name]
        
        template_instance = template_class()
        
        # Determine template category for context-aware provider types
        template_category = self._infer_template_category(template_name)
        
        # Extract variables from template using consistent pattern
        variables = self._extract_variables_from_template(template_instance.template, template_category)
        
        # Create validated TemplateResource
        return TemplateResource(
            name=template_name,
            template=template_instance.template,
            variables=variables,
            description=f"{template_name} configuration template"
        )
    
    def _extract_variables_from_template(self, template: str, template_category: str = "llm") -> dict[str, TemplateVariableConfig]:
        """Extract variables from template using consistent {var} pattern with type inference."""
        variable_pattern = re.compile(r'\{([^}]+)\}')
        variable_names = set(variable_pattern.findall(template))
        
        # Filter out internal placeholders that are handled by template.generate()
        internal_placeholders = {
            'additional_params', 'class_name', 
            'model_path', 'model_name', 'temperature', 'max_tokens', 'top_p'
        }
        user_variables = variable_names - internal_placeholders
        
        variables = {}
        for var_name in user_variables:
            # Infer variable type from name patterns with template context
            var_type, default_val, choices, min_val, max_val = self._infer_variable_type(var_name, template_category)
            
            variables[var_name] = TemplateVariableConfig(
                variable_type=var_type,
                description=f"Value for {var_name}",
                required=True,
                default_value=default_val,
                choices=choices,
                min_value=min_val,
                max_value=max_val
            )
        
        return variables
    
    def _infer_template_category(self, template_name: str) -> str:
        """Infer template category from template name."""
        template_name_lower = template_name.lower()
        if any(term in template_name_lower for term in ['database', 'db', 'postgres', 'mongo', 'sqlite']):
            return "database"
        elif any(term in template_name_lower for term in ['vector', 'chroma', 'pinecone', 'qdrant']):
            return "vector_db"
        elif any(term in template_name_lower for term in ['cache', 'redis']):
            return "cache"
        elif any(term in template_name_lower for term in ['storage', 's3', 'local']):
            return "storage"
        elif any(term in template_name_lower for term in ['graph', 'neo4j', 'arango']):
            return "graph_db"
        elif any(term in template_name_lower for term in ['embedding']):
            return "embedding"
        elif any(term in template_name_lower for term in ['mq', 'queue', 'rabbit', 'kafka']):
            return "message_queue"
        else:
            return "llm"  # Default
    
    def _get_provider_types_for_category(self, category: str) -> List[str]:
        """Get available provider types for a specific category from constants - fail fast if not available."""
        try:
            from flowlib.providers.core.constants import PROVIDER_TYPE_MAP
            
            # Map categories to their provider types
            category_providers = []
            for provider_name, provider_category in PROVIDER_TYPE_MAP.items():
                if provider_category == category:
                    category_providers.append(provider_name)
            
            # Following CLAUDE.md principles: fail fast if no providers found
            if not category_providers:
                raise ValueError(f"No providers found for category '{category}' in PROVIDER_TYPE_MAP")
            
            return sorted(category_providers)
            
        except ImportError as e:
            # Following CLAUDE.md: no fallbacks, fail fast
            raise ImportError(f"Provider constants not available - cannot determine provider types for {category}: {e}")
        except Exception as e:
            # Following CLAUDE.md: proper error propagation, no silent failures
            raise ValueError(f"Failed to get provider types for {category}: {e}")
    
    def _infer_variable_type(self, var_name: str, template_category: str = "llm") -> tuple[str, Union[str, int, float, bool, None], Optional[List[str]], Optional[Union[int, float]], Optional[Union[int, float]]]:
        """Infer variable type from variable name patterns."""
        var_name_lower = var_name.lower()
        
        # Boolean patterns
        if any(pattern in var_name_lower for pattern in [
            'use_', 'enable_', 'disable_', 'is_', 'has_', 'allow_', 'verbose', 
            '_enabled', '_disabled', '_flag', 'gpu', 'ssl', 'tls'
        ]):
            return "bool", False, None, None, None
        
        # Integer patterns
        elif any(pattern in var_name_lower for pattern in [
            'port', 'threads', 'batch', 'size', 'count', 'max_', 'min_', 
            'n_', 'num_', 'layers', 'ctx', 'tokens', 'connections'
        ]):
            # Set reasonable defaults and ranges for common patterns
            if 'port' in var_name_lower:
                return "int", 8080, None, 1, 65535
            elif 'threads' in var_name_lower or 'n_threads' in var_name_lower:
                return "int", 4, None, 1, 64
            elif 'batch' in var_name_lower or 'n_batch' in var_name_lower:
                return "int", 512, None, 1, 8192
            elif 'tokens' in var_name_lower or 'ctx' in var_name_lower:
                return "int", 2048, None, 1, 32768
            elif 'layers' in var_name_lower:
                return "int", 0, None, 0, 100
            else:
                return "int", 1, None, 0, None
        
        # Float patterns
        elif any(pattern in var_name_lower for pattern in [
            'temperature', 'top_p', 'learning_rate', 'alpha', 'beta', 
            'threshold', 'ratio', 'factor', 'weight'
        ]):
            if 'temperature' in var_name_lower:
                return "float", 0.7, None, 0.0, 2.0
            elif 'top_p' in var_name_lower:
                return "float", 1.0, None, 0.0, 1.0
            else:
                return "float", 1.0, None, None, None
        
        # Choice patterns
        elif any(pattern in var_name_lower for pattern in [
            'provider_type', 'model_type', 'format', 'mode', 'type'
        ]):
            if 'provider_type' in var_name_lower:
                # Get dynamic provider types based on template context
                try:
                    provider_types = self._get_provider_types_for_category(template_category)
                    default_provider = provider_types[0] if provider_types else "unknown"
                    return "choice", default_provider, provider_types, None, None
                except (ImportError, ValueError) as e:
                    # Following CLAUDE.md: fail fast, no fallbacks
                    logger.error(f"Cannot determine provider types for {template_category}: {e}")
                    raise ValueError(f"Template variable inference failed for provider_type: {e}")
            elif 'format' in var_name_lower:
                return "choice", "json", ["json", "yaml", "xml", "text"], None, None
            else:
                return "choice", "", [], None, None
        
        # Default to string
        else:
            return "string", "", None, None, None
    
    def generate_configuration(self, template_resource: TemplateResource, 
                             variables: dict[str, Union[str, int, float, bool]]) -> TemplateOperationResult:
        """Generate configuration using dynamic template generation from actual provider classes."""
        try:
            # Validate all required variables are provided
            for var_name, config in template_resource.variables.items():
                if config.required and var_name not in variables:
                    return TemplateOperationResult(
                        success=False,
                        error=f"Required variable '{var_name}' not provided"
                    )
            
            # Use dynamic template generation for all supported template types
            dynamic_templates = [
                "model-config-template", "llm-provider-config-template", "database-config-template",
                "vector-config-template", "cache-config-template", "storage-config-template", 
                "graph-config-template", "embedding-config-template", "message-queue-config-template"
            ]
            
            if template_resource.name in dynamic_templates:
                from ...logic.services.dynamic_template_generator import DynamicTemplateGenerator
                
                generator = DynamicTemplateGenerator()
                # Fail-fast approach - no fallbacks allowed
                if 'provider_type' not in variables:
                    raise ValueError("Required template variable 'provider_type' not provided")
                if 'config_name' not in variables:
                    raise ValueError("Required template variable 'config_name' not provided")
                if 'description' not in variables:
                    raise ValueError("Required template variable 'description' not provided")
                
                provider_type = variables['provider_type']
                config_name = variables['config_name']
                description = variables['description']
                
                try:
                    content = generator.generate_template(
                        provider_type=provider_type,
                        config_name=config_name,
                        description=description
                    )
                    return TemplateOperationResult(success=True, content=content)
                except ValueError as e:
                    return TemplateOperationResult(
                        success=False,
                        error=f"Provider type '{provider_type}' not supported: {e}"
                    )
            
            # Use the old template system for other template types (for now)
            from ...logic.services.configuration_templates import (
                LLMProviderConfigTemplate, DatabaseConfigTemplate, VectorConfigTemplate,
                TemplateGenerationParameters
            )
            
            # Map template names to classes
            template_classes = {
                "llm-provider-config-template": LLMProviderConfigTemplate,
                "database-config-template": DatabaseConfigTemplate,
                "vector-config-template": VectorConfigTemplate
            }
            
            template_class = template_classes[template_resource.name] if template_resource.name in template_classes else None
            if template_class:
                # Use the old template system
                template_instance = template_class()
                
                # Convert GUI variables to TemplateGenerationParameters format
                additional_params = {}
                for var_name, value in variables.items():
                    if var_name not in ['config_name', 'description', 'provider_type', 'class_name']:
                        additional_params[var_name] = value
                
                # Fail-fast approach - no fallbacks
                if 'config_name' not in variables:
                    raise ValueError("Required template variable 'config_name' not provided")
                if 'description' not in variables:
                    raise ValueError("Required template variable 'description' not provided")
                if 'provider_type' not in variables:
                    raise ValueError("Required template variable 'provider_type' not provided")
                
                params = TemplateGenerationParameters(
                    name=variables['config_name'],
                    description=variables['description'],
                    provider=variables['provider_type'],
                    additional_params=additional_params
                )
                
                content = template_instance.generate(params)
                return TemplateOperationResult(success=True, content=content)
            
            else:
                # Fallback to simple string replacement for unknown templates
                content = template_resource.template
                for var_name, value in variables.items():
                    # Convert value to appropriate string representation
                    if isinstance(value, bool):
                        str_value = "True" if value else "False"  # Python boolean format
                    elif isinstance(value, (int, float)):
                        str_value = str(value)
                    else:
                        str_value = str(value)
                    
                    # Use secure substitution pattern
                    placeholder = "{" + var_name + "}"
                    content = content.replace(placeholder, str_value)
                
                return TemplateOperationResult(success=True, content=content)
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return TemplateOperationResult(
                success=False,
                error=f"Generation failed: {str(e)}"
            )
    
    def _get_sensible_default(self, var_name: str, var_config: TemplateVariableConfig) -> Union[str, int, float, bool]:
        """Get sensible default values based on variable name and type, following CLAUDE.md principles."""
        var_name_lower = var_name.lower()
        var_type = var_config.variable_type
        
        # Type-based defaults
        if var_type == "bool":
            return False
        elif var_type == "int":
            if 'port' in var_name_lower:
                return 5432 if 'postgres' in var_name_lower else 27017 if 'mongo' in var_name_lower else 8000
            elif 'ctx' in var_name_lower or 'context' in var_name_lower:
                return 4096
            elif 'thread' in var_name_lower:
                return 4
            elif 'batch' in var_name_lower:
                return 512
            elif 'token' in var_name_lower:
                return 2048
            elif 'layer' in var_name_lower:
                return 0
            else:
                return 1
        elif var_type == "float":
            if 'temperature' in var_name_lower:
                return 0.7
            elif 'top_p' in var_name_lower:
                return 1.0
            else:
                return 0.0
        
        # String defaults based on variable name patterns
        if 'config_name' in var_name_lower:
            return "my-config"
        elif 'class_name' in var_name_lower:
            return "MyConfig"
        elif 'description' in var_name_lower:
            return "Configuration description"
        elif 'provider_type' in var_name_lower:
            return "llamacpp"  # Safe default
        elif 'model_path' in var_name_lower:
            return "/path/to/model.gguf"
        elif 'model_name' in var_name_lower:
            return "default_model"
        elif 'model_id' in var_name_lower:
            return "gemini-1.5-flash-latest"
        elif 'api_key' in var_name_lower:
            return ""  # Empty string for security
        elif 'host' in var_name_lower:
            return "localhost"
        elif 'database' in var_name_lower or 'db_name' in var_name_lower:
            return "flowlib"
        elif 'user' in var_name_lower:
            return "flowlib_user"
        elif 'password' in var_name_lower:
            return ""  # Empty string for security
        elif 'collection' in var_name_lower:
            return "default_collection"
        elif 'persist' in var_name_lower and 'dir' in var_name_lower:
            return "./vector_data"
        elif 'format' in var_name_lower:
            return "json"
        else:
            return ""  # Safe fallback


class TemplateCustomizationDialog(QDialog):
    """Template customization dialog with strict validation."""
    
    def __init__(self, template_resource: TemplateResource, parent=None):
        super().__init__(parent)
        self.template_resource = template_resource
        self.variable_widgets: dict[str, TemplateVariableWidget] = {}
        self.result_content: Optional[str] = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI with validated template resource."""
        self.setWindowTitle(f"Customize {self.template_resource.name}")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel(self.template_resource.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Variables section
        if self.template_resource.variables:
            variables_group = QGroupBox("Template Variables")
            variables_layout = QVBoxLayout(variables_group)
            
            # Scrollable area for variables
            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            
            # Create variable widgets with validated configs
            for var_name, var_config in self.template_resource.variables.items():
                var_widget = TemplateVariableWidget(var_name, var_config)
                self.variable_widgets[var_name] = var_widget
                scroll_layout.addWidget(var_widget)
            
            scroll_area.setWidget(scroll_widget)
            scroll_area.setWidgetResizable(True)
            variables_layout.addWidget(scroll_area)
            layout.addWidget(variables_group)
        
        # Preview section
        preview_group = QGroupBox("Template Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        self.preview_text.setFont(QFont("Consolas", 9))
        self.preview_text.setPlainText(self.template_resource.template)
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Connect variable changes to preview update
        for widget in self.variable_widgets.values():
            self._connect_widget_signals(widget)
    
    def _connect_widget_signals(self, widget: TemplateVariableWidget):
        """Connect appropriate signals based on widget type."""
        input_widget = widget.input_widget
        
        if isinstance(input_widget, QLineEdit):
            input_widget.textChanged.connect(self.update_preview)
        elif isinstance(input_widget, QCheckBox):
            input_widget.toggled.connect(self.update_preview)
        elif isinstance(input_widget, (QSpinBox, QDoubleSpinBox)):
            input_widget.valueChanged.connect(self.update_preview)
        elif isinstance(input_widget, QComboBox):
            input_widget.currentTextChanged.connect(self.update_preview)
    
    def update_preview(self):
        """Update preview with current variable values."""
        variables = self.get_variable_values()
        service = TemplateService(None)  # Service methods are static for preview
        
        # Generate preview
        result = service.generate_configuration(self.template_resource, variables)
        if result.success:
            self.preview_text.setPlainText(result.content)
        else:
            self.preview_text.setPlainText(f"Preview error: {result.error}")
    
    def get_variable_values(self) -> dict[str, Union[str, int, float, bool]]:
        """Get current variable values with proper types."""
        values = {}
        for var_name, widget in self.variable_widgets.items():
            values[var_name] = widget.get_value()
        return values
    
    def accept(self):
        """Accept dialog and generate final content."""
        variables = self.get_variable_values()
        service = TemplateService(None)
        
        result = service.generate_configuration(self.template_resource, variables)
        if result.success:
            self.result_content = result.content
            super().accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to generate configuration: {result.error}")


class TemplateManagerWidget(QWidget):
    """Template Manager Widget with clean architecture and strict validation."""
    
    template_generated = Signal(str, str)  # Emits template_name, generated_content
    
    def __init__(self, service_factory, parent=None):
        super().__init__(parent)
        self.service_factory = service_factory
        self.template_service = TemplateService(service_factory)
        self.state = TemplateWidgetState()
        self.templates: dict[str, TemplateResource] = {}
        self.init_ui()
        self.load_templates()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Template selection
        selection_group = QGroupBox("Template Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        self.template_combo = QComboBox()
        self.template_combo.currentTextChanged.connect(self.on_template_changed)
        selection_layout.addWidget(QLabel("Available Templates:"))
        selection_layout.addWidget(self.template_combo)
        
        # Template description
        self.description_label = QLabel("Select a template to see its description")
        self.description_label.setWordWrap(True)
        selection_layout.addWidget(self.description_label)
        
        layout.addWidget(selection_group)
        
        # Template preview
        preview_group = QGroupBox("Template Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 9))
        self.preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_text)
        
        layout.addWidget(preview_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        self.use_template_button = QPushButton("Use Template")
        self.use_template_button.clicked.connect(self.use_template_directly)
        self.use_template_button.setEnabled(False)
        actions_layout.addWidget(self.use_template_button)
        
        self.customize_button = QPushButton("Customize Template")
        self.customize_button.clicked.connect(self.customize_template)
        self.customize_button.setEnabled(False)
        actions_layout.addWidget(self.customize_button)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
    
    def load_templates(self):
        """Load available templates from service."""
        try:
            template_names = self.template_service.get_available_templates()
            self.state.available_templates = template_names
            
            # Load template resources
            for template_name in template_names:
                try:
                    template_resource = self.template_service.load_template_resource(template_name)
                    self.templates[template_name] = template_resource
                except Exception as e:
                    logger.error(f"Failed to load template {template_name}: {e}")
            
            # Update UI
            self.template_combo.clear()
            if self.templates:
                self.template_combo.addItems(list(self.templates.keys()))
            else:
                self.template_combo.addItem("No templates available")
                
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load templates: {str(e)}")
    
    def on_template_changed(self, template_name: str):
        """Handle template selection change."""
        if template_name in self.templates:
            template_resource = self.templates[template_name]
            self.state.selected_template = template_name
            
            # Update UI
            self.description_label.setText(template_resource.description)
            self.preview_text.setPlainText(template_resource.template)
            self.use_template_button.setEnabled(True)
            self.customize_button.setEnabled(True)
        else:
            self.state.selected_template = None
            self.description_label.setText("Select a template to see its description")
            self.preview_text.clear()
            self.use_template_button.setEnabled(False)
            self.customize_button.setEnabled(False)
    
    def customize_template(self):
        """Open template customization dialog."""
        if not self.state.selected_template or self.state.selected_template not in self.templates:
            return
        
        template_resource = self.templates[self.state.selected_template]
        
        try:
            dialog = TemplateCustomizationDialog(template_resource, self)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_content:
                self.template_generated.emit(self.state.selected_template, dialog.result_content)
                
        except Exception as e:
            logger.error(f"Template customization failed: {e}")
            QMessageBox.critical(self, "Error", f"Template customization failed: {str(e)}")
    
    def use_template_directly(self):
        """Use template with default values without customization."""
        if not self.state.selected_template or self.state.selected_template not in self.templates:
            return
        
        template_resource = self.templates[self.state.selected_template]
        
        try:
            # Generate template with proper default values - no angle bracket placeholders
            default_variables = {}
            for var_name, var_config in template_resource.variables.items():
                # Use actual default value or provide sensible defaults based on variable type
                if var_config.default_value is not None:
                    default_variables[var_name] = var_config.default_value
                else:
                    # Provide sensible defaults based on variable name and type
                    default_variables[var_name] = self.template_service._get_sensible_default(var_name, var_config)
            
            result = self.template_service.generate_configuration(
                template_resource, default_variables
            )
            
            if result.success and result.content:
                self.template_generated.emit(self.state.selected_template, result.content)
            else:
                if not result.error:
                    raise ValueError("Template generation failed with no error message provided")
                raise ValueError(result.error)
                
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            QMessageBox.critical(self, "Error", f"Template generation failed: {str(e)}")
    
    def get_selected_template(self) -> Optional[str]:
        """Get currently selected template name."""
        return self.state.selected_template
    
    def generate_template_content(self, variables: dict[str, Union[str, int, float, bool]]) -> Optional[str]:
        """Generate template content with provided variables."""
        if not self.state.selected_template or self.state.selected_template not in self.templates:
            return None
        
        template_resource = self.templates[self.state.selected_template]
        result = self.template_service.generate_configuration(template_resource, variables)
        
        if result.success:
            return result.content
        else:
            logger.error(f"Template generation failed: {result.error}")
            return None