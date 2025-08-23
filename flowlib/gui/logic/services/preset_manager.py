"""
Preset Manager using flowlib resource system.

Replaces stub preset management with real flowlib resource templates
and configuration generation using the resource registry and template system.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import List, Optional, Literal, Union
from datetime import datetime
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from .models import OperationResult
from .error_boundaries import handle_service_errors, ServiceError
from .async_qt_helper import AsyncServiceMixin


class VariableConfig(StrictBaseModel):
    """Pydantic model for preset variable configuration."""
    # Inherits strict configuration from StrictBaseModel
    
    type: Literal['string', 'integer', 'number', 'boolean'] = Field(description="Variable data type")
    description: str = Field(description="Variable description")
    required: bool = Field(default=False, description="Whether variable is required")
    default: Optional[Union[str, int, float, bool]] = Field(default=None, description="Default value")
    min: Optional[float] = Field(default=None, description="Minimum value for numeric types")
    max: Optional[float] = Field(default=None, description="Maximum value for numeric types")
    sensitive: bool = Field(default=False, description="Whether variable contains sensitive data")


class PresetData(StrictBaseModel):
    """Pydantic model for preset data structure."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(description="Preset display name")
    type: str = Field(description="Configuration type (e.g., 'llm_config', 'database_config')")
    description: str = Field(default="", description="Preset description")
    category: str = Field(default="other", description="Preset category")
    template: dict[str, Union[str, int, float, bool]] = Field(description="Configuration template")
    variables: dict[str, VariableConfig] = Field(default_factory=dict, description="Variable definitions")


class PresetSummary(StrictBaseModel):
    """Pydantic model for preset summary information."""
    # Inherits strict configuration from StrictBaseModel
    
    id: str = Field(description="Preset identifier")
    name: str = Field(description="Preset display name")
    type: str = Field(description="Configuration type")
    description: str = Field(description="Preset description")
    category: str = Field(description="Preset category")
    variable_count: int = Field(description="Number of variables")
    has_sensitive_vars: bool = Field(description="Whether preset has sensitive variables")
    created: str = Field(description="Creation timestamp")

logger = logging.getLogger(__name__)


class PresetManager(AsyncServiceMixin):
    """Real preset manager using flowlib resource system."""
    
    def __init__(self, service_factory):
        super().__init__()
        self.service_factory = service_factory
        self._initialized = False
        self._preset_cache = {}
        
    async def initialize(self) -> bool:
        """Initialize the preset manager."""
        try:
            # Ensure service factory registry is initialized
            await self.service_factory._ensure_registry_initialized()
            
            # Load built-in presets
            await self._load_builtin_presets()
            
            self._initialized = True
            logger.info("PresetManager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PresetManager: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the preset manager."""
        self._initialized = False
        self._preset_cache.clear()
    
    async def _generate_dynamic_presets(self) -> dict[str, dict]:
        """Generate presets dynamically from available providers."""
        try:
            from flowlib.providers.core.constants import PROVIDER_TYPE_MAP
            
            # Group providers by category
            providers_by_category = {}
            for provider_name, category in PROVIDER_TYPE_MAP.items():
                if category not in providers_by_category:
                    providers_by_category[category] = []
                providers_by_category[category].append(provider_name)
            
            # Generate basic preset for each category with first available provider
            presets = {}
            
            for category, provider_list in providers_by_category.items():
                if not provider_list:
                    continue
                
                # Use first available provider as default
                default_provider = provider_list[0]
                preset_id = f"basic_{category}"
                
                # Generate preset based on category
                preset_data = self._generate_preset_for_category(category, default_provider, provider_list)
                if preset_data:
                    presets[preset_id] = preset_data
            
            return presets
            
        except ImportError:
            logger.error("Failed to import provider constants")
            raise ServiceError("Provider constants not available")
        except Exception as e:
            logger.error(f"Failed to generate dynamic presets: {e}")
            raise ServiceError(f"Dynamic preset generation failed: {e}")
    
    def _generate_preset_for_category(self, category: str, default_provider: str, available_providers: List[str]) -> Optional[dict]:
        """Generate a preset configuration for a specific provider category."""
        category_configs = {
            "llm": {
                "name": "Basic LLM Configuration",
                "type": "llm_config",
                "description": f"Basic LLM provider configuration using {default_provider}",
                "category": "llm",
                "template": {
                    "provider_type": default_provider
                },
                "variables": {
                    "provider_type": {
                        "type": "string",
                        "description": "LLM provider type",
                        "required": True,
                        "default": default_provider
                    }
                }
            },
            "database": {
                "name": "Basic Database Configuration", 
                "type": "database_config",
                "description": f"Basic database provider configuration using {default_provider}",
                "category": "database",
                "template": {
                    "provider_type": default_provider,
                    "host": "localhost",
                    "database": "flowlib"
                },
                "variables": {
                    "provider_type": {
                        "type": "string",
                        "description": "Database provider type",
                        "required": True,
                        "default": default_provider
                    },
                    "host": {
                        "type": "string",
                        "description": "Database host",
                        "required": True,
                        "default": "localhost"
                    }
                }
            },
            "vector_db": {
                "name": "Basic Vector Database Configuration",
                "type": "vector_config", 
                "description": f"Basic vector database configuration using {default_provider}",
                "category": "vector",
                "template": {
                    "provider_type": default_provider,
                    "collection_name": "default_collection"
                },
                "variables": {
                    "provider_type": {
                        "type": "string",
                        "description": "Vector database provider type",
                        "required": True,
                        "default": default_provider
                    }
                }
            },
            "cache": {
                "name": "Basic Cache Configuration",
                "type": "cache_config",
                "description": f"Basic cache configuration using {default_provider}",
                "category": "cache", 
                "template": {
                    "provider_type": default_provider
                },
                "variables": {
                    "provider_type": {
                        "type": "string",
                        "description": "Cache provider type",
                        "required": True,
                        "default": default_provider
                    }
                }
            },
            "embedding": {
                "name": "Basic Embedding Configuration",
                "type": "embedding_config",
                "description": f"Basic embedding configuration using {default_provider}",
                "category": "embedding",
                "template": {
                    "provider_type": default_provider
                },
                "variables": {
                    "provider_type": {
                        "type": "string",
                        "description": "Embedding provider type", 
                        "required": True,
                        "default": default_provider
                    }
                }
            }
        }
        
        return category_configs[category] if category in category_configs else None
    
    async def _load_builtin_presets(self) -> None:
        """Load built-in presets dynamically from available providers."""
        try:
            # Generate presets dynamically from available providers
            builtin_presets = await self._generate_dynamic_presets()
            
            # Validate and cache presets using Pydantic models
            validated_presets = {}
            for preset_id, preset_raw in builtin_presets.items():
                try:
                    validated_preset = PresetData(**preset_raw)
                    validated_presets[preset_id] = validated_preset
                except Exception as e:
                    logger.error(f"Invalid built-in preset '{preset_id}': {e}")
                    raise ServiceError(f"Invalid built-in preset data: {e}")
            
            self._preset_cache.update(validated_presets)
            logger.info(f"Loaded {len(validated_presets)} validated built-in presets")
            
            # Try to load custom presets from filesystem
            await self._load_custom_presets()
            
        except Exception as e:
            logger.error(f"Failed to load built-in presets: {e}")
            raise ServiceError(f"Failed to load built-in presets: {e}", 
                             context={'operation': '_load_builtin_presets'})    
    async def _load_custom_presets(self) -> None:
        """Load custom presets from flowlib directory."""
        try:
            preset_dir = Path.home() / ".flowlib" / "presets"
            if not preset_dir.exists():
                return
            
            custom_count = 0
            for preset_file in preset_dir.glob("*.yaml"):
                try:
                    with open(preset_file, 'r') as f:
                        preset_raw = yaml.safe_load(f)
                    
                    preset_id = preset_file.stem
                    # Validate custom preset using Pydantic
                    validated_preset = PresetData(**preset_raw)
                    self._preset_cache[preset_id] = validated_preset
                    custom_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load/validate custom preset {preset_file}: {e}")
            
            if custom_count > 0:
                logger.info(f"Loaded {custom_count} custom presets")
                
        except Exception as e:
            logger.warning(f"Failed to load custom presets: {e}")
    
    @handle_service_errors("get_available_presets")
    async def get_available_presets(self) -> OperationResult:
        """Get available configuration presets."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'get_available_presets'})
        
        try:
            presets = []
            for preset_id, preset_data in self._preset_cache.items():
                # Use validated Pydantic model instead of .get() patterns
                preset_summary = PresetSummary(
                    id=preset_id,
                    name=preset_data.name,
                    type=preset_data.type,
                    description=preset_data.description,
                    category=preset_data.category,
                    variable_count=len(preset_data.variables),
                    has_sensitive_vars=any(var.sensitive for var in preset_data.variables.values()),
                    created=datetime.now().strftime("%Y-%m-%d %H:%M")
                )
                presets.append(preset_summary.model_dump())
            
            # Sort by category and name
            presets.sort(key=lambda x: (x['category'], x['name']))
            
            return OperationResult(
                success=True,
                message=f"Found {len(presets)} available presets",
                data={
                    'presets': presets,
                    'categories': list(set(p['category'] for p in presets)),
                    'total_count': len(presets)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get available presets: {e}")
            raise ServiceError(f"Failed to get available presets: {e}", 
                             context={'operation': 'get_available_presets'})
    
    @handle_service_errors("get_preset_details")
    async def get_preset_details(self, preset_id: str) -> OperationResult:
        """Get detailed information about a preset."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'get_preset_details'})
        
        try:
            if preset_id not in self._preset_cache:
                raise ServiceError(f"Preset '{preset_id}' not found", 
                                 context={'preset_id': preset_id})
            
            preset_data = self._preset_cache[preset_id]
            
            return OperationResult(
                success=True,
                message=f"Retrieved details for preset '{preset_id}'",
                data={
                    'preset_id': preset_id,
                    'preset_data': preset_data.model_dump(),
                    'template_fields': list(preset_data.template.keys()),
                    'variable_fields': list(preset_data.variables.keys()),
                    'required_variables': [
                        var_name for var_name, var_config in preset_data.variables.items()
                        if var_config.required
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get preset details for '{preset_id}': {e}")
            raise ServiceError(f"Failed to get preset details: {e}", 
                             context={'preset_id': preset_id})
    
    @handle_service_errors("validate_preset_variables")
    async def validate_preset_variables(self, preset_id: str, variables: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Validate variables for a preset without generating configuration."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'validate_preset_variables'})
        
        try:
            if preset_id not in self._preset_cache:
                raise ServiceError(f"Preset '{preset_id}' not found", 
                                 context={'preset_id': preset_id})
            
            preset_data = self._preset_cache[preset_id]
            
            # Validate required variables using Pydantic model
            missing_required = []
            for var_name, var_config in preset_data.variables.items():
                if var_config.required and var_name not in variables:
                    missing_required.append(var_name)
            
            if missing_required:
                raise ServiceError(f"Missing required variables: {', '.join(missing_required)}", 
                                 context={'preset_id': preset_id, 'missing_variables': missing_required})
            
            # Validate variable types and constraints using Pydantic constraints
            validation_results = {}
            for var_name, var_value in variables.items():
                if var_name in preset_data.variables:
                    var_config = preset_data.variables[var_name]
                    
                    try:
                        # Type validation
                        if var_config.type == 'integer':
                            var_value = int(var_value)
                        elif var_config.type == 'number':
                            var_value = float(var_value)
                        elif var_config.type == 'boolean':
                            var_value = bool(var_value)
                        # string type needs no conversion
                        
                        # Range validation
                        if var_config.min is not None and var_value < var_config.min:
                            raise ServiceError(f"Variable '{var_name}' must be >= {var_config.min}", 
                                             context={'preset_id': preset_id, 'variable': var_name})
                        
                        if var_config.max is not None and var_value > var_config.max:
                            raise ServiceError(f"Variable '{var_name}' must be <= {var_config.max}", 
                                             context={'preset_id': preset_id, 'variable': var_name})
                        
                        validation_results[var_name] = {
                            'valid': True,
                            'converted_value': var_value,
                            'type': var_config.type
                        }
                        
                    except (ValueError, TypeError) as e:
                        validation_results[var_name] = {
                            'valid': False,
                            'error': f"Variable '{var_name}' must be {var_config.type}: {e}",
                            'type': var_config.type
                        }
            
            # Check if all validations passed
            all_valid = all(result['valid'] for result in validation_results.values())
            validation_errors = [result['error'] for result in validation_results.values() if not result['valid']]
            
            return OperationResult(
                success=all_valid,
                message=f"Validation {'passed' if all_valid else 'failed'} for preset '{preset_id}'" + (
                    f": {'; '.join(validation_errors)}" if validation_errors else ""
                ),
                data={
                    'preset_id': preset_id,
                    'validation_results': validation_results,
                    'all_valid': all_valid,
                    'validation_errors': validation_errors,
                    'validated_variables': variables,
                    'validation_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to validate variables for preset '{preset_id}': {e}")
            raise ServiceError(f"Variable validation failed: {e}", 
                             context={'preset_id': preset_id})
    
    @handle_service_errors("generate_from_preset")
    async def generate_from_preset(self, preset_id: str, variables: dict[str, Union[str, int, float, bool]], 
                                 config_name: str) -> OperationResult:
        """Generate configuration from preset template."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'generate_from_preset'})
        
        try:
            if preset_id not in self._preset_cache:
                raise ServiceError(f"Preset '{preset_id}' not found", 
                                 context={'preset_id': preset_id})
            
            preset_data = self._preset_cache[preset_id]
            
            # Validate required variables using Pydantic model
            missing_required = []
            for var_name, var_config in preset_data.variables.items():
                if var_config.required and var_name not in variables:
                    missing_required.append(var_name)
            
            if missing_required:
                raise ServiceError(f"Missing required variables: {', '.join(missing_required)}", 
                                 context={'preset_id': preset_id, 'missing_variables': missing_required})
            
            # Generate configuration from template using validated Pydantic model
            template = preset_data.template.copy()
            
            # Substitute and validate variables using Pydantic constraints
            for var_name, var_value in variables.items():
                if var_name in preset_data.variables:
                    var_config = preset_data.variables[var_name]
                    
                    # Type validation using Pydantic model constraints
                    try:
                        if var_config.type == 'integer':
                            var_value = int(var_value)
                        elif var_config.type == 'number':
                            var_value = float(var_value)
                        elif var_config.type == 'boolean':
                            var_value = bool(var_value)
                        # string type needs no conversion
                        
                        # Range validation using Pydantic constraints
                        if var_config.min is not None and var_value < var_config.min:
                            raise ServiceError(f"Variable '{var_name}' must be >= {var_config.min}", 
                                             context={'preset_id': preset_id, 'variable': var_name})
                        
                        if var_config.max is not None and var_value > var_config.max:
                            raise ServiceError(f"Variable '{var_name}' must be <= {var_config.max}", 
                                             context={'preset_id': preset_id, 'variable': var_name})
                        
                        # Update template
                        template[var_name] = var_value
                        
                    except (ValueError, TypeError) as e:
                        raise ServiceError(f"Variable '{var_name}' must be {var_config.type}: {e}", 
                                         context={'preset_id': preset_id, 'variable': var_name})
            
            # Fill in defaults for missing non-required variables using Pydantic model
            for var_name, var_config in preset_data.variables.items():
                if var_name not in template and var_config.default is not None:
                    template[var_name] = var_config.default
            
            # Create configuration resource using validated preset type
            config_type = preset_data.type
            
            try:
                from flowlib.resources.models.config_resource import (
                    LLMConfigResource, DatabaseConfigResource, VectorDBConfigResource,
                    CacheConfigResource, StorageConfigResource, EmbeddingConfigResource,
                    GraphDBConfigResource, MessageQueueConfigResource
                )
                
                # Map preset types to resource classes
                resource_classes = {
                    'llm_config': LLMConfigResource,
                    'database_config': DatabaseConfigResource,
                    'vector_db_config': VectorDBConfigResource,
                    'cache_config': CacheConfigResource,
                    'storage_config': StorageConfigResource,
                    'embedding_config': EmbeddingConfigResource,
                    'graph_db_config': GraphDBConfigResource,
                    'message_queue_config': MessageQueueConfigResource,
                }
                
                if config_type not in resource_classes:
                    raise ServiceError(f"Unknown configuration type: {config_type}", 
                                     context={'preset_id': preset_id, 'config_type': config_type})
                
                resource_class = resource_classes[config_type]
                
                # Create configuration resource
                config_resource = resource_class(
                    name=config_name,
                    type=config_type,
                    **template
                )
                
                # Register with resource registry
                from flowlib.resources.registry.registry import resource_registry
                resource_registry.register(config_name, config_resource)
                
                return OperationResult(
                    success=True,
                    message=f"Configuration '{config_name}' generated successfully from preset '{preset_id}'",
                    data={
                        'config_name': config_name,
                        'preset_id': preset_id,
                        'config_type': config_type,
                        'template_used': template,
                        'variables_applied': variables,
                        'generation_timestamp': datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                raise ServiceError(f"Failed to create configuration resource: {e}", 
                                 context={'preset_id': preset_id, 'config_type': config_type})
            
        except Exception as e:
            logger.error(f"Failed to generate configuration from preset '{preset_id}': {e}")
            raise ServiceError(f"Configuration generation failed: {e}", 
                             context={'preset_id': preset_id, 'config_name': config_name})
    
    @handle_service_errors("save_custom_preset")
    async def save_custom_preset(self, preset_id: str, preset_data: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Save a custom preset to filesystem."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'save_custom_preset'})
        
        try:
            # Validate preset data structure
            required_fields = ['name', 'type', 'description', 'template']
            for field in required_fields:
                if field not in preset_data:
                    raise ServiceError(f"Missing required field: {field}", 
                                     context={'preset_id': preset_id, 'missing_field': field})
            
            # Create presets directory
            preset_dir = Path.home() / ".flowlib" / "presets"
            preset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save preset file
            preset_file = preset_dir / f"{preset_id}.yaml"
            with open(preset_file, 'w') as f:
                yaml.dump(preset_data, f, default_flow_style=False, indent=2)
            
            # Validate and update cache with Pydantic model
            try:
                validated_preset = PresetData(**preset_data)
                self._preset_cache[preset_id] = validated_preset
            except Exception as validation_error:
                raise ServiceError(f"Invalid preset data: {validation_error}", 
                                 context={'preset_id': preset_id})
            
            return OperationResult(
                success=True,
                message=f"Custom preset '{preset_id}' saved successfully",
                data={
                    'preset_id': preset_id,
                    'preset_file': str(preset_file),
                    'save_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to save custom preset '{preset_id}': {e}")
            raise ServiceError(f"Failed to save custom preset: {e}", 
                             context={'preset_id': preset_id})
    
    @handle_service_errors("delete_custom_preset")
    async def delete_custom_preset(self, preset_id: str) -> OperationResult:
        """Delete a custom preset."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'delete_custom_preset'})
        
        try:
            preset_dir = Path.home() / ".flowlib" / "presets"
            preset_file = preset_dir / f"{preset_id}.yaml"
            
            if not preset_file.exists():
                raise ServiceError(f"Custom preset '{preset_id}' not found", 
                                 context={'preset_id': preset_id})
            
            # Remove file
            preset_file.unlink()
            
            # Remove from cache
            if preset_id in self._preset_cache:
                del self._preset_cache[preset_id]
            
            return OperationResult(
                success=True,
                message=f"Custom preset '{preset_id}' deleted successfully",
                data={
                    'preset_id': preset_id,
                    'deletion_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to delete custom preset '{preset_id}': {e}")
            raise ServiceError(f"Failed to delete custom preset: {e}", 
                             context={'preset_id': preset_id})
    
    @handle_service_errors("get_preset_categories")
    async def get_preset_categories(self) -> OperationResult:
        """Get all available preset categories."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'get_preset_categories'})
        
        try:
            categories = set()
            category_counts = {}
            
            for preset_data in self._preset_cache.values():
                # Use validated Pydantic model instead of .get() pattern
                category = preset_data.category
                categories.add(category)
                # Use proper dict access with default initialization
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
            
            category_list = [
                {
                    'name': category,
                    'display_name': category.title(),
                    'preset_count': category_counts[category]
                }
                for category in sorted(categories)
            ]
            
            return OperationResult(
                success=True,
                message=f"Found {len(categories)} preset categories",
                data={
                    'categories': category_list,
                    'total_categories': len(categories),
                    'total_presets': sum(category_counts.values())
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get preset categories: {e}")
            raise ServiceError(f"Failed to get preset categories: {e}", 
                             context={'operation': 'get_preset_categories'})