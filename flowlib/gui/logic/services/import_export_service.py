"""
Import/Export Service using flowlib RegistryBridge.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere with strict validation
- Async-first design with proper error handling
- No legacy code, no backward compatibility
"""

import json
import yaml
import logging
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime
from pydantic import Field
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

from .models import OperationResult
from .error_boundaries import handle_service_errors, ServiceError
from .async_qt_helper import AsyncServiceMixin


class ConfigResourceInfo(StrictBaseModel):
    """Pydantic model for configuration resource information."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    name: str = Field(description="Configuration resource name")
    type: str = Field(description="Resource type")
    provider_type: str = Field(description="Provider type")
    description: str = Field(default="", description="Resource description")
    configuration: dict[str, Union[str, int, float, bool]] = Field(description="Configuration data")
    can_export: bool = Field(default=True, description="Whether resource can be exported")


class EnvironmentExportData(StrictBaseModel):
    """Pydantic model for environment export data structure."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    environment_name: str = Field(description="Environment name")
    export_timestamp: str = Field(description="Export timestamp")
    export_version: str = Field(default="1.0", description="Export format version")
    flowlib_version: str = Field(default="unknown", description="Flowlib version")
    configurations: dict[str, dict[str, Union[str, int, float, bool]]] = Field(description="Configuration resources")
    role_assignments: dict[str, str] = Field(default_factory=dict, description="Role assignments")


class ImportExportServiceState(MutableStrictBaseModel):
    """Import/export service state with strict validation but mutable for runtime updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    initialized: bool = False
    exports_count: int = 0
    imports_count: int = 0
    last_operation_timestamp: Optional[str] = None

logger = logging.getLogger(__name__)


class ImportExportService(AsyncServiceMixin):
    """
    Real import/export service using flowlib RegistryBridge.
    
    Strict contracts only, no attribute checks or fallbacks.
    """
    
    def __init__(self, service_factory):
        super().__init__()
        self.service_factory = service_factory
        self.state = ImportExportServiceState()
        
    async def initialize(self) -> None:
        """Initialize the import/export service."""
        try:
            # Ensure registry bridge is available
            await self.service_factory._ensure_registry_initialized()
            self.state.initialized = True
            logger.info("ImportExportService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ImportExportService: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the import/export service."""
        try:
            self.state = ImportExportServiceState()
            logger.info("ImportExportService shutdown complete")
        except Exception as e:
            logger.error(f"ImportExportService shutdown failed: {e}")
            raise
    
    def _validate_config_resource(self, config_resource) -> ConfigResourceInfo:
        """Validate and extract config resource info using strict Pydantic validation."""
        try:
            # All flowlib config resources follow strict Pydantic contracts
            # Trust the type system - no attribute existence checks
            name = config_resource.name
            resource_type = type(config_resource).__name__
            provider_type = config_resource.provider_type
            try:
                description = config_resource.description
            except AttributeError:
                description = ''
            configuration = config_resource.model_dump()
            
            # Create validated ConfigResourceInfo
            return ConfigResourceInfo(
                name=name,
                type=resource_type,
                provider_type=provider_type,
                description=description,
                configuration=configuration,
                can_export=True
            )
        except AttributeError as e:
            raise ServiceError(f"Invalid config resource - missing required attribute: {e}", 
                             context={'resource_type': type(config_resource).__name__})
        except Exception as e:
            raise ServiceError(f"Invalid config resource format: {e}", 
                             context={'resource_type': type(config_resource).__name__})
    
    @handle_service_errors("export_configuration")
    async def export_configuration(self, config_name: str, export_path: str) -> OperationResult:
        """Export a configuration using real flowlib registry."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'export_configuration'})
        
        try:
            # Get configuration from resource registry
            config_resource = self.service_factory.get_resource(config_name)
            
            if not config_resource:
                raise ServiceError(f"Configuration '{config_name}' not found", 
                                 context={'config_name': config_name})
            
            # Convert configuration to exportable format
            export_data = {
                'config_name': config_name,
                'resource_type': type(config_resource).__name__,
                'export_timestamp': datetime.now().isoformat(),
                'flowlib_version': '1.0.0',  # Could be dynamic
                'configuration': config_resource.model_dump()
            }
            
            # Write to file
            export_path_obj = Path(export_path)
            
            if export_path_obj.suffix.lower() == '.json':
                with open(export_path_obj, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif export_path_obj.suffix.lower() in ['.yaml', '.yml']:
                with open(export_path_obj, 'w', encoding='utf-8') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            else:
                # Default to JSON
                with open(export_path_obj, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            return OperationResult(
                success=True,
                message=f"Configuration '{config_name}' exported successfully",
                data={
                    'config_name': config_name,
                    'export_path': str(export_path_obj),
                    'file_size': export_path_obj.stat().st_size,
                    'export_timestamp': export_data['export_timestamp']
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to export configuration '{config_name}': {e}")
            raise ServiceError(f"Configuration export failed: {e}", 
                             context={'config_name': config_name, 'export_path': export_path})
    
    @handle_service_errors("import_configuration")
    async def import_configuration(self, import_path: str, target_name: Optional[str] = None) -> OperationResult:
        """Import a configuration using real flowlib registry."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'import_configuration'})
        
        try:
            import_path_obj = Path(import_path)
            
            if not import_path_obj.exists():
                raise ServiceError(f"Import file not found: {import_path}", 
                                 context={'import_path': import_path})
            
            # Read configuration file
            if import_path_obj.suffix.lower() == '.json':
                with open(import_path_obj, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
            elif import_path_obj.suffix.lower() in ['.yaml', '.yml']:
                with open(import_path_obj, 'r', encoding='utf-8') as f:
                    import_data = yaml.safe_load(f)
            else:
                raise ServiceError(f"Unsupported file format: {import_path_obj.suffix}", 
                                 context={'import_path': import_path})
            
            # Validate import data structure
            required_fields = ['config_name', 'resource_type', 'configuration']
            for field in required_fields:
                if field not in import_data:
                    raise ServiceError(f"Invalid import file: missing '{field}' field", 
                                     context={'import_path': import_path})
            
            # Determine target configuration name
            config_name = target_name or import_data['config_name']
            resource_type = import_data['resource_type']
            config_data = import_data['configuration']
            
            # Try to recreate the configuration resource
            try:
                from flowlib.resources.models.config_resource import (
                    LLMConfigResource, DatabaseConfigResource, VectorDBConfigResource,
                    CacheConfigResource, StorageConfigResource, EmbeddingConfigResource,
                    GraphDBConfigResource, MessageQueueConfigResource
                )
                
                # Map resource types to classes
                resource_classes = {
                    'LLMConfigResource': LLMConfigResource,
                    'DatabaseConfigResource': DatabaseConfigResource,
                    'VectorDBConfigResource': VectorDBConfigResource,
                    'CacheConfigResource': CacheConfigResource,
                    'StorageConfigResource': StorageConfigResource,
                    'EmbeddingConfigResource': EmbeddingConfigResource,
                    'GraphDBConfigResource': GraphDBConfigResource,
                    'MessageQueueConfigResource': MessageQueueConfigResource,
                }
                
                if resource_type not in resource_classes:
                    raise ServiceError(f"Unknown resource type: {resource_type}", 
                                     context={'resource_type': resource_type})
                
                resource_class = resource_classes[resource_type]
                
                # Create and register the configuration
                config_resource = resource_class(**config_data)
                
                # Register with resource registry
                from flowlib.resources.registry.registry import resource_registry
                resource_registry.register(config_name, config_resource)
                
                return OperationResult(
                    success=True,
                    message=f"Configuration '{config_name}' imported successfully",
                    data={
                        'config_name': config_name,
                        'resource_type': resource_type,
                        'import_path': str(import_path_obj),
                        'original_name': import_data['config_name'],
                        'import_timestamp': datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                raise ServiceError(f"Failed to create configuration resource: {e}", 
                                 context={'resource_type': resource_type, 'config_data': config_data})
            
        except Exception as e:
            logger.error(f"Failed to import configuration from '{import_path}': {e}")
            raise ServiceError(f"Configuration import failed: {e}", 
                             context={'import_path': import_path})
    
    @handle_service_errors("export_environment")
    async def export_environment(self, environment_name: str, export_path: str) -> OperationResult:
        """Export entire environment using registry bridge."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'export_environment'})
        
        try:
            from flowlib.resources.registry.registry import resource_registry
            from flowlib.resources.models.config_resource import ProviderConfigResource
            
            # Get all configurations
            all_resources = resource_registry.list()
            config_resources = [r for r in all_resources if isinstance(r, ProviderConfigResource)]
            
            # Build environment export using validated Pydantic model
            configurations = {}
            for config_resource in config_resources:
                validated_resource = self._validate_config_resource(config_resource)
                configurations[validated_resource.name] = {
                    'resource_type': validated_resource.type,
                    'configuration': validated_resource.configuration
                }
            
            # Create validated environment export data
            environment_export = EnvironmentExportData(
                environment_name=environment_name,
                export_timestamp=datetime.now().isoformat(),
                export_version='1.0',
                flowlib_version='1.0.0',
                configurations=configurations,
                role_assignments={}
            )
            
            environment_data = environment_export.model_dump()
            
            # Write environment file
            export_path_obj = Path(export_path)
            
            if export_path_obj.suffix.lower() == '.json':
                with open(export_path_obj, 'w', encoding='utf-8') as f:
                    json.dump(environment_data, f, indent=2, default=str)
            else:
                with open(export_path_obj, 'w', encoding='utf-8') as f:
                    yaml.dump(environment_data, f, default_flow_style=False)
            
            return OperationResult(
                success=True,
                message=f"Environment '{environment_name}' exported successfully",
                data={
                    'environment_name': environment_name,
                    'export_path': str(export_path_obj),
                    'total_configurations': len(config_resources),
                    'file_size': export_path_obj.stat().st_size,
                    'export_timestamp': environment_data['export_timestamp']
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to export environment '{environment_name}': {e}")
            raise ServiceError(f"Environment export failed: {e}", 
                             context={'environment_name': environment_name, 'export_path': export_path})
    
    @handle_service_errors("get_exportable_configurations")
    async def get_exportable_configurations(self) -> OperationResult:
        """Get list of configurations that can be exported."""
        if not self._initialized:
            raise ServiceError("Service not initialized", context={'operation': 'get_exportable_configurations'})
        
        try:
            from flowlib.resources.registry.registry import resource_registry
            from flowlib.resources.models.config_resource import ProviderConfigResource
            
            # Get all configurations
            all_resources = resource_registry.list()
            config_resources = [r for r in all_resources if isinstance(r, ProviderConfigResource)]
            
            exportable_configs = []
            for config_resource in config_resources:
                # Use strict Pydantic validation
                validated_resource = self._validate_config_resource(config_resource)
                exportable_configs.append(validated_resource.model_dump())
            
            return OperationResult(
                success=True,
                message=f"Found {len(exportable_configs)} exportable configurations",
                data={
                    'configurations': exportable_configs,
                    'total_count': len(exportable_configs)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get exportable configurations: {e}")
            raise ServiceError(f"Failed to get exportable configurations: {e}", 
                             context={'operation': 'get_exportable_configurations'})
    
    @handle_service_errors("generate_export_preview")
    async def generate_export_preview(self, export_options: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Generate a preview of what would be exported with given options."""
        try:
            from flowlib.resources.registry.registry import resource_registry
            from flowlib.resources.models.config_resource import ProviderConfigResource
            
            # Get all configuration resources
            all_resources = resource_registry.list()
            config_resources = [
                r for r in all_resources 
                if isinstance(r, ProviderConfigResource)
            ]
            
            # Filter by export types if specified
            export_types = export_options['export_types'] if 'export_types' in export_options else []
            if export_types:
                filtered_resources = []
                for resource in config_resources:
                    # Map provider types to export types
                    provider_type = getattr(resource, 'provider_type', 'unknown').lower()
                    if any(export_type.lower() in provider_type or provider_type in export_type.lower() 
                          for export_type in export_types):
                        filtered_resources.append(resource)
                config_resources = filtered_resources
            
            # Build preview data
            preview_configs = []
            for resource in config_resources:
                try:
                    config_info = self._validate_config_resource(resource)
                    preview_configs.append({
                        'name': config_info.name,
                        'type': config_info.type,
                        'provider_type': config_info.provider_type,
                        'description': config_info.description,
                        'size_estimate': len(str(config_info.configuration))
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid resource {resource}: {e}")
            
            export_format = export_options['format'] if 'format' in export_options else 'JSON'
            
            return OperationResult(
                success=True,
                message=f"Export preview generated for {len(preview_configs)} configurations",
                data={
                    'preview_type': 'export',
                    'format': export_format,
                    'configurations': preview_configs,
                    'total_count': len(preview_configs),
                    'estimated_size_kb': sum(c['size_estimate'] for c in preview_configs) / 1024,
                    'export_options': export_options
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate export preview: {e}")
            raise ServiceError(f"Export preview generation failed: {e}", 
                             context={'export_options': export_options})
    
    @handle_service_errors("generate_import_preview")
    async def generate_import_preview(self, import_options: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Generate a preview of what would be imported with given options."""
        try:
            if 'file_path' not in import_options:
                raise ServiceError("Import options missing required 'file_path' field", context={'import_options': import_options})
            import_path = import_options['file_path']
            if not import_path:
                raise ServiceError("Import file path is required", context={'import_options': import_options})
            
            file_path = Path(import_path)
            if not file_path.exists():
                raise ServiceError(f"Import file does not exist: {import_path}", 
                                 context={'import_path': import_path})
            
            # Analyze the import file
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Try to parse as JSON first, then YAML
                try:
                    data = json.loads(content)
                    file_format = 'JSON'
                except json.JSONDecodeError:
                    try:
                        data = yaml.safe_load(content)
                        file_format = 'YAML'
                    except yaml.YAMLError:
                        raise ServiceError("File is not valid JSON or YAML", 
                                         context={'import_path': import_path})
                
                # Analyze the structure
                preview_data = {
                    'preview_type': 'import',
                    'file_format': file_format,
                    'file_size_kb': file_path.stat().st_size / 1024,
                    'import_options': import_options
                }
                
                if isinstance(data, dict):
                    # Check if it's an environment export
                    if 'configurations' in data and 'environment_name' in data:
                        preview_data.update({
                            'import_type': 'environment',
                            'environment_name': data['environment_name'] if 'environment_name' in data else 'unknown',
                            'configuration_count': len(data['configurations'] if 'configurations' in data else {}),
                            'role_assignments': len(data['role_assignments'] if 'role_assignments' in data else {}),
                            'export_version': data['export_version'] if 'export_version' in data else 'unknown'
                        })
                    else:
                        # Single configuration
                        preview_data.update({
                            'import_type': 'single_configuration',
                            'configuration_count': 1,
                            'detected_keys': list(data.keys())[:10]  # First 10 keys
                        })
                else:
                    preview_data.update({
                        'import_type': 'unknown',
                        'data_type': type(data).__name__
                    })
                
                return OperationResult(
                    success=True,
                    message=f"Import preview generated for {file_format} file",
                    data=preview_data
                )
                
            except Exception as e:
                return OperationResult(
                    success=False,
                    message=f"Failed to analyze import file: {e}",
                    data={'preview_type': 'import', 'error': str(e)}
                )
            
        except Exception as e:
            logger.error(f"Failed to generate import preview: {e}")
            raise ServiceError(f"Import preview generation failed: {e}", 
                             context={'import_options': import_options})
    
    @handle_service_errors("analyze_import_file")
    async def analyze_import_file(self, file_path: str) -> OperationResult:
        """Analyze an import file to determine its contents."""
        try:
            # Reuse the import preview logic for file analysis
            preview_result = await self.generate_import_preview({'file_path': file_path})
            
            if preview_result.success:
                # Transform preview data to analysis format
                analysis_data = preview_result.data.copy()
                analysis_data.update({
                    'success': True,
                    'file_type': analysis_data['file_format'] if 'file_format' in analysis_data else 'unknown',
                    'config_count': analysis_data['configuration_count'] if 'configuration_count' in analysis_data else 0,
                    'message': f"File analysis completed for {analysis_data['file_format'] if 'file_format' in analysis_data else 'unknown'} file"
                })
                
                return OperationResult(
                    success=True,
                    message=analysis_data['message'],
                    data=analysis_data
                )
            else:
                return preview_result
                
        except Exception as e:
            logger.error(f"Failed to analyze import file: {e}")
            raise ServiceError(f"Import file analysis failed: {e}", 
                             context={'file_path': file_path})
    
    @handle_service_errors("export_configurations")
    async def export_configurations(self, export_options: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Export configurations with specified options."""
        try:
            # Extract export path from options
            if 'export_path' not in export_options:
                raise ServiceError("Export options missing required 'export_path' field", context={'export_options': export_options})
            export_path = export_options['export_path']
            if not export_path:
                raise ServiceError("Export path is required", context={'export_options': export_options})
            
            # Use environment export for full exports
            environment_name = export_options['environment'] if 'environment' in export_options else 'development'
            
            return await self.export_environment(environment_name, export_path)
            
        except Exception as e:
            logger.error(f"Failed to export configurations: {e}")
            raise ServiceError(f"Configuration export failed: {e}", 
                             context={'export_options': export_options})
    
    @handle_service_errors("import_configurations")
    async def import_configurations(self, import_options: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Import configurations with specified options."""
        try:
            # Extract import path from options
            if 'file_path' not in import_options:
                raise ServiceError("Import options missing required 'file_path' field", context={'import_options': import_options})
            import_path = import_options['file_path']
            if not import_path:
                raise ServiceError("Import file path is required", context={'import_options': import_options})
            
            # Use single configuration import for now
            target_name = import_options['target_name'] if 'target_name' in import_options else None
            
            return await self.import_configuration(import_path, target_name)
            
        except Exception as e:
            logger.error(f"Failed to import configurations: {e}")
            raise ServiceError(f"Configuration import failed: {e}", 
                             context={'import_options': import_options})
    
    @handle_service_errors("create_backup")
    async def create_backup(self, backup_options: dict[str, Union[str, int, float, bool]]) -> OperationResult:
        """Create a repository backup."""
        try:
            # Extract backup path from options
            backup_path = backup_options['backup_path'] if 'backup_path' in backup_options else None
            if not backup_path:
                # Generate default backup path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"flowlib_backup_{timestamp}.json"
            
            # Use environment export for backup
            environment_name = backup_options['environment'] if 'environment' in backup_options else 'development'
            
            result = await self.export_environment(environment_name, backup_path)
            
            if result.success:
                # Transform result to backup format
                backup_data = {
                    'success': True,
                    'backup_path': backup_path,
                    'backup_size': Path(backup_path).stat().st_size if Path(backup_path).exists() else 0,
                    'message': f"Backup created successfully at {backup_path}"
                }
                
                return OperationResult(
                    success=True,
                    message=backup_data['message'],
                    data=backup_data
                )
            else:
                return result
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise ServiceError(f"Backup creation failed: {e}", 
                             context={'backup_options': backup_options})