"""
Flowlib Integration Service.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere with strict validation
- Async-first design with proper error handling
- No legacy code, no backward compatibility
"""

import logging
import re
from typing import List, Optional
from pathlib import Path
from pydantic import Field
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

logger = logging.getLogger(__name__)


class FlowlibIntegrationState(MutableStrictBaseModel):
    """Flowlib integration state with strict validation but mutable for runtime updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    registry_loaded: bool = False
    configurations_synchronized: bool = False
    last_sync_count: int = 0


class ConfigurationRole(StrictBaseModel):
    """Configuration role assignment with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    role_name: str
    config_id: str
    config_type: str


class ConfigurationMetadata(StrictBaseModel):
    """Configuration metadata with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    config_id: str
    config_type: str
    file_path: str
    content: str
    role_assignments: List[str] = Field(default_factory=list)


class ConfigurationInfoData(StrictBaseModel):
    """Configuration information data with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    config_id: str = Field(description="Configuration identifier")
    config_type: str = Field(description="Configuration type")
    role_assignments: List[str] = Field(default_factory=list, description="Role assignment names")


class ConfigurationListItem(StrictBaseModel):
    """Configuration list item with strict validation."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    name: str = Field(description="Configuration name")
    type: str = Field(description="Configuration type")
    file_path: str = Field(description="Configuration file path")
    content: str = Field(description="Configuration content")


class RepositoryData(StrictBaseModel):
    """Repository data structure with strict validation."""
    # Inherits strict configuration from StrictBaseModel
    
    role_assignments: dict[str, str] = Field(default_factory=dict)
    configurations: dict[str, ConfigurationMetadata] = Field(default_factory=dict)


class FlowlibIntegrationService:
    """
    Service for integrating GUI configurations with flowlib registry.
    
    Async-first design with strict type safety and no fallbacks.
    """
    
    def __init__(self):
        self.state = FlowlibIntegrationState()
        self._config_directory = Path.home() / ".flowlib" / "configs"
        self._initialized = False
        self._ensure_config_directory()
        self._initialized = True
    
    def _ensure_config_directory(self) -> None:
        """Ensure configuration directory exists."""
        self._config_directory.mkdir(parents=True, exist_ok=True)
    
    async def synchronize_configurations(self) -> RepositoryData:
        """Synchronize configurations from filesystem with registry."""
        try:
            configurations = {}
            role_assignments = {}
            
            # Scan configuration files (skip __init__.py and other non-config files)
            config_files = [f for f in self._config_directory.glob("*.py") 
                          if f.name != "__init__.py" and not f.name.startswith("__")]
            
            for config_file in config_files:
                try:
                    # First check if file has valid configuration decorators
                    if not self._has_valid_configuration_decorator(config_file):
                        logger.debug(f"Skipping {config_file.name}: no valid configuration decorators")
                        continue
                        
                    metadata = await self._analyze_configuration_file(config_file)
                    configurations[metadata.config_id] = metadata
                    
                    # Extract role assignments from decorators
                    for role in metadata.role_assignments:
                        role_assignments[role] = metadata.config_id
                        
                except Exception as e:
                    logger.warning(f"Skipping invalid configuration file {config_file}: {e}")
            
            repository_data = RepositoryData(
                role_assignments=role_assignments,
                configurations=configurations
            )
            
            self.state.last_sync_count = len(configurations)
            self.state.configurations_synchronized = True
            
            logger.info(f"Synchronized {len(configurations)} configurations")
            return repository_data
            
        except Exception as e:
            logger.error(f"Configuration synchronization failed: {e}")
            raise
    
    async def _analyze_configuration_file(self, file_path: Path) -> ConfigurationMetadata:
        """Analyze a configuration file and extract metadata."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract configuration ID and type from decorators
            config_info = self._extract_configuration_info(content)
            
            return ConfigurationMetadata(
                config_id=config_info.config_id,
                config_type=config_info.config_type,
                file_path=str(file_path),
                content=content,
                role_assignments=config_info.role_assignments
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze configuration file {file_path}: {e}")
            raise
    
    def _has_valid_configuration_decorator(self, file_path: Path) -> bool:
        """Check if file contains valid configuration decorators."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Quick check for configuration decorator patterns
            decorator_patterns = [
                r'@llm_config\s*\(',
                r'@model_config\s*\(',
                r'@database_config\s*\(',
                r'@vector_db_config\s*\(',
                r'@cache_config\s*\(',
                r'@storage_config\s*\(',
                r'@embedding_config\s*\(',
                r'@graph_db_config\s*\(',
                r'@message_queue_config\s*\('
            ]
            
            for pattern in decorator_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _extract_configuration_info(self, content: str) -> ConfigurationInfoData:
        """Extract configuration information from content."""
        config_id = ''
        config_type = 'unknown'
        role_assignments = []
        
        # Extract decorator information using strict patterns
        decorator_patterns = [
            (r'@llm_config\s*\(\s*["\']([^"\']+)["\']', 'llm'),
            (r'@model_config\s*\(\s*["\']([^"\']+)["\']', 'model'),
            (r'@database_config\s*\(\s*["\']([^"\']+)["\']', 'database'),
            (r'@vector_db_config\s*\(\s*["\']([^"\']+)["\']', 'vector'),
            (r'@cache_config\s*\(\s*["\']([^"\']+)["\']', 'cache'),
            (r'@storage_config\s*\(\s*["\']([^"\']+)["\']', 'storage'),
            (r'@embedding_config\s*\(\s*["\']([^"\']+)["\']', 'embedding'),
            (r'@graph_db_config\s*\(\s*["\']([^"\']+)["\']', 'graph'),
            (r'@message_queue_config\s*\(\s*["\']([^"\']+)["\']', 'message_queue')
        ]
        
        for pattern, found_config_type in decorator_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                config_id = match.group(1)
                config_type = found_config_type
                role_assignments.append(config_id)
                break
        
        if not config_id:
            raise ValueError("Configuration file must have a valid decorator")
        
        return ConfigurationInfoData(
            config_id=config_id,
            config_type=config_type,
            role_assignments=role_assignments
        )
    
    async def load_registry_environment(self, environment: str) -> None:
        """Load environment into flowlib registry."""
        try:
            from flowlib.config.registry_bridge import load_repository_environment
            
            repository_data = await self.synchronize_configurations()
            
            # Convert to format expected by registry bridge
            registry_data = {
                "role_assignments": repository_data.role_assignments,
                "configurations": {
                    config_id: {
                        "type": metadata.config_type,
                        "file_path": metadata.file_path,
                        "content": metadata.content
                    }
                    for config_id, metadata in repository_data.configurations.items()
                }
            }
            
            load_repository_environment(environment, registry_data)
            
            self.state.registry_loaded = True
            logger.info(f"Loaded environment '{environment}' into flowlib registry")
            
        except Exception as e:
            logger.error(f"Failed to load registry environment: {e}")
            raise
    
    async def create_configuration_file(self, config_id: str, content: str, config_type: str) -> Path:
        """Create a new configuration file."""
        try:
            # Validate configuration content
            self._validate_configuration_content(content, config_type)
            
            # Generate filename
            filename = f"{config_id.replace('-', '_')}.py"
            file_path = self._config_directory / filename
            
            # Write configuration file
            file_path.write_text(content, encoding='utf-8')
            
            logger.info(f"Created configuration file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to create configuration file: {e}")
            raise
    
    async def update_configuration_file(self, config_id: str, content: str, config_type: str) -> Path:
        """Update an existing configuration file."""
        try:
            # Validate configuration content
            self._validate_configuration_content(content, config_type)
            
            # Find existing file
            config_files = list(self._config_directory.glob("*.py"))
            target_file = None
            
            for config_file in config_files:
                try:
                    file_content = config_file.read_text(encoding='utf-8')
                    file_config_info = self._extract_configuration_info(file_content)
                    if file_config_info.config_id == config_id:
                        target_file = config_file
                        break
                except Exception:
                    continue
            
            if not target_file:
                # Create new file if not found
                return await self.create_configuration_file(config_id, content, config_type)
            
            # Update existing file
            target_file.write_text(content, encoding='utf-8')
            
            logger.info(f"Updated configuration file: {target_file}")
            return target_file
            
        except Exception as e:
            logger.error(f"Failed to update configuration file: {e}")
            raise
    
    async def delete_configuration_file(self, config_id: str) -> bool:
        """Delete a configuration file."""
        try:
            # Find and delete file
            config_files = list(self._config_directory.glob("*.py"))
            
            for config_file in config_files:
                try:
                    content = config_file.read_text(encoding='utf-8')
                    config_info = self._extract_configuration_info(content)
                    if config_info.config_id == config_id:
                        config_file.unlink()
                        logger.info(f"Deleted configuration file: {config_file}")
                        return True
                except Exception:
                    continue
            
            logger.warning(f"Configuration file not found for ID: {config_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete configuration file: {e}")
            raise
    
    def _validate_configuration_content(self, content: str, config_type: str) -> None:
        """Validate configuration content structure."""
        try:
            # Check for required imports
            required_imports = self._get_required_imports(config_type)
            for required_import in required_imports:
                if required_import not in content:
                    raise ValueError(f"Missing required import: {required_import}")
            
            # Check for proper decorator
            expected_decorators = {
                'llm': '@llm_config',
                'model': '@model_config',
                'database': '@database_config',
                'vector': '@vector_db_config',
                'cache': '@cache_config',
                'storage': '@storage_config',
                'embedding': '@embedding_config'
            }
            
            expected_decorator = expected_decorators.get(config_type)
            if expected_decorator and expected_decorator not in content:
                raise ValueError(f"Missing required decorator: {expected_decorator}")
            
            # Check for class definition
            if 'class ' not in content:
                raise ValueError("Configuration must define a class")
            
            # Check for proper base class based on config type
            if config_type == 'model':
                if 'ResourceBase' not in content:
                    raise ValueError("Model configuration must define a class inheriting from ResourceBase")
            else:
                if 'ConfigResource' not in content:
                    raise ValueError("Configuration must define a class inheriting from ConfigResource")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _get_required_imports(self, config_type: str) -> List[str]:
        """Get required imports for configuration type."""
        base_imports = [
            'from flowlib.resources.decorators.decorators import'
        ]
        
        # Different imports for different config types
        if config_type == 'model':
            base_imports.append('from flowlib.resources.models.base import')
        else:
            base_imports.append('from flowlib.resources.models.config_resource import')
        
        type_specific = {
            'llm': ['llm_config', 'LLMConfigResource'],
            'model': ['model_config', 'ResourceBase'],
            'database': ['database_config', 'DatabaseConfigResource'],
            'vector': ['vector_db_config', 'VectorDBConfigResource'],
            'cache': ['cache_config', 'CacheConfigResource'],
            'storage': ['storage_config', 'StorageConfigResource'],
            'embedding': ['embedding_config', 'EmbeddingConfigResource'],
            'graph': ['graph_db_config', 'GraphDBConfigResource'],
            'message_queue': ['message_queue_config', 'MessageQueueConfigResource']
        }
        
        specific_imports = type_specific[config_type] if config_type in type_specific else []
        return base_imports + specific_imports
    
    async def get_available_configurations(self) -> List[ConfigurationMetadata]:
        """Get list of available configurations."""
        try:
            repository_data = await self.synchronize_configurations()
            return list(repository_data.configurations.values())
            
        except Exception as e:
            logger.error(f"Failed to get available configurations: {e}")
            raise
    
    async def get_configuration_by_id(self, config_id: str) -> Optional[ConfigurationMetadata]:
        """Get specific configuration by ID."""
        try:
            repository_data = await self.synchronize_configurations()
            return repository_data.configurations.get(config_id)
            
        except Exception as e:
            logger.error(f"Failed to get configuration {config_id}: {e}")
            raise
    
    async def list_configurations(self) -> List[ConfigurationListItem]:
        """List configurations in format expected by ConfigurationService."""
        try:
            configurations = await self.get_available_configurations()
            return [
                ConfigurationListItem(
                    name=config.config_id,
                    type=config.config_type,
                    file_path=config.file_path,
                    content=config.content
                )
                for config in configurations
            ]
        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return []
    
    async def load_configuration(self, config_id: str) -> Optional[str]:
        """Load configuration content by config ID."""
        try:
            config_metadata = await self.get_configuration_by_id(config_id)
            return config_metadata.content if config_metadata else None
            
        except Exception as e:
            logger.error(f"Failed to load configuration '{config_id}': {e}")
            return None
    
    async def delete_configuration(self, config_id: str) -> 'OperationResult':
        """Delete configuration by config ID."""
        try:
            from .models import OperationResult, OperationType
            
            # Delete the configuration file
            success = await self.delete_configuration_file(config_id)
            
            return OperationResult(
                success=success,
                message=f"Configuration '{config_id}' {'deleted successfully' if success else 'not found'}",
                operation_type=OperationType.DELETE
            )
            
        except Exception as e:
            from .models import OperationResult
            logger.error(f"Failed to delete configuration '{config_id}': {e}")
            return OperationResult(
                success=False,
                message=f"Failed to delete configuration: {e}"
            )
    
    async def configuration_exists(self, config_id: str) -> bool:
        """Check if configuration exists."""
        try:
            config_metadata = await self.get_configuration_by_id(config_id)
            return config_metadata is not None
            
        except Exception as e:
            logger.error(f"Failed to check configuration existence '{config_id}': {e}")
            return False
    
    async def save_configuration(self, config_id: str, content: str) -> 'OperationResult':
        """Save configuration - wrapper around create/update methods."""
        try:
            from .models import OperationResult, OperationType
            
            # Extract config type from content
            config_info = self._extract_configuration_info(content)
            config_type = config_info.config_type
            
            # Check if configuration exists
            existing_config = await self.get_configuration_by_id(config_id)
            
            if existing_config:
                # Update existing configuration
                file_path = await self.update_configuration_file(config_id, content, config_type)
                operation = OperationType.UPDATE
            else:
                # Create new configuration
                file_path = await self.create_configuration_file(config_id, content, config_type)
                operation = OperationType.CREATE
            
            return OperationResult(
                success=True,
                message=f"Configuration '{config_id}' saved successfully",
                operation_type=operation,
                data={"file_path": str(file_path)}
            )
            
        except Exception as e:
            from .models import OperationResult
            logger.error(f"Failed to save configuration '{config_id}': {e}")
            return OperationResult(
                success=False,
                message=f"Failed to save configuration: {e}"
            )
    
    def get_service_state(self) -> dict:
        """Get current service state."""
        return {
            "registry_loaded": self.state.registry_loaded,
            "configurations_synchronized": self.state.configurations_synchronized,
            "last_sync_count": self.state.last_sync_count,
            "config_directory": str(self._config_directory)
        }
    
    async def shutdown(self) -> None:
        """Shutdown service and cleanup resources."""
        try:
            self.state = FlowlibIntegrationState()
            logger.info("Flowlib integration service shutdown complete")
            
        except Exception as e:
            logger.error(f"Service shutdown failed: {e}")
            raise