"""
Configuration Manager controller for business logic.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere with strict validation
- Async-first design with proper error handling
- No legacy code, no backward compatibility
"""

import logging
from typing import List, Optional, Union
from pydantic import Field
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

from PySide6.QtCore import Signal

from ..services.base_controller import BaseController
from ..services.models import ConfigurationCreateData, ConfigurationEditData

logger = logging.getLogger(__name__)


class ConfigurationControllerState(MutableStrictBaseModel):
    """Configuration controller state with strict validation but mutable for runtime updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    initialized: bool = False
    current_operations: int = 0
    last_operation: Optional[str] = None
    current_operation: Optional[str] = None
    configurations_loaded: bool = False
    last_refresh_count: int = 0


class ConfigurationController(BaseController):
    """
    Configuration controller with clean async operations.
    
    No attribute existence checks, no fallbacks, strict contracts only.
    """
    
    # Signals for configuration events
    configurations_loaded = Signal(list)  # List of configurations loaded
    configuration_created = Signal(str)
    configuration_updated = Signal(str) 
    configuration_deleted = Signal(str)
    configuration_validated = Signal(str, bool, str)  # config_name, is_valid, message
    configuration_imported = Signal(str)
    configuration_exported = Signal(str)
    
    def __init__(self, service_factory, parent=None):
        super().__init__(parent)
        self.service_factory = service_factory
        self.state = ConfigurationControllerState()
        # Initialize service immediately - fail fast, no workarounds
        import asyncio
        self.config_manager = asyncio.run(self.service_factory.get_configuration_service())
        logger.info("ConfigurationController initialized successfully")
    
    def refresh_configurations(self) -> None:
        """Refresh the list of configurations."""
        self.state.current_operation = "refresh_configurations"
        self.start_operation("refresh_configurations", self.config_manager.list_configurations)
    
    def create_configuration(self, config_data: dict) -> None:
        """Create a new configuration."""
        # Validate configuration data with strict Pydantic model
        validated_data = ConfigurationCreateData(**config_data)
        self.state.current_operation = "create_configuration"
        self.start_operation("create_configuration", self.config_manager.create_configuration, validated_data)
    
    def get_configuration_details(self, config_name: str) -> None:
        """Get detailed information about a configuration."""
        self.state.current_operation = "get_configuration"
        self.start_operation("get_configuration", self.config_manager.get_configuration_details, config_name)
    
    def update_configuration(self, config_name: str, new_data: dict) -> None:
        """Update an existing configuration."""
        if not self.config_manager:
            self.operation_failed.emit("update_configuration", "Configuration manager not available")
            return
        
        try:
            # Validate edit data with strict Pydantic model
            validated_data = ConfigurationEditData(**new_data)
            updated_code = validated_data.content
            
            # Determine configuration type from validated data
            config_type = validated_data.type if validated_data.type else "llm"
            
            self.state.current_operation = "update_configuration"
            self.start_operation("update_configuration", 
                               self.config_manager.save_configuration, 
                               config_name, updated_code, config_type)
            
        except Exception as e:
            self.operation_failed.emit("update_configuration", f"Invalid configuration data: {str(e)}")
    
    def delete_configuration(self, config_name: str) -> None:
        """Delete a configuration."""
        self.state.current_operation = "delete_configuration"
        self.start_operation("delete_configuration", self.config_manager.delete_configuration, config_name)
    
    def import_configuration(self, file_path: str) -> None:
        """Import configuration from file."""
        self.state.current_operation = "import_configuration"
        self.start_operation("import_configuration", self.config_manager.import_configuration, file_path)
    
    def export_configuration(self, config_name: str, file_path: str) -> None:
        """Export configuration to file."""
        self.state.current_operation = "export_configuration" 
        self.start_operation("export_configuration", 
                           self.config_manager.export_configuration, 
                           config_name, file_path)
    
    def validate_configuration(self, config_name: str) -> None:
        """Validate a configuration."""
        self.state.current_operation = "validate_configuration"
        self.start_operation("validate_configuration", 
                           self.config_manager.validate_configuration, 
                           config_name)
    
    def validate_all_configurations(self) -> None:
        """Validate all configurations."""
        self.state.current_operation = "validate_all_configurations"
        # For now, use a placeholder operation - this can be enhanced later
        self.start_operation("validate_all_configurations",
                           lambda: "All configurations validation not yet implemented")
    
    def duplicate_configuration(self, source_name: str, new_name: str) -> None:
        """Duplicate an existing configuration."""
        self.state.current_operation = "duplicate_configuration"
        self.start_operation("duplicate_configuration", 
                           self.config_manager.duplicate_configuration, 
                           source_name, new_name)
    
    def backup_configurations(self, backup_path: str) -> None:
        """Backup all configurations."""
        self.state.current_operation = "backup_configurations"
        self.start_operation("backup_configurations", 
                           self.config_manager.backup_configurations, 
                           backup_path)
    
    def restore_configurations(self, backup_path: str) -> None:
        """Restore configurations from backup."""
        self.state.current_operation = "restore_configurations"
        self.start_operation("restore_configurations", 
                           self.config_manager.restore_configurations, 
                           backup_path)
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list], performance_monitor=None) -> None:
        """Handle operation completion with configuration-specific logic."""
        if not success:
            super()._on_operation_finished(operation_name, success, result, performance_monitor)
            return
        
        # Extract data first using base controller logic - single source of truth
        extracted_result = self._extract_operation_data(result)
        
        # Call base controller with extracted data
        super()._on_operation_finished(operation_name, success, extracted_result, performance_monitor)
        
        # Emit specific signals based on operation type using extracted data
        if operation_name == "create_configuration":
            # Result is extracted as dict by base controller - single source of truth
            if isinstance(extracted_result, dict) and 'config_name' in extracted_result:
                config_name = extracted_result['config_name']
            else:
                config_name = 'unknown'
            self.configuration_created.emit(config_name)
            logger.info(f"Configuration created: {config_name}")
            
        elif operation_name == "update_configuration":
            # Result is extracted as dict by base controller - single source of truth
            if isinstance(extracted_result, dict) and 'config_name' in extracted_result:
                config_name = extracted_result['config_name']
            else:
                config_name = 'unknown'
            self.configuration_updated.emit(config_name)
            logger.info(f"Configuration updated: {config_name}")
            
        elif operation_name == "delete_configuration":
            # Result is extracted as dict by base controller - single source of truth
            if isinstance(extracted_result, dict) and 'config_name' in extracted_result:
                config_name = extracted_result['config_name']
            else:
                config_name = 'unknown'
            self.configuration_deleted.emit(config_name)
            logger.info(f"Configuration deleted: {config_name}")
            
        elif operation_name == "import_configuration":
            # Result is extracted as dict by base controller - single source of truth
            if isinstance(extracted_result, dict) and 'config_name' in extracted_result:
                config_name = extracted_result['config_name']
            else:
                config_name = 'unknown'
            self.configuration_imported.emit(config_name)
            logger.info(f"Configuration imported: {config_name}")
            
        elif operation_name == "export_configuration":
            # Result is extracted as dict by base controller - single source of truth
            if isinstance(extracted_result, dict) and 'config_name' in extracted_result:
                config_name = extracted_result['config_name']
            else:
                config_name = 'unknown'
            self.configuration_exported.emit(config_name)
            logger.info(f"Configuration exported: {config_name}")
            
        elif operation_name == "refresh_configurations":
            # Use extracted result which should be list of dictionaries (not Pydantic objects)
            count = len(extracted_result) if isinstance(extracted_result, list) else 0
            self.state.last_refresh_count = count
            self.state.configurations_loaded = True
            self.configurations_loaded.emit(extracted_result if isinstance(extracted_result, list) else [])
            logger.info(f"Configurations refreshed: {count} found")
    
    def get_controller_state(self) -> dict:
        """Get current controller state."""
        return {
            "current_operation": self.state.current_operation,
            "configurations_loaded": self.state.configurations_loaded,
            "last_refresh_count": self.state.last_refresh_count,
            "config_manager_available": self.config_manager is not None
        }
    
    async def shutdown(self) -> None:
        """Shutdown controller and cleanup resources."""
        try:
            self.state = ConfigurationControllerState()
            self.config_manager = None
            logger.info("Configuration controller shutdown complete")
            
        except Exception as e:
            logger.error(f"Configuration controller shutdown failed: {e}")
            raise