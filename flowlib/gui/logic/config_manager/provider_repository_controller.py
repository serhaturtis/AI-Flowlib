"""
Provider Repository controller for business logic.

Handles provider repository management operations and coordinates
between the UI layer and service layer.
"""

import logging
from typing import List, Optional, Union
from PySide6.QtCore import Signal
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from ..services.base_controller import BaseController

logger = logging.getLogger(__name__)


class EnvironmentSwitchResult(StrictBaseModel):
    """Pydantic model for environment switch operation results."""
    # Inherits strict configuration from StrictBaseModel
    
    success: bool = Field(description="Switch success status")
    environment: str = Field(description="New environment name")
    message: str = Field(default="", description="Switch operation message")


class RoleAssignmentResult(StrictBaseModel):
    """Pydantic model for role assignment operation results."""
    # Inherits strict configuration from StrictBaseModel
    
    success: bool = Field(description="Assignment success status")
    role: str = Field(description="Role name")
    provider: str = Field(description="Provider name")
    message: str = Field(default="", description="Assignment operation message")


class RoleUnassignmentResult(StrictBaseModel):
    """Pydantic model for role unassignment operation results."""
    # Inherits strict configuration from StrictBaseModel
    
    success: bool = Field(description="Unassignment success status")
    role: str = Field(description="Role name")
    message: str = Field(default="", description="Unassignment operation message")


class BackupCreationResult(StrictBaseModel):
    """Pydantic model for backup creation operation results."""
    # Inherits strict configuration from StrictBaseModel
    
    success: bool = Field(description="Backup creation success status")
    backup_path: str = Field(description="Path to created backup file")
    backup_size: int = Field(default=0, description="Size of backup in bytes")
    message: str = Field(default="", description="Backup operation message")


class ProviderRepositoryController(BaseController):
    """
    Controller for provider repository management operations.
    
    Provides business logic for:
    - Repository overview and status
    - Environment switching
    - Role assignments and management
    """
    
    # Specific signals for repository operations
    overview_loaded = Signal(dict)  # Repository overview data
    environment_switched = Signal(str)  # New environment name
    role_assigned = Signal(str, str)  # role_name, provider_name
    role_unassigned = Signal(str)  # role_name
    backup_created = Signal(str)  # backup_path
    
    def __init__(self, service_factory, parent=None):
        super().__init__(parent)
        self.service_factory = service_factory
        # Initialize service immediately - fail fast, no workarounds
        import asyncio
        self.repository_service = asyncio.run(self.service_factory.get_repository_service())
        logger.info("ProviderRepositoryController initialized successfully")
    
    def get_repository_overview(self):
        """Get repository overview and status information."""
        self.start_operation("get_overview", self.repository_service.get_repository_overview)
    
    def switch_environment(self, environment_name: str):
        """Switch to a different environment."""
        logger.info(f"Switching to environment: {environment_name}")
        self.start_operation("switch_environment", self.repository_service.switch_environment, environment_name)
    
    def get_available_environments(self):
        """Get list of available environments."""
        self.start_operation("get_environments", self.repository_service.get_environments)
    
    def assign_role(self, role_name: str, config_name: str):
        """Assign a role to a configuration using RoleManager."""
        
        logger.info(f"Assign role requested: {role_name} -> {config_name}")
        
        # Create a wrapper function to use RoleManager directly
        async def assign_role_wrapper():
            try:
                from flowlib.config.role_manager import role_manager
                
                success = role_manager.assign_role(role_name, config_name)
                
                if success:
                    return {
                        "success": True,
                        "role": role_name,
                        "config": config_name
                    }
                else:
                    return {
                        "success": False,
                        "role": role_name,
                        "config": config_name,
                        "error": f"Failed to assign role '{role_name}' to configuration '{config_name}'"
                    }
            except Exception as e:
                logger.error(f"Failed to assign role: {e}")
                return {
                    "success": False,
                    "role": role_name,
                    "config": config_name,
                    "error": str(e)
                }
        
        # Call the wrapper function
        self.start_operation("assign_role", assign_role_wrapper)
    
    def unassign_role(self, role_name: str):
        """Unassign a role using RoleManager."""
        
        logger.info(f"Unassign role requested: {role_name}")
        
        # Create a wrapper function to use RoleManager directly
        async def unassign_role_wrapper():
            try:
                from flowlib.config.role_manager import role_manager
                
                success = role_manager.unassign_role(role_name)
                
                if success:
                    return {
                        "success": True,
                        "role": role_name
                    }
                else:
                    return {
                        "success": False,
                        "role": role_name,
                        "error": f"Failed to unassign role '{role_name}' (may not have been assigned)"
                    }
            except Exception as e:
                logger.error(f"Failed to unassign role: {e}")
                return {
                    "success": False,
                    "role": role_name,
                    "error": str(e)
                }
        
        # Call the wrapper function
        self.start_operation("unassign_role", unassign_role_wrapper)
    
    def create_backup(self, backup_location: Optional[str] = None):
        """Create a repository backup."""
        logger.info(f"Create backup requested: {backup_location}")
        
        # Use the import/export service for backup functionality
        import_export_service = self.service_factory.get_import_export_service()
        if import_export_service:
            # Create a full backup using the import/export service
            self.start_operation("create_backup", import_export_service.create_full_backup)
        else:
            self.operation_failed.emit("create_backup", "Import/Export service not available for backup creation")
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list], performance_monitor=None):
        """Handle operation completion with specific logic."""
        # Extract data first using the same method as base controller
        extracted_result = self._extract_operation_data(result)
        
        super()._on_operation_finished(operation_name, success, result)
        
        if not success:
            return
        
        # Emit specific signals based on operation type using validated Pydantic models
        try:
            if operation_name == "get_overview":
                logger.debug(f"Emitting overview data: type={type(extracted_result)}, keys={getattr(extracted_result, 'keys', lambda: 'N/A')()}")
                self.overview_loaded.emit(extracted_result)
            elif operation_name == "switch_environment":
                # Base controller automatically extracts data from OperationResult
                if isinstance(extracted_result, dict):
                    # Check for extracted data from OperationResult
                    if 'environment_id' in extracted_result:
                        self.environment_switched.emit(extracted_result['environment_id'])
                    else:
                        # Legacy dict format
                        switch_result = EnvironmentSwitchResult(**extracted_result)
                        self.environment_switched.emit(switch_result.environment)
                else:
                    logger.warning(f"Unexpected environment switch result type: {type(extracted_result)}")
                    self.environment_switched.emit("unknown")
            elif operation_name == "assign_role":
                # Base controller automatically extracts data from OperationResult
                if isinstance(result, dict):
                    # Check for extracted data from OperationResult - fail-fast approach
                    if 'role' not in result:
                        raise ValueError("assign_role operation result missing required 'role' field")
                    if 'config' not in result:
                        raise ValueError("assign_role operation result missing required 'config' field")
                    
                    role = result['role']
                    config = result['config']
                    self.role_assigned.emit(role, config)
                else:
                    raise ValueError(f"assign_role operation returned invalid result type: {type(result)}")
            elif operation_name == "unassign_role":
                # Base controller automatically extracts data from OperationResult
                if isinstance(result, dict):
                    # Check for extracted data from OperationResult - fail-fast approach
                    if 'role' not in result:
                        raise ValueError("unassign_role operation result missing required 'role' field")
                    role = result['role']
                    self.role_unassigned.emit(role)
                else:
                    raise ValueError(f"unassign_role operation returned invalid result type: {type(result)}")
            elif operation_name == "create_backup":
                # Validate result as BackupCreationResult
                if isinstance(result, dict):
                    backup_result = BackupCreationResult(**result)
                    self.backup_created.emit(backup_result.backup_path)
                else:
                    logger.warning(f"Unexpected backup creation result type: {type(result)}")
                    self.backup_created.emit(str(result))
                    
        except Exception as e:
            logger.error(f"Failed to validate {operation_name} result: {e}")
            # Emit error recovery signals following CLAUDE.md error propagation
            if operation_name == "switch_environment":
                self.environment_switched.emit("unknown")
            elif operation_name == "assign_role":
                self.role_assigned.emit("unknown", "unknown")
            elif operation_name == "unassign_role":
                self.role_unassigned.emit("unknown")
            elif operation_name == "create_backup":
                self.backup_created.emit("unknown")
    
    def is_service_available(self) -> bool:
        """Check if the repository service is available."""
        return self.repository_service is not None