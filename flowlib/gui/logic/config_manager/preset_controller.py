"""
Preset Manager controller for business logic.

Handles preset management operations and coordinates
between the UI layer and service layer.
"""

import logging
from typing import List, Optional, Union
from PySide6.QtCore import Signal

from ..services.base_controller import BaseController
from ..services.models import PresetData

logger = logging.getLogger(__name__)


class PresetController(BaseController):
    """
    Controller for preset management operations.
    
    Provides business logic for:
    - Preset creation and management
    - Variable substitution and validation
    - Preset application and testing
    """
    
    # Specific signals for preset operations
    presets_loaded = Signal(list)  # List of presets
    preset_created = Signal(object)  # Created preset
    preset_applied = Signal(str, dict)  # preset_name, result
    preset_deleted = Signal(str)  # Deleted preset name
    variables_extracted = Signal(str, list)  # preset_name, variables
    
    def __init__(self, service_factory):
        super().__init__(service_factory)
        # Initialize service immediately - fail fast, no workarounds
        import asyncio
        self.preset_manager = asyncio.run(self.service_factory.get_preset_service())
        asyncio.run(self.preset_manager.initialize())
        logger.info("PresetController initialized successfully")
    
    def refresh_presets(self):
        """Refresh the list of presets."""
        # Get the preset templates from the preset manager
        self.start_operation("refresh_presets", self.preset_manager.get_available_presets)
    
    def create_preset(self, preset_data: dict[str, Union[str, int, float, bool]]):
        """Create a new preset."""
        # Validate preset data using Pydantic model
        validated_preset = PresetData(**preset_data)
        logger.info(f"Create preset requested: {validated_preset.name}")
        # Generate preset_id from validated name
        preset_name = validated_preset.name
        preset_data = validated_preset.model_dump()  # Use validated data
        preset_id = preset_name.lower().replace(' ', '_').replace('-', '_')
        preset_id = ''.join(c for c in preset_id if c.isalnum() or c == '_')
        
        self.start_operation("create_preset", self.preset_manager.save_custom_preset, preset_id, preset_data)
    
    def apply_preset(self, preset_name: str, variables: dict[str, Union[str, int, float, bool]], config_name: str = None):
        """Apply a preset with variable substitution."""
        logger.info(f"Apply preset requested: {preset_name}")
        
        # Generate config name if not provided
        if not config_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = f"preset_{preset_name}_{timestamp}"
        
        self.start_operation("apply_preset", self.preset_manager.generate_from_preset, preset_name, variables, config_name)
    
    def delete_preset(self, preset_name: str):
        """Delete a preset."""
        logger.info(f"Delete preset requested: {preset_name}")
        self.start_operation("delete_preset", self.preset_manager.delete_custom_preset, preset_name)
    
    def get_preset_details(self, preset_name: str):
        """Get detailed information about a preset."""
        self.start_operation("get_preset_details", self.preset_manager.get_preset_details, preset_name)
    
    def extract_variables(self, preset_name: str):
        """Extract variables from a preset template."""
        # Variables are included in get_preset_details response
        self.start_operation("extract_variables", self.preset_manager.get_preset_details, preset_name)
    
    def validate_preset(self, preset_name: str, variables: dict[str, Union[str, int, float, bool]]):
        """Validate a preset with given variables."""
        logger.info(f"Validate preset requested: {preset_name}")
        # Use dedicated validation method
        self.start_operation("validate_preset", self.preset_manager.validate_preset_variables, preset_name, variables)
    
    def export_preset(self, preset_name: str, export_path: str):
        """Export a preset to a file."""
        logger.info(f"Export preset requested: {preset_name} -> {export_path}")
        # For now, get preset details and save manually
        import json
        from pathlib import Path
        
        async def export_task():
            # Get result and let base controller handle success/failure
            result = await self.preset_manager.get_preset_details(preset_name)
            
            # Only perform file operation if this is called directly (not through base controller)
            # Base controller will handle success status properly
            if hasattr(result, 'success') and result.success and hasattr(result, 'data'):
                Path(export_path).write_text(json.dumps(result.data['preset'], indent=2))
            
            return result
        
        self.start_operation("export_preset", export_task)
    
    def import_preset(self, import_path: str):
        """Import a preset from a file."""
        logger.info(f"Import preset requested: {import_path}")
        # Load preset data and save as custom preset
        import json
        from pathlib import Path
        
        async def import_task():
            preset_data = json.loads(Path(import_path).read_text())
            validated_preset = PresetData(**preset_data)
            preset_id = validated_preset.id if validated_preset.id else Path(import_path).stem
            return await self.preset_manager.save_custom_preset(preset_id, preset_data)
        
        self.start_operation("import_preset", import_task)
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list], performance_monitor=None):
        """Handle operation completion with specific logic."""
        if not success:
            super()._on_operation_finished(operation_name, success, result, performance_monitor)
            return
        
        # Extract data first using base controller logic - single source of truth
        extracted_result = self._extract_operation_data(result)
        
        # Call base controller with extracted data
        super()._on_operation_finished(operation_name, success, extracted_result, performance_monitor)
        
        # Emit specific signals based on operation type using extracted data
        if operation_name == "refresh_presets":
            # For refresh_presets, the result should contain presets list
            if isinstance(extracted_result, dict) and 'presets' in extracted_result:
                self.presets_loaded.emit(extracted_result['presets'])
            else:
                # Fallback if the result is directly the presets list
                self.presets_loaded.emit(extracted_result if isinstance(extracted_result, list) else [])
        elif operation_name == "create_preset":
            self.preset_created.emit(extracted_result)
            # Also refresh the preset list to show the new preset
            self.refresh_presets()
        elif operation_name == "apply_preset":
            preset_name = "unknown"  # Would need to store this context
            self.preset_applied.emit(preset_name, extracted_result if isinstance(extracted_result, dict) else {'result': extracted_result})
        elif operation_name == "delete_preset":
            self.preset_deleted.emit(str(extracted_result))
        elif operation_name == "extract_variables":
            preset_name = "unknown"  # Would need to store this context
            
            # Extract variables from preset details response
            variables = []
            if isinstance(extracted_result, dict) and 'preset_data' in extracted_result:
                preset_data = extracted_result['preset_data']
                if 'variables' in preset_data:
                    # Convert variables dict to list format expected by UI
                    for var_name, var_config in preset_data['variables'].items():
                        var_dict = var_config.copy() if isinstance(var_config, dict) else {}
                        var_dict['name'] = var_name
                        variables.append(var_dict)
            
            self.variables_extracted.emit(preset_name, variables)
    
