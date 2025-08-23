"""
Knowledge Plugin Manager controller for business logic.

Handles knowledge plugin operations and coordinates
between the UI layer and service layer.
"""

import logging
from typing import List, Optional, Union
from PySide6.QtCore import Signal

from ..services.base_controller import BaseController
from ..services.models import KnowledgePluginConfig

logger = logging.getLogger(__name__)


class KnowledgePluginController(BaseController):
    """
    Controller for knowledge plugin management operations.
    
    Provides business logic for:
    - Plugin discovery and listing
    - Plugin generation from documents
    - Plugin validation and testing
    """
    
    # Specific signals for knowledge plugin operations
    plugins_loaded = Signal(list)  # List of plugins
    plugin_generated = Signal(object)  # Generated plugin details
    plugin_deleted = Signal(str)  # Deleted plugin name
    plugin_tested = Signal(str, dict)  # plugin_name, test_results
    generation_progress = Signal(str, int)  # status_message, percentage
    domain_strategies_loaded = Signal(list)  # Available domain strategies
    
    def __init__(self, service_factory):
        super().__init__(service_factory)
        # Initialize service immediately - fail fast, no workarounds
        import asyncio
        self.plugin_manager = asyncio.run(self.service_factory.get_knowledge_plugin_service())
        asyncio.run(self.plugin_manager.initialize())
        logger.info("KnowledgePluginController initialized successfully")
    
    def refresh_plugins(self):
        """Refresh the list of knowledge plugins."""
        self.start_operation("refresh_plugins", self.plugin_manager.list_plugins)
    
    def generate_plugin(self, plugin_config: dict[str, Union[str, int, float, bool]]):
        """Generate a new knowledge plugin using validated Pydantic model."""
        # Validate plugin config using Pydantic model
        validated_config = KnowledgePluginConfig(**plugin_config)
        logger.info(f"Generate plugin requested: {validated_config.name}")
        self.start_operation("generate_plugin", self.plugin_manager.generate_plugin, validated_config.model_dump())
    
    def delete_plugin(self, plugin_name: str):
        """Delete a knowledge plugin."""
        logger.info(f"Delete plugin requested: {plugin_name}")
        self.start_operation("delete_plugin", self.plugin_manager.delete_plugin, plugin_name)
    
    def get_plugin_details(self, plugin_name: str):
        """Get detailed information about a plugin."""
        self.start_operation("get_plugin_details", self.plugin_manager.get_plugin_details, plugin_name)
    
    def validate_plugin(self, plugin_name: str):
        """Validate a knowledge plugin."""
        logger.info(f"Validate plugin requested: {plugin_name}")
        self.start_operation("validate_plugin", self.plugin_manager.validate_plugin, plugin_name)
    
    def test_plugin(self, plugin_name: str, test_options: dict[str, Union[str, int, float, bool]] = None):
        """Test a knowledge plugin."""
        logger.info(f"Test plugin requested: {plugin_name}")
        if test_options is None:
            test_options = {}
        
        # Use the plugin manager's validate method as a test
        self.start_operation("test_plugin", self.plugin_manager.validate_plugin, plugin_name)
    
    def get_available_extractors(self):
        """Get list of available knowledge extractors."""
        self.start_operation("get_extractors", self.plugin_manager.get_available_extractors)
    
    def get_available_domain_strategies(self):
        """Get list of available domain strategies from flowlib registry."""
        from flowlib.knowledge.plugin_generation.domain_strategies import domain_strategy_registry
        
        # Get available strategies from flowlib registry
        strategies_data = domain_strategy_registry.list_available_strategies()
        
        # Convert to format expected by UI
        strategies_list = []
        for domain_enum, metadata in strategies_data.items():
            strategies_list.append({
                'id': domain_enum.value,  # e.g., "generic", "software_engineering"
                'name': metadata['name'],  # e.g., "Generic", "Software Engineering"
                'description': metadata['description']  # Full description
            })
        
        # Emit the loaded strategies
        self.domain_strategies_loaded.emit(strategies_list)
    
    def preview_plugin_generation(self, plugin_config: dict[str, Union[str, int, float, bool]]):
        """Preview what a plugin generation would produce using validated Pydantic model."""
        # Validate plugin config using Pydantic model
        validated_config = KnowledgePluginConfig(**plugin_config)
        logger.info(f"Preview generation requested: {validated_config.name}")
        self.start_operation("preview_generation", self.plugin_manager.preview_generation, validated_config.model_dump())
    
    def get_plugin_manifest(self, plugin_name: str):
        """Get the manifest of a plugin."""
        self.start_operation("get_manifest", self.plugin_manager.get_plugin_manifest, plugin_name)
    
    def update_plugin_manifest(self, plugin_name: str, manifest_data: dict[str, Union[str, int, float, bool]]):
        """Update a plugin's manifest."""
        logger.info(f"Update manifest requested: {plugin_name}")
        self.start_operation("update_manifest", self.plugin_manager.update_plugin_manifest, plugin_name, manifest_data)
    
    def _on_operation_finished(self, operation_name: str, success: bool, result: Union[str, int, float, bool, dict, list], performance_monitor=None):
        """Handle operation completion with specific logic."""
        super()._on_operation_finished(operation_name, success, result, performance_monitor)
        
        if not success:
            return
        
        # Emit specific signals based on operation type
        if operation_name == "refresh_plugins":
            self.plugins_loaded.emit(result)
        elif operation_name == "generate_plugin":
            self.plugin_generated.emit(result)
        elif operation_name == "delete_plugin":
            self.plugin_deleted.emit(str(result))
        elif operation_name == "test_plugin":
            plugin_name = "unknown"  # Would need to store this context
            test_results = result if isinstance(result, dict) else {'result': result}
            self.plugin_tested.emit(plugin_name, test_results)
    
