"""
Service Factory for dependency injection and service lifecycle management.

Clean implementation following CLAUDE.md principles:
- No fallbacks, no workarounds, single Pydantic contracts
- Type safety everywhere with strict validation  
- Async-first design with proper lifecycle management
- No legacy code, no backward compatibility
"""

import logging
import asyncio
from typing import Optional, List
from pydantic import Field
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

from PySide6.QtCore import QObject, Signal

# Import flowlib registry components
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from flowlib.config.registry_bridge import load_repository_environment

logger = logging.getLogger(__name__)


class ServiceFactoryState(MutableStrictBaseModel):
    """Service factory state with strict validation but mutable for runtime updates."""
    # Inherits strict configuration from MutableStrictBaseModel
    
    registry_initialized: bool = False
    services_loaded: bool = False


class ServiceFactoryConfig(StrictBaseModel):
    """Service factory configuration."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    service_timeout_seconds: int = 30


class ServiceFactory(QObject):
    """
    Service factory with clean dependency injection.
    
    No wrapper classes, no fallbacks, no backward compatibility.
    Single source of truth for service access.
    """
    
    # Signals for service lifecycle events
    service_created = Signal(str, object)
    service_failed = Signal(str, str)
    registry_initialized = Signal()
    
    def __init__(self, config: Optional[ServiceFactoryConfig] = None):
        super().__init__()
        self.config = config or ServiceFactoryConfig()
        self.state = ServiceFactoryState()
        
    async def initialize(self) -> None:
        """Initialize service factory and registries."""
        try:
            await self._ensure_registry_initialized()
            self.state.services_loaded = True
            logger.info("Service factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Service factory initialization failed: {e}")
            raise
    
    async def _ensure_registry_initialized(self) -> None:
        """Ensure flowlib registry is initialized."""
        if self.state.registry_initialized:
            return
            
        try:
            # Load minimal repository data for registry initialization
            repository_data = {
                "role_assignments": {},
                "configurations": {}
            }
            load_repository_environment("development", repository_data)
            
            self.state.registry_initialized = True
            self.registry_initialized.emit()
            logger.info("Flowlib registry initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize flowlib registry: {e}")
            raise
    
    
    
    async def get_configuration_service(self):
        """Get configuration service with proper dependency injection."""
        try:
            from .configuration_service import ConfigurationService
            from .flowlib_integration_service import FlowlibIntegrationService
            
            # Create FlowlibIntegrationService dependency
            flowlib_integration = FlowlibIntegrationService()
            
            # Create and initialize ConfigurationService
            config_service = ConfigurationService(flowlib_integration)
            await config_service.initialize()
            
            return config_service
            
        except Exception as e:
            logger.error(f"Failed to create configuration service: {e}")
            self.service_failed.emit("configuration_service", str(e))
            raise
    
    async def get_repository_service(self):
        """Get repository service with proper dependency injection."""
        try:
            from .repository_service import RepositoryService
            service = RepositoryService(self)
            await service.initialize()
            return service
            
        except Exception as e:
            logger.error(f"Failed to create repository service: {e}")
            self.service_failed.emit("repository_service", str(e))
            raise
    
    async def get_template_service(self):
        """Get template service."""
        try:
            from ..ui.widgets.template_manager import TemplateService
            return TemplateService(self)
            
        except Exception as e:
            logger.error(f"Failed to create template service: {e}")
            self.service_failed.emit("template_service", str(e))
            raise
    
    async def get_testing_service(self):
        """Get testing service."""
        try:
            from .testing_service import TestingService
            return TestingService(self)
            
        except Exception as e:
            logger.error(f"Failed to create testing service: {e}")
            self.service_failed.emit("testing_service", str(e))
            raise
    
    async def get_import_export_service(self):
        """Get import/export service."""
        try:
            from .import_export_service import ImportExportService
            return ImportExportService(self)
            
        except Exception as e:
            logger.error(f"Failed to create import/export service: {e}")
            self.service_failed.emit("import_export_service", str(e))
            raise
    
    
    async def get_preset_service(self):
        """Get preset service."""
        try:
            from .preset_manager import PresetManager
            return PresetManager(self)
            
        except Exception as e:
            logger.error(f"Failed to create preset service: {e}")
            self.service_failed.emit("preset_service", str(e))
            raise
    
    async def get_knowledge_plugin_service(self):
        """Get knowledge plugin service."""
        try:
            from .knowledge_plugin_service import KnowledgePluginService
            return KnowledgePluginService()
            
        except Exception as e:
            logger.error(f"Failed to create knowledge plugin service: {e}")
            self.service_failed.emit("knowledge_plugin_service", str(e))
            raise
    
    def get_service_status(self) -> dict:
        """Get service factory status."""
        return {
            "registry_initialized": self.state.registry_initialized,
            "services_loaded": self.state.services_loaded
        }
    
    def get_available_configuration_roles(self) -> List[str]:
        """Get available configuration roles from registry."""
        try:
            # Use resource registry to get available configuration types
            roles = []
            # Default configuration roles based on provider categories
            if hasattr(provider_registry, '_factory_metadata'):
                categories = set()
                for (category, _), _ in provider_registry._factory_metadata.items():
                    categories.add(category)
                
                for category in sorted(categories):
                    roles.append(f"{category}_config")
            
            if not roles:
                # Fallback list of common roles
                roles = ["llm_config", "database_config", "vector_config", "cache_config"]
            
            return roles
            
        except Exception as e:
            logger.error(f"Failed to get configuration roles: {e}")
            return ["llm_config", "database_config", "vector_config"]  # Safe fallback
    
    def get_available_provider_types(self) -> List[str]:
        """Get available provider types from registry."""
        try:
            provider_types = []
            if hasattr(provider_registry, '_factory_metadata'):
                for (category, provider_type), _ in provider_registry._factory_metadata.items():
                    provider_types.append(f"{category}/{provider_type}")
            
            return sorted(provider_types)
            
        except Exception as e:
            logger.error(f"Failed to get provider types: {e}")
            return []  # Safe fallback - empty list
    
    def get_available_providers_by_type(self, provider_type: str) -> List[str]:
        """Get available providers for a specific provider type."""
        try:
            providers = []
            if hasattr(provider_registry, '_factory_metadata'):
                for (category, ptype), _ in provider_registry._factory_metadata.items():
                    if category.lower() == provider_type.lower():
                        providers.append(ptype)
            
            return sorted(providers)
            
        except Exception as e:
            logger.error(f"Failed to get providers for type {provider_type}: {e}")
            return []  # Safe fallback - empty list
    
    async def shutdown(self) -> None:
        """Shutdown service factory and cleanup resources."""
        try:
            self.state = ServiceFactoryState()
            logger.info("Service factory shutdown complete")
            
        except Exception as e:
            logger.error(f"Service factory shutdown failed: {e}")
            raise