"""
Agent configuration management component.

This module handles agent configuration preparation, validation,
and management operations that were previously in BaseAgent.
"""

import logging
from typing import Any, Dict, Optional, Union

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ConfigurationError
from flowlib.agent.models.config import AgentConfig

logger = logging.getLogger(__name__)


class AgentConfigManager(AgentComponent):
    """Handles agent configuration management.
    
    This component is responsible for:
    - Configuration preparation and validation
    - Configuration merging and defaults
    - Configuration access and management
    """
    
    def __init__(self, name: str = "config_manager"):
        """Initialize the configuration manager.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self._config: Optional[AgentConfig] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the configuration manager."""
        logger.info("Configuration manager initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the configuration manager."""
        logger.info("Configuration manager shutdown")
    
    def prepare_config(self, config: Optional[Union[Dict[str, Any], AgentConfig]] = None) -> AgentConfig:
        """Prepare configuration for the agent.
        
        This is the SINGLE path for configuration construction.
        
        Args:
            config: Configuration dictionary or AgentConfig instance
            
        Returns:
            Prepared AgentConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid or cannot be built
        """
        try:
            # If already an AgentConfig, validate and return
            if isinstance(config, AgentConfig):
                self._config = config
                return config
            
            # If dict, create AgentConfig from dict
            elif isinstance(config, dict):
                self._config = AgentConfig(**config)
                return self._config
            
            # If None, create default config
            elif config is None:
                logger.warning("No config provided, creating default AgentConfig")
                self._config = AgentConfig(
                    name="default_agent",
                    persona="Default helpful assistant",
                    provider_name="llamacpp"
                )
                return self._config
            
            # Invalid config type
            else:
                raise ConfigurationError(f"Invalid config type: {type(config)}. Expected dict or AgentConfig.")
                
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to prepare agent configuration: {e}") from e
    
    @property
    def config(self) -> Optional[AgentConfig]:
        """Get the current configuration.
        
        Returns:
            Current AgentConfig instance
        """
        return self._config
    
    def update_config(self, config: Union[Dict[str, Any], AgentConfig]) -> AgentConfig:
        """Update the current configuration.
        
        Args:
            config: New configuration
            
        Returns:
            Updated AgentConfig instance
        """
        return self.prepare_config(config)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.
        
        Returns:
            Configuration as dictionary
        """
        if not self._config:
            return {}
        return self._config.model_dump()