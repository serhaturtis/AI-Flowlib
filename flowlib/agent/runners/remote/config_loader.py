"""
Configuration loading utilities for remote agent components.
"""

import yaml  # type: ignore[import-untyped]
import logging
import os
from typing import Optional

from .config_models import RemoteConfig

logger = logging.getLogger(__name__)

def load_remote_config(config_path: Optional[str] = None) -> RemoteConfig:
    """
    Loads remote configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. 
                     Defaults to 'remote_config.yaml' in the current directory.

    Returns:
        A RemoteConfig object populated with the settings.
    """
    config_path = config_path or "remote_config.yaml"
    config_data = {}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Loaded remote configuration from: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load or parse config file '{config_path}': {e}", exc_info=True)
            logger.warning("Proceeding with default remote configurations.")
            config_data = {} # Use defaults if file loading fails
    else:
        logger.warning(f"Configuration file not found at '{config_path}'. Using default remote configurations.")
        config_data = {}

    try:
        # Parse loaded data (or empty dict for defaults) into the Pydantic model
        # Pydantic handles default values defined in the models.
        config = RemoteConfig.model_validate(config_data or {})
        return config
    except Exception as e:
        logger.error(f"Failed to validate remote configuration: {e}", exc_info=True)
        logger.warning("Falling back to default remote configurations due to validation error.")
        # Fallback to default model if validation fails
        return RemoteConfig() 