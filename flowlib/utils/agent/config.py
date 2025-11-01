"""
Utilities for loading and validating agent configurations.
"""

import logging
import os

import yaml  # type: ignore[import-untyped]

from flowlib.agent.core.errors import ConfigurationError
from flowlib.agent.models.config import AgentConfig

# Removed ProviderType import - using config-driven provider access
from flowlib.providers.core.factory import create_and_initialize_provider
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.models.model_resource import ModelResource
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)


def initialize_resources_from_config(config: AgentConfig) -> None:
    """
    Initialize resources (e.g., model resources) from the agent configuration.
    This function reads the 'resource_config' section and registers resources to the resource registry.
    All contract violations fail fast.

    Args:
        config: The agent configuration
    Raises:
        ConfigurationError: If resource configuration is invalid or contract is violated
    """
    # No fallbacks - explicit attribute check
    if not hasattr(config, "resource_config"):
        logger.warning("No resource configuration found in agent config")
        return

    resource_config = config.resource_config
    if not resource_config:
        logger.warning("No resource configuration found in agent config")
        return
    for resource_type, resources in resource_config.model_dump().items():
        logger.info(f"Initializing resources of type '{resource_type}' from configuration")
        for resource_name, resource_info in resources.items():
            if not isinstance(resource_info, dict):
                raise ConfigurationError(
                    message=f"Resource '{resource_name}' of type '{resource_type}' must be a dict.",
                    config_key=f"resource_config.{resource_type}.{resource_name}",
                )
            if "provider" not in resource_info:
                raise ConfigurationError(
                    message=f"Resource '{resource_name}' of type '{resource_type}' missing required 'provider' field.",
                    config_key=f"resource_config.{resource_type}.{resource_name}",
                )
            if "config" not in resource_info:
                raise ConfigurationError(
                    message=f"Resource '{resource_name}' of type '{resource_type}' missing required 'config' field.",
                    config_key=f"resource_config.{resource_type}.{resource_name}",
                )
            provider = resource_info["provider"]
            config_section = resource_info["config"]
            if not provider or not config_section:
                raise ConfigurationError(
                    message=f"Resource '{resource_name}' of type '{resource_type}' missing required 'provider' or 'config' field.",
                    config_key=f"resource_config.{resource_type}.{resource_name}",
                )
            if resource_registry.contains(resource_name):
                logger.warning(
                    f"Resource '{resource_name}' of type '{resource_type}' already exists. Skipping initialization."
                )
                continue
            resource_registry.register(
                name=resource_name,
                obj=ModelResource(
                    name=resource_name,
                    type="model",
                    provider_type=provider,
                    config=config_section,
                    model_path=None,
                    model_name=resource_name,
                ),
                resource_type=resource_type,
            )
            logger.info(f"Initialized resource '{resource_name}' of type '{resource_type}'")
    logger.info("Resource initialization from configuration complete")


def load_agent_config(filepath: str) -> AgentConfig:
    """Loads agent configuration from a YAML file.

    Reads the specified YAML file, parses its content, and validates it
    against the AgentConfig Pydantic model.

    Args:
        filepath: The path to the YAML configuration file.

    Returns:
        A validated AgentConfig instance.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ConfigurationError: If the file cannot be parsed as YAML or if the
                            content does not conform to the AgentConfig model.
    """
    logger.info(f"Attempting to load agent configuration from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found at: {filepath}")

    try:
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        if not isinstance(config_dict, dict):
            raise ConfigurationError(f"Failed to parse YAML or file is empty: {filepath}")

        # Validate and instantiate AgentConfig using Pydantic
        # Pydantic will raise validation errors if the structure/types are wrong
        loaded_config = AgentConfig(**config_dict)
        logger.info(f"Successfully loaded and validated AgentConfig from {filepath}")
        return loaded_config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{filepath}': {e}", exc_info=True)
        raise ConfigurationError(
            message=f"Error parsing YAML configuration file: {filepath}", cause=e
) from e
    except Exception as e:
        # Catch Pydantic ValidationErrors and other potential issues
        logger.error(f"Error validating configuration from '{filepath}': {e}", exc_info=True)
        raise ConfigurationError(
            message=f"Error validating configuration loaded from file: {filepath}", cause=e
) from e


async def initialize_providers_from_config(config: AgentConfig) -> None:
    """
    Initialize providers from the agent configuration.
    This function extracts provider configurations from the AgentConfig and initializes them using a single Pydantic settings object per provider. All contract violations fail fast.

    Args:
        config: The agent configuration
    Raises:
        ConfigurationError: If provider configuration is invalid or contract is violated
    """
    provider_config = config.provider_config
    if not provider_config:
        logger.warning("No provider configuration found in agent config")
        return

    for provider_type, providers in provider_config.model_dump().items():
        logger.info(f"Initializing providers of type '{provider_type}' from configuration")
        for provider_name, provider_info in providers.items():
            # Each provider must have implementation and settings
            if "implementation" not in provider_info:
                raise ConfigurationError(
                    message=f"Provider '{provider_name}' of type '{provider_type}' missing required 'implementation' field.",
                    config_key=f"provider_config.{provider_type}.{provider_name}",
                )
            if "settings" not in provider_info:
                raise ConfigurationError(
                    message=f"Provider '{provider_name}' of type '{provider_type}' missing required 'settings' field.",
                    config_key=f"provider_config.{provider_type}.{provider_name}",
                )
            implementation = provider_info["implementation"]
            settings_dict = provider_info["settings"]
            if not implementation or not settings_dict:
                raise ConfigurationError(
                    message=f"Provider '{provider_name}' of type '{provider_type}' missing required 'implementation' or 'settings' field.",
                    config_key=f"provider_config.{provider_type}.{provider_name}",
                )
            # Lookup settings class from provider registry metadata
            meta_key = (provider_type, implementation)
            if meta_key in provider_registry._factory_metadata:
                meta = provider_registry._factory_metadata[meta_key]
                settings_class = meta["settings_class"] if "settings_class" in meta else None
            else:
                meta = None
                settings_class = None
            if not settings_class:
                raise ConfigurationError(
                    message=f"No registered settings class for provider '{provider_name}' of type '{provider_type}' (implementation '{implementation}').",
                    config_key=f"provider_config.{provider_type}.{provider_name}.settings",
                )
            if not callable(settings_class):
                raise ConfigurationError(
                    message=f"Settings class for provider '{provider_name}' is not callable: {type(settings_class)}",
                    config_key=f"provider_config.{provider_type}.{provider_name}.settings",
                )
            try:
                settings_obj = settings_class(**settings_dict)
            except Exception as e:
                raise ConfigurationError(
                    message=f"Failed to instantiate settings for provider '{provider_name}' of type '{provider_type}': {e}",
                    config_key=f"provider_config.{provider_type}.{provider_name}.settings",
                    cause=e,
) from e
            # Create and initialize the provider
            try:
                await create_and_initialize_provider(
                    provider_type=provider_type,
                    name=provider_name,
                    implementation=implementation,
                    register=True,
                    settings=settings_obj,
                )
                logger.info(
                    f"Initialized provider '{provider_name}' of type '{provider_type}' with implementation '{implementation}'"
                )
            except Exception as e:
                raise ConfigurationError(
                    message=f"Failed to initialize provider '{provider_name}' of type '{provider_type}': {e}",
                    config_key=f"provider_config.{provider_type}.{provider_name}",
                    cause=e,
) from e
    logger.info("Provider initialization from configuration complete")
