"""Registry Bridge for Configuration Integration.

This module provides a bridge between the GUI configuration system and
the flowlib registry system, enabling runtime configuration loading and
environment switching without restart.
"""

import logging
from typing import Any, cast

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.config_resource import (
    CacheConfigResource,
    DatabaseConfigResource,
    EmbeddingConfigResource,
    GraphDBConfigResource,
    LLMConfigResource,
    MessageQueueConfigResource,
    StorageConfigResource,
    VectorDBConfigResource,
)
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.model_resource import ModelResource
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)


class RegistryBridge:
    """Bridge between GUI configuration system and flowlib registries.

    This class provides the interface for loading configurations from the
    GUI configuration system into the flowlib registry system, enabling
    runtime environment switching and configuration management.
    """

    def __init__(self) -> None:
        """Initialize the registry bridge."""
        # Track environment resources for cleanup
        self._environment_resources: dict[str, list[tuple]] = {}
        self._current_environment: str | None = None

    def load_environment(self, environment: str, repository_data: dict[str, Any]) -> None:
        """Load all configurations for an environment from repository data.

        Args:
            environment: Name of the environment to load
            repository_data: Dictionary containing role assignments and configurations
                Expected format:
                {
                    "role_assignments": {
                        "role_name": "config_id",
                        ...
                    },
                    "configurations": {
                        "config_id": {
                            "type": "llm_config",
                            "provider_type": "llamacpp",
                            "settings": {...}
                        },
                        ...
                    }
                }

        Raises:
            ValueError: If repository data format is invalid
            TypeError: If configuration types are not supported
        """
        if "role_assignments" not in repository_data or "configurations" not in repository_data:
            raise ValueError("Repository data must contain 'role_assignments' and 'configurations'")

        # Clear existing environment
        if self._current_environment:
            self.clear_environment(self._current_environment)

        role_assignments = repository_data["role_assignments"]
        configurations = repository_data["configurations"]

        logger.info(
            f"Loading environment '{environment}' with {len(role_assignments)} role assignments"
        )

        # Load each role assignment
        for role_name, config_id in role_assignments.items():
            if config_id not in configurations:
                logger.warning(f"Configuration '{config_id}' not found for role '{role_name}'")
                continue

            config_data = configurations[config_id]
            self._register_config_from_data(environment, role_name, config_id, config_data)

        self._current_environment = environment
        logger.info(f"Successfully loaded environment '{environment}'")

    def _register_config_from_data(
        self, environment: str, role_name: str, canonical_name: str, config_data: dict[str, Any]
    ) -> None:
        """Register a configuration from repository data with proper role assignment.

        Args:
            environment: Environment name for tracking
            role_name: Role name for the configuration (e.g., "knowledge-extraction")
            canonical_name: Canonical config name (e.g., "my-phi4-model")
            config_data: Configuration data from repository
        """
        try:
            if "type" not in config_data:
                raise ValueError(
                    f"Configuration for role '{role_name}' missing required 'type' field"
                )
            config_type = config_data["type"]

            # Create resource object based on type
            resource_obj = self._create_resource_from_config(config_data, canonical_name)

            # Determine resource type for registry
            resource_type = self._map_config_type_to_resource_type(config_type)

            # Determine aliases - if role_name differs from canonical_name, it's an alias
            aliases = []
            if role_name != canonical_name:
                aliases = [role_name]

            # Register with resource registry using canonical name and role alias
            resource_registry.register_with_aliases(
                canonical_name=canonical_name,
                obj=resource_obj,
                aliases=aliases,
                resource_type=resource_type,
            )

            # Track both canonical and alias for environment cleanup
            canonical_key = (resource_type, canonical_name)
            if environment not in self._environment_resources:
                self._environment_resources[environment] = []
            self._environment_resources[environment].append(canonical_key)

            if aliases:
                # Also track alias keys for cleanup (they use separate storage)
                for alias in aliases:
                    alias_key = (resource_type, alias)
                    self._environment_resources[environment].append(alias_key)

            logger.debug(
                f"Registered {config_type} configuration '{canonical_name}' with role '{role_name}' for environment '{environment}'"
            )

        except Exception as e:
            logger.error(f"Failed to register configuration '{role_name}': {e}")
            raise

    def _create_resource_from_config(self, config_data: dict[str, Any], canonical_name: str) -> Any:
        """Create a resource object from configuration data.

        Args:
            config_data: Configuration data from repository
            canonical_name: Canonical name for the resource

        Returns:
            Resource object instance

        Raises:
            ValueError: If configuration type is not supported
        """
        config_type = config_data["type"]
        if "provider_type" not in config_data:
            raise KeyError("Required 'provider_type' missing from config_data")
        provider_type = config_data["provider_type"]
        if "settings" not in config_data:
            raise KeyError("Required 'settings' missing from config_data")
        settings = config_data["settings"]

        # If we have configuration file content, extract settings from it
        content = config_data["content"] if "content" in config_data else None
        if content and content.strip():
            try:
                extracted_settings = self._extract_settings_from_content(content, config_type)
                if extracted_settings:
                    # Use extracted settings, but allow explicit settings to override
                    settings = {**extracted_settings, **settings}
                    logger.debug(
                        f"Extracted settings from content for {config_type}: {list(extracted_settings.keys())}"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract settings from content for {config_type}: {e}")
                # Fall back to using just the settings dict

        # UNIFIED SETTINGS EXTRACTION: All provider types use the same pattern
        # Extract nested settings dict if present (from AST parsing with Field(default_factory=lambda: {...}))
        nested_settings = settings["settings"] if "settings" in settings else {}
        # Merge with any top-level settings, but nested settings take precedence
        final_settings = {
            **{k: v for k, v in settings.items() if k not in ["provider_type", "settings"]},
            **nested_settings,
        }

        # Create appropriate resource type - all use unified constructor pattern
        if config_type == "llm_config":
            return LLMConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
                n_threads=final_settings.get("n_threads"),
                n_batch=final_settings.get("n_batch"),
                use_gpu=final_settings.get("use_gpu"),
                n_gpu_layers=final_settings.get("n_gpu_layers"),
                chat_format=final_settings.get("chat_format"),
                verbose=final_settings.get("verbose"),
                timeout=final_settings.get("timeout"),
                max_concurrent_models=final_settings.get("max_concurrent_models"),
            )
        elif config_type == "model_config":
            # Model configuration - dynamically determine the correct model class
            # Remove provider_type from settings to avoid duplicate if it exists
            model_settings = {
                k: v for k, v in settings.items() if k not in ["provider_type", "name", "type"]
            }

            # Use provider_type to determine the correct model class dynamically
            provider_type = settings.get("provider_type", "unknown")
            model_class = self._get_model_class_for_provider(provider_type)

            if model_class:
                # Create model instance using the proper model class
                try:
                    model_instance = model_class(**model_settings)
                    # Register as ResourceBase
                    resource_registry.register(
                        name=canonical_name, obj=model_instance, resource_type=ResourceType.MODEL
                    )
                except Exception as e:
                    logger.warning(f"Failed to create model instance for {canonical_name}: {e}")
                    # Fallback to ModelResource wrapper
                    fallback_model = ModelResource(
                        name=canonical_name,
                        type=ResourceType.MODEL,
                        provider_type=provider_type,
                        model_path=model_settings.get("model_path"),
                        model_name=model_settings.get("model_name", canonical_name),
                        chat_format=model_settings.get("chat_format"),
                        config=model_settings,
                    )
                    resource_registry.register(
                        name=canonical_name, obj=fallback_model, resource_type=ResourceType.MODEL
                    )
            else:
                # Unknown provider type - use generic ModelResource wrapper
                generic_model = ModelResource(
                    name=canonical_name,
                    type=ResourceType.MODEL,
                    provider_type=provider_type,
                    model_path=model_settings.get("model_path"),
                    model_name=model_settings.get("model_name", canonical_name),
                    chat_format=model_settings.get("chat_format"),
                    config=model_settings,
                )
                resource_registry.register(
                    name=canonical_name, obj=generic_model, resource_type=ResourceType.MODEL
                )
        elif config_type == "database_config":
            return DatabaseConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        elif config_type == "vector_db_config":
            return VectorDBConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        elif config_type == "cache_config":
            return CacheConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        elif config_type == "storage_config":
            return StorageConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        elif config_type == "embedding_config":
            return EmbeddingConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        elif config_type == "graph_db_config":
            return GraphDBConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        elif config_type == "message_queue_config":
            return MessageQueueConfigResource(
                name=canonical_name,
                type=config_type,
                provider_type=provider_type,
                settings=final_settings,  # Unified: always pass as settings dict
            )
        else:
            raise ValueError(f"Unsupported configuration type: {config_type}")

    def _get_model_class_for_provider(self, provider_type: str) -> type | None:
        """Get the appropriate model class for a provider type.

        Args:
            provider_type: The provider type (e.g., 'llamacpp', 'google_ai')

        Returns:
            Model class or None if not found
        """
        # Dynamic mapping of provider types to their model classes
        provider_to_model_class = {
            "llamacpp": "flowlib.providers.llm.models.LlamaModelConfig",
            "google_ai": "flowlib.providers.llm.models.GoogleAIModelConfig",
            # Add more mappings as needed
        }

        class_path = provider_to_model_class.get(provider_type)
        if not class_path:
            return None

        try:
            # Dynamically import the model class
            module_path, class_name = class_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            result = getattr(module, class_name)
            return cast(type | None, result)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to import model class for provider '{provider_type}': {e}")
            return None

    def _extract_settings_from_content(self, content: str, config_type: str) -> dict[str, Any]:
        """Extract configuration settings from Python file content.

        Args:
            content: Python configuration file content
            config_type: Type of configuration

        Returns:
            Dictionary of extracted settings
        """
        import ast

        settings = {}

        try:
            # Parse the Python content
            parsed = ast.parse(content)

            # Find the configuration class
            for node in ast.walk(parsed):
                if isinstance(node, ast.ClassDef):
                    # Look for class with proper decorator
                    has_config_decorator = False
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                            decorator_name = decorator.func.id
                            expected_decorators = {
                                "llm_config": "llm_config",
                                "model_config": "model_config",
                                "vector_db_config": "vector_db_config",
                                "embedding_config": "embedding_config",
                            }
                            if decorator_name in expected_decorators.values():
                                has_config_decorator = True
                                break

                    if not has_config_decorator:
                        continue

                    # Extract field definitions from the class
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            field_name = item.target.id

                            # Extract default value if present
                            if item.value:
                                try:
                                    if isinstance(item.value, ast.Call):
                                        # Handle Field() calls
                                        if (
                                            isinstance(item.value.func, ast.Name)
                                            and item.value.func.id == "Field"
                                        ):
                                            # Look for default value as first positional arg
                                            if item.value.args:
                                                settings[field_name] = ast.literal_eval(
                                                    item.value.args[0]
                                                )

                                            # Look for default_factory in keywords
                                            for keyword in item.value.keywords:
                                                if keyword.arg == "default_factory":
                                                    if isinstance(keyword.value, ast.Lambda):
                                                        # Handle lambda: {...} for dict defaults
                                                        if isinstance(
                                                            keyword.value.body, (ast.Dict, ast.Call)
                                                        ):
                                                            try:
                                                                default_val = ast.literal_eval(
                                                                    keyword.value.body
                                                                )
                                                                settings[field_name] = default_val
                                                            except (
                                                                ValueError,
                                                                TypeError,
                                                                SyntaxError,
                                                            ) as eval_error:
                                                                # Skip complex default values that can't be statically evaluated
                                                                logger.debug(
                                                                    f"Could not evaluate default value for field '{field_name}': {eval_error}"
                                                                )
                                    else:
                                        # Direct value assignment
                                        settings[field_name] = ast.literal_eval(item.value)

                                except (ValueError, TypeError):
                                    # Can't evaluate the value, skip it
                                    pass

            # For model configs, keep config dict separate (don't merge into main settings)
            # The config dict should be passed as-is to ModelResource.config field

            logger.debug(
                f"Extracted {len(settings)} settings from content: {list(settings.keys())}"
            )
            return settings

        except Exception as e:
            logger.warning(f"Failed to parse configuration content: {e}")
            return {}

    def _map_config_type_to_resource_type(self, config_type: str) -> str:
        """Map configuration type to resource type.

        Args:
            config_type: Configuration type from repository

        Returns:
            Resource type for registry
        """
        mapping = {
            "llm_config": ResourceType.LLM_CONFIG,
            "model_config": ResourceType.MODEL_CONFIG,
            "database_config": ResourceType.DATABASE_CONFIG,
            "vector_db_config": ResourceType.VECTOR_DB_CONFIG,
            "cache_config": ResourceType.CACHE_CONFIG,
            "storage_config": ResourceType.STORAGE_CONFIG,
            "embedding_config": ResourceType.EMBEDDING_CONFIG,
            "graph_db_config": ResourceType.GRAPH_DB_CONFIG,
            "message_queue_config": ResourceType.MESSAGE_QUEUE_CONFIG,
        }

        if config_type not in mapping:
            raise ValueError(f"Unknown config type: {config_type}")

        return mapping[config_type]

    def clear_environment(self, environment: str) -> None:
        """Clear resources registered for a specific environment.

        Args:
            environment: Name of the environment to clear
        """
        if environment not in self._environment_resources:
            logger.debug(f"No resources found for environment '{environment}'")
            return

        keys_to_clear = self._environment_resources[environment]

        for key in keys_to_clear:
            resource_type, resource_name = key
            try:
                # Remove from resource registry
                resource_registry.remove(resource_name)
                logger.debug(f"Removed resource '{resource_name}' from environment '{environment}'")
            except Exception as e:
                logger.warning(f"Failed to remove resource '{resource_name}': {e}")

        # Clear environment tracking
        del self._environment_resources[environment]

        if self._current_environment == environment:
            self._current_environment = None

        logger.info(f"Cleared environment '{environment}'")

    def switch_environment(self, new_environment: str, repository_data: dict[str, Any]) -> None:
        """Switch to a different environment.

        Args:
            new_environment: Name of the new environment
            repository_data: Repository data for the new environment
        """
        logger.info(f"Switching from '{self._current_environment}' to '{new_environment}'")

        # Clear current environment
        if self._current_environment:
            self.clear_environment(self._current_environment)

        # Load new environment
        self.load_environment(new_environment, repository_data)

    def get_current_environment(self) -> str | None:
        """Get the currently loaded environment.

        Returns:
            Name of the current environment, or None if no environment is loaded
        """
        return self._current_environment

    def list_environment_resources(self, environment: str) -> list[str]:
        """List resources registered for an environment.

        Args:
            environment: Name of the environment

        Returns:
            List of resource names for the environment
        """
        if environment not in self._environment_resources:
            return []

        return [name for _, name in self._environment_resources[environment]]

    async def register_configuration(
        self,
        config_id: str,
        config_type: str,
        provider_type: str,
        settings: dict[str, Any],
        environment: str,
    ) -> None:
        """Register a configuration with the flowlib registry.

        Args:
            config_id: Unique configuration identifier
            config_type: Type of configuration (llm_config, model_config, etc.)
            provider_type: Provider type (llamacpp, postgresql, etc.)
            settings: Configuration settings
            environment: Environment name
        """
        try:
            config_data = {
                "type": config_type,
                "provider_type": provider_type,
                "settings": settings,
            }

            # Additional validation for model configurations
            if config_type == "model_config":
                await self._validate_model_config_registration(config_id, config_data, environment)

            # Use config_type as role_name and config_id as canonical_name
            role_name = config_type.replace("_", "-")
            self._register_config_from_data(environment, role_name, config_id, config_data)
            logger.info(
                f"Registered {config_type} configuration {config_id} for environment {environment}"
            )

        except Exception as e:
            logger.error(f"Failed to register configuration {config_id}: {e}")
            raise

    async def _validate_model_config_registration(
        self, config_id: str, config_data: dict[str, Any], environment: str
    ) -> None:
        """Validate model configuration before registration.

        Args:
            config_id: Configuration ID
            config_data: Configuration data
            environment: Environment name
        """
        if "provider_type" not in config_data:
            raise ValueError(f"Model configuration '{config_id}' must specify provider_type")
        provider_type = config_data["provider_type"]

        # Check if required model settings are present
        settings = config_data["settings"] if "settings" in config_data else {}
        if "model_path" not in settings and "model_name" not in settings:
            logger.warning(
                f"Model configuration '{config_id}' should specify either model_path or model_name"
            )

        logger.debug(
            f"Validated model configuration '{config_id}' for provider type '{provider_type}'"
        )

    def get_environment_configurations(self, environment: str) -> dict[str, dict[str, Any]]:
        """Get all configurations for a specific environment.

        Args:
            environment: Environment name

        Returns:
            Dictionary mapping config names to their data, categorized by provider and model configs
        """
        configurations: dict[str, dict[str, Any]] = {}

        if environment not in self._environment_resources:
            return configurations

        # Get all resource names for this environment
        for resource_type, resource_name in self._environment_resources[environment]:
            try:
                # Get the resource from registry
                resource = resource_registry.get(resource_name)
                if resource:
                    # Convert resource back to configuration data
                    config_data = {
                        "type": resource.type if hasattr(resource, "type") else resource_type,
                        "provider_type": resource.provider_type
                        if hasattr(resource, "provider_type")
                        else "unknown",
                        "settings": self._extract_settings_from_resource(resource),
                    }

                    # Add category for better organization
                    if resource_type == "model_config":
                        config_data["category"] = "model"
                    elif resource_type in [
                        "llm_config",
                        "database_config",
                        "vector_db_config",
                        "cache_config",
                        "storage_config",
                        "embedding_config",
                        "graph_db_config",
                        "message_queue_config",
                    ]:
                        config_data["category"] = "provider"
                    else:
                        config_data["category"] = "other"

                    configurations[resource_name] = config_data

            except Exception as e:
                logger.warning(f"Failed to get configuration for {resource_name}: {e}")

        return configurations

    def _extract_settings_from_resource(self, resource: ResourceBase) -> dict[str, Any]:
        """Extract settings from a resource object.

        Args:
            resource: Resource object

        Returns:
            Settings dictionary
        """
        settings = {}

        try:
            # Get all attributes except standard pydantic/metadata fields
            skip_fields = {"name", "type", "provider_type", "__dict__", "__class__"}

            if hasattr(resource, "model_dump"):
                # Pydantic v2 model
                data = resource.model_dump()
                settings = {k: v for k, v in data.items() if k not in skip_fields and v is not None}
            elif hasattr(resource, "__dict__"):
                # Regular object
                settings = {
                    k: v
                    for k, v in resource.__dict__.items()
                    if k not in skip_fields and not k.startswith("_")
                }

            # Special handling for model resources
            if hasattr(resource, "get_model_settings"):
                # ModelResource has a specialized method
                model_settings = resource.get_model_settings()
                settings.update(model_settings)

        except Exception as e:
            logger.warning(f"Failed to extract settings from resource: {e}")

        return settings

    async def update_role_assignments(self, environment: str, assignments: dict[str, Any]) -> None:
        """Update role assignments for an environment.

        Args:
            environment: Environment name
            assignments: Dictionary of role assignments
        """
        try:
            # Clear existing assignments for this environment
            self.clear_environment(environment)

            # Rebuild assignment data in expected format
            repository_data = {
                "role_assignments": {
                    role_id: assignment.config_id for role_id, assignment in assignments.items()
                },
                "configurations": {},
            }

            # For each assignment, we need the configuration data
            for role_id, assignment in assignments.items():
                config_id = assignment.config_id
                # This would need to be loaded from somewhere - for now create minimal data
                # Try to determine if this is a provider or model config based on role_id
                config_type = "model_config" if "model" in role_id.lower() else "llm_config"
                repository_data["configurations"][config_id] = {
                    "type": config_type,
                    "provider_type": "unknown",
                    "settings": {},
                }

            # Load the environment with new assignments
            self.load_environment(environment, repository_data)

            logger.info(f"Updated role assignments for environment {environment}")

        except Exception as e:
            logger.error(f"Failed to update role assignments for {environment}: {e}")
            raise

    def get_role_assignments(self, environment: str) -> dict[str, str]:
        """Get role assignments for an environment.

        Args:
            environment: Environment name

        Returns:
            Dictionary mapping role names to config IDs
        """
        assignments: dict[str, str] = {}

        if environment not in self._environment_resources:
            return assignments

        # Convert resource tracking back to role assignments
        for _resource_type, resource_name in self._environment_resources[environment]:
            assignments[resource_name] = resource_name  # In this context, role name = config ID

        return assignments

    def validate_repository_data(self, repository_data: dict[str, Any]) -> bool:
        """Validate repository data format.

        Args:
            repository_data: Repository data to validate

        Returns:
            True if data is valid, False otherwise
        """
        try:
            if "role_assignments" not in repository_data or "configurations" not in repository_data:
                return False

            role_assignments = repository_data["role_assignments"]
            configurations = repository_data["configurations"]

            if not isinstance(role_assignments, dict) or not isinstance(configurations, dict):
                return False

            # Validate that all assigned configs exist
            for role_name, config_id in role_assignments.items():
                if config_id not in configurations:
                    logger.warning(
                        f"Configuration '{config_id}' referenced by role '{role_name}' not found"
                    )
                    return False

            # Validate configuration format
            for config_id, config_data in configurations.items():
                if not isinstance(config_data, dict):
                    return False

                if "type" not in config_data:
                    logger.warning(f"Configuration '{config_id}' missing 'type' field")
                    return False

            # Additional validation for model configurations
            if not self._validate_model_provider_references(configurations):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating repository data: {e}")
            return False

    def _validate_model_provider_references(
        self, configurations: dict[str, dict[str, Any]]
    ) -> bool:
        """Validate that model configurations reference valid provider configurations.

        Args:
            configurations: Dictionary of configurations to validate

        Returns:
            True if all model configurations have valid provider references
        """
        try:
            # Get all provider configurations
            provider_configs = {
                config_id: config_data
                for config_id, config_data in configurations.items()
                if (
                    "type" in config_data
                    and config_data["type"]
                    in [
                        "llm_config",
                        "database_config",
                        "vector_db_config",
                        "cache_config",
                        "storage_config",
                        "embedding_config",
                        "graph_db_config",
                        "message_queue_config",
                    ]
                )
            }

            # Get all model configurations
            model_configs = {
                config_id: config_data
                for config_id, config_data in configurations.items()
                if ("type" in config_data and config_data["type"] == "model_config")
            }

            # Validate each model configuration
            for model_id, model_data in model_configs.items():
                if "provider_type" not in model_data:
                    logger.warning(
                        f"Model configuration '{model_id}' missing required 'provider_type' field"
                    )
                    continue
                provider_type = model_data["provider_type"]

                # Check if there's a provider configuration with matching provider_type
                matching_provider = None
                for provider_id, provider_data in provider_configs.items():
                    # Fail-fast validation - provider config must have provider_type
                    if "provider_type" not in provider_data:
                        raise ValueError(
                            f"Provider configuration '{provider_id}' missing required 'provider_type' field"
                        )
                    if provider_data["provider_type"] == provider_type:
                        matching_provider = provider_id
                        break

                if not matching_provider:
                    logger.warning(
                        f"Model configuration '{model_id}' references provider_type '{provider_type}' "
                        f"but no matching provider configuration found"
                    )
                    # This is a warning, not an error - models can reference providers registered elsewhere
                    continue

                logger.debug(
                    f"Model '{model_id}' references provider '{matching_provider}' (type: {provider_type})"
                )

            return True

        except Exception as e:
            logger.error(f"Error validating model-provider references: {e}")
            return False


# Global registry bridge instance
registry_bridge = RegistryBridge()


def load_repository_environment(environment: str, repository_data: dict[str, Any]) -> None:
    """Convenience function to load an environment from repository data.

    Args:
        environment: Name of the environment
        repository_data: Repository data to load
    """
    registry_bridge.load_environment(environment, repository_data)


def switch_repository_environment(new_environment: str, repository_data: dict[str, Any]) -> None:
    """Convenience function to switch environments.

    Args:
        new_environment: Name of the new environment
        repository_data: Repository data for the new environment
    """
    registry_bridge.switch_environment(new_environment, repository_data)


def clear_current_environment() -> None:
    """Convenience function to clear the current environment."""
    current = registry_bridge.get_current_environment()
    if current:
        registry_bridge.clear_environment(current)
