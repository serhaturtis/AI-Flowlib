from collections.abc import Callable
from typing import Any

from .provider_base import ProviderBase

# Removed ProviderType import - using config-driven provider access
from .registry import provider_registry


def provider(
    name: str, provider_type: str = "llm", *, settings_class: type | None = None, **metadata: Any
) -> Callable[[type], type]:
    """
    Register a class as a provider factory.
    Enforces contract: only ProviderBase subclasses (pydantic v2) can be registered.
    Requires a settings_class argument (Pydantic v2 class) for contract-compliant configuration.
    Fails fast if not provided.
    """
    if settings_class is None:
        raise TypeError(
            f"Provider '{name}' must supply a 'settings_class' argument (Pydantic v2 class)"
        )

    # Validate decorator settings if provided
    if "settings" in metadata:
        settings_value = metadata["settings"]
        if not isinstance(settings_value, (dict, settings_class)):
            raise TypeError(
                f"Decorator-defined settings for '{name}' must be a dict or an instance of {settings_class.__name__}, got {type(settings_value)}"
            )

    def decorator(cls: type) -> type:
        if provider_registry is None:
            raise RuntimeError("Provider registry not initialized")
        # Check contract
        if not isinstance(cls, type) or not issubclass(cls, ProviderBase):
            raise TypeError(
                f"Provider '{name}' must be a ProviderBase subclass (pydantic v2), got {type(cls)}"
            )

        # Create factory function
        def factory(runtime_settings_dict: dict[str, Any] | None = None) -> Any:
            final_settings_arg = None

            # Priority 1: Runtime settings from configuration (e.g., agent_config.yaml)
            if runtime_settings_dict is not None:
                if settings_class:
                    try:
                        final_settings_arg = settings_class(**runtime_settings_dict)
                    except Exception as e:
                        raise ValueError(
                            f"Error parsing runtime_settings for '{name}' with {settings_class.__name__}: {e}. Input: {runtime_settings_dict}"
                        ) from e
                else:
                    # No settings_class defined, but runtime_settings were given. Pass as dict.
                    final_settings_arg = runtime_settings_dict
            else:
                # Priority 2: Settings defined as an argument to the decorator itself
                decorator_arg_settings = metadata.get("settings")
                if decorator_arg_settings is not None:
                    if settings_class and isinstance(decorator_arg_settings, dict):
                        try:
                            final_settings_arg = settings_class(**decorator_arg_settings)
                        except Exception as e:
                            raise ValueError(
                                f"Error parsing decorator-defined settings for '{name}' with {settings_class.__name__}: {e}"
                            ) from e
                    elif (
                        isinstance(
                            decorator_arg_settings, settings_class if settings_class else object
                        )
                        or not settings_class
                    ):
                        final_settings_arg = (
                            decorator_arg_settings  # Already a model or no model to check against
                        )
                    else:
                        raise TypeError(
                            f"Decorator-defined settings for '{name}' for provider '{name}' must be a dict or an instance of {settings_class.__name__}, got {type(decorator_arg_settings)}"
                        )
                # Priority 3: Default instantiation of settings_class if no settings provided from runtime or decorator args
                elif settings_class:
                    try:
                        final_settings_arg = settings_class()  # Use Pydantic model defaults
                    except Exception as e:
                        raise ValueError(
                            f"Error instantiating default settings for '{name}' with {settings_class.__name__}: {e}"
                        ) from e
                else:
                    # No runtime_settings, no decorator_settings, no settings_class. Pass empty dict.
                    final_settings_arg = {}

            # Prepare kwargs for the provider, excluding 'settings' if it was a decorator metadata key, as it's now handled in final_settings_arg
            provider_init_kwargs = {k: v for k, v in metadata.items() if k != "settings"}

            # Ensure logger is available or handle absence
            # logger.debug(f"Provider '{name}': Instantiating with settings from '{source_of_settings}'. Settings object: {type(final_settings_arg)}")
            return cls(
                name=name,
                provider_type=provider_type,
                settings=final_settings_arg,
                **provider_init_kwargs,
            )

        provider_registry.register_factory(
            name=name, factory=factory, provider_type=provider_type, settings_class=settings_class, **metadata
        )
        # Add dynamic attributes to the class with proper type annotation
        cls.__provider_name__ = name  # type: ignore[attr-defined]
        cls.__provider_type__ = provider_type  # type: ignore[attr-defined]
        cls.__provider_metadata__ = metadata  # type: ignore[attr-defined]
        cls.settings_class = settings_class  # type: ignore[attr-defined]

        return cls

    return decorator


# Specialized provider decorators for convenience
def llm_provider(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an LLM provider factory."""
    return provider(name, "llm", **metadata)


def db_provider(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a database provider factory."""
    return provider(name, "database", **metadata)


def vector_db_provider(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a vector database provider factory."""
    return provider(name, "vector_db", **metadata)


def cache_provider(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a cache provider factory."""
    return provider(name, "cache", **metadata)


def storage_provider(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a storage provider factory."""
    return provider(name, "storage", **metadata)


def message_queue_provider(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a message queue provider factory."""
    return provider(name, "message_queue", **metadata)
