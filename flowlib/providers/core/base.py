"""Provider base implementation with enhanced configuration and lifecycle management.

This module provides the foundation for all providers with improved
configuration, initialization, and error handling.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.core.models import StrictBaseModel

from .provider_base import ProviderBase

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='ProviderSettings')  # For settings type


# Changed to StrictBaseModel - enforcing CLAUDE.md principles
class ProviderSettings(StrictBaseModel):
    """Base settings for providers.
    
    Contains only truly common fields that apply to ALL provider types.
    API-specific fields (api_key, api_base, rate limiting) should be added
    directly to provider classes that need them.
    """
    # Inherits strict configuration from StrictBaseModel

    # Common timeout and retry settings (apply to all providers)
    timeout: float = Field(default=300.0, description="Operation timeout in seconds (5 minutes default)")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts on failure")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retry attempts in seconds")

    # Common logging settings
    verbose: bool = Field(default=False, description="Enable verbose logging for debugging")

    # Advanced settings for customization
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom provider-specific settings")

    def merge(self, other: Union['ProviderSettings', Dict[str, Any]]) -> 'ProviderSettings':
        """Merge with another settings object.
        
        Args:
            other: Settings to merge with
            
        Returns:
            New settings instance with merged values
        """
        if isinstance(other, dict):
            # Convert dict to settings
            other_settings = self.__class__(**other)
        else:
            other_settings = other

        # Start with current settings
        merged_dict = self.model_dump()

        # Update with other settings (only non-None values)
        for key, value in other_settings.model_dump().items():
            if value is not None:
                if key == "custom_settings":
                    # Merge custom settings
                    merged_dict["custom_settings"].update(value)
                else:
                    merged_dict[key] = value

        return self.__class__(**merged_dict)

    def with_overrides(self, **kwargs: Any) -> 'ProviderSettings':
        """Create new settings with overrides.
        
        Args:
            **kwargs: Settings to override
            
        Returns:
            New settings instance with overrides
        """
        settings_dict = self.model_dump()
        settings_dict.update(kwargs)
        return self.__class__(**settings_dict)


class RetryConfig(StrictBaseModel):
    """Retry configuration model for provider operations."""

    max_retries: int = Field(description="Maximum number of retry attempts")
    retry_delay_seconds: float = Field(description="Delay between retry attempts in seconds")
    timeout_seconds: Optional[float] = Field(default=None, description="Timeout for individual operations in seconds")

    @classmethod
    def from_settings_and_params(
        cls,
        settings: ProviderSettings,
        retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None
    ) -> 'RetryConfig':
        """Create retry config from settings and optional parameter overrides.

        Args:
            settings: Provider settings to use as base
            retries: Optional override for max_retries
            retry_delay: Optional override for retry delay
            timeout: Optional override for timeout

        Returns:
            Configured RetryConfig instance
        """
        return cls(
            max_retries=retries if retries is not None else settings.max_retries,
            retry_delay_seconds=retry_delay if retry_delay is not None else settings.retry_delay_seconds,
            timeout_seconds=timeout if timeout is not None else settings.timeout
        )


class Provider(ProviderBase[T]):
    """Base class for all providers with enhanced lifecycle management.
    
    This class provides:
    1. Consistent initialization and cleanup pattern
    2. Configuration via settings models
    3. Clean error handling
    4. Asynchronous execution with retry and timeout capabilities
    """
    def __init__(
        self,
        name: str,
        provider_type: str,
        settings: Optional[Any] = None,
        **kwargs: Any
    ):
        # Create default settings if none provided
        if settings is None:
            settings = cast(T, self._default_settings())
        else:
            settings = cast(T, settings)

        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._initialized = False
        self._setup_lock = asyncio.Lock()
        logger.debug(f"Created provider: {name} ({self.provider_type}) with settings: {self.settings}")

    @property
    def initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    def _default_settings(self) -> ProviderSettings:
        """Create default settings instance.
        
        Returns:
            Default settings for this provider
            
        Raises:
            TypeError: If the provider class doesn't have proper settings type information
        """
        # First check if the provider has a settings_class attribute (from decorator)
        if hasattr(self.__class__, 'settings_class') and self.__class__.settings_class:
            settings_instance = self.__class__.settings_class()
            if not isinstance(settings_instance, ProviderSettings):
                raise TypeError(f"Settings class must return ProviderSettings instance, got {type(settings_instance)}")
            return settings_instance

        # Look through the MRO to find a class that inherits from Provider with generic args
        for base in self.__class__.__mro__:
            if hasattr(base, '__orig_bases__'):
                for orig_base in base.__orig_bases__:
                    if hasattr(orig_base, '__origin__') and hasattr(orig_base, '__args__'):
                        # Check if this is Provider[SomeSettings]
                        origin = orig_base.__origin__
                        if (hasattr(origin, '__name__') and
                            origin.__name__ == 'Provider' and
                            orig_base.__args__):
                            settings_type = orig_base.__args__[0]
                            settings_instance = settings_type()
                            if not isinstance(settings_instance, ProviderSettings):
                                raise TypeError(f"Settings type must return ProviderSettings instance, got {type(settings_instance)}")
                            return settings_instance
                        # Check if this is SomeProvider that inherits from Provider[SomeSettings]
                        elif hasattr(origin, '__mro__'):
                            for parent in origin.__mro__:
                                if hasattr(parent, '__orig_bases__'):
                                    for parent_base in parent.__orig_bases__:
                                        if (hasattr(parent_base, '__origin__') and
                                            hasattr(parent_base, '__args__') and
                                            hasattr(parent_base.__origin__, '__name__') and
                                            parent_base.__origin__.__name__ == 'Provider'):
                                            settings_type = parent_base.__args__[0]
                                            settings_instance = settings_type()
                                            if not isinstance(settings_instance, ProviderSettings):
                                                raise TypeError(f"Settings type must return ProviderSettings instance, got {type(settings_instance)}")
                                            return settings_instance

        # If no settings type found, raise helpful error
        raise TypeError(
            f"Provider class {self.__class__.__name__} must specify settings type.\n"
            f"Use either: @provider(settings_class=YourSettings) decorator or "
            f"class {self.__class__.__name__}(Provider[{self.__class__.__name__}Settings])\n"
            f"This enforces single source of truth for provider configuration."
        )

    async def initialize(self) -> None:
        """Initialize the provider.
        
        This method:
        1. Ensures the provider is only initialized once
        2. Provides thread safety with a lock
        3. Standardizes the initialization pattern
        
        Raises:
            ProviderError: If initialization fails
        """
        if self._initialized:
            return

        async with self._setup_lock:
            # Double-checked locking pattern
            if self._initialized:
                return  # type: ignore[unreachable]

            try:
                await self._initialize()
                self._initialized = True
                logger.info(f"Provider '{self.name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize provider '{self.name}': {str(e)}")
                raise ProviderError(
                    message=f"Failed to initialize provider: {str(e)}",
                    context=ErrorContext.create(
                        flow_name="provider_base",
                        error_type="InitializationError",
                        error_location="initialize",
                        component=self.name,
                        operation="provider_initialization"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="initialize",
                        retry_count=0
                    ),
                    cause=e
                )

    async def shutdown(self) -> None:
        """Close provider resources.
        
        This method:
        1. Ensures clean shutdown of provider resources
        2. Only attempts shutdown if previously initialized
        3. Handles shutdown errors gracefully
        """
        if not self._initialized:
            return

        try:
            await self._shutdown()
            self._initialized = False
            logger.info(f"Provider '{self.name}' shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down provider '{self.name}': {str(e)}")
            # We don't re-raise the error to allow graceful shutdown

    async def _initialize(self) -> None:
        """Concrete initialization logic implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize().")

    async def _shutdown(self) -> None:
        """Concrete shutdown logic implemented by subclasses.
        
        Default implementation does nothing.
        """
        pass

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update provider settings using flowlib's functional pattern.
        
        This method creates a new frozen settings instance with the updated values,
        following flowlib's immutability principles. All providers inherit this
        method, ensuring consistent configuration update behavior.
        
        Args:
            new_settings: Dictionary of settings to update
            
        Raises:
            ValueError: If settings validation fails
            TypeError: If settings cannot be merged
        """
        if not new_settings:
            return

        try:
            # Create new settings using the functional pattern
            updated_settings = self.settings.with_overrides(**new_settings)

            # Use object.__setattr__ to bypass frozen constraint on self
            object.__setattr__(self, 'settings', updated_settings)

            logger.debug(f"Updated settings for provider '{self.name}': {new_settings}")

        except Exception as e:
            logger.error(f"Failed to update settings for provider '{self.name}': {e}")
            raise ValueError(f"Failed to update provider settings: {e}") from e

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        *args: Any,
        retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> Any:
        """Execute an operation with retry and timeout handling.
        
        Args:
            operation: Async callable to execute
            *args: Arguments for the operation
            retries: Number of retries (defaults to settings)
            retry_delay: Delay between retries in seconds (defaults to settings)
            timeout: Timeout in seconds (defaults to settings)
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Operation result
            
        Raises:
            ProviderError: If operation fails after retries or times out
        """
        # Create strict retry configuration using Pydantic model
        # Since T is bound to ProviderSettings, this cast is safe
        provider_settings = self.settings if isinstance(self.settings, ProviderSettings) else ProviderSettings()
        retry_config = RetryConfig.from_settings_and_params(
            settings=provider_settings,
            retries=retries,
            retry_delay=retry_delay,
            timeout=timeout
        )

        # Ensure provider is initialized
        if not self._initialized:
            await self.initialize()

        # Execute with retries
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt <= retry_config.max_retries:
            try:
                # Execute with timeout if specified
                if retry_config.timeout_seconds:
                    return await asyncio.wait_for(
                        operation(*args, **kwargs),
                        timeout=retry_config.timeout_seconds
                    )
                else:
                    return await operation(*args, **kwargs)

            except asyncio.TimeoutError as e:
                logger.warning(f"Provider {self.name} operation timed out after {retry_config.timeout_seconds}s")
                last_error = e
                break  # Don't retry on timeout

            except Exception as e:
                attempt += 1
                last_error = e

                if attempt <= retry_config.max_retries:
                    logger.warning(
                        f"Provider {self.name} operation failed (attempt {attempt}/{retry_config.max_retries}): {str(e)}"
                    )
                    # Wait before retrying
                    await asyncio.sleep(retry_config.retry_delay_seconds)
                else:
                    # Max retries reached
                    break

        # If we get here, all retries failed or timed out
        error_msg = f"Provider operation failed after {attempt} attempt(s)"
        logger.error(f"{error_msg}: {str(last_error)}")

        # Create strict error context
        error_context = ErrorContext.create(
            flow_name="provider_operation",
            error_type="ProviderError",
            error_location=f"{self.__class__.__name__}.execute_with_retry",
            component=self.name,
            operation="execute_with_retry"
        )

        provider_context = ProviderErrorContext(
            provider_name=self.name,
            provider_type=self.provider_type,
            operation="execute_with_retry",
            retry_count=attempt
        )

        raise ProviderError(
            message=error_msg,
            context=error_context,
            provider_context=provider_context,
            cause=last_error
        )

    async def test_connection(self) -> Dict[str, Any]:
        """Test the provider connection and return connection status.
        
        This method provides a standardized interface for testing provider connections.
        It attempts to initialize the provider if not already initialized, then calls
        the provider-specific check_connection method.
        
        Returns:
            Dict containing:
                - success: Boolean indicating if connection test passed
                - message: Human-readable message describing the result
                - provider_type: The type of provider being tested
                - provider_name: The name of the provider instance
                - error_details: Optional error details if connection failed
                
        Raises:
            ProviderError: If the provider doesn't implement check_connection
        """
        result = {
            "success": False,
            "message": "Connection test failed",
            "provider_type": self.provider_type,
            "provider_name": self.name,
            "error_details": None
        }

        try:
            # Ensure provider is initialized
            if not self._initialized:
                await self.initialize()

            # Check if provider implements check_connection
            if not hasattr(self, 'check_connection'):
                raise ProviderError(
                    message=f"Provider {self.__class__.__name__} does not implement check_connection method",
                    context=ErrorContext.create(
                        flow_name="provider_base",
                        error_type="NotImplementedError",
                        error_location="test_connection",
                        component=self.name,
                        operation="provider_connection_test"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type=self.provider_type,
                        operation="test_connection",
                        retry_count=0
                    )
                )

            # Call provider-specific connection check
            is_connected = await self.check_connection()

            if is_connected:
                result.update({
                    "success": True,
                    "message": f"Connection to {self.provider_type} provider '{self.name}' successful"
                })
            else:
                result.update({
                    "message": f"Connection to {self.provider_type} provider '{self.name}' failed - connection is not active"
                })

        except Exception as e:
            logger.error(f"Connection test failed for provider {self.name}: {str(e)}")
            result.update({
                "message": f"Connection test failed for {self.provider_type} provider '{self.name}': {str(e)}",
                "error_details": str(e)
            })

        return result
