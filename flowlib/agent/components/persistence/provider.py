"""
Provider-based state persistence.

This module provides a provider-based implementation of the state persister
interface, allowing for flexible storage backends.
"""

import logging
from typing import cast

from pydantic import Field

from flowlib.agent.core.errors import StatePersistenceError
from flowlib.agent.models.state import AgentState
from flowlib.providers.core.base import ProviderSettings

from .base import BaseStatePersister

logger = logging.getLogger(__name__)


class ProviderStatePersisterSettings(ProviderSettings):
    """Settings for ProviderStatePersister."""

    provider_name: str = Field(..., min_length=1, description="Name of the provider to use")


class ProviderStatePersister(BaseStatePersister[ProviderStatePersisterSettings]):
    """Provider-based implementation of state persistence.

    Uses a registered provider to store and retrieve agent states.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        name: str = "provider_state_persister",
        settings: ProviderStatePersisterSettings | None = None,
    ):
        """Initialize provider state persister.

        Args:
            provider_name: Name of the provider to use (for backward compatibility)
            name: Name of this persister instance
            settings: Settings instance or will be created from provider_name
        """
        # Handle backward compatibility
        if settings is None and provider_name:
            settings = ProviderStatePersisterSettings(provider_name=provider_name)
        elif settings is None:
            raise ValueError("Either provider_name or settings must be provided")

        super().__init__(name=name, settings=settings)
        self._provider: BaseStatePersister | None = None

    async def _initialize(self) -> None:
        """Initialize persister by getting the provider from registry."""
        try:
            from flowlib.providers.core.registry import provider_registry

            provider_name = self.settings.provider_name
            # For now, use a default config approach - this may need to be updated
            # based on the actual provider registry API for named providers
            provider = await provider_registry.get_by_config(provider_name)
            if not provider:
                raise StatePersistenceError(
                    message=f"Provider not found: {provider_name}", operation="initialize"
                )
            self._provider = cast(BaseStatePersister, provider)
            logger.debug(f"Using provider: {provider_name}")
        except Exception as e:
            error_msg = f"Error initializing provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(message=error_msg, operation="initialize", cause=e) from e

    async def _save_state_impl(
        self, state: AgentState, metadata: dict[str, str] | None = None
    ) -> bool:
        """Save agent state using provider.

        Args:
            state: Agent state to save
            metadata: Optional metadata to save with the state

        Returns:
            True if state was saved successfully
        """
        try:
            if self._provider is None:
                raise RuntimeError(f"{self.name} not initialized - provider is None")

            # Save state directly - the provider handles conversion
            await self._provider.save_state(state, metadata)

            logger.debug(f"State saved using provider: {state.task_id}")
            return True

        except Exception as e:
            error_msg = f"Error saving state using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg, operation="save", task_id=state.task_id, cause=e
) from e

    async def _load_state_impl(self, task_id: str) -> AgentState | None:
        """Load agent state using provider.

        Args:
            task_id: Task ID to load state for

        Returns:
            Loaded state or None if not found
        """
        try:
            if self._provider is None:
                raise RuntimeError(f"{self.name} not initialized - provider is None")

            # Load state from provider
            # Load state directly - the provider returns AgentState
            state = await self._provider.load_state(task_id)

            if not state:
                logger.warning(f"State not found: {task_id}")
                return None

            return state

        except Exception as e:
            error_msg = f"Error loading state using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg, operation="load", task_id=task_id, cause=e
) from e

    async def _delete_state_impl(self, task_id: str) -> bool:
        """Delete agent state using provider.

        Args:
            task_id: Task ID to delete state for

        Returns:
            True if state was deleted successfully
        """
        try:
            if self._provider is None:
                raise RuntimeError(f"{self.name} not initialized - provider is None")

            # Delete state from provider
            await self._provider.delete_state(task_id)

            logger.debug(f"State deleted using provider: {task_id}")
            return True

        except Exception as e:
            error_msg = f"Error deleting state using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg, operation="delete", task_id=task_id, cause=e
) from e

    async def _list_states_impl(
        self, filter_criteria: dict[str, str] | None = None
    ) -> list[dict[str, str]]:
        """List available states using provider.

        Args:
            filter_criteria: Optional criteria to filter by

        Returns:
            List of state metadata dictionaries
        """
        try:
            if self._provider is None:
                raise RuntimeError(f"{self.name} not initialized - provider is None")

            # List states from provider
            result = await self._provider.list_states(filter_criteria)

            # Type validation following flowlib's no-fallbacks principle
            if not isinstance(result, list):
                raise StatePersistenceError(
                    message=f"Provider returned invalid type: expected list, got {type(result)}",
                    operation="list",
                    context={"filter_criteria": filter_criteria},
                )

            # Validate each item is a dict with string keys and values
            validated_result: list[dict[str, str]] = []
            for item in result:
                if not isinstance(item, dict):
                    raise StatePersistenceError(
                        message=f"Provider returned invalid item type: expected dict, got {type(item)}",
                        operation="list",
                        context={"filter_criteria": filter_criteria},
                    )
                # Convert all values to strings for consistency
                validated_item = {str(k): str(v) for k, v in item.items()}
                validated_result.append(validated_item)

            return validated_result

        except Exception as e:
            error_msg = f"Error listing states using provider: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(message=error_msg, operation="list", cause=e) from e
