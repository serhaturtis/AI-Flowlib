"""
Base implementation for state persistence.

This module provides a base implementation of the state persistence interface
with common functionality that can be extended by specific implementations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, TypeVar

# Import ProviderType constants
# Removed ProviderType import - using config-driven provider access
# Removed BaseComponent import
from flowlib.agent.core.errors import StatePersistenceError
from flowlib.agent.models.state import AgentState

# Import Provider and ProviderSettings
from flowlib.providers.core.base import Provider, ProviderSettings

logger = logging.getLogger(__name__)

# Define a TypeVar for the specific settings class used by concrete persisters
SettingsType = TypeVar('SettingsType', bound=ProviderSettings)

# Updated inheritance: Inherits from Provider, the Interface, and is Generic
class BaseStatePersister(Provider[SettingsType]):
    """Base implementation of state persistence, now Provider-based.
    
    Acts as an abstract base for concrete state persisters, providing Provider lifecycle/settings.
    Subclasses should provide their specific Settings class (subclass of ProviderSettings).
    Example: class MyPersister(BaseStatePersister): ...
    """

    def __init__(self, name: str = "base_state_persister", settings: Optional[SettingsType] = None):
        """Initialize base state persister.
        
        Args:
            name: Component/Provider name
            settings: Provider settings instance matching SettingsType
        """
        # Call Provider.__init__
        # Specify provider_type for categorization in the registry
        super().__init__(name=name, settings=settings, provider_type="state_persister")
        # Logger is now handled by the Provider base class if needed, or use self.name
        # self._logger = logging.getLogger(f"{__name__}.{self.name}")

    # Remove initialize/shutdown/initialized property - handled by Provider base class
    # async def _initialize_impl(self) -> None: ...
    # async def _shutdown_impl(self) -> None: ...
    # @property def initialized(self) -> bool: ...

    # Keep the StatePersistenceInterface methods, ensure they call the _impl methods
    async def save_state(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Save agent state.
        
        This method updates the state's timestamp and delegates to the
        implementation-specific _save_state_impl method.
        """
        if not state.task_id:
            logger.warning(f"Cannot save state without task_id for persister '{self.name}'")
            return False

        # Ensure provider is initialized (Provider base class handles this)
        # if not self.initialized: await self.initialize()

        try:
            # Update the model's timestamp using the proper update method
            state._update_model(updated_at=datetime.now())
            # Use execute_with_retry from Provider base? Or handle retries here?
            # For now, directly call the implementation.
            return await self._save_state_impl(state, metadata)
        except Exception as e:
            # Error handling should be improved, potentially leveraging Provider context
            error_msg = f"Error saving state via '{self.name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StatePersistenceError(
                message=error_msg,
                operation="save",
                task_id=state.task_id,
                cause=e
            )

    async def load_state(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        """Load agent state.
        
        Delegates to the implementation-specific _load_state_impl method.
        """
        # if not self.initialized: await self.initialize()
        try:
            return await self._load_state_impl(task_id)
        except Exception as e:
            error_msg = f"Error loading state '{task_id}' via '{self.name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StatePersistenceError(
                message=error_msg,
                operation="load",
                task_id=task_id,
                cause=e
            )

    async def delete_state(
        self,
        task_id: str
    ) -> bool:
        """Delete agent state.
        
        Delegates to the implementation-specific _delete_state_impl method.
        """
        # if not self.initialized: await self.initialize()
        try:
            return await self._delete_state_impl(task_id)
        except Exception as e:
            error_msg = f"Error deleting state '{task_id}' via '{self.name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StatePersistenceError(
                message=error_msg,
                operation="delete",
                task_id=task_id,
                cause=e
            )

    async def list_states(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """List available states.
        
        Delegates to the implementation-specific _list_states_impl method.
        """
        # if not self.initialized: await self.initialize()
        try:
            return await self._list_states_impl(filter_criteria)
        except Exception as e:
            error_msg = f"Error listing states via '{self.name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise StatePersistenceError(
                message=error_msg,
                operation="list",
                cause=e
            )

    # Keep abstract implementation methods required by StatePersistenceInterface
    # These must be implemented by concrete subclasses (like RedisStatePersister)

    async def _save_state_impl(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        raise NotImplementedError("_save_state_impl must be implemented by subclasses")

    async def _load_state_impl(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        raise NotImplementedError("_load_state_impl must be implemented by subclasses")

    async def _delete_state_impl(
        self,
        task_id: str
    ) -> bool:
        raise NotImplementedError("_delete_state_impl must be implemented by subclasses")

    async def _list_states_impl(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        raise NotImplementedError("_list_states_impl must be implemented by subclasses")
