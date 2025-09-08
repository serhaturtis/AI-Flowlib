"""
Agent state management component.

This module handles agent state persistence, loading, and management
operations that were previously in BaseAgent.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import StatePersistenceError, ConfigurationError, NotInitializedError
from flowlib.agent.models.state import AgentState, ConversationTurn
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.agent.components.persistence.factory import create_state_persister

logger = logging.getLogger(__name__)


class AgentStateManager(AgentComponent):
    """Handles agent state management and persistence.
    
    This component is responsible for:
    - State creation and initialization
    - State persistence operations (save/load/delete)
    - State listing and management
    """
    
    def __init__(self, 
                 state_persister: Optional[BaseStatePersister] = None,
                 name: str = "state_manager"):
        """Initialize the state manager.
        
        Args:
            state_persister: Optional state persister
            name: Component name
        """
        super().__init__(name)
        self._state_persister = state_persister
        self._current_state: Optional[AgentState] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the state manager."""
        if self._state_persister:
            await self._state_persister.initialize()
        logger.info("State manager initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the state manager."""
        if self._state_persister:
            await self._state_persister.shutdown()
        logger.info("State manager shutdown")
    
    def setup_persister(self, config: AgentConfig) -> None:
        """Setup state persister from configuration.
        
        Args:
            config: Agent configuration
        """
        if config.state_config and not self._state_persister:
            self._state_persister = create_state_persister(
                persister_type=config.state_config.persistence_type,
                **config.state_config.model_dump(exclude={"persistence_type"})
            )
    
    async def create_state(self, task_description: str = "") -> AgentState:
        """Create a new agent state.
        
        Args:
            task_description: Initial task description
            
        Returns:
            New AgentState instance
        """
        self._current_state = AgentState(task_description=task_description)
        logger.info(f"Created new agent state with task_id: {self._current_state.task_id}")
        return self._current_state
    
    async def load_state(self, task_id: str) -> AgentState:
        """Load agent state from persistence.
        
        Args:
            task_id: Task ID to load
            
        Returns:
            Loaded AgentState
            
        Raises:
            StatePersistenceError: If loading fails
        """
        if not self._state_persister:
            raise StatePersistenceError("No state persister configured", "load", task_id)
        
        try:
            state_data = await self._state_persister.load_state(task_id)
            if state_data is None:
                raise StatePersistenceError(f"No state found for task {task_id}", "load", task_id)
            self._current_state = AgentState(initial_state_data=state_data)
            logger.info(f"Loaded agent state for task_id: {task_id}")
            return self._current_state
        except Exception as e:
            raise StatePersistenceError(f"Failed to load state for task {task_id}: {e}", "load", task_id) from e
    
    async def save_state(self) -> None:
        """Save current agent state.
        
        Raises:
            StatePersistenceError: If saving fails
        """
        if not self._current_state:
            raise StatePersistenceError("No current state to save", "save")
        
        if not self._state_persister:
            raise StatePersistenceError("No state persister configured", "save")
        
        try:
            await self._state_persister.save_state(
                self._current_state,
                self._current_state.model_dump()
            )
            logger.debug(f"Saved state for task_id: {self._current_state.task_id}")
        except Exception as e:
            raise StatePersistenceError(f"Failed to save state: {e}", "save", self._current_state.task_id) from e
    
    async def delete_state(self, task_id: Optional[str] = None) -> None:
        """Delete agent state.
        
        Args:
            task_id: Task ID to delete, or current state if None
            
        Raises:
            StatePersistenceError: If deletion fails
        """
        if not self._state_persister:
            raise StatePersistenceError("No state persister configured", "delete")
        
        target_task_id = task_id or (self._current_state.task_id if self._current_state else None)
        
        if not target_task_id:
            raise StatePersistenceError("No task ID specified for deletion", "delete")
        
        try:
            await self._state_persister.delete_state(target_task_id)
            logger.info(f"Deleted state for task_id: {target_task_id}")
            
            # Clear current state if we deleted it
            if self._current_state and self._current_state.task_id == target_task_id:
                self._current_state = None
        except Exception as e:
            raise StatePersistenceError(f"Failed to delete state for task {target_task_id}: {e}", "delete", target_task_id) from e
    
    async def list_states(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """List available states.
        
        Args:
            filter_criteria: Optional filter criteria
            
        Returns:
            List of state metadata
            
        Raises:
            StatePersistenceError: If listing fails
        """
        if not self._state_persister:
            raise StatePersistenceError("No state persister configured", "list")
        
        try:
            return await self._state_persister.list_states(filter_criteria)
        except Exception as e:
            raise StatePersistenceError(f"Failed to list states: {e}", "list") from e
    
    @property
    def current_state(self) -> Optional[AgentState]:
        """Get the current state.
        
        Returns:
            Current AgentState instance
        """
        return self._current_state
    
    @current_state.setter
    def current_state(self, state: AgentState) -> None:
        """Set the current state.
        
        Args:
            state: New AgentState instance
        """
        self._current_state = state
    
    def should_auto_load(self, config: AgentConfig) -> bool:
        """Check if state should be auto-loaded.
        
        Args:
            config: Agent configuration
            
        Returns:
            True if auto-load is enabled and task_id is provided
        """
        return (config.state_config and 
                config.state_config.auto_load and 
                config.task_id is not None)
    
    def should_auto_save(self, config: AgentConfig) -> bool:
        """Check if state should be auto-saved.
        
        Args:
            config: Agent configuration
            
        Returns:
            True if auto-save is enabled
        """
        return (config.state_config and 
                config.state_config.auto_save)
    
    async def add_conversation_turn(self, turn: ConversationTurn) -> None:
        """Add conversation turn to session history.
        
        Args:
            turn: Conversation turn to add
            
        Raises:
            NotInitializedError: If state not initialized
        """
        if not self._current_state:
            raise NotInitializedError("State not initialized", "add_conversation_turn")
        
        # Get current model
        model = self._current_state.as_model()
        
        # Add turn to session conversation history
        model.session.conversation_history.append(turn)
        
        # Update state with new session
        self._current_state._update_model(session=model.session)
        
    async def get_session_context(self) -> Dict[str, Any]:
        """Get complete session context for tool execution.
        
        Returns:
            Session context dictionary
        """
        if not self._current_state:
            return {
                "session_id": None,
                "conversation_history": [],
                "shared_context": {},
                "working_directory": os.getcwd(),
                "agent_name": None
            }
        
        model = self._current_state.as_model()
        
        # Get agent name from config manager if available
        agent_name = None
        if self._registry:
            config_manager = self._registry.get("config_manager")
            if config_manager and hasattr(config_manager, 'config'):
                agent_name = config_manager.config.name
        
        return {
            "session_id": model.session.session_id,
            "conversation_history": [
                turn.model_dump() for turn in model.session.conversation_history
            ],
            "shared_context": model.session.shared_context,
            "working_directory": model.session.working_directory,
            "agent_name": agent_name,
            "collaborating_agents": model.session.collaborating_agents
        }
    
    async def update_shared_context(self, context_updates: Dict[str, Any]) -> None:
        """Update shared context in session.
        
        Args:
            context_updates: Updates to apply to shared context
            
        Raises:
            NotInitializedError: If state not initialized
        """
        if not self._current_state:
            raise NotInitializedError("State not initialized", "update_shared_context")
        
        model = self._current_state.as_model()
        
        # Update shared context
        model.session.shared_context.update(context_updates)
        
        # Update state
        self._current_state._update_model(session=model.session)