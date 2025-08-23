"""
Persistence interface definitions.

This module defines the protocols and interfaces for state persistence operations.
"""

from typing import Dict, List, Optional, Protocol
from flowlib.agent.models.state import AgentState


class StatePersistenceInterface(Protocol):
    """Interface for state persistence.
    
    Defines the methods for saving and loading agent state.
    """
    
    async def save_state(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Save agent state.
        
        Args:
            state: Agent state to save
            metadata: Optional metadata to save with the state
            
        Returns:
            True if state was saved successfully
            
        Raises:
            StatePersistenceError: If saving fails
        """
        ...
    
    async def load_state(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        """Load agent state.
        
        Args:
            task_id: Task ID to load state for
            
        Returns:
            Loaded state or None if not found
            
        Raises:
            StatePersistenceError: If loading fails
        """
        ...
    
    async def delete_state(
        self,
        task_id: str
    ) -> bool:
        """Delete agent state.
        
        Args:
            task_id: Task ID to delete state for
            
        Returns:
            True if state was deleted successfully
            
        Raises:
            StatePersistenceError: If deletion fails
        """
        ...
    
    async def list_states(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """List available states.
        
        Args:
            filter_criteria: Optional criteria to filter by
            
        Returns:
            List of state metadata dictionaries
            
        Raises:
            StatePersistenceError: If listing fails
        """
        ... 