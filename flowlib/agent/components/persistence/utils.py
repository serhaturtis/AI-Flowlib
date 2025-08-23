"""
Utility functions related to agent state persistence.
"""

import logging
from typing import Dict, List, Optional

# Import factory and base persister type
from .factory import create_state_persister
from .base import BaseStatePersister

logger = logging.getLogger(__name__)


async def list_saved_states_metadata(
    persister_type: str,
    **kwargs
) -> List[Dict[str, str]]:
    """Lists metadata for available saved agent states using the specified persister.

    Creates a temporary persister instance based on the type and config,
    lists the states, and shuts down the persister.

    Args:
        persister_type: The type of persister (e.g., 'file', 'database').
        **kwargs: Configuration arguments required by the specific persister
                  (e.g., base_path='./states' for 'file').

    Returns:
        A list of dictionaries, where each dictionary contains metadata for a saved state.
        Returns an empty list if the persister fails to initialize or list states.
    """
    persister: Optional[BaseStatePersister] = None
    try:
        # Create and initialize the persister
        persister = create_state_persister(persister_type=persister_type, **kwargs)
        await persister.initialize()
        
        # List states using the persister's implementation
        states_metadata = await persister.list_states()
        return states_metadata
        
    except Exception as e:
        logger.error(f"Failed to list saved states using {persister_type} persister: {e}", exc_info=True)
        # Return empty list on error to allow calling code to handle gracefully
        return [] 
    finally:
        # Ensure persister is shut down even if errors occurred
        if persister and persister.initialized:
            try:
                await persister.shutdown()
            except Exception as shutdown_err:
                logger.error(f"Error shutting down temporary persister: {shutdown_err}", exc_info=True) 