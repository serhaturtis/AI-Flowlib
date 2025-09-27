"""
File-based state persistence.

This module provides a simple file-based implementation of the state persister
interface, storing agent states as JSON files on disk.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from flowlib.agent.core.errors import StatePersistenceError
from flowlib.agent.models.state import AgentState
from .base import BaseStatePersister
from flowlib.providers.core.base import ProviderSettings

logger = logging.getLogger(__name__)


class FileStatePersisterSettings(ProviderSettings):
    """Settings specific to the FileStatePersister."""
    directory: str = "./states"


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class FileStatePersister(BaseStatePersister[FileStatePersisterSettings]):
    """File-based implementation of state persistence.
    
    Stores agent states as JSON files in a configurable directory.
    Each state is stored in a separate file named after the task ID.
    """
    def __init__(self, settings: FileStatePersisterSettings):
        if not settings or not getattr(settings, "directory", None):
            raise ValueError("FileStatePersister requires a 'settings' argument with a 'directory' field.")
        super().__init__("file_state_persister", settings=settings)
    
    async def _initialize(self) -> None:
        """Initialize persister by creating the directory if needed."""
        # Create directory if it doesn't exist
        os.makedirs(self.settings.directory, exist_ok=True)
        logger.debug(f"Using state directory: {self.settings.directory}")
    
    async def _save_state_impl(
        self,
        state: AgentState,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Save agent state to file.
        
        Args:
            state: Agent state to save
            metadata: Optional metadata to save with the state
            
        Returns:
            True if state was saved successfully
        """
        try:
            # Get state as dictionary for JSON serialization
            state_dict = state.model_dump()
            
            # Ensure directory exists
            os.makedirs(self.settings.directory, exist_ok=True)
            
            # Create file path
            file_path = os.path.join(self.settings.directory, f"{state.task_id}.json")
            
            # Write to file using custom encoder for datetime objects
            with open(file_path, "w") as f:
                json.dump(state_dict, f, cls=DateTimeEncoder, indent=2)
                
            # Save metadata if provided
            if metadata:
                metadata_path = os.path.join(self.settings.directory, f"{state.task_id}.meta.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, cls=DateTimeEncoder)
                
            logger.debug(f"State saved to {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving state to file: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="save",
                task_id=state.task_id,
                cause=e
            )
    
    async def _load_state_impl(
        self,
        task_id: str
    ) -> Optional[AgentState]:
        """Load agent state from file.
        
        Args:
            task_id: Task ID to load state for
            
        Returns:
            Loaded state or None if not found
        """
        try:
            # Create file path
            file_path = os.path.join(self.settings.directory, f"{task_id}.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"State file not found: {file_path}")
                return None
            
            # Read from file
            with open(file_path, "r") as f:
                state_json = f.read()
                
            # Parse JSON
            state_dict = json.loads(state_json)
            
            # Create AgentState directly with the state data
            # This fixes the 'dict' object has no attribute 'data' error
            # by correctly passing the dictionary as the data parameter
            if "task_description" not in state_dict:
                raise StatePersistenceError(
                    f"Corrupted state file for task {task_id}: missing required 'task_description' field",
                    "load_state"
                )
            task_description = state_dict["task_description"]
            return AgentState(
                task_description=task_description,
                task_id=task_id,
                initial_state_data=state_dict
            )
            
        except Exception as e:
            error_msg = f"Error loading state from file: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="load",
                task_id=task_id,
                cause=e
            )
    
    async def _delete_state_impl(
        self,
        task_id: str
    ) -> bool:
        """Delete agent state file.
        
        Args:
            task_id: Task ID to delete state for
            
        Returns:
            True if state was deleted successfully
        """
        try:
            # Create file paths
            state_path = os.path.join(self.settings.directory, f"{task_id}.json")
            metadata_path = os.path.join(self.settings.directory, f"{task_id}.meta.json")
            
            # Delete state file if it exists
            if os.path.exists(state_path):
                os.remove(state_path)
                
            # Delete metadata file if it exists
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            logger.debug(f"State deleted: {task_id}")
            return True
            
        except Exception as e:
            error_msg = f"Error deleting state file: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="delete",
                task_id=task_id,
                cause=e
            )
    
    async def _list_states_impl(
        self,
        filter_criteria: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """List available states.
        
        Args:
            filter_criteria: Optional criteria to filter by
            
        Returns:
            List of state metadata dictionaries
        """
        try:
            # Get all JSON files in the directory
            state_files = [f for f in os.listdir(self.settings.directory) if f.endswith(".json") and not f.endswith(".meta.json")]
            
            # Extract task IDs
            task_ids = [os.path.splitext(f)[0] for f in state_files]
            
            # Create metadata for each state
            result = []
            for task_id in task_ids:
                # Check if metadata file exists
                metadata_path = os.path.join(self.settings.directory, f"{task_id}.meta.json")
                metadata = {}
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                
                # Add task ID to metadata
                metadata["task_id"] = task_id
                
                # Apply filter criteria if provided
                if filter_criteria:
                    matches = True
                    for key, value in filter_criteria.items():
                        if key not in metadata or metadata[key] != value:
                            matches = False
                            break
                    
                    if not matches:
                        continue
                
                result.append(metadata)
            
            return result
            
        except Exception as e:
            error_msg = f"Error listing state files: {str(e)}"
            logger.error(error_msg)
            raise StatePersistenceError(
                message=error_msg,
                operation="list",
                cause=e
            ) 