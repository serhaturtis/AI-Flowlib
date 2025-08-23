"""
State Persister adapter implementations that leverage existing DB/Cache providers
for use in distributed agent scenarios.
"""

import logging
from typing import Optional, Dict, List

from pydantic import Field
from flowlib.core.models import StrictBaseModel
from typing import Union
# Import Provider base class
from flowlib.providers.core.base import Provider

from .base import BaseStatePersister
from flowlib.agent.models.state import AgentState

# Import provider types
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.cache.base import CacheProvider
from flowlib.providers.db.base import DBProvider
from flowlib.providers.graph.base import GraphDBProvider
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.core.decorators import provider

# Import correct error classes from core.errors instead of non-existent exceptions module
from ...core.errors import ProviderError, ErrorContext

logger = logging.getLogger(__name__)


class StateTTLInfo(StrictBaseModel):
    """State TTL information model."""
    
    ttl_seconds: Optional[int] = Field(default=None, description="TTL in seconds for state storage")
    
    @classmethod
    def extract_from_state(cls, state: 'AgentState') -> 'StateTTLInfo':
        """Extract TTL information from agent state.
        
        Args:
            state: Agent state to extract TTL from
            
        Returns:
            TTL information model
        """
        ttl_seconds = None
        
        # Explicit path checking without fallbacks
        if (state.config and 
            state.config.state_config and 
            hasattr(state.config.state_config, 'ttl_seconds')):
            ttl_seconds = state.config.state_config.ttl_seconds
            
        return cls(ttl_seconds=ttl_seconds)


# --- Redis State Persister ---

# No environment variables
class RedisStatePersisterSettings(StrictBaseModel):
    """Settings specific to the RedisStatePersister."""
    
    redis_provider_name: str = Field(..., min_length=1, description="The registered name of the RedisProvider instance to use.")
    key_prefix: str = Field("agent_state:", description="Prefix for keys in Redis where state is stored.")

@provider(name="redis", provider_type="state_persister", settings_class=RedisStatePersisterSettings)
# Corrected inheritance: Only inherit from BaseStatePersister (which is a Provider)
class RedisStatePersister(BaseStatePersister[RedisStatePersisterSettings]): 
    """State persister using a Redis cache provider as the backend."""

    # __init__ should call super(), which calls BaseStatePersister/Provider init
    def __init__(self, name: str, settings: RedisStatePersisterSettings):
        super().__init__(name=name, settings=settings)
        self._redis_provider: Optional[CacheProvider] = None

    # Change initialize/shutdown to match Provider base class pattern (_initialize / _shutdown)
    async def _initialize(self):
        # Removed redundant check for self.initialized, Provider base handles this
        try:
            logger.debug(f"Initializing RedisStatePersister '{self.name}'...")
            provider = await provider_registry.get_by_config("default-cache")
            if not isinstance(provider, CacheProvider):
                 raise TypeError(f"Provider '{self.settings.redis_provider_name}' is not a valid CacheProvider instance.")
            if not provider.initialized:
                 logger.debug(f"Initializing underlying RedisProvider '{self.settings.redis_provider_name}'...")
                 await provider.initialize()
            self._redis_provider = provider
        except Exception as e:
            logger.error(f"Failed to initialize RedisStatePersister '{self.name}' with provider '{self.settings.redis_provider_name}': {e}", exc_info=True)
            self._redis_provider = None
            raise

    # Renamed to _save_state_impl to match BaseStatePersister
    async def _save_state_impl(self, state: AgentState, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not self.initialized or not self._redis_provider: # Check Provider's flag
             # Initialize should be called automatically by Provider base class if needed
             raise RuntimeError(f"RedisStatePersister '{self.name}' is not initialized or redis provider is missing.")
        state_id = state.task_id # BaseStatePersister already checks this
        redis_key = f"{self.settings.key_prefix}{state_id}"
        try:
            state_json = state.model_dump_json(indent=None)
            # Extract TTL using strict Pydantic model
            ttl_info = StateTTLInfo.extract_from_state(state)
            await self._redis_provider.set(redis_key, state_json, ttl_seconds=ttl_info.ttl_seconds)
            logger.debug(f"Saved state for task '{state_id}' to Redis key '{redis_key}' via persister '{self.name}'")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Failed to save state '{state_id}' via persister '{self.name}': {e}", exc_info=True)
            return False # Indicate failure

    # Renamed to _load_state_impl to match BaseStatePersister
    async def _load_state_impl(self, state_id: str) -> Optional[AgentState]:
        if not self.initialized or not self._redis_provider:
             raise RuntimeError(f"RedisStatePersister '{self.name}' is not initialized or redis provider is missing.")
        redis_key = f"{self.settings.key_prefix}{state_id}"
        try:
            state_json = await self._redis_provider.get(redis_key)
            if state_json is None:
                 return None # Return None if not found, BaseStatePersister handles error wrapping if needed
            state = AgentState.model_validate_json(state_json)
            logger.debug(f"Loaded state for task '{state_id}' from Redis key '{redis_key}' via persister '{self.name}'")
            return state
        except Exception as e:
            logger.error(f"Failed to load state '{state_id}' via persister '{self.name}': {e}", exc_info=True)
            raise # Re-raise exception, BaseStatePersister handles wrapping

    # Add _delete_state_impl and _list_states_impl if needed, or rely on NotImplementedError
    async def _delete_state_impl(self, task_id: str) -> bool:
        if not self.initialized or not self._redis_provider:
             raise RuntimeError(f"RedisStatePersister '{self.name}' is not initialized or redis provider is missing.")
        redis_key = f"{self.settings.key_prefix}{task_id}"
        try:
            deleted_count = await self._redis_provider.delete(redis_key)
            logger.debug(f"Deleted state for task '{task_id}' (key: '{redis_key}', count: {deleted_count}) via persister '{self.name}'")
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete state '{task_id}' via persister '{self.name}': {e}", exc_info=True)
            return False
            
    # Implement _list_states_impl if required by the interface
    async def _list_states_impl(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        # This might be complex with Redis keys, potentially using SCAN
        # Returning NotImplementedError for now if not critical path
        raise NotImplementedError("_list_states_impl not implemented for RedisStatePersister")

    # Change initialize/shutdown to match Provider base class pattern (_initialize / _shutdown)
    async def _shutdown(self):
         logger.info(f"Shutting down RedisStatePersister '{self.name}'. Underlying provider '{self.settings.redis_provider_name}' shutdown managed by registry.")
         self._redis_provider = None

# --- MongoDB State Persister ---

class MongoStatePersisterSettings(StrictBaseModel):
    """Settings for the MongoStatePersister."""
    mongo_provider_name: str = Field(..., min_length=1, description="Registered name of the MongoDBProvider instance.")
    database_name: str = Field(..., min_length=1, description="Name of the MongoDB database to use.")
    collection_name: str = Field("agent_states", description="Name of the collection to store agent states.")

@provider(name="mongodb", provider_type="state_persister", settings_class=MongoStatePersisterSettings)
# Corrected inheritance
class MongoStatePersister(BaseStatePersister[MongoStatePersisterSettings]):
    """State persister using a MongoDB provider as the backend."""

    def __init__(self, name: str, settings: MongoStatePersisterSettings):
        super().__init__(name=name, settings=settings)
        self._mongo_provider: Optional[DBProvider] = None

    async def _initialize(self):
        # Removed redundant check for self.initialized
        try:
            logger.debug(f"Initializing MongoStatePersister '{self.name}'...")
            provider = await provider_registry.get_by_config("default-database")
            if not isinstance(provider, DBProvider):
                 raise TypeError(f"Provider '{self.settings.mongo_provider_name}' is not a valid DBProvider instance.")
            if not provider.initialized:
                 await provider.initialize()
            self._mongo_provider = provider
        except Exception as e:
            logger.error(f"Failed to initialize MongoStatePersister '{self.name}': {e}", exc_info=True)
            self._mongo_provider = None
            raise
            
    async def _save_state_impl(self, state: AgentState, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not self.initialized or not self._mongo_provider: raise RuntimeError(f"{self.name} not initialized.")
        state_id = state.task_id
        try:
            state_dict = state.model_dump(mode='json')
            state_dict["_id"] = state_id # Use task_id as MongoDB _id
            if metadata: state_dict["metadata"] = metadata # Add metadata if provided
            
            # Assume the underlying DBProvider has an upsert method
            # We need to check the actual DBProvider interface for exact method signature
            result = await self._mongo_provider.upsert_document(
                database_name=self.settings.database_name,
                collection_name=self.settings.collection_name,
                document=state_dict,
                filter_query={"_id": state_id} 
            )
            logger.debug(f"Saved state '{state_id}' via {self.name}. Result: {result}")
            # Check result based on provider's return value (e.g., upserted_id, matched_count)
            return result is not None # Basic check, adjust based on actual return
        except Exception as e:
            logger.error(f"Failed to save state '{state_id}' via {self.name}: {e}", exc_info=True)
            return False

    async def _load_state_impl(self, state_id: str) -> Optional[AgentState]:
        if not self.initialized or not self._mongo_provider: raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Assume find_document_by_id method exists
            state_dict = await self._mongo_provider.find_document_by_id(
                database_name=self.settings.database_name,
                collection_name=self.settings.collection_name,
                document_id=state_id
            )
            if state_dict is None: return None
            # Remove metadata if present before validating AgentState
            metadata = None
            if "metadata" in state_dict:
                metadata = state_dict.pop("metadata")
            state = AgentState.model_validate(state_dict)
            logger.debug(f"Loaded state '{state_id}' via {self.name}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state '{state_id}' via {self.name}: {e}", exc_info=True)
            raise
            
    async def _delete_state_impl(self, task_id: str) -> bool:
        if not self.initialized or not self._mongo_provider: raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Assume delete_document_by_id exists
            result = await self._mongo_provider.delete_document_by_id(
                database_name=self.settings.database_name,
                collection_name=self.settings.collection_name,
                document_id=task_id
            )
            logger.debug(f"Deleted state '{task_id}' via {self.name}. Result: {result}")
            return result is not None and result > 0 # Check if delete was successful
        except Exception as e:
            logger.error(f"Failed to delete state '{task_id}' via {self.name}: {e}", exc_info=True)
            return False
            
    async def _list_states_impl(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        # Implementation depends heavily on how metadata/filtering is stored/queried in Mongo
        raise NotImplementedError("_list_states_impl not implemented for MongoStatePersister")

    async def _shutdown(self):
        logger.info(f"Shutting down MongoStatePersister '{self.name}'.")
        self._mongo_provider = None

# --- PostgreSQL State Persister ---

class PostgresStatePersisterSettings(StrictBaseModel):
    """Settings for the PostgresStatePersister."""
    postgres_provider_name: str = Field(..., min_length=1, description="Registered name of the PostgresProvider instance.")
    table_name: str = Field("agent_states", description="Name of the table to store agent states.")
    id_column: str = Field("task_id", description="Name of the primary key column (holding the task_id).")
    data_column: str = Field("state_data", description="Name of the JSONB column holding the agent state.")

@provider(name="postgres", provider_type="state_persister", settings_class=PostgresStatePersisterSettings)
# Corrected inheritance
class PostgresStatePersister(BaseStatePersister[PostgresStatePersisterSettings]):
    """State persister using a PostgreSQL provider as the backend."""

    def __init__(self, name: str, settings: PostgresStatePersisterSettings):
        super().__init__(name=name, settings=settings)
        self._postgres_provider: Optional[DBProvider] = None

    async def _initialize(self):
        # Removed redundant check for self.initialized
        try:
            logger.debug(f"Initializing PostgresStatePersister '{self.name}'...")
            provider = await provider_registry.get_by_config("default-database")
            if not isinstance(provider, DBProvider):
                 raise TypeError(f"Provider '{self.settings.postgres_provider_name}' is not a valid DBProvider instance.")
            if not provider.initialized:
                 await provider.initialize()
            self._postgres_provider = provider
        except Exception as e:
            logger.error(f"Failed to initialize PostgresStatePersister '{self.name}': {e}", exc_info=True)
            self._postgres_provider = None
            raise

    async def _save_state_impl(self, state: AgentState, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not self.initialized or not self._postgres_provider: raise RuntimeError(f"{self.name} not initialized.")
        state_id = state.task_id
        try:
            state_json_str = state.model_dump_json(indent=None)
            record_data = {
                self.settings.id_column: state_id,
                self.settings.data_column: state_json_str
                # Add metadata to a separate column if needed
            }
            if metadata: 
                # Need a way to store metadata, e.g., another JSONB column? 
                logger.warning(f"Metadata provided but not stored by {self.name}")
                
            # Assume upsert_record method exists
            result = await self._postgres_provider.upsert_record(
                 table_name=self.settings.table_name,
                 record_data=record_data,
                 conflict_target=self.settings.id_column 
            )
            logger.debug(f"Saved state '{state_id}' via {self.name}. Result: {result}")
            return result is not None # Basic check
        except Exception as e:
            logger.error(f"Failed to save state '{state_id}' via {self.name}: {e}", exc_info=True)
            return False

    async def _load_state_impl(self, state_id: str) -> Optional[AgentState]:
        if not self.initialized or not self._postgres_provider: raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Assume find_record_by_field exists
            record = await self._postgres_provider.find_record_by_field(
                table_name=self.settings.table_name,
                field_name=self.settings.id_column,
                field_value=state_id
            )
            if record is None: return None
            if self.settings.data_column not in record:
                raise ValueError(f"State data column '{self.settings.data_column}' missing from record for '{state_id}'")
            state_data = record[self.settings.data_column]
            if state_data is None: 
                 raise ValueError(f"State data column '{self.settings.data_column}' missing for '{state_id}'.")
                 
            # Handle potential JSON string or dict/list from DB driver
            if isinstance(state_data, str):
                 state = AgentState.model_validate_json(state_data)
            elif isinstance(state_data, (dict, list)): # Handle direct JSON/JSONB types
                 state = AgentState.model_validate(state_data)
            else:
                 raise TypeError(f"Unexpected type {type(state_data)} for state data.")
                 
            logger.debug(f"Loaded state '{state_id}' via {self.name}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state '{state_id}' via {self.name}: {e}", exc_info=True)
            raise
            
    async def _delete_state_impl(self, task_id: str) -> bool:
        if not self.initialized or not self._postgres_provider: raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Assume delete_record_by_field exists
            result = await self._postgres_provider.delete_record_by_field(
                table_name=self.settings.table_name,
                field_name=self.settings.id_column,
                field_value=task_id
            )
            logger.debug(f"Deleted state '{task_id}' via {self.name}. Result: {result}")
            return result is not None and result > 0 # Check if delete was successful
        except Exception as e:
            logger.error(f"Failed to delete state '{task_id}' via {self.name}: {e}", exc_info=True)
            return False
            
    async def _list_states_impl(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        # Needs implementation based on how metadata/filtering is handled
        raise NotImplementedError("_list_states_impl not implemented for PostgresStatePersister")

    async def _shutdown(self):
        logger.info(f"Shutting down PostgresStatePersister '{self.name}'.")
        self._postgres_provider = None 