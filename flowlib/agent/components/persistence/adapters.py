"""
State Persister adapter implementations that leverage existing DB/Cache providers
for use in distributed agent scenarios.
"""

import logging
from typing import Dict, List, Optional, cast

from pydantic import Field

from flowlib.agent.models.state import AgentState, AgentStateModel
from flowlib.core.models import StrictBaseModel

# Import provider types
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.cache.base import CacheProvider
from flowlib.providers.core.base import ProviderSettings
from flowlib.providers.core.decorators import provider
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.db.base import DBProvider
from flowlib.providers.db.mongodb.provider import MongoDBProvider

# Import Provider base class
from .base import BaseStatePersister

# Import correct error classes from core.errors instead of non-existent exceptions module

logger = logging.getLogger(__name__)


class StateTTLInfo(StrictBaseModel):
    """State TTL information model."""

    ttl_seconds: Optional[int] = Field(default=None, description="TTL in seconds for state storage")

    @classmethod
    def extract_from_state(cls, state: AgentState) -> 'StateTTLInfo':
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
class RedisStatePersisterSettings(ProviderSettings):
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
    async def _initialize(self) -> None:
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
            state_json = state.model_dump_json()
            # Extract TTL using strict Pydantic model
            ttl_info = StateTTLInfo.extract_from_state(state)
            await self._redis_provider.set(redis_key, state_json, ttl=ttl_info.ttl_seconds)
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
            # Import here to avoid circular imports
            from flowlib.agent.models.state import AgentStateModel
            state_model = AgentStateModel.model_validate_json(state_json)
            state = AgentState(initial_state_data=state_model)
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
    async def _shutdown(self) -> None:
         logger.info(f"Shutting down RedisStatePersister '{self.name}'. Underlying provider '{self.settings.redis_provider_name}' shutdown managed by registry.")
         self._redis_provider = None

# --- MongoDB State Persister ---

class MongoStatePersisterSettings(ProviderSettings):
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
        self._mongo_provider: Optional[MongoDBProvider] = None

    async def _initialize(self) -> None:
        # Removed redundant check for self.initialized
        try:
            logger.debug(f"Initializing MongoStatePersister '{self.name}'...")
            provider = await provider_registry.get_by_config("default-database")
            if not isinstance(provider, DBProvider):
                 raise TypeError(f"Provider '{self.settings.mongo_provider_name}' is not a valid DBProvider instance.")
            if not provider.initialized:
                 await provider.initialize()
            self._mongo_provider = cast(MongoDBProvider, provider)
        except Exception as e:
            logger.error(f"Failed to initialize MongoStatePersister '{self.name}': {e}", exc_info=True)
            self._mongo_provider = None
            raise

    async def _save_state_impl(self, state: AgentState, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not self.initialized or not self._mongo_provider:
            raise RuntimeError(f"{self.name} not initialized.")
        state_id = state.task_id
        try:
            state_dict = state.model_dump()
            state_dict["_id"] = state_id # Use task_id as MongoDB _id
            if metadata:
                state_dict["metadata"] = metadata  # Add metadata if provided

            # Implement upsert logic: try update first, then insert if needed
            update_count = await self._mongo_provider.update_document(
                collection=self.settings.collection_name,
                query={"_id": state_id},
                update={"$set": state_dict},
                upsert=True
            )
            logger.debug(f"Saved state '{state_id}' via {self.name}. Update count: {update_count}")
            # Check result based on update count - should be >= 0 for successful operation
            return update_count >= 0
        except Exception as e:
            logger.error(f"Failed to save state '{state_id}' via {self.name}: {e}", exc_info=True)
            return False

    async def _load_state_impl(self, state_id: str) -> Optional[AgentState]:
        if not self.initialized or not self._mongo_provider:
            raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Query for the document by _id
            results = await self._mongo_provider.execute_query(
                collection=self.settings.collection_name,
                query={"_id": state_id}
            )
            state_dict = results[0] if results else None
            if state_dict is None:
                return None
            # Remove metadata if present before validating AgentState
            if "metadata" in state_dict:
                state_dict.pop("metadata")
            # Import here to avoid circular imports
            from flowlib.agent.models.state import AgentStateModel
            state_model = AgentStateModel.model_validate(state_dict)
            state = AgentState(initial_state_data=state_model)
            logger.debug(f"Loaded state '{state_id}' via {self.name}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state '{state_id}' via {self.name}: {e}", exc_info=True)
            raise

    async def _delete_state_impl(self, task_id: str) -> bool:
        if not self.initialized or not self._mongo_provider:
            raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Delete document by _id
            result = await self._mongo_provider.delete_document(
                collection=self.settings.collection_name,
                query={"_id": task_id}
            )
            logger.debug(f"Deleted state '{task_id}' via {self.name}. Result: {result}")
            return result is not None and result > 0 # Check if delete was successful
        except Exception as e:
            logger.error(f"Failed to delete state '{task_id}' via {self.name}: {e}", exc_info=True)
            return False

    async def _list_states_impl(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        # Implementation depends heavily on how metadata/filtering is stored/queried in Mongo
        raise NotImplementedError("_list_states_impl not implemented for MongoStatePersister")

    async def _shutdown(self) -> None:
        logger.info(f"Shutting down MongoStatePersister '{self.name}'.")
        self._mongo_provider = None

# --- PostgreSQL State Persister ---

class PostgresStatePersisterSettings(ProviderSettings):
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

    async def _initialize(self) -> None:
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
        if not self.initialized or not self._postgres_provider:
            raise RuntimeError(f"{self.name} not initialized.")
        state_id = state.task_id
        try:
            state_json_str = state.model_dump_json()
            record_data = {
                self.settings.id_column: state_id,
                self.settings.data_column: state_json_str
                # Add metadata to a separate column if needed
            }
            if metadata:
                # Need a way to store metadata, e.g., another JSONB column?
                logger.warning(f"Metadata provided but not stored by {self.name}")

            # Use actual SQL with execute method for upsert
            upsert_sql = f"""
                INSERT INTO {self.settings.table_name} ({self.settings.id_column}, {self.settings.data_column})
                VALUES (%(task_id)s, %(state_data)s)
                ON CONFLICT ({self.settings.id_column})
                DO UPDATE SET {self.settings.data_column} = EXCLUDED.{self.settings.data_column}
            """
            result = await self._postgres_provider.execute(
                upsert_sql,
                {"task_id": state_id, "state_data": record_data[self.settings.data_column]}
            )
            logger.debug(f"Saved state '{state_id}' via {self.name}. Result: {result}")
            return result is not None # Basic check
        except Exception as e:
            logger.error(f"Failed to save state '{state_id}' via {self.name}: {e}", exc_info=True)
            return False

    async def _load_state_impl(self, state_id: str) -> Optional[AgentState]:
        if not self.initialized or not self._postgres_provider:
            raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Use actual SQL with execute method to find record
            select_sql = f"""
                SELECT {self.settings.data_column}
                FROM {self.settings.table_name}
                WHERE {self.settings.id_column} = %(task_id)s
            """
            records = await self._postgres_provider.execute(
                select_sql,
                {"task_id": state_id}
            )
            record = records[0] if records else None
            if record is None:
                return None
            if self.settings.data_column not in record:
                raise ValueError(f"State data column '{self.settings.data_column}' missing from record for '{state_id}'")
            state_data = record[self.settings.data_column]
            if state_data is None:
                 raise ValueError(f"State data column '{self.settings.data_column}' missing for '{state_id}'.")

            # Handle potential JSON string or dict/list from DB driver
            if isinstance(state_data, str):
                 state_model = AgentStateModel.model_validate_json(state_data)
                 state = AgentState(initial_state_data=state_model)
            elif isinstance(state_data, (dict, list)): # Handle direct JSON/JSONB types
                 state_model = AgentStateModel.model_validate(state_data)
                 state = AgentState(initial_state_data=state_model)
            else:
                 raise TypeError(f"Unexpected type {type(state_data)} for state data.")

            logger.debug(f"Loaded state '{state_id}' via {self.name}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state '{state_id}' via {self.name}: {e}", exc_info=True)
            raise

    async def _delete_state_impl(self, task_id: str) -> bool:
        if not self.initialized or not self._postgres_provider:
            raise RuntimeError(f"{self.name} not initialized.")
        try:
            # Use actual SQL with execute method to delete record
            delete_sql = f"""
                DELETE FROM {self.settings.table_name}
                WHERE {self.settings.id_column} = %(task_id)s
            """
            result = await self._postgres_provider.execute(
                delete_sql,
                {"task_id": task_id}
            )
            logger.debug(f"Deleted state '{task_id}' via {self.name}. Result: {result}")
            return result is not None and result > 0 # Check if delete was successful
        except Exception as e:
            logger.error(f"Failed to delete state '{task_id}' via {self.name}: {e}", exc_info=True)
            return False

    async def _list_states_impl(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        # Needs implementation based on how metadata/filtering is handled
        raise NotImplementedError("_list_states_impl not implemented for PostgresStatePersister")

    async def _shutdown(self) -> None:
        logger.info(f"Shutting down PostgresStatePersister '{self.name}'.")
        self._postgres_provider = None
