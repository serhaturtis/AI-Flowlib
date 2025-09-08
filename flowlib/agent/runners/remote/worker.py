"""
Agent Worker Service

Consumes tasks from a message queue, executes the agent, 
and publishes results back.
"""

import logging
import asyncio
import signal
import argparse
import yaml
import os
from typing import Optional, Type, Any
try:
    import aio_pika
    from aio_pika.message import AbstractIncomingMessage
except ImportError:
    aio_pika = None
    AbstractIncomingMessage = None

from pydantic import BaseModel, Field, ConfigDict

# Ensure provider implementations are imported so decorators run
from flowlib.providers.mq.rabbitmq.provider import RabbitMQProvider, RabbitMQProviderSettings
# NOTE: Add imports for other needed providers (like Redis, Mongo, Postgres state persisters) 
# if they aren't implicitly imported elsewhere before use.

# Corrected relative imports (using ... instead of ....)
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.mq.base import MQProvider, MessageMetadata
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.agent.runners.remote.models import AgentTaskMessage, AgentResultMessage
from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.models.state import AgentState
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.runners.autonomous import run_autonomous # Reusing autonomous runner logic
# Import the config loader
from flowlib.agent.runners.remote.config_loader import load_remote_config, RemoteConfig 
# Example persister import from the new location:
from flowlib.agent.components.persistence.adapters import RedisStatePersister 

logger = logging.getLogger(__name__)


class DefaultAgentConfigData(BaseModel):
    """Default agent configuration data model."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(default="default_worker_agent", description="Agent name")
    persona: str = Field(default="A remote worker agent executing distributed tasks", description="Agent persona")
    provider_name: str = Field(default="llamacpp", description="Provider name")


# Store loaded worker config globally within the worker process for access
# This is a simple approach; dependency injection could be used for more complex apps
_worker_config: Optional["WorkerServiceConfig"] = None

def load_base_agent_config() -> AgentConfig:
    """Loads the base AgentConfig from the path specified in the worker config."""
    global _worker_config
    if not _worker_config or not _worker_config.base_agent_config_path:
        logger.warning("Base agent config path not specified in worker config. Using default AgentConfig().")
        return AgentConfig(name="default_worker_agent", persona="A remote worker agent executing distributed tasks", provider_name="llamacpp")
    
    config_path = _worker_config.base_agent_config_path
    if not os.path.exists(config_path):
        logger.warning(f"Base agent config file not found at '{config_path}'. Using default AgentConfig().")
        return AgentConfig(name="default_worker_agent", persona="A remote worker agent executing distributed tasks", provider_name="llamacpp")
        
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Ensure required fields are present with defaults using Pydantic
        config_data = config_data or {}
        default_config = DefaultAgentConfigData()
        
        # Merge with defaults using explicit field access
        merged_config = {
            "name": config_data["name"] if "name" in config_data else default_config.name,
            "persona": config_data["persona"] if "persona" in config_data else default_config.persona,
            "provider_name": config_data["provider_name"] if "provider_name" in config_data else default_config.provider_name,
            **{k: v for k, v in config_data.items() if k not in ["name", "persona", "provider_name"]}
        }
            
        base_config = AgentConfig.model_validate(merged_config)
        logger.info(f"Loaded base agent configuration from: {config_path}")
        return base_config
    except Exception as e:
        logger.error(f"Failed to load or validate base agent config '{config_path}': {e}", exc_info=True)
        logger.warning("Using default AgentConfig() due to error.")
        return AgentConfig(name="default_worker_agent", persona="A remote worker agent executing distributed tasks", provider_name="llamacpp")


class AgentWorker:
    """Manages the lifecycle and task processing for a remote agent worker."""

    def __init__(
        self,
        mq_provider_name: str,
        state_persister_name: str, # Registered name of the StatePersister instance
        task_queue_name: str,
        results_queue_name: str,
        # Configurations like state_provider_name (e.g., "redis-backend-1")
        # needed by the specific persister should be part of the persister's
        # own configuration when it's registered.
    ):
        self.mq_provider_name = mq_provider_name
        self.state_persister_name = state_persister_name # Name to lookup in registry
        self.task_queue_name = task_queue_name
        self.results_queue_name = results_queue_name
        
        self._mq_provider: Optional[MQProvider] = None
        self._state_persister: Optional[BaseStatePersister] = None
        self._consumer_tag: Optional[str] = None
        self._shutdown_requested = asyncio.Event()
        self._consumer_task: Optional[asyncio.Task] = None

    async def _initialize_providers(self):
        """Initialize Message Queue and State Persister providers."""
        try:
            logger.info("Initializing Message Queue provider...")
            self._mq_provider = await provider_registry.get(
                "message_queue", self.mq_provider_name
            )
            if not self._mq_provider or not isinstance(self._mq_provider, MQProvider):
                 raise ValueError(f"Invalid MQ provider: {self.mq_provider_name}")
            if not self._mq_provider.initialized:
                 await self._mq_provider.initialize()
            logger.info(f"Message Queue provider '{self.mq_provider_name}' initialized.")

            logger.info("Initializing State Persister...")
            # Look up the pre-configured and registered StatePersister instance by name
            self._state_persister = await provider_registry.get(
               "state_persister", self.state_persister_name 
            )
            
            if not self._state_persister or not isinstance(self._state_persister, BaseStatePersister):
                 raise ValueError(f"Registered provider '{self.state_persister_name}' is not a valid State Persister.")
            
            # Assume the persister adapter handles its own initialization if needed 
            # (e.g., initializing its underlying DB/Cache provider)
            # Check if it has an async initialize method
            if hasattr(self._state_persister, 'initialize') and asyncio.iscoroutinefunction(self._state_persister.initialize):
                 # Check if it requires initialization (simple check for an internal flag)
                 needs_init = not getattr(self._state_persister, '_is_initialized', True)
                 if needs_init:
                      logger.info(f"Calling initialize() on State Persister '{self.state_persister_name}'.")
                      await self._state_persister.initialize()
                 
            logger.info(f"State Persister '{self.state_persister_name}' retrieved and ready.")

        except Exception as e:
            logger.exception(f"Failed to initialize providers: {e}")
            raise

    async def handle_task_message(self, message: AbstractIncomingMessage) -> None:
        """Handles an incoming task message from the queue."""
        logger.info(f"Received raw message. Attempting to decode Task.")
        task_msg: Optional[AgentTaskMessage] = None
        agent: Optional[BaseAgent] = None
        status = "FAILURE"
        result_data = None
        error_message = None
        final_state_id = None

        try:
            logger.info(f"Received task message, delivery_tag={message.delivery_tag}")
            # 1. Deserialize Task Message
            try:
                task_msg = AgentTaskMessage.model_validate_json(message.body)
                logger.info(f"Processing task_id: {task_msg.task_id}")
            except Exception as deser_err:
                logger.error(f"Failed to deserialize task message: {deser_err}", exc_info=True)
                error_message = f"Invalid task message format: {deser_err}"
                status = "INVALID_MESSAGE"
                logger.error("Acknowledging invalid message.")
                await message.ack()
                return # Cannot proceed

            if not self._state_persister:
                 raise RuntimeError("State persister not available.")

            # 2. Load or Create Agent State
            agent_state: Optional[AgentState] = None
            agent_config: Optional[AgentConfig] = None
            try:
                # Determine the configuration first, potentially merging task overrides
                base_config = load_base_agent_config() # Load base config using helper
                if task_msg.agent_config:
                    logger.debug(f"Merging task-specific config overrides for task '{task_msg.task_id}'")
                    merged_config_dict = base_config.model_dump()
                    merged_config_dict.update(task_msg.agent_config)
                    try:
                         agent_config = AgentConfig.model_validate(merged_config_dict)
                    except Exception as config_val_err:
                         logger.warning(f"Failed to validate merged config for task '{task_msg.task_id}': {config_val_err}. Using base config.")
                         agent_config = base_config
                else:
                    agent_config = base_config

                # Now, load or create the state
                if task_msg.initial_state_id:
                    logger.info(f"Loading state for task_id: {task_msg.task_id} from state_id: {task_msg.initial_state_id}")
                    agent_state = await self._state_persister.load(task_msg.initial_state_id)
                    # Use the derived agent_config - state doesn't contain config info
                    logger.info(f"Loaded existing state for task '{task_msg.task_id}', using derived agent config.")
                else:
                    logger.info(f"Creating new state for task_id: {task_msg.task_id}")
                    agent_state = AgentState(
                        task_id=task_msg.task_id,
                        task_description=task_msg.task_description,
                        initial_state_data=task_msg.initial_state_data
                    )

            except FileNotFoundError:
                logger.error(f"Initial state ID '{task_msg.initial_state_id}' not found.")
                error_message = f"Initial state ID '{task_msg.initial_state_id}' not found."
            except Exception as state_err:
                logger.exception(f"Error loading/creating state for task '{task_msg.task_id}': {state_err}")
                error_message = f"Failed to load/create state: {state_err}"

            if error_message:
                raise Exception(error_message)
                
            # 3. Initialize BaseAgent
            if not agent_state or not agent_config:
                 raise RuntimeError(f"Failed to establish valid agent state/config for task '{task_msg.task_id}'")
                 
            agent = BaseAgent(config=agent_config, initial_state=agent_state, state_persister=self._state_persister)
            await agent.initialize()
            logger.info(f"Agent initialized for task '{task_msg.task_id}'.")

            # 4. Execute Agent Logic (using autonomous runner)
            # Make sure run_autonomous uses the provided agent instance and its state
            final_state = await run_autonomous(agent=agent) # Reuses the loop logic
            final_state_id = final_state.task_id # Should be same as task_id
            
            # Check final state for completion/errors
            if final_state.is_complete:
                 status = "SUCCESS"
                 # Extract relevant result data - more comprehensive than just last message
                 result_data = {
                     "is_complete": True,
                     "progress": final_state.progress,
                     "final_message": final_state.system_messages[-1] if final_state.system_messages else "Task completed.",
                     # Add other relevant fields from state if needed
                     # "extracted_data": final_state.get_data("some_key") 
                 } 
                 logger.info(f"Task '{task_msg.task_id}' completed successfully.")
            elif final_state.errors:
                 status = "FAILURE" # Consider it a failure if errors occurred, even if not is_complete
                 error_message = f"Agent finished with errors: {'; '.join(final_state.errors)}"
                 result_data = { # Still provide some context
                     "is_complete": False,
                     "progress": final_state.progress,
                     "errors": final_state.errors
                 }
                 logger.error(f"Task '{task_msg.task_id}' finished with errors: {final_state.errors}")
            else:
                 # Finished run (e.g., max cycles) but not complete and no explicit errors recorded
                 status = "INCOMPLETE"
                 error_message = "Agent finished run (e.g., max cycles reached) but task is not marked complete."
                 result_data = {
                     "is_complete": False,
                     "progress": final_state.progress,
                     "final_message": final_state.system_messages[-1] if final_state.system_messages else None,
                 }
                 logger.warning(f"Task '{task_msg.task_id}' finished incomplete.")
                 
            # Final save (run_autonomous might already do this, but ensure)
            await agent.save_state() # Ensure final state is saved

        except Exception as e:
            logger.exception(f"Error processing task '{task_msg.task_id if task_msg else 'N/A'}': {e}")
            status = "FAILURE"
            error_message = f"Worker error during task processing: {str(e)}" # Clarify worker error
            if agent and agent._state_manager.current_state: # Try to get final state ID even on error
                 final_state_id = agent._state_manager.current_state.task_id
                 try:
                      # Attempt to save state even if processing failed mid-way
                      await self._state_persister.save(agent._state_manager.current_state) 
                 except Exception as save_err:
                      logger.error(f"Failed to save error state: {save_err}")
                 # Include partial state data in result if possible?
                 result_data = {
                     "is_complete": agent._state_manager.current_state.is_complete,
                     "progress": agent._state_manager.current_state.progress,
                     "errors": agent._state_manager.current_state.errors + [f"Processing Exception: {str(e)}"]
                 }
            else: # If error happened before agent/state was initialized
                 final_state_id = task_msg.task_id if task_msg else None # Best guess

        # 5. Publish Result Message
        if task_msg and self._mq_provider:
            result_msg = AgentResultMessage(
                task_id=task_msg.task_id,
                status=status,
                final_state_id=final_state_id,
                result_data=result_data,
                error_message=error_message,
                correlation_id=task_msg.correlation_id
            )
            target_queue = task_msg.reply_to_queue or self.results_queue_name
            try:
                await self._mq_provider.publish(
                    queue=target_queue,
                    message=result_msg,
                    persistent=True,
                    correlation_id=task_msg.correlation_id # Pass correlation ID if needed by provider
                )
                logger.info(f"Published result for task '{task_msg.task_id}' to queue '{target_queue}'. Status: {status}")
            except Exception as pub_err:
                logger.exception(f"Failed to publish result for task '{task_msg.task_id}': {pub_err}")
                # If publish fails, we cannot report success/failure back easily.
                # NACKing might cause reprocessing, which could be bad.
                # Log error and ACK anyway? Log and raise to stop worker? Log and ACK.
                logger.error("Acknowledging task despite result publish failure.")
        else:
             logger.error("Could not publish result: Task message or MQ provider missing.")

        # 6. Acknowledge Original Task Message
        logger.info(f"Acknowledging task message, delivery_tag={message.delivery_tag}")
        await message.ack()

    async def start(self):
        """Start the worker: initialize providers and begin consuming messages."""
        logger.info("Starting Agent Worker...")
        self._shutdown_requested.clear()
        await self._initialize_providers()

        if not self._mq_provider:
             logger.error("MQ Provider not initialized. Worker cannot start.")
             return
             
        logger.info(f"Starting message consumption from queue: '{self.task_queue_name}'")
        self._consumer_tag = await self._mq_provider.consume(
            queue_name=self.task_queue_name,
            callback=self.handle_task_message 
        )
        logger.info(f"Message consumption started with consumer tag: {self._consumer_tag}")
        # Keep running until shutdown is requested
        await self._shutdown_requested.wait()
        logger.info("Shutdown requested, stopping worker...")

    async def stop(self):
        """Request the worker to stop consuming and shut down gracefully."""
        logger.info("Requesting worker shutdown...")
        self._shutdown_requested.set()

        if self._consumer_task and not self._consumer_task.done():
             self._consumer_task.cancel()
             
        if self._mq_provider and self._consumer_tag:
            try:
                logger.info(f"Stopping message consumption (consumer tag: {self._consumer_tag})...")
                await self._mq_provider.stop_consuming(self._consumer_tag)
                self._consumer_tag = None
            except Exception as e:
                logger.exception(f"Error stopping MQ consumption: {e}")
        
        # Shutdown providers (optional, could be handled globally)
        # if self._state_persister:
        #     await self._state_persister.shutdown()
        # if self._mq_provider:
        #     await self._mq_provider.shutdown()
            
        logger.info("Agent Worker stopped.")


async def run_worker_service(
    config_path: Optional[str] = None # Add argument for config file path
):
    """Main entry point to run the agent worker service."""
    global _worker_config # Allow modification of global
    
    # Load configuration from YAML file
    remote_config = load_remote_config(config_path)
    _worker_config = remote_config.worker # Store globally for helper access
    
    # Use loaded config
    worker = AgentWorker(
        mq_provider_name=_worker_config.mq_provider_name,
        state_persister_name=_worker_config.state_persister_name,
        task_queue_name=_worker_config.task_queue,
        results_queue_name=_worker_config.results_queue
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Signal received, initiating graceful shutdown...")
        stop_event.set()

    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        # Start the worker and wait for it to run until stop is requested
        start_task = asyncio.create_task(worker.start())
        # Wait for either the worker to finish (unexpectedly) or stop_event to be set
        await asyncio.wait(
            [start_task, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        # If stop_event was set, initiate graceful stop
        if stop_event.is_set():
            await worker.stop()
        else:
             logger.warning("Worker start task finished unexpectedly.")
             # Check if the task had an exception
             if start_task.done() and start_task.exception():
                  raise start_task.exception()
                  
    except asyncio.CancelledError:
         logger.info("Worker service run cancelled.")
    except Exception as e:
         logger.exception("Unhandled exception in worker service run:", exc_info=True)
    finally:
         # Ensure final stop sequence is called if not already stopped
         if not worker._shutdown_requested.is_set():
              await worker.stop()
         # Remove signal handlers
         for sig in (signal.SIGINT, signal.SIGTERM):
              loop.remove_signal_handler(sig)
         logger.info("Worker service finished.")

# Example usage (if run as a script - requires env vars for config):
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run the Agent Worker service.")
    parser.add_argument(
        "-c", "--config", 
        default="remote_config.yaml", 
        help="Path to the remote configuration YAML file (default: remote_config.yaml)"
    )
    args = parser.parse_args()
    
    # --- Configuration is loaded via Environment Variables by Pydantic Settings ---
    # Ensure ENV VARS are set for:
    # - RabbitMQ provider (e.g., FLOWLIB__PROVIDERS__MQ__RABBITMQ__DEFAULT_MQ__HOST)
    # - Redis provider (e.g., FLOWLIB__PROVIDERS__CACHE__REDIS__SHARED_CACHE__HOST)
    # - StatePersister settings (e.g., FLOWLIB__PROVIDERS__STATE_PERSISTER__REDIS__WORKER_STATE_STORE__REDIS_PROVIDER_NAME)
    # The names "default_mq", "shared_cache", "worker_state_store" are examples and must match
    # the names specified in the loaded remote_config.yaml (worker section).
    # --------------------------------------------------------------------------
    
    try:
        # Load config and run the service
        asyncio.run(run_worker_service(config_path=args.config))
    except KeyboardInterrupt:
        logger.info("Exiting.") 