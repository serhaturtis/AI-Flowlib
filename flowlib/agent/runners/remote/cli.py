"""
Command-Line Interface for interacting with the Remote Agent system.

Currently provides functionality to submit tasks to the agent task queue.
"""

import argparse
import asyncio
import uuid
import logging
from typing import Optional

from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.mq.base import MQProvider
from .models import AgentTaskMessage
from .config_loader import load_remote_config
# Import all provider modules to trigger decorator registration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def submit_task(
    mq_provider_name: str,
    queue_name: str,
    task_description: str,
    task_id: Optional[str] = None,
    initial_state_id: Optional[str] = None,
    reply_to: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Constructs and publishes an AgentTaskMessage to the specified queue."""
    
    mq_provider: Optional[MQProvider] = None
    try:
        logger.info(f"Getting MQ provider: {mq_provider_name}")
        # Providers are now auto-registered via imports above
        
        provider = await provider_registry.get_by_config(mq_provider_name)
        mq_provider = provider if isinstance(provider, MQProvider) else None
        if not mq_provider or not isinstance(mq_provider, MQProvider):
            raise ValueError(f"Could not find or initialize MQ provider '{mq_provider_name}'. Is it configured and registered?")
        if not mq_provider.initialized:
             await mq_provider.initialize()
             
        # Generate IDs if not provided
        task_id = task_id or str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Create the message
        task_message = AgentTaskMessage(
            task_id=task_id,
            task_description=task_description,
            initial_state_id=initial_state_id,
            initial_state_data=None,
            agent_config=None,
            reply_to_queue=reply_to,
            correlation_id=correlation_id
        )
        
        logger.info(f"Publishing task {task_id} to queue '{queue_name}'")
        # Create metadata with correlation_id
        from flowlib.providers.mq.base import MessageMetadata
        metadata = MessageMetadata(correlation_id=correlation_id)

        await mq_provider.publish(
            exchange="",  # Use default exchange
            routing_key=queue_name,
            message=task_message,
            metadata=metadata
        )
        
        logger.info(f"Task {task_id} submitted successfully.")
        print(f"Submitted Task ID: {task_id}")
        print(f"Correlation ID: {correlation_id}")
        
    except Exception as e:
        logger.exception(f"Failed to submit task: {e}")
        print(f"Error submitting task: {e}")
    finally:
        # Optional: Shutdown the specific provider instance?
        # Or rely on global shutdown elsewhere.
        # if mq_provider and mq_provider.initialized:
        #     await mq_provider.shutdown()
        pass 

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a task to the remote agent worker queue.")
    
    # Add config file argument first
    parser.add_argument(
        "-c", "--config", 
        default="remote_config.yaml", 
        help="Path to the remote configuration YAML file (default: remote_config.yaml)"
    )
    
    parser.add_argument("-d", "--description", required=True, help="The task description for the agent.")
    # Remove arguments that will now come from config file
    # parser.add_argument("--mq-provider", help="Name of the registered MQ provider instance to use.")
    # parser.add_argument("--queue", help="Name of the task queue to publish to.")
    parser.add_argument("--task-id", help="Optional specific ID for the task (default: auto-generated UUID).")
    parser.add_argument("--state-id", help="Optional ID of a previous state to resume from.")
    parser.add_argument("--reply-to", help="Optional queue name for the worker to send the result message to.")
    parser.add_argument("--correlation-id", help="Optional correlation ID (default: auto-generated UUID).")
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    remote_config = load_remote_config(args.config)
    cli_config = remote_config.cli
    
    try:
        asyncio.run(submit_task(
            # Get MQ provider and queue name from loaded config
            mq_provider_name=cli_config.mq_provider_name,
            queue_name=cli_config.task_queue,
            # Other args remain from CLI input
            task_description=args.description,
            task_id=args.task_id,
            initial_state_id=args.state_id,
            reply_to=args.reply_to,
            correlation_id=args.correlation_id
        ))
    except KeyboardInterrupt:
        logger.info("CLI interrupted.")

# This allows running the CLI directly, assuming environment and providers are set up
# Example: python -m flowlib.agent.remote.cli -c path/to/remote_config.yaml --description "Summarize..."
# Ensure ENV VARS are set for the MQ provider specified in remote_config.yaml (e.g., FLOWLIB__PROVIDERS__MQ__RABBITMQ__DEFAULT_MQ__HOST)
if __name__ == "__main__":
    # Configuration for MQ provider is loaded via environment variables
    # by the RabbitMQSettings model when the provider is initialized by the registry.
    # YAML config specifies *which* configured provider to use.
    main() 