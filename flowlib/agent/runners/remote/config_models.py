"""
Pydantic models for parsing the remote configuration YAML file (e.g., remote_config.yaml).
"""

from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from typing import Optional

class WorkerServiceConfig(StrictBaseModel):
    """Configuration specific to the Agent Worker service."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    mq_provider_name: str = Field("rabbitmq", description="Registered name of the MQ provider instance to use.")
    state_persister_name: str = Field("redis", description="Registered name of the State Persister instance to use.")
    task_queue: str = Field("agent_tasks", description="Name of the queue to consume tasks from.")
    results_queue: str = Field("agent_results", description="Default name of the queue to publish results to.")
    base_agent_config_path: Optional[str] = Field("./agent_config.yaml", description="Path to the base YAML file for default agent configuration.")

class CLIToolConfig(StrictBaseModel):
    """Configuration specific to the CLI tool."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    mq_provider_name: str = Field("rabbitmq", description="Registered name of the MQ provider instance to use.")
    task_queue: str = Field("agent_tasks", description="Default name of the task queue to publish to.")

class RemoteConfig(StrictBaseModel):
    """Root model for the remote configuration file."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    worker: WorkerServiceConfig = Field(default_factory=lambda: WorkerServiceConfig(
        mq_provider_name="rabbitmq",
        state_persister_name="redis",
        task_queue="agent_tasks",
        results_queue="agent_results",
        base_agent_config_path="./agent_config.yaml"
    ))
    cli: CLIToolConfig = Field(default_factory=lambda: CLIToolConfig(
        mq_provider_name="rabbitmq",
        task_queue="agent_tasks"
    ))
    # Add other sections as needed (e.g., result_processor) 