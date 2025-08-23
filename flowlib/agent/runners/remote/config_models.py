"""
Pydantic models for parsing the remote configuration YAML file (e.g., remote_config.yaml).
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class WorkerServiceConfig(BaseModel):
    """Configuration specific to the Agent Worker service."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    mq_provider_name: str = Field("rabbitmq", description="Registered name of the MQ provider instance to use.")
    state_persister_name: str = Field("redis", description="Registered name of the State Persister instance to use.")
    task_queue: str = Field("agent_tasks", description="Name of the queue to consume tasks from.")
    results_queue: str = Field("agent_results", description="Default name of the queue to publish results to.")
    base_agent_config_path: Optional[str] = Field("./agent_config.yaml", description="Path to the base YAML file for default agent configuration.")

class CLIToolConfig(BaseModel):
    """Configuration specific to the CLI tool."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    mq_provider_name: str = Field("rabbitmq", description="Registered name of the MQ provider instance to use.")
    task_queue: str = Field("agent_tasks", description="Default name of the task queue to publish to.")

class RemoteConfig(BaseModel):
    """Root model for the remote configuration file."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    worker: WorkerServiceConfig = Field(default_factory=WorkerServiceConfig)
    cli: CLIToolConfig = Field(default_factory=CLIToolConfig)
    # Add other sections as needed (e.g., result_processor) 