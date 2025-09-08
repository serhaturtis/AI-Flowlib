"""Clean interfaces for dependency inversion.

This module defines protocol interfaces that eliminate circular dependencies
by removing concrete class dependencies. All components depend only on
these interfaces, never on concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any, Dict, List, Optional, runtime_checkable, Type
from datetime import datetime


@runtime_checkable
class PromptTemplate(Protocol):
    """Protocol defining the interface for prompt templates.
    
    Any class decorated with @prompt must implement this interface.
    """
    template: str
    config: Dict[str, Any]


@runtime_checkable
class Provider(Protocol):
    """Provider interface eliminating circular dependencies.
    
    All provider implementations must implement this interface.
    No concrete dependencies - only interface contracts.
    """
    
    name: str
    provider_type: str
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        ...
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        ...


@runtime_checkable
class LLMProvider(Provider, Protocol):
    """LLM Provider specific interface."""
    
    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from prompt."""
        ...
    
    async def generate_structured(self, prompt: PromptTemplate, output_type: Type, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> Any:
        """Generate structured output."""
        ...


@runtime_checkable
class VectorProvider(Provider, Protocol):
    """Vector Database Provider interface."""
    
    async def add_vectors(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Add vectors to the database."""
        ...
    
    async def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        ...


@runtime_checkable
class GraphProvider(Provider, Protocol):
    """Graph Database Provider interface."""
    
    async def add_entity(self, entity_data: Dict[str, Any]) -> str:
        """Add entity to graph."""
        ...
    
    async def add_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Dict[str, Any]) -> str:
        """Add relationship to graph."""
        ...


@runtime_checkable
class DatabaseProvider(Provider, Protocol):
    """Database Provider interface."""
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute database query."""
        ...
    
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch single record."""
        ...


@runtime_checkable
class CacheProvider(Provider, Protocol):
    """Cache Provider interface."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ...


@runtime_checkable
class Resource(Protocol):
    """Resource interface eliminating circular dependencies.
    
    All resource implementations must implement this interface.
    """
    
    name: str
    resource_type: str
    
    def get_data(self) -> Any:
        """Get resource data."""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get resource metadata."""
        ...


@runtime_checkable
class Configuration(Resource, Protocol):
    """Configuration resource interface."""
    
    def get_settings(self) -> Dict[str, Any]:
        """Get configuration settings."""
        ...
    
    def get_provider_type(self) -> str:
        """Get provider type for this configuration."""
        ...


@runtime_checkable
class PromptResource(Resource, Protocol):
    """Prompt resource interface."""
    
    def get_template(self) -> str:
        """Get prompt template."""
        ...
    
    def format(self, **kwargs) -> str:
        """Format prompt with variables."""
        ...


@runtime_checkable
class ModelResource(Resource, Protocol):
    """Model resource interface."""
    
    def get_model_class(self) -> type:
        """Get model class."""
        ...
    
    def create_instance(self, **kwargs) -> Any:
        """Create model instance."""
        ...


@runtime_checkable
class Flow(Protocol):
    """Flow interface eliminating circular dependencies.
    
    All flow implementations must implement this interface.
    """
    
    name: str
    description: Optional[str]
    
    async def execute(self, input_data: Any, **kwargs) -> Any:
        """Execute the flow with input data."""
        ...
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get input schema for validation."""
        ...
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema for validation."""
        ...


@runtime_checkable
class AgentFlow(Flow, Protocol):
    """Agent-specific flow interface."""
    
    async def run_pipeline(self, input_data: Any) -> Any:
        """Run the agent pipeline."""
        ...


@runtime_checkable
class Stage(Protocol):
    """Stage interface for pipeline components."""
    
    name: str
    
    async def execute(self, context: Any) -> Any:
        """Execute stage with context."""
        ...


@runtime_checkable
class ToolProvider(Provider, Protocol):
    """Tool provider interface for structured tool calling.
    
    Tool providers enable agents to execute tools autonomously through
    structured generation and validation. Each tool provider implements
    this interface to provide schema-driven tool execution.
    """
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get JSON schema for LLM tool calling.
        
        Returns:
            Tool schema in OpenAI function calling format
        """
        ...
    
    def get_parameter_model(self) -> type:
        """Get Pydantic model class for parameter validation.
        
        Returns:
            Pydantic model class for tool parameters
        """
        ...
    
    async def execute_tool(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """Execute the tool with validated parameters.
        
        Args:
            parameters: Tool parameters (pre-validated)
            context: Optional execution context
            
        Returns:
            Tool execution result
        """
        ...


@runtime_checkable
class Memory(Protocol):
    """Memory interface for agent memory systems."""
    
    async def store(self, key: str, value: Any, **kwargs) -> None:
        """Store value in memory."""
        ...
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Any]:
        """Retrieve value from memory."""
        ...
    
    async def search(self, query: str, **kwargs) -> List[Any]:
        """Search memory."""
        ...


@runtime_checkable
class Planning(Protocol):
    """Planning interface for agent planning systems."""
    
    async def create_plan(self, objective: str, context: Dict[str, Any]) -> Any:
        """Create execution plan."""
        ...
    
    async def execute_plan(self, plan: Any) -> Any:
        """Execute a plan."""
        ...


# Factory Protocol for dynamic creation
@runtime_checkable
class Factory(Protocol):
    """Factory interface for dynamic object creation."""
    
    def create(self, type_name: str, config: Dict[str, Any]) -> Any:
        """Create object of specified type with configuration."""
        ...
    
    def supports(self, type_name: str) -> bool:
        """Check if factory supports creating this type."""
        ...


# Container Protocol for dependency injection
@runtime_checkable
class Container(Protocol):
    """Container interface for dependency injection."""
    
    async def get_provider(self, config_name: str) -> Provider:
        """Get provider by configuration name."""
        ...
    
    def get_resource(self, name: str, resource_type: Optional[str] = None) -> Resource:
        """Get resource by name and type."""
        ...
    
    def get_flow(self, name: str) -> Flow:
        """Get flow by name."""
        ...
    
    def register(self, item_type: str, name: str, factory: Any, metadata: Dict[str, Any]) -> None:
        """Register item in container."""
        ...


# Event Protocol for registration
@runtime_checkable
class RegistrationEvent(Protocol):
    """Registration event interface."""
    
    item_type: str
    name: str
    factory: Any
    metadata: Dict[str, Any]


# Type aliases for clarity
ProviderFactory = Any  # Callable[[], Provider]
ResourceFactory = Any  # Callable[[], Resource] 
FlowFactory = Any     # Callable[[], Flow]