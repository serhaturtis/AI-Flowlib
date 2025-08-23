"""Dependency injection container for agent components."""

import asyncio
import logging
from typing import Type, TypeVar, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from collections import defaultdict

from flowlib.agent.core.interfaces import ComponentInterface
from flowlib.agent.core.errors import ComponentError, AgentError
from flowlib.agent.core.base import AgentComponent

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=ComponentInterface)


class ComponentRegistration(BaseModel):
    """Registration information for a component."""
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True, strict=True, arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Component name")
    component: AgentComponent = Field(..., description="Component instance")
    dependencies: List[str] = Field(default_factory=list, description="Component dependencies")


class ComponentContainer:
    """Manages component lifecycle and dependencies with proper initialization order."""
    
    def __init__(self):
        self._components: dict[str, AgentComponent] = {}
        self._providers: dict[str, object] = {}
        self._dependencies: dict[str, List[str]] = defaultdict(list)
        self._initialized: set = set()
        self._initializing: set = set()
        
    async def register_component(
        self, 
        name: str, 
        component: AgentComponent,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Register a component with its dependencies."""
        if name in self._components:
            raise ComponentError(f"Component '{name}' already registered", "ComponentContainer")
            
        self._components[name] = component
        if dependencies:
            self._dependencies[name] = dependencies
            
        logger.debug(f"Registered component: {name}")
    
    async def register_provider(self, name: str, provider: object) -> None:
        """Register a provider (LLM, database, etc.)."""
        self._providers[name] = provider
        logger.debug(f"Registered provider: {name}")
    
    def get_component(self, component_type: Type[T]) -> T:
        """Type-safe component retrieval."""
        for component in self._components.values():
            if isinstance(component, component_type):
                return component
        raise ComponentError(f"No component of type {component_type.__name__} found", "ComponentContainer")
    
    def get_component_by_name(self, name: str) -> AgentComponent:
        """Get component by name."""
        if name not in self._components:
            raise ComponentError(f"Component '{name}' not found", "ComponentContainer")
        return self._components[name]
    
    def get_provider(self, name: str) -> object:
        """Get provider by name."""
        if name not in self._providers:
            raise ComponentError(f"Provider '{name}' not found", "ComponentContainer")
        return self._providers[name]
    
    async def initialize_all(self) -> None:
        """Initialize all components in dependency order."""
        logger.info("Initializing all components...")
        
        # Topological sort for dependency order
        initialization_order = self._get_initialization_order()
        
        for component_name in initialization_order:
            await self._initialize_component(component_name)
            
        logger.info(f"Initialized {len(self._initialized)} components")
    
    async def shutdown_all(self) -> None:
        """Shutdown all components in reverse dependency order."""
        logger.info("Shutting down all components...")
        
        # Shutdown in reverse order
        shutdown_order = list(reversed(list(self._initialized)))
        
        for component_name in shutdown_order:
            try:
                component = self._components[component_name]
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                logger.debug(f"Shutdown component: {component_name}")
            except Exception as e:
                logger.error(f"Error shutting down {component_name}: {e}")
        
        self._initialized.clear()
        logger.info("Component shutdown complete")
    
    async def _initialize_component(self, name: str) -> None:
        """Initialize a single component with dependency checking."""
        if name in self._initialized:
            return
            
        if name in self._initializing:
            raise ComponentError(f"Circular dependency detected involving '{name}'", "ComponentContainer")
            
        self._initializing.add(name)
        
        try:
            # Initialize dependencies first
            dependencies = self._dependencies[name] if name in self._dependencies else None
            if dependencies:
                for dep_name in dependencies:
                    await self._initialize_component(dep_name)
            
            # Initialize the component
            component = self._components[name]
            if hasattr(component, 'initialize'):
                await component.initialize()
                
            self._initialized.add(name)
            logger.debug(f"Initialized component: {name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize component '{name}': {e}")
            raise ComponentError(f"Component initialization failed: {name}", "ComponentContainer") from e
        finally:
            self._initializing.discard(name)
    
    def _get_initialization_order(self) -> List[str]:
        """Get components in dependency order using topological sort."""
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        all_components = set(self._components.keys())
        
        # Calculate in-degrees
        for component in all_components:
            if component not in self._dependencies:
                continue
            for dep in self._dependencies[component]:
                in_degree[component] += 1
        
        # Start with components that have no dependencies
        queue = [comp for comp in all_components if in_degree[comp] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent components
            for component, deps in self._dependencies.items():
                if current in deps:
                    in_degree[component] -= 1
                    if in_degree[component] == 0:
                        queue.append(component)
        
        if len(result) != len(all_components):
            missing = all_components - set(result)
            raise ComponentError(
                f"Circular dependencies detected: {missing}",
                component_name=str(missing),
                operation="dependency_resolution"
            )
            
        return result
    
    def get_dependency_graph(self) -> dict[str, List[str]]:
        """Get the current dependency graph for debugging."""
        return dict(self._dependencies)
    
    def get_component_status(self) -> dict[str, str]:
        """Get status of all components."""
        status = {}
        for name in self._components:
            if name in self._initialized:
                status[name] = "initialized"
            elif name in self._initializing:
                status[name] = "initializing"
            else:
                status[name] = "not_initialized"
        return status


class ComponentBuilder:
    """Builder pattern for constructing components with dependencies."""
    
    def __init__(self, container: ComponentContainer):
        self.container = container
        self._pending_registrations: List[tuple] = []
    
    def add_component(
        self, 
        name: str, 
        component: AgentComponent,
        dependencies: Optional[List[str]] = None
    ) -> 'ComponentBuilder':
        """Add a component to be registered."""
        self._pending_registrations.append((name, component, dependencies))
        return self
    
    def add_provider(self, name: str, provider: object) -> 'ComponentBuilder':
        """Add a provider to be registered."""
        self._pending_registrations.append((name, provider, None, True))
        return self
    
    async def build(self) -> ComponentContainer:
        """Register all components and providers."""
        for registration in self._pending_registrations:
            if len(registration) == 4 and registration[3]:  # Provider
                await self.container.register_provider(registration[0], registration[1])
            else:  # Component
                await self.container.register_component(*registration[:3])
        
        await self.container.initialize_all()
        return self.container


class AgentComponentRegistry:
    """Registry for standard agent components with default configurations."""
    
    @staticmethod
    def create_standard_container() -> ComponentContainer:
        """Create a container with standard agent components."""
        return ComponentContainer()
    
    @staticmethod
    async def setup_memory_components(
        container: ComponentContainer,
        config: dict
    ) -> None:
        """Setup memory subsystem components."""
        from flowlib.agent.components.memory.working import WorkingMemory
        from flowlib.agent.components.memory.vector import VectorMemory
        from flowlib.agent.components.memory.knowledge import KnowledgeMemory
        
        # Working memory (no dependencies)
        working_memory = WorkingMemory()
        await container.register_component("working_memory", working_memory)
        
        # Vector memory (depends on working memory for fallback)
        enable_vector = config["enable_vector_memory"] if "enable_vector_memory" in config else None
        if enable_vector is None or enable_vector:
            vector_memory = VectorMemory()
            await container.register_component(
                "vector_memory", 
                vector_memory,
                dependencies=["working_memory"]
            )
        
        # Knowledge memory (depends on vector memory)
        enable_knowledge = config["enable_knowledge_memory"] if "enable_knowledge_memory" in config else None
        if enable_knowledge is None or enable_knowledge:
            knowledge_memory = KnowledgeMemory()
            deps = ["working_memory"]
            enable_vector = config["enable_vector_memory"] if "enable_vector_memory" in config else None
            if enable_vector is None or enable_vector:
                deps.append("vector_memory")
            await container.register_component(
                "knowledge_memory",
                knowledge_memory,
                dependencies=deps
            )
    
    @staticmethod
    async def setup_planning_components(
        container: ComponentContainer,
        config: dict
    ) -> None:
        """Setup planning subsystem components."""
        from flowlib.agent.components.planning import AgentPlanner
        
        planning = AgentPlanner(config)
        await container.register_component(
            "planning",
            planning,
            dependencies=["working_memory"]  # Planning needs memory
        )
    
    @staticmethod
    async def setup_reflection_components(
        container: ComponentContainer,
        config: dict
    ) -> None:
        """Setup reflection subsystem components."""
        from flowlib.agent.components.reflection.base import AgentReflection
        
        reflection = AgentReflection()
        await container.register_component(
            "reflection",
            reflection,
            dependencies=["working_memory", "planning"]
        )