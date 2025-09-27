"""Task execution coordinator without decomposition bypass.

This module provides pure coordination for executing pre-decomposed TODOs,
following the architectural principle that ALL tasks go through decomposition first.
"""

import logging
from typing import List, Optional, Any, cast

from flowlib.agent.core.base import AgentComponent
from .models import TaskExecutionResult, ToolExecutionContext
from ..models import TodoItem
from flowlib.flows.registry import flow_registry
from flowlib.agent.core.context.models import ExecutionContext

logger = logging.getLogger(__name__)


class TaskExecutionComponent(AgentComponent):
    """Coordinates execution of pre-decomposed TODOs.
    
    This is NOT a flow - it's a coordinator/service class.
    It ONLY handles execution coordination.
    It does NOT perform task decomposition or classification.
    ALL tasks must go through decomposition first.
    """
    
    def __init__(self, name: str = "task_executor", activity_stream: Optional[Any] = None) -> None:
        """Initialize the task executor.
        
        Args:
            name: Component name
            activity_stream: Optional activity stream for real-time updates
        """
        super().__init__(name)
        self._activity_stream = activity_stream
    
    async def _initialize_impl(self) -> None:
        """Initialize the task executor."""
        logger.info("Task executor initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the task executor."""
        logger.info("Task executor shutdown")
    
    def _get_default_agent_role(self) -> str:
        """Get default agent role for safety.

        SECURITY: Always returns "general_purpose" for minimal access.
        Role must be explicitly set by user - never inferred from persona.

        Returns:
            "general_purpose" - the most restrictive role
        """
        # SECURITY: Never infer role from persona - always use most restrictive default
        # User must explicitly assign role through configuration
        return "general_purpose"
    
    async def execute_decomposed_todos(
        self,
        todos: List[TodoItem],
        context: ToolExecutionContext
    ) -> TaskExecutionResult:
        """Execute pre-decomposed TODOs through execution flow.
        
        Args:
            todos: Pre-generated TODO items from decomposition system
            context: Execution context
            
        Returns:
            Aggregated execution result
        """
        self._check_initialized()
        
        # Ensure agent role is set in context
        if not context.agent_role:
            # SECURITY: Use default restrictive role if not explicitly set
            context.agent_role = self._get_default_agent_role()
            logger.warning(f"No agent role specified - using default {context.agent_role} for safety")
        
        # Get execution flow from registry and create proper input
        from .execution_flow import TodoBatchExecutionInput, TaskExecutionFlow
        execution_flow = cast(TaskExecutionFlow, flow_registry.get("task-execution"))

        # Create input model for the flow
        flow_input = TodoBatchExecutionInput(todos=todos, context=context)
        result = await execution_flow.execute_todo_batch(flow_input)
        return cast(TaskExecutionResult, result)
    
    async def execute_todos(
        self,
        context: ExecutionContext
    ) -> TaskExecutionResult:
        """Execute TODOs from unified execution context.
        
        Args:
            context: Unified execution context containing TODOs and all necessary information
            
        Returns:
            Aggregated execution result
        """
        self._check_initialized()
        
        # Extract TODOs from context
        todos = context.task.todos
        
        # Create ToolExecutionContext for backward compatibility with existing flow
        import uuid
        from .models import ToolExecutionSharedData
        
        # SECURITY: Get explicitly configured role or use default restrictive role
        # Never infer from persona to prevent privilege escalation
        from flowlib.config.role_manager import role_manager
        
        agent_id = context.session.agent_name
        profile_name = role_manager.get_role_assignment(agent_id)
        
        if profile_name:
            # Get the profile config and extract the agent role
            from flowlib.resources.registry.registry import resource_registry
            try:
                profile_config = resource_registry.get(profile_name)
                # Access the agent_role from the resource's data
                if hasattr(profile_config, 'agent_role'):
                    agent_role = profile_config.agent_role
                elif hasattr(profile_config, 'data') and isinstance(profile_config.data, dict):
                    agent_role = profile_config.data.get('agent_role', self._get_default_agent_role())
                else:
                    agent_role = self._get_default_agent_role()
            except KeyError:
                logger.warning(f"Agent profile '{profile_name}' not found for agent '{agent_id}' - using default role")
                agent_role = self._get_default_agent_role()
        else:
            logger.warning(f"No role assignment found for agent '{agent_id}' - using default role")
            agent_role = self._get_default_agent_role()
        
        tool_context = ToolExecutionContext(
            working_directory=context.session.working_directory,
            timeout_seconds=context.component.execution_timeout,
            agent_id=context.session.agent_name,
            agent_persona=context.session.agent_persona,
            agent_role=agent_role,
            session_id=context.session.session_id,
            task_id=context.task.description[:50],  # Use task description as ID
            execution_id=str(uuid.uuid4()),
            parent_execution_id=None,
            execution_depth=0,
            previous_results=[],
            shared_data=ToolExecutionSharedData()
        )
        
        # Use existing execution method
        return await self.execute_decomposed_todos(todos, tool_context)
    
    def get_available_tools(self, agent_role: Optional[str] = None) -> List[str]:
        """Get list of tools available for the specified agent role.
        
        Args:
            agent_role: Agent role to check (must be explicitly provided)
            
        Returns:
            List of tool names accessible to the agent role
        """
        if not agent_role:
            # SECURITY: Return minimal tools if no role specified
            agent_role = self._get_default_agent_role()
            logger.warning(f"No agent role specified - returning tools for {agent_role}")
        
        # Import here to avoid circular imports
        from .registry import tool_registry
        return tool_registry.list_tools_for_role(agent_role)
    
    def validate_tool_access(self, tool_name: str, agent_role: Optional[str] = None) -> bool:
        """Validate if an agent can access a specific tool.
        
        Args:
            tool_name: Name of the tool to check
            agent_role: Agent role to check (must be explicitly provided)
            
        Returns:
            True if access is allowed, False otherwise
        """
        if not agent_role:
            # SECURITY: Use most restrictive role if not specified
            agent_role = self._get_default_agent_role()
            logger.warning(f"No agent role specified - validating against {agent_role}")
        
        # Import here to avoid circular imports
        from .tool_role_manager import tool_role_manager
        try:
            tool_role_manager.validate_tool_access(agent_role, tool_name)
            return True
        except Exception:
            return False
    
