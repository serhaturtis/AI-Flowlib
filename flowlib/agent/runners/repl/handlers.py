"""Command handlers for agent REPL."""

from typing import Dict, Any, List, Optional
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from flowlib.agent.runners.repl.commands import CommandHandler, Command, CommandType
from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.components.task.decomposition.manager import TodoManager


class MemoryStats(StrictBaseModel):
    """Memory statistics model."""
    model_config = ConfigDict(extra="forbid")
    
    count: int = Field(default=0, description="Count of items")
    collections: int = Field(default=0, description="Number of collections")
    nodes: int = Field(default=0, description="Number of nodes")
    edges: int = Field(default=0, description="Number of edges")
    node_types: List[str] = Field(default_factory=list, description="Node type names")


class FlowInfo(StrictBaseModel):
    """Flow information model."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Flow name")
    description: str = Field(default="No description", description="Flow description")


class MemoryItem(StrictBaseModel):
    """Memory item model."""
    model_config = ConfigDict(extra="forbid")
    
    content: str = Field(default="Unknown", description="Item content")


class MCPServerInfo(StrictBaseModel):
    """MCP server information model."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Server name")
    description: str = Field(default="No description", description="Server description")


class MCPStatus(StrictBaseModel):
    """MCP status model."""
    model_config = ConfigDict(extra="forbid")
    
    connected: bool = Field(default=False, description="Connection status")
    server: str = Field(default="None", description="Server name")
    protocol_version: str = Field(default="Unknown", description="Protocol version")
    tool_count: int = Field(default=0, description="Number of tools")
    resource_count: int = Field(default=0, description="Number of resources")


class MCPTool(StrictBaseModel):
    """MCP tool model."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Tool name")
    description: str = Field(default="No description", description="Tool description")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Tool parameters")


class MCPResource(StrictBaseModel):
    """MCP resource model."""
    model_config = ConfigDict(extra="forbid")
    
    uri: str = Field(description="Resource URI")
    description: str = Field(default="No description", description="Resource description")
    mimeType: Optional[str] = Field(default=None, description="MIME type")


class DefaultCommandHandler(CommandHandler):
    """Default handler for basic commands."""

    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        return command.type in [CommandType.USER, CommandType.SYSTEM]

    async def handle(self, command: Command, context: Dict[str, Any]) -> None:
        """Handle the command."""
        if command.type == CommandType.SYSTEM:
            # System commands are handled by the registry
            return None
        
        # User commands are passed through
        return None


class AgentCommandHandler(CommandHandler):
    """Handler for agent-specific commands."""
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        # Handle special agent commands
        if command.type == CommandType.USER:
            # Check for special patterns
            if command.raw_input.startswith("!"):
                return True
            if command.raw_input.startswith("?"):
                return True
        return False
    
    async def handle(self, command: Command, context: Dict[str, Any]) -> Optional[str]:
        """Handle agent-specific commands."""
        if "agent" not in context:
            return "No agent available."
        agent = context["agent"]
        
        # Handle ! commands (direct flow execution)
        if command.raw_input.startswith("!"):
            flow_command = command.raw_input[1:].strip()
            return await self._execute_flow(flow_command, agent, context)
        
        # Handle ? commands (introspection)
        elif command.raw_input.startswith("?"):
            query = command.raw_input[1:].strip()
            return await self._introspect(query, agent, context)
        
        return None
    
    async def _execute_flow(self, flow_command: str, agent: BaseAgent, context: Dict[str, Any]) -> str:
        """Execute a flow directly."""
        parts = flow_command.split(maxsplit=1)
        if not parts:
            return "Usage: !flow_name [arguments]"
        
        flow_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            # Parse arguments as JSON if possible
            import json
            try:
                flow_args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                # Treat as string argument
                flow_args = {"input": args}
            
            # Execute flow
            result = await agent.execute_flow(flow_name, flow_args)
            
            # Update stats
            context["session_stats"]["flows_executed"] += 1
            
            return f"Flow '{flow_name}' executed successfully:\n{result}"
            
        except Exception as e:
            return f"Error executing flow '{flow_name}': {str(e)}"
    
    async def _introspect(self, query: str, agent: BaseAgent, context: Dict[str, Any]) -> str:
        """Introspect agent state."""
        if not query:
            return "Usage: ?[state|memory|context|flows|tools]"
        
        query = query.lower()
        
        if query == "state":
            return await self._show_agent_state(agent)
        elif query == "memory":
            return await self._show_memory_details(agent)
        elif query == "context":
            return self._show_context(context)
        elif query == "flows":
            return await self._show_available_flows(agent)
        elif query == "tools":
            return await self._show_available_tools(agent)
        else:
            return f"Unknown introspection query: {query}"
    
    async def _show_agent_state(self, agent: BaseAgent) -> str:
        """Show current agent state."""
        if not hasattr(agent, "state"):
            return "Agent has no state."
        
        state = agent.state
        status = state.status if hasattr(state, 'status') else 'Unknown'
        mode = state.mode if hasattr(state, 'mode') else 'Unknown'
        active_flows = state.active_flows if hasattr(state, 'active_flows') else []
        context_items = state.context if hasattr(state, 'context') else {}
        return f"""**Agent State:**
  Status: {status}
  Mode: {mode}
  Active Flows: {len(active_flows)}
  Context Items: {len(context_items)}
"""
    
    async def _show_memory_details(self, agent: BaseAgent) -> str:
        """Show detailed memory information."""
        if not hasattr(agent, "memory"):
            return "Agent has no memory system."
        
        memory = agent.memory
        details = []
        
        # Working memory
        if hasattr(memory, "working_memory"):
            wm = memory.working_memory
            recent = await wm.get_recent(5)
            details.append(f"**Working Memory** ({len(recent)} recent items):")
            for item in recent:
                memory_item = MemoryItem.model_validate(item) if isinstance(item, dict) else MemoryItem(content=str(item))
                details.append(f"  • {memory_item.content[:50]}...")
        
        # Vector memory
        if hasattr(memory, "vector_memory"):
            vm = memory.vector_memory
            stats_raw = await vm.get_stats()
            stats = MemoryStats.model_validate(stats_raw) if isinstance(stats_raw, dict) else MemoryStats()
            details.append("\n**Vector Memory**:")
            details.append(f"  Embeddings: {stats.count}")
            details.append(f"  Collections: {stats.collections}")
        
        # Knowledge graph
        if hasattr(memory, "knowledge_graph"):
            kg = memory.knowledge_graph
            stats_raw = await kg.get_stats()
            stats = MemoryStats.model_validate(stats_raw) if isinstance(stats_raw, dict) else MemoryStats()
            details.append("\n**Knowledge Graph**:")
            details.append(f"  Nodes: {stats.nodes}")
            details.append(f"  Edges: {stats.edges}")
            details.append(f"  Node Types: {', '.join(stats.node_types)}")
        
        return "\n".join(details)
    
    def _show_context(self, context: Dict[str, Any]) -> str:
        """Show current context."""
        # Filter sensitive items
        safe_context = {
            k: v for k, v in context.items()
            if k not in ["agent", "config", "command_history"]
        }
        
        import json
        return f"**Current Context:**\n```json\n{json.dumps(safe_context, indent=2, default=str)}\n```"
    
    async def _show_available_flows(self, agent: BaseAgent) -> str:
        """Show flows available to the agent."""
        if hasattr(agent, "flow_runner"):
            flows = await agent.flow_runner.list_available_flows()
            if flows:
                flow_list = "**Available Flows:**\n"
                for flow_data in flows:
                    flow = FlowInfo.model_validate(flow_data) if isinstance(flow_data, dict) else FlowInfo(name=str(flow_data))
                    flow_list += f"  • {flow.name}: {flow.description}\n"
                return flow_list
        
        return "No flows available."
    
    async def _show_available_tools(self, agent: BaseAgent) -> str:
        """Show tools available to the agent."""
        try:
            from flowlib.agent.components.task.execution.registry import tool_registry
            
            tools = tool_registry.list_tools()
            if tools:
                tool_list = "**Available Tools:**\n"
                for tool_name in tools:
                    try:
                        metadata = tool_registry.get_metadata(tool_name)
                        if metadata is None:
                            tool_list += f"  • {tool_name}: Available\n"
                            continue

                        description = getattr(metadata, 'description', 'No description available')
                        category = getattr(metadata, 'category', 'general')
                        tool_list += f"  • {tool_name} ({category}): {description}\n"
                    except Exception:
                        tool_list += f"  • {tool_name}: Available\n"
                return tool_list
            
        except Exception as e:
            return f"Error getting tools: {str(e)}"
        
        return "No tools available."


class MCPCommandHandler(CommandHandler):
    """Handler for MCP (Model Context Protocol) commands."""
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        if command.type == CommandType.SLASH:
            return command.name in ["mcp", "tools", "resources"]
        return False
    
    async def handle(self, command: Command, context: Dict[str, Any]) -> Optional[str]:
        """Handle MCP-related commands."""
        if "agent" not in context:
            return "No agent available."
        agent = context["agent"]
        
        if command.name == "mcp":
            return await self._handle_mcp_command(command.args, agent, context)
        elif command.name == "tools":
            return await self._list_mcp_tools(agent, context)
        elif command.name == "resources":
            return await self._list_mcp_resources(agent, context)
        
        return None
    
    async def _handle_mcp_command(self, args: List[str], agent: BaseAgent, context: Dict[str, Any]) -> str:
        """Handle MCP subcommands."""
        if not args:
            return """**MCP Commands:**
  /mcp connect <server>    - Connect to MCP server
  /mcp disconnect         - Disconnect from MCP server
  /mcp status            - Show MCP connection status
  /mcp servers           - List available MCP servers
"""
        
        subcommand = args[0]
        
        if subcommand == "connect":
            if len(args) < 2:
                return "Usage: /mcp connect <server>"
            server = args[1]
            return await self._connect_mcp_server(server, agent)
        
        elif subcommand == "disconnect":
            return await self._disconnect_mcp_server(agent)
        
        elif subcommand == "status":
            return await self._show_mcp_status(agent)
        
        elif subcommand == "servers":
            return await self._list_mcp_servers(agent)
        
        else:
            return f"Unknown MCP subcommand: {subcommand}"
    
    async def _connect_mcp_server(self, server: str, agent: BaseAgent) -> str:
        """Connect to an MCP server."""
        if not hasattr(agent, "mcp_client"):
            return "Agent does not support MCP."
        
        try:
            await agent.mcp_client.connect(server)
            return f"Connected to MCP server: {server}"
        except Exception as e:
            return f"Failed to connect to MCP server: {str(e)}"
    
    async def _disconnect_mcp_server(self, agent: BaseAgent) -> str:
        """Disconnect from MCP server."""
        if not hasattr(agent, "mcp_client"):
            return "Agent does not support MCP."
        
        try:
            await agent.mcp_client.disconnect()
            return "Disconnected from MCP server."
        except Exception as e:
            return f"Failed to disconnect: {str(e)}"
    
    async def _show_mcp_status(self, agent: BaseAgent) -> str:
        """Show MCP connection status."""
        if not hasattr(agent, "mcp_client"):
            return "Agent does not support MCP."
        
        client = agent.mcp_client
        status_raw = await client.get_status()
        status = MCPStatus.model_validate(status_raw) if isinstance(status_raw, dict) else MCPStatus()
        
        return f"""**MCP Status:**
  Connected: {status.connected}
  Server: {status.server}
  Protocol Version: {status.protocol_version}
  Tools Available: {status.tool_count}
  Resources Available: {status.resource_count}
"""
    
    async def _list_mcp_servers(self, agent: BaseAgent) -> str:
        """List available MCP servers."""
        if not hasattr(agent, "mcp_client"):
            return "Agent does not support MCP."
        
        servers = await agent.mcp_client.list_servers()
        if not servers:
            return "No MCP servers configured."
        
        server_list = "**Available MCP Servers:**\n"
        for server_data in servers:
            server = MCPServerInfo.model_validate(server_data) if isinstance(server_data, dict) else MCPServerInfo(name=str(server_data))
            server_list += f"  • {server.name}: {server.description}\n"
        
        return server_list
    
    async def _list_mcp_tools(self, agent: BaseAgent, context: Dict[str, Any]) -> str:
        """List MCP tools."""
        if not hasattr(agent, "mcp_client") or not agent.mcp_client.connected:
            return "No MCP connection. Use /mcp connect <server> first."
        
        tools = await agent.mcp_client.list_tools()
        if not tools:
            return "No MCP tools available."
        
        tool_list = "**MCP Tools:**\n"
        for tool_data in tools:
            tool = MCPTool.model_validate(tool_data) if isinstance(tool_data, dict) else MCPTool(name=str(tool_data))
            tool_list += f"  • {tool.name}: {tool.description}\n"
            if context.get('verbose') and tool.parameters:
                tool_list += f"    Parameters: {tool.parameters}\n"
        
        return tool_list
    
    async def _list_mcp_resources(self, agent: BaseAgent, context: Dict[str, Any]) -> str:
        """List MCP resources."""
        if not hasattr(agent, "mcp_client") or not agent.mcp_client.connected:
            return "No MCP connection. Use /mcp connect <server> first."
        
        resources = await agent.mcp_client.list_resources()
        if not resources:
            return "No MCP resources available."
        
        resource_list = "**MCP Resources:**\n"
        for resource_data in resources:
            resource = MCPResource.model_validate(resource_data) if isinstance(resource_data, dict) else MCPResource(uri=str(resource_data))
            resource_list += f"  • {resource.uri}: {resource.description}\n"
            if context.get('verbose') and resource.mimeType:
                resource_list += f"    Type: {resource.mimeType}\n"
        
        return resource_list


class ToolCommandHandler(CommandHandler):
    """Handler for REPL tool execution commands."""
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        if command.type == CommandType.SLASH:
            # Handle /tool command
            return command.name == "tool"
        elif command.type == CommandType.USER:
            # Handle @tool_name commands
            if command.raw_input.startswith("@"):
                tool_name = command.raw_input[1:].split()[0]
                try:
                    from flowlib.agent.components.task.execution.registry import tool_registry
                    return tool_registry.contains(tool_name)
                except Exception:
                    return False
        return False
    
    async def handle(self, command: Command, context: Dict[str, Any]) -> Any:
        """Handle tool-related commands."""
        if command.type == CommandType.SLASH and command.name == "tool":
            return await self._handle_tool_command(command.args, context)
        elif command.type == CommandType.USER and command.raw_input.startswith("@"):
            return await self._execute_tool_directly(command.raw_input, context)
        
        return None
    
    async def _handle_tool_command(self, args: List[str], context: Dict[str, Any]) -> str:
        """Handle /tool subcommands."""
        if not args:
            return """**Tool Commands:**
  /tool list                    - List available tools
  /tool describe <tool_name>    - Describe a specific tool
  /tool execute <tool_name> <args> - Execute a tool with arguments
  
**Direct Tool Execution:**
  @<tool_name> <args>          - Execute tool directly
  
**Example:**
  @read file_path=/path/to/file.py
  @bash command="ls -la"
  @grep pattern="class.*:" path="." include="*.py"
"""
        
        subcommand = args[0]
        
        if subcommand == "list":
            return await self._list_tools()
        elif subcommand == "describe":
            if len(args) < 2:
                return "Usage: /tool describe <tool_name>"
            return await self._describe_tool(args[1])
        elif subcommand == "execute":
            if len(args) < 2:
                return "Usage: /tool execute <tool_name> [args...]"
            tool_name = args[1]
            tool_args = " ".join(args[2:]) if len(args) > 2 else ""
            return await self._execute_tool(tool_name, tool_args, context)
        else:
            return f"Unknown tool subcommand: {subcommand}"
    
    async def _list_tools(self) -> str:
        """List all available tools."""
        try:
            from flowlib.agent.components.task.execution.registry import tool_registry
            
            tools = tool_registry.list_tools()
            if not tools:
                return "No tools available."
            
            tool_list = "**Available Tools:**\n"
            for tool_name in sorted(tools):
                try:
                    metadata = tool_registry.get_metadata(tool_name)
                    if metadata is None:
                        tool_list += f"  • {tool_name}: Available\n"
                        continue

                    description = getattr(metadata, 'description', 'No description available')
                    category = getattr(metadata, 'category', 'general')
                    tool_list += f"  • {tool_name} ({category}): {description}\n"
                except Exception:
                    tool_list += f"  • {tool_name}: Available\n"
            
            return tool_list
            
        except Exception as e:
            return f"Error listing tools: {str(e)}"
    
    async def _describe_tool(self, tool_name: str) -> str:
        """Describe a specific tool."""
        try:
            from flowlib.agent.components.task.execution.registry import tool_registry
            
            if not tool_registry.contains(tool_name):
                return f"Tool '{tool_name}' not found."

            metadata = tool_registry.get_metadata(tool_name)
            if metadata is None:
                return f"No metadata available for tool '{tool_name}'."

            # Note: Parameter schema not available through registry - would need tool instance

            description = f"**Tool: {tool_name}**\n\n"
            description += f"Description: {metadata.description}\n"
            # Note: category and version may not be available in current ToolMetadata

            # Check for additional metadata attributes if they exist
            if hasattr(metadata, 'category') and metadata.category:
                description += f"Category: {metadata.category}\n"
            if hasattr(metadata, 'version') and metadata.version:
                description += f"Version: {metadata.version}\n"
            description += "\n"

            # Note: Parameter details would require creating tool instance
            description += "Use the tool name in your TODO to see how it works in practice.\n"
            
            description += "\n**Usage Examples:**\n"
            description += f"  @{tool_name} param1=value1 param2=value2\n"
            description += f"  /tool execute {tool_name} param1=value1\n"
            
            return description
            
        except Exception as e:
            return f"Error describing tool '{tool_name}': {str(e)}"
    
    async def _execute_tool_directly(self, command_text: str, context: Dict[str, Any]) -> str:
        """Execute a tool directly from @tool_name syntax."""
        # Parse @tool_name param1=value1 param2=value2
        parts = command_text[1:].split(maxsplit=1)  # Remove @ and split
        if not parts:
            return "Invalid tool command."
        
        tool_name = parts[0]
        args_text = parts[1] if len(parts) > 1 else ""
        
        return await self._execute_tool(tool_name, args_text, context)
    
    async def _execute_tool(self, tool_name: str, args_text: str, context: Dict[str, Any]) -> str:
        """Execute a tool with given arguments."""
        try:
            from flowlib.agent.components.task.execution.orchestration import tool_orchestrator, ToolExecutionRequest
            
            if not tool_orchestrator.get_available_tools():
                return "No tools available. Tool system may not be initialized."
            
            if tool_name not in tool_orchestrator.get_available_tools():
                return f"Tool '{tool_name}' not found. Use '/tool list' to see available tools."
            
            # Parse arguments
            try:
                args = self._parse_tool_args(args_text)
            except ValueError as e:
                return f"Error parsing arguments: {str(e)}"
            
            # Create execution context
            exec_context = tool_orchestrator.create_execution_context(
                agent_id="repl-agent",
                agent_persona="Interactive REPL Agent",
                working_directory=".",
                session_id="repl-session"
            )
            
            # Create and execute request
            request = ToolExecutionRequest(
                tool_name=tool_name,
                raw_parameters=args,
                context=exec_context
            )
            
            response = await tool_orchestrator.execute_tool(request)
            
            # Format result
            if response.is_success():
                output = f"✅ **{tool_name}** executed successfully"
                content = response.get_display_content()
                if content:
                    output += f"\n\n{content}"
                return output
            elif response.is_error():
                error_msg = response.error.error_message if response.error else "Unknown error"
                return f"❌ **{tool_name}** failed: {error_msg}"
            else:
                return f"⚠️ **{tool_name}** completed with status: {response.status.value}"
                
        except Exception as e:
            return f"❌ Error executing tool '{tool_name}': {str(e)}"
    
    def _parse_tool_args(self, args_text: str) -> Dict[str, Any]:
        """Parse tool arguments from string."""
        if not args_text.strip():
            return {}
        
        args = {}
        
        # Parse key=value pairs with proper handling of JSON values
        import re
        import json
        
        # Regular expression to match key=value pairs where value might be JSON
        # This handles quoted strings, arrays, objects, and simple values
        pattern = r'(\w+)=((?:"[^"]*"|\'[^\']*\'|\[.*?\]|\{.*?\}|[^"\'\s]\S*(?:\s+(?![\w]+=)[^\s]+)*|\S+))'
        
        matches = re.findall(pattern, args_text)
        
        for key, value in matches:
            key = key.strip()
            value = value.strip()
            
            # Try to parse value as JSON first
            try:
                parsed_value = json.loads(value)
                args[key] = parsed_value
            except json.JSONDecodeError:
                # Remove quotes if present and treat as string
                if value.startswith('"') and value.endswith('"'):
                    args[key] = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    args[key] = value[1:-1]
                else:
                    # Try to convert to appropriate type
                    if value.lower() in ("true", "false"):
                        args[key] = value.lower() == "true"
                    elif value.isdigit():
                        args[key] = int(value)
                    elif value.replace(".", "").isdigit():
                        args[key] = float(value)
                    else:
                        args[key] = value
        
        # Handle any leftover text that doesn't match key=value pattern as positional args
        remaining_text = args_text
        for key, value in matches:
            remaining_text = remaining_text.replace(f"{key}={value}", "").strip()
        
        if remaining_text:
            # If there's remaining text, treat it as content
            args["content"] = remaining_text
        
        return args


class TodoCommandHandler(CommandHandler):
    """Handler for TODO management commands."""
    
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        if command.type == CommandType.SLASH:
            return command.name in ["todo", "todos"]
        elif command.type == CommandType.USER:
            # Handle #todo commands for quick TODO creation
            return command.raw_input.startswith("#todo")
        return False
    
    async def handle(self, command: Command, context: Dict[str, Any]) -> Any:
        """Handle TODO-related commands."""
        if command.type == CommandType.SLASH:
            return await self._handle_todo_command(command.args, context)
        elif command.type == CommandType.USER and command.raw_input.startswith("#todo"):
            return await self._quick_todo(command.raw_input, context)
        
        return None
    
    async def _handle_todo_command(self, args: List[str], context: Dict[str, Any]) -> str:
        """Handle /todo subcommands."""
        agent = context["agent"] if "agent" in context else None
        if not agent or not hasattr(agent, "_engine"):
            return "No agent with TODO support available."
        
        todo_manager = agent._engine.get_todo_manager()
        
        if not args:
            return """**TODO Commands:**
  /todo list                    - List current TODOs
  /todo add <content>           - Add a new TODO
  /todo complete <id>           - Mark TODO as completed
  /todo fail <id> <reason>      - Mark TODO as failed
  /todo delete <id>             - Delete a TODO
  /todo progress                - Show progress summary
  /todo export [format]         - Export TODOs (json/markdown)
  /todo clear                   - Clear all TODOs
  
**Quick TODO Creation:**
  #todo <content>               - Quick add TODO
  
**Examples:**
  /todo add "Fix the music generation bug"
  #todo Review the generated album
  /todo complete abc-123
"""
        
        subcommand = args[0]
        
        if subcommand == "list":
            return await self._list_todos(todo_manager)
        elif subcommand == "add":
            if len(args) < 2:
                return "Usage: /todo add <content>"
            content = " ".join(args[1:])
            return await self._add_todo(todo_manager, content)
        elif subcommand == "complete":
            if len(args) < 2:
                return "Usage: /todo complete <id>"
            return await self._complete_todo(todo_manager, args[1])
        elif subcommand == "fail":
            if len(args) < 3:
                return "Usage: /todo fail <id> <reason>"
            todo_id = args[1]
            reason = " ".join(args[2:])
            return await self._fail_todo(todo_manager, todo_id, reason)
        elif subcommand == "delete":
            if len(args) < 2:
                return "Usage: /todo delete <id>"
            return await self._delete_todo(todo_manager, args[1])
        elif subcommand == "progress":
            return await self._show_progress(todo_manager)
        elif subcommand == "export":
            format_type = args[1] if len(args) > 1 else "markdown"
            return await self._export_todos(todo_manager, format_type)
        elif subcommand == "clear":
            return await self._clear_todos(todo_manager)
        else:
            return f"Unknown TODO subcommand: {subcommand}"
    
    async def _quick_todo(self, command_text: str, context: Dict[str, Any]) -> str:
        """Handle quick TODO creation with #todo."""
        agent = context["agent"] if "agent" in context else None
        if not agent or not hasattr(agent, "_engine"):
            return "No agent with TODO support available."
        
        # Extract content after #todo
        content = command_text[5:].strip()  # Remove "#todo"
        if not content:
            return "Usage: #todo <content>"
        
        todo_manager = agent._engine.get_todo_manager()
        return await self._add_todo(todo_manager, content)
    
    async def _list_todos(self, todo_manager: Optional[TodoManager]) -> str:
        """List current TODOs."""
        if not todo_manager:
            return "TODO manager not available."

        current_list = todo_manager.get_current_list()
        if not current_list or not current_list.items:
            return "No TODOs in current list."
        
        from ...components.task import TodoStatus
        
        result = ["**Current TODOs:**\n"]
        
        # Group by status
        for status in TodoStatus:
            todos = current_list.get_by_status(status)
            if not todos:
                continue
            
            status_emoji = {
                TodoStatus.PENDING: "⏳",
                TodoStatus.IN_PROGRESS: "🔄", 
                TodoStatus.COMPLETED: "✅",
                TodoStatus.FAILED: "❌",
                TodoStatus.CANCELLED: "🚫",
                TodoStatus.BLOCKED: "⛔"
            }
            
            emoji = status_emoji[status] if status in status_emoji else '📋'
            result.append(f"## {emoji} {status.value.title().replace('_', ' ')}")
            
            for todo in sorted(todos, key=lambda t: t.created_at):
                priority_emoji = {
                    "urgent": "🚨",
                    "high": "⚠️", 
                    "medium": "📋",
                    "low": "📝"
                }
                
                emoji = priority_emoji[todo.priority.value] if todo.priority.value in priority_emoji else "📋"
                result.append(f"  {emoji} `{todo.id[:8]}` {todo.content}")
                
                if todo.error_message:
                    result.append(f"    ❌ {todo.error_message}")
        
        return "\n".join(result)
    
    async def _add_todo(self, todo_manager: Optional[TodoManager], content: str) -> str:
        """Add a new TODO."""
        if not todo_manager:
            return "TODO manager not available."

        from ...components.task import TodoPriority
        
        # Determine priority from content
        priority = TodoPriority.MEDIUM
        if "urgent" in content.lower() or "asap" in content.lower():
            priority = TodoPriority.URGENT
        elif "important" in content.lower() or "high" in content.lower():
            priority = TodoPriority.HIGH
        elif "low" in content.lower() or "later" in content.lower():
            priority = TodoPriority.LOW
        
        todo_id = todo_manager.add_todo(content, priority=priority)
        if todo_id:
            return f"✅ Added TODO `{todo_id[:8]}`: {content}"
        else:
            return "❌ Failed to add TODO"
    
    async def _complete_todo(self, todo_manager: Optional[TodoManager], todo_id: str) -> str:
        """Mark TODO as completed."""
        if not todo_manager:
            return "TODO manager not available."

        current_list = todo_manager.get_current_list()
        if not current_list:
            return "No current TODO list."
        
        # Find TODO by partial ID
        todo = None
        for item in current_list.items:
            if item.id.startswith(todo_id):
                todo = item
                break
        
        if not todo:
            return f"TODO not found: {todo_id}"
        
        if todo_manager.mark_todo_completed(todo.id):
            return f"✅ Marked TODO as completed: {todo.content}"
        else:
            return f"❌ Failed to complete TODO: {todo_id}"
    
    async def _fail_todo(self, todo_manager: Optional[TodoManager], todo_id: str, reason: str) -> str:
        """Mark TODO as failed."""
        if not todo_manager:
            return "TODO manager not available."

        current_list = todo_manager.get_current_list()
        if not current_list:
            return "No current TODO list."
        
        # Find TODO by partial ID
        todo = None
        for item in current_list.items:
            if item.id.startswith(todo_id):
                todo = item
                break
        
        if not todo:
            return f"TODO not found: {todo_id}"
        
        if todo_manager.mark_todo_failed(todo.id, reason):
            return f"❌ Marked TODO as failed: {todo.content}\nReason: {reason}"
        else:
            return f"❌ Failed to mark TODO as failed: {todo_id}"
    
    async def _delete_todo(self, todo_manager: Optional[TodoManager], todo_id: str) -> str:
        """Delete a TODO."""
        if todo_manager is None:
            raise RuntimeError("TODO manager not available")

        current_list = todo_manager.get_current_list()
        if not current_list:
            return "No current TODO list."
        
        # Find TODO by partial ID
        todo = None
        for item in current_list.items:
            if item.id.startswith(todo_id):
                todo = item
                break
        
        if not todo:
            return f"TODO not found: {todo_id}"
        
        if current_list.remove_todo(todo.id):
            return f"🗑️ Deleted TODO: {todo.content}"
        else:
            return f"❌ Failed to delete TODO: {todo_id}"
    
    async def _show_progress(self, todo_manager: Optional[TodoManager]) -> str:
        """Show progress summary."""
        if todo_manager is None:
            raise RuntimeError("TODO manager not available")

        current_list = todo_manager.get_current_list()
        if not current_list:
            return "No current TODO list."
        
        summary_obj = current_list.get_status_summary()
        # Convert to dict-like format for compatibility
        summary = {
            'total': summary_obj.total,
            'progress': summary_obj.completed / summary_obj.total if summary_obj.total > 0 else 0,
            'completed': summary_obj.completed,
            'pending': summary_obj.pending,
            'in_progress': summary_obj.in_progress,
            'failed': summary_obj.failed,
            'blocked': summary_obj.blocked
        }
        
        progress_bar = ""
        if summary["total"] > 0:
            progress_pct = int(summary["progress"] * 100)
            filled = int(progress_pct / 10)
            progress_bar = "█" * filled + "░" * (10 - filled)
        
        return f"""**TODO Progress Summary:**

📊 Progress: {progress_bar} {int((summary['progress'] if 'progress' in summary else 0) * 100)}%

📈 **Statistics:**
  • Total: {summary['total'] if 'total' in summary else 0}
  • Completed: {summary['completed'] if 'completed' in summary else 0}
  • In Progress: {summary['in_progress'] if 'in_progress' in summary else 0}
  • Pending: {summary['pending'] if 'pending' in summary else 0}
  • Failed: {summary['failed'] if 'failed' in summary else 0}
  • Blocked: {summary['blocked'] if 'blocked' in summary else 0}
"""
    
    async def _export_todos(self, todo_manager: Optional[TodoManager], format_type: str) -> str:
        """Export TODOs."""
        if todo_manager is None:
            raise RuntimeError("TODO manager not available")

        current_list = todo_manager.get_current_list()
        if not current_list:
            return "No current TODO list to export."
        
        try:
            exported = todo_manager.export_list(format_type)
            return f"**Exported TODOs ({format_type}):**\n\n```{format_type}\n{exported}\n```"
        except ValueError as e:
            return f"❌ Export failed: {str(e)}"
    
    async def _clear_todos(self, todo_manager: Optional[TodoManager]) -> str:
        """Clear all TODOs."""
        if todo_manager is None:
            raise RuntimeError("TODO manager not available")

        current_list = todo_manager.get_current_list()
        if not current_list:
            return "No current TODO list to clear."
        
        todo_count = len(current_list.items)
        todo_manager.create_list()  # Create new empty list
        
        return f"🗑️ Cleared {todo_count} TODOs from the list."