"""Command system for agent REPL."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Callable, Any
from enum import Enum


class CommandType(Enum):
    """Types of commands in the REPL."""
    SLASH = "slash"  # Commands starting with /
    META = "meta"    # Commands starting with @
    SYSTEM = "system"  # System commands like exit, clear
    USER = "user"    # Regular user input


@dataclass
class Command:
    """Represents a parsed command."""
    type: CommandType
    name: str
    args: List[str]
    raw_input: str
    

class CommandHandler(ABC):
    """Base class for command handlers."""
    
    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the command."""
        pass
    
    @abstractmethod
    async def handle(self, command: Command, context: Dict[str, Any]) -> Any:
        """Handle the command and return result."""
        pass


class CommandRegistry:
    """Registry for command handlers and parsing."""
    
    def __init__(self) -> None:
        self.handlers: List[CommandHandler] = []
        self.slash_commands: Dict[str, Callable] = {}
        self.meta_commands: Dict[str, Callable] = {}
        
        # Register built-in commands
        self._register_builtin_commands()
    
    def _register_builtin_commands(self) -> None:
        """Register built-in slash commands."""
        self.slash_commands.update({
            "help": self._show_help,
            "clear": self._clear_screen,
            "history": self._show_history,
            "save": self._save_conversation,
            "load": self._load_conversation,
            "mode": self._change_mode,
            "model": self._change_model,
            "flows": self._list_flows,
            "memory": self._memory_status,
            "tools": self._list_tools,
            "exit": self._exit_repl,
            "quit": self._exit_repl,
        })
        
        self.meta_commands.update({
            "debug": self._toggle_debug,
            "verbose": self._toggle_verbose,
            "stream": self._toggle_streaming,
            "stats": self._show_stats,
        })
    
    def parse_command(self, user_input: str) -> Command:
        """Parse user input into a command."""
        user_input = user_input.strip()
        
        if not user_input:
            return Command(CommandType.USER, "", [], "")
        
        # Check for slash commands
        if user_input.startswith("/"):
            match = re.match(r'^/(\w+)(?:\s+(.*))?$', user_input)
            if match:
                cmd_name = match.group(1)
                args = match.group(2).split() if match.group(2) else []
                return Command(CommandType.SLASH, cmd_name, args, user_input)
        
        # Check for meta commands
        elif user_input.startswith("@"):
            match = re.match(r'^@(\w+)(?:\s+(.*))?$', user_input)
            if match:
                cmd_name = match.group(1)
                args = match.group(2).split() if match.group(2) else []
                return Command(CommandType.META, cmd_name, args, user_input)
        
        # Check for system commands
        elif user_input.lower() in ["exit", "quit", "clear", "cls"]:
            return Command(CommandType.SYSTEM, user_input.lower(), [], user_input)
        
        # Regular user input
        return Command(CommandType.USER, "", [], user_input)
    
    def register_handler(self, handler: CommandHandler) -> None:
        """Register a command handler."""
        self.handlers.append(handler)

    def register_slash_command(self, name: str, handler: Callable) -> None:
        """Register a slash command."""
        self.slash_commands[name] = handler

    def register_meta_command(self, name: str, handler: Callable) -> None:
        """Register a meta command."""
        self.meta_commands[name] = handler
    
    async def execute_command(self, command: Command, context: Dict[str, Any]) -> Any:
        """Execute a parsed command."""
        # Handle slash commands
        if command.type == CommandType.SLASH:
            if command.name in self.slash_commands:
                return await self._call_handler(self.slash_commands[command.name], command, context)
            else:
                return f"Unknown command: /{command.name}. Type /help for available commands."
        
        # Handle meta commands  
        elif command.type == CommandType.META:
            if command.name in self.meta_commands:
                return await self._call_handler(self.meta_commands[command.name], command, context)
            else:
                return f"Unknown meta command: @{command.name}"
        
        # Handle system commands
        elif command.type == CommandType.SYSTEM:
            if command.name in ["exit", "quit"]:
                context["should_exit"] = True
                return "Goodbye!"
            elif command.name in ["clear", "cls"]:
                return self._clear_screen(command, context)
        
        # Let handlers process other commands
        for handler in self.handlers:
            if handler.can_handle(command):
                return await handler.handle(command, context)
        
        # Default to user input (will be processed by agent)
        return None
    
    async def _call_handler(self, handler: Callable, command: Command, context: Dict[str, Any]) -> Any:
        """Call a handler function."""
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            return await handler(command, context)
        else:
            return handler(command, context)
    
    # Built-in command implementations
    def _show_help(self, command: Command, context: Dict[str, Any]) -> str:
        """Show available commands."""
        help_text = """
📚 **Available Commands**

**Slash Commands:**
  /help              - Show this help message
  /clear             - Clear the screen
  /history           - Show conversation history
  /save [filename]   - Save conversation to file
  /load [filename]   - Load conversation from file
  /mode [mode]       - Change agent mode (chat, task, debug)
  /model [name]      - Change the active model
  /flows             - List available flows
  /memory            - Show memory status
  /tools             - List available tools
  /tool              - Tool management commands
  /todo              - TODO management commands
  /mcp               - MCP server management
  /exit, /quit       - Exit the REPL

**Meta Commands:**
  @debug            - Toggle debug mode
  @verbose          - Toggle verbose output
  @stream           - Toggle response streaming
  @stats            - Show session statistics

**Tool Execution:**
  @<tool_name> <args> - Execute tool directly
  Examples:
    @read file_path="/path/to/file.py"
    @bash command="ls -la"
    @grep pattern="class.*:" path="." include="*.py"

**Special Commands:**
  !<flow_name> <args> - Execute flow directly
  ?[state|memory|context|flows|tools] - Introspect agent
  #todo <content>     - Quick add TODO

**Shortcuts:**
  Ctrl+C            - Cancel current operation
  Ctrl+D            - Exit REPL
  ↑/↓               - Navigate command history
"""
        return help_text.strip()
    
    def _clear_screen(self, command: Command, context: Dict[str, Any]) -> str:
        """Clear the screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        return ""
    
    def _show_history(self, command: Command, context: Dict[str, Any]) -> str:
        """Show command history."""
        history = context["command_history"] if "command_history" in context else []
        if not history:
            return "No command history yet."
        
        return "**Command History:**\n" + "\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(history[-20:]))
    
    async def _save_conversation(self, command: Command, context: Dict[str, Any]) -> str:
        """Save conversation to file."""
        filename = command.args[0] if command.args else "conversation.json"
        # Implementation would save conversation
        return f"Conversation saved to {filename}"
    
    async def _load_conversation(self, command: Command, context: Dict[str, Any]) -> str:
        """Load conversation from file."""
        filename = command.args[0] if command.args else "conversation.json"
        # Implementation would load conversation
        return f"Conversation loaded from {filename}"
    
    def _change_mode(self, command: Command, context: Dict[str, Any]) -> str:
        """Change agent mode."""
        if not command.args:
            current_mode = context["mode"] if "mode" in context else "chat"
            return f"Current mode: {current_mode}\nAvailable modes: chat, task, debug"
        
        mode = command.args[0].lower()
        if mode in ["chat", "task", "debug"]:
            context["mode"] = mode
            return f"Mode changed to: {mode}"
        else:
            return f"Invalid mode: {mode}. Available modes: chat, task, debug"
    
    def _change_model(self, command: Command, context: Dict[str, Any]) -> str:
        """Change the active model."""
        if not command.args:
            current_model = context["model"] if "model" in context else "default"
            return f"Current model: {current_model}"
        
        model = command.args[0]
        context["model"] = model
        return f"Model changed to: {model}"
    
    async def _list_flows(self, command: Command, context: Dict[str, Any]) -> str:
        """List available flows."""
        # Get flows from registry
        from flowlib.flows.registry.registry import flow_registry
        flows = flow_registry.get_all_flow_metadata()
        
        if not flows:
            return "No flows registered."
        
        flow_list = "**Available Flows:**\n"
        for flow_name, flow_metadata in flows.items():
            description = getattr(flow_metadata, 'description', 'No description')
            flow_list += f"  • {flow_name}: {description}\n"
        
        return flow_list
    
    async def _memory_status(self, command: Command, context: Dict[str, Any]) -> str:
        """Show memory status."""
        if "agent" not in context:
            return "No agent available."
        agent = context["agent"]
        if not hasattr(agent, "memory"):
            return "No memory system available."
        
        # Get memory stats
        stats = await agent.memory.get_stats()
        
        working_items = stats["working_memory_items"] if "working_memory_items" in stats else 0
        vector_items = stats["vector_memory_items"] if "vector_memory_items" in stats else 0
        graph_nodes = stats["knowledge_graph_nodes"] if "knowledge_graph_nodes" in stats else 0
        graph_edges = stats["knowledge_graph_edges"] if "knowledge_graph_edges" in stats else 0
        storage_mb = stats["total_storage_mb"] if "total_storage_mb" in stats else 0
        
        return f"""**Memory Status:**
  Working Memory: {working_items} items
  Vector Memory: {vector_items} items
  Knowledge Graph: {graph_nodes} nodes, {graph_edges} edges
  Total Storage: {storage_mb:.2f} MB
"""
    
    async def _list_tools(self, command: Command, context: Dict[str, Any]) -> str:
        """List available tools."""
        if "agent" not in context:
            return "No agent available."
        
        tools = context["available_tools"] if "available_tools" in context else []
        if not tools:
            return "No tools available."
        
        tool_list = "**Available Tools:**\n"
        for tool in tools:
            description = tool["description"] if "description" in tool else "No description"
            tool_list += f"  • {tool['name']}: {description}\n"
        
        return tool_list
    
    def _exit_repl(self, command: Command, context: Dict[str, Any]) -> str:
        """Exit the REPL."""
        context["should_exit"] = True
        return "Goodbye!"
    
    def _toggle_debug(self, command: Command, context: Dict[str, Any]) -> str:
        """Toggle debug mode."""
        current_debug = context["debug"] if "debug" in context else False
        debug = not current_debug
        context["debug"] = debug
        return f"Debug mode: {'ON' if debug else 'OFF'}"
    
    def _toggle_verbose(self, command: Command, context: Dict[str, Any]) -> str:
        """Toggle verbose output."""
        current_verbose = context["verbose"] if "verbose" in context else False
        verbose = not current_verbose
        context["verbose"] = verbose
        return f"Verbose mode: {'ON' if verbose else 'OFF'}"
    
    def _toggle_streaming(self, command: Command, context: Dict[str, Any]) -> str:
        """Toggle response streaming."""
        current_stream = context["stream"] if "stream" in context else True
        stream = not current_stream
        context["stream"] = stream
        return f"Response streaming: {'ON' if stream else 'OFF'}"
    
    def _show_stats(self, command: Command, context: Dict[str, Any]) -> str:
        """Show session statistics."""
        stats = context["session_stats"] if "session_stats" in context else {}
        
        message_count = stats["message_count"] if "message_count" in stats else 0
        tokens_used = stats["tokens_used"] if "tokens_used" in stats else 0
        tools_called = stats["tools_called"] if "tools_called" in stats else 0
        flows_executed = stats["flows_executed"] if "flows_executed" in stats else 0
        duration = stats["duration"] if "duration" in stats else "N/A"
        
        return f"""**Session Statistics:**
  Messages: {message_count}
  Tokens Used: {tokens_used}
  Tools Called: {tools_called}
  Flows Executed: {flows_executed}
  Session Duration: {duration}
"""