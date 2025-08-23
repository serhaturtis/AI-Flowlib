"""Main REPL implementation for interactive agent sessions."""

import asyncio
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from pydantic import BaseModel, Field, ConfigDict

try:
    import readline  # For command history
except ImportError:
    readline = None  # Windows fallback

from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table

from flowlib.agent.core import Agent
from flowlib.agent.models.state import AgentState
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.runners.repl.commands import CommandRegistry, CommandType
from flowlib.agent.runners.repl.handlers import DefaultCommandHandler, AgentCommandHandler, ToolCommandHandler, TodoCommandHandler


class SessionStats(BaseModel):
    """Session statistics model."""
    model_config = ConfigDict(extra="forbid")
    
    message_count: int = Field(default=0, description="Number of messages processed")
    tokens_used: int = Field(default=0, description="Number of tokens used")
    tools_called: int = Field(default=0, description="Number of tools called")
    flows_executed: int = Field(default=0, description="Number of flows executed")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")


class REPLContext(BaseModel):
    """REPL context model."""
    model_config = ConfigDict(extra="forbid")
    
    should_exit: bool = Field(default=False, description="Whether to exit the REPL")
    command_history: List[str] = Field(default_factory=list, description="Command history")
    session_stats: SessionStats = Field(default_factory=SessionStats, description="Session statistics")
    mode: str = Field(default="chat", description="Current mode")
    debug: bool = Field(default=False, description="Debug mode enabled")
    verbose: bool = Field(default=False, description="Verbose mode enabled")
    stream: bool = Field(default=True, description="Streaming mode enabled")


class AgentStatsData(BaseModel):
    """Agent statistics data model."""
    model_config = ConfigDict(extra="forbid")
    
    flows_executed: int = Field(default=0, description="Number of flows executed")


class ModeEmojis(BaseModel):
    """Mode emoji mapping model."""
    model_config = ConfigDict(extra="forbid")
    
    chat: str = Field(default="💬", description="Chat mode emoji")
    task: str = Field(default="🎯", description="Task mode emoji")
    debug: str = Field(default="🐛", description="Debug mode emoji")


class AgentREPL:
    """Interactive REPL for agent conversations."""
    
    def __init__(
        self,
        agent: Optional[Agent] = None,
        config: Optional[AgentConfig] = None,
        history_file: Optional[str] = None
    ):
        self.agent = agent
        self.config = config
        self.console = Console()
        self.command_registry = CommandRegistry()
        self.repl_context = REPLContext()
        self.context: Dict[str, Any] = {
            "agent": agent,
            "config": config,
            "should_exit": self.repl_context.should_exit,
            "command_history": self.repl_context.command_history,
            "session_stats": self.repl_context.session_stats.model_dump(),
            "mode": self.repl_context.mode,
            "debug": self.repl_context.debug,
            "verbose": self.repl_context.verbose,
            "stream": self.repl_context.stream
        }
        
        # Setup history
        self.history_file = history_file or os.path.expanduser("~/.flowlib_agent_history")
        self._setup_history()
        
        # Register command handlers
        self._register_handlers()
        
        # Initialize agent if needed
        if not self.agent and self.config:
            self._initialize_agent()
    
    def _setup_history(self):
        """Setup command history."""
        if readline:
            # Configure readline
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind("set editing-mode emacs")
            
            # Load history
            try:
                if os.path.exists(self.history_file):
                    readline.read_history_file(self.history_file)
            except Exception:
                pass
            
            # Set history size
            readline.set_history_length(1000)
    
    def _save_history(self):
        """Save command history."""
        if readline:
            try:
                readline.write_history_file(self.history_file)
            except Exception:
                pass
    
    def _register_handlers(self):
        """Register command handlers."""
        self.command_registry.register_handler(DefaultCommandHandler())
        self.command_registry.register_handler(AgentCommandHandler())
        self.command_registry.register_handler(ToolCommandHandler())
        self.command_registry.register_handler(TodoCommandHandler())
    
    def _initialize_agent(self):
        """Initialize the agent."""
        from flowlib.agent.core import Agent
        
        self.agent = Agent(config=self.config)
        self.context["agent"] = self.agent
    
    async def start(self):
        """Start the REPL session."""
        # Show welcome message
        self._show_welcome()
        
        # Main REPL loop
        try:
            await self._repl_loop()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
        except EOFError:
            self.console.print("\n[yellow]End of input[/yellow]")
        finally:
            self._save_history()
            self._show_goodbye()
    
    def _show_welcome(self):
        """Show welcome message."""
        welcome = Panel(
            "[bold cyan]🤖 Flowlib Agent REPL[/bold cyan]\n\n"
            "Welcome to the interactive agent session!\n"
            "Type [bold]/help[/bold] for available commands.\n"
            "Type your message to start a conversation.",
            title="Welcome",
            border_style="cyan"
        )
        self.console.print(welcome)
        self.console.print()
    
    def _show_goodbye(self):
        """Show goodbye message."""
        # Calculate session duration
        duration = datetime.now() - self.repl_context.session_stats.start_time
        
        stats_table = Table(title="Session Summary", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats = self.repl_context.session_stats
        stats_table.add_row("Messages", str(stats.message_count))
        stats_table.add_row("Tokens Used", str(stats.tokens_used))
        stats_table.add_row("Tools Called", str(stats.tools_called))
        stats_table.add_row("Flows Executed", str(stats.flows_executed))
        stats_table.add_row("Duration", str(duration).split('.')[0])
        
        self.console.print()
        self.console.print(stats_table)
        self.console.print("\n[cyan]Thanks for using Flowlib Agent! Goodbye! 👋[/cyan]")
    
    async def _repl_loop(self):
        """Main REPL loop."""
        while not self.context["should_exit"]:
            try:
                # Get user input
                user_input = self._get_input()
                
                if user_input is None:  # EOF
                    break
                
                # Add to command history
                self.context["command_history"].append(user_input)
                
                # Parse command
                command = self.command_registry.parse_command(user_input)
                
                # Execute command
                result = await self.command_registry.execute_command(command, self.context)
                
                # Handle result
                if result is not None:
                    # Command produced output
                    self._display_output(result)
                elif command.type == CommandType.USER and command.raw_input:
                    # Regular user message - send to agent
                    await self._handle_user_message(command.raw_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit[/yellow]")
                continue
            except Exception as e:
                self.console.print(f"\n[red]Error: {str(e)}[/red]")
                if self.repl_context.debug:
                    import traceback
                    self.console.print(traceback.format_exc())
    
    def _get_input(self) -> Optional[str]:
        """Get user input with prompt."""
        mode = self.repl_context.mode
        mode_emojis = ModeEmojis()
        mode_emoji = mode_emojis.chat  # Default
        if mode == "task":
            mode_emoji = mode_emojis.task
        elif mode == "debug":
            mode_emoji = mode_emojis.debug
        
        try:
            return Prompt.ask(f"\n[bold cyan]{mode_emoji} You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            return None
    
    def _display_output(self, output: Any):
        """Display output to console."""
        if isinstance(output, str):
            if output.strip():  # Don't display empty strings
                self.console.print(Markdown(output))
        else:
            self.console.print(output)
    
    async def _handle_user_message(self, message: str):
        """Handle a user message by sending to agent."""
        if not self.agent:
            self.console.print("[red]No agent initialized![/red]")
            return
        
        # Update stats
        self.repl_context.session_stats.message_count += 1
        self.context["session_stats"]["message_count"] = self.repl_context.session_stats.message_count
        
        # Set up real-time activity streaming
        if self.repl_context.stream:
            # Create a real-time activity display handler
            activity_buffer = []
            
            def activity_handler(activity_text: str):
                activity_buffer.append(activity_text)
                self.console.print(activity_text)
            
            # Set up agent's activity stream to output in real-time
            if hasattr(self.agent, 'set_activity_stream_handler'):
                self.agent.set_activity_stream_handler(activity_handler)
            
            response = await self._get_agent_response(message)
        else:
            response = await self._get_agent_response(message)
        
        # Display response
        self.console.print(f"\n[bold green]🤖 Assistant[/bold green]")
        self._display_output(response)
    
    async def _get_agent_response(self, message: str) -> str:
        """Get response from agent."""
        try:
            # Send message to agent
            response = await self.agent.process_message(
                message=message,
                context=self.context
            )
            
            # Update stats if available  
            if isinstance(response, dict) and "stats" in response:
                stats_data = AgentStatsData.model_validate(response["stats"])
                self.repl_context.session_stats.flows_executed += stats_data.flows_executed
                self.context["session_stats"]["flows_executed"] = self.repl_context.session_stats.flows_executed
            
            # Build the complete response with activity details
            output_parts = []
            
            # Show activity if available
            if isinstance(response, dict) and "activity" in response and response["activity"]:
                output_parts.append(response["activity"])
            
            # Extract main content
            content = ""
            if hasattr(response, "content"):
                content = response.content
            elif isinstance(response, dict) and "content" in response:
                content = response["content"]
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            # Add content if it's not empty and not just "Task completed"
            if content and content != "Task completed":
                if output_parts:  # If we have activity, add some spacing
                    output_parts.append("")  # Empty line
                output_parts.append(content)
            
            return "\n".join(output_parts) if output_parts else content
                
        except Exception as e:
            error_msg = f"Error getting agent response: {str(e)}"
            if self.repl_context.debug:
                import traceback
                error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg


class InteractiveAgent:
    """High-level interface for creating interactive agent sessions."""
    
    @classmethod
    def create_repl(
        cls,
        agent_type: str = "default",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentREPL:
        """Create a REPL with specified agent type."""
        from ...registry import agent_registry
        
        # Get agent class
        agent_class = agent_registry.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Create config
        default_config = {
            "name": "repl_agent", 
            "persona": "Interactive REPL agent",
            "provider_name": "mock"  # Default provider for REPL
        }
        default_config.update(config or {})
        agent_config = AgentConfig(**default_config)
        
        # Create agent
        agent = agent_class(config=agent_config)
        
        # Create REPL
        return AgentREPL(agent=agent, config=agent_config, **kwargs)
    
    @classmethod
    async def start_session(
        cls,
        agent_type: str = "default",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Start an interactive session with specified agent."""
        repl = cls.create_repl(agent_type, config, **kwargs)
        await repl.start()


# Convenience function
async def start_agent_repl(agent=None, config=None, **kwargs):
    """Start an agent REPL session."""
    repl = AgentREPL(agent=agent, config=config, **kwargs)
    await repl.start()