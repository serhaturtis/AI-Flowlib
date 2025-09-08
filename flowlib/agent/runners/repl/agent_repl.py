"""Main REPL implementation for interactive agent sessions."""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

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

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.models.state import AgentState, AgentStats
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.core.thread_manager import AgentThreadPoolManager
from flowlib.agent.core.models.messages import (
    AgentMessage, AgentResponse, AgentMessageType, AgentMessagePriority
)
from flowlib.agent.runners.repl.commands import CommandRegistry, CommandType
from flowlib.agent.runners.repl.handlers import DefaultCommandHandler, AgentCommandHandler, ToolCommandHandler, TodoCommandHandler
from flowlib.agent.core.agent_activity_formatter import AgentActivityFormatter


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
        agent_id: str,
        config: Optional[AgentConfig] = None,
        history_file: Optional[str] = None
    ):
        self.agent_id = agent_id
        if not config:
            raise ValueError("AgentConfig is required - no default agent configuration allowed")
        self.config = config
        self.console = Console()
        self.command_registry = CommandRegistry()
        self.repl_context = REPLContext()
        
        # Queue-based agent communication
        self.thread_pool_manager = AgentThreadPoolManager()
        self.agent: Optional[BaseAgent] = None
        
        self.context: Dict[str, Any] = {
            "agent_id": agent_id,
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
        
        # Agent initialization handled by initialize() method
    
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
    
    async def initialize(self):
        """Initialize the REPL with queue-based agent."""
        # Create agent in its own thread
        self.agent = self.thread_pool_manager.create_agent(
            self.agent_id,
            self.config
        )
        
        # Start response router
        router = self.thread_pool_manager.get_response_router(self.agent_id)
        if router:
            await router.start()
            self.console.print(f"[green]✓[/green] Response router started")
        else:
            self.console.print(f"[red]✗[/red] No response router found")
        
        self.console.print(f"[green]✓[/green] Agent {self.agent_id} initialized and running")
    
    async def shutdown(self):
        """Shutdown the REPL and agent."""
        # Shutdown agent
        self.thread_pool_manager.shutdown_agent(self.agent_id)
        self.console.print(f"[yellow]Agent {self.agent_id} stopped[/yellow]")
    
    async def start(self):
        """Start the REPL session."""
        # Initialize agent
        await self.initialize()
        
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
            # Ensure cleanup
            await self.shutdown()
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
        
        # Get response from agent via queue
        response = await self._get_agent_response(message)
        
        # Display response
        self.console.print(f"\n[bold green]🤖 Assistant[/bold green]")
        self._display_output(response)
    
    async def _get_agent_response(self, message: str) -> str:
        """Get response from agent using queue-based communication."""
        try:
            logger.debug(f"[REPL] Creating agent message for: {message[:50]}...")
            
            # Create agent message
            agent_message = AgentMessage(
                message_type=AgentMessageType.CONVERSATION,
                content=message,
                context={
                    "session_stats": self.context["session_stats"],
                    "mode": self.context["mode"]
                },
                response_queue_id=f"repl_{self.agent_id}",
                priority=AgentMessagePriority.NORMAL,
                timeout=None  # Disabled timeout
            )
            
            logger.debug(f"[REPL] Agent message created with ID: {agent_message.message_id}")
            
            # Show immediate acknowledgment
            self.console.print("[dim]Message received, processing...[/dim]")
            
            # Send to agent queue
            logger.debug(f"[REPL] Sending message to agent queue...")
            message_id = await self.thread_pool_manager.send_message(
                self.agent_id, agent_message
            )
            logger.debug(f"[REPL] Message sent to agent queue with ID: {message_id}")
            
            # Wait for response
            logger.debug(f"[REPL] Waiting for response from agent...")
            response = await self.thread_pool_manager.wait_for_response(
                self.agent_id, message_id, agent_message.timeout
            )
            logger.debug(f"[REPL] Received response from agent: success={response.success}")
            
            # Handle response
            if not response.success:
                return f"[red]Agent error: {response.error}[/red]"
            
            # Extract content from response
            content = ""
            response_data = response.response_data
            if isinstance(response_data, dict) and "content" in response_data:
                content = response_data["content"]
            else:
                content = str(response_data)
            
            # Update stats
            self.repl_context.session_stats.flows_executed += 1
            self.context["session_stats"]["flows_executed"] = self.repl_context.session_stats.flows_executed
            
            # Build output with activity stream
            output_parts = []
            
            # Show activity if available
            if response.activity_stream:
                formatted_activity = AgentActivityFormatter.format_activity_stream(response.activity_stream)
                if formatted_activity.strip():
                    output_parts.append(formatted_activity)
            
            # Update conversation history
            # Conversation history is managed by agent state manager - no duplication needed
            
            # Add main content
            if content and content != "Task completed":
                if output_parts:  # If we have activity, add some spacing
                    output_parts.append("")  # Empty line
                output_parts.append(content)
            
            return "\n".join(output_parts) if output_parts else content
                
        except TimeoutError:
            return f"[red]Response timeout[/red]"
        except Exception as e:
            import traceback
            error_msg = f"Error getting agent response: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
            return error_msg


# Main entry point
async def start_agent_repl(agent_id: str, config: AgentConfig, **kwargs):
    """Start an agent REPL session."""
    repl = AgentREPL(agent_id=agent_id, config=config, **kwargs)
    await repl.start()