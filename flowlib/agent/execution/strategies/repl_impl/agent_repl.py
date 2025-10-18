"""Main REPL implementation for interactive agent sessions."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import readline  # For command history
    READLINE_AVAILABLE = True
except ImportError:
    # Readline not available on Windows - create dummy namespace
    class DummyReadline:
        """Dummy readline for platforms without readline support."""
        def set_completer(self, func: Any) -> None: pass
        def parse_and_bind(self, string: str) -> None: pass

    readline = DummyReadline()  # type: ignore[assignment]
    READLINE_AVAILABLE = False

from pydantic import ConfigDict, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.core.models.messages import (
    AgentMessage,
    AgentMessagePriority,
    AgentMessageType,
    AgentResponseData,
)
from flowlib.agent.core.thread_manager import AgentThreadPoolManager
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.execution.strategies.repl_impl.commands import (
    CommandRegistry,
    CommandType,
)
from flowlib.agent.execution.strategies.repl_impl.handlers import (
    AgentCommandHandler,
    DefaultCommandHandler,
    TodoCommandHandler,
    ToolCommandHandler,
)
from flowlib.core.models import MutableStrictBaseModel, StrictBaseModel

logger = logging.getLogger(__name__)


class SessionStats(MutableStrictBaseModel):
    """Session statistics model."""

    message_count: int = Field(default=0, description="Number of messages processed")
    tokens_used: int = Field(default=0, description="Number of tokens used")
    tools_called: int = Field(default=0, description="Number of tools called")
    flows_executed: int = Field(default=0, description="Number of flows executed")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")


class REPLContext(StrictBaseModel):
    """REPL context model."""
    model_config = ConfigDict(extra="forbid")

    should_exit: bool = Field(default=False, description="Whether to exit the REPL")
    command_history: List[str] = Field(default_factory=list, description="Command history")
    session_stats: SessionStats = Field(default_factory=SessionStats, description="Session statistics")
    mode: str = Field(default="chat", description="Current mode")
    debug: bool = Field(default=False, description="Debug mode enabled")
    verbose: bool = Field(default=False, description="Verbose mode enabled")
    stream: bool = Field(default=True, description="Streaming mode enabled")


class ModeEmojis(StrictBaseModel):
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

    def _setup_history(self) -> None:
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

    def _save_history(self) -> None:
        """Save command history."""
        if readline:
            try:
                readline.write_history_file(self.history_file)
            except Exception:
                pass

    def _register_handlers(self) -> None:
        """Register command handlers."""
        self.command_registry.register_handler(DefaultCommandHandler())
        self.command_registry.register_handler(AgentCommandHandler())
        self.command_registry.register_handler(ToolCommandHandler())
        self.command_registry.register_handler(TodoCommandHandler())

    async def initialize(self) -> None:
        """Initialize the REPL with queue-based agent."""
        # Create agent in its own thread
        self.agent = self.thread_pool_manager.create_agent(
            self.agent_id,
            self.config
        )

        # Add agent to context for command handlers
        self.context["agent"] = self.agent

        # Start response router
        router = self.thread_pool_manager.get_response_router(self.agent_id)
        if router:
            await router.start()
            self.console.print("[green]✓[/green] Response router started")
        else:
            self.console.print("[red]✗[/red] No response router found")

        self.console.print(f"[green]✓[/green] Agent {self.agent_id} initialized and running")

    async def shutdown(self) -> None:
        """Shutdown the REPL and agent."""
        # Shutdown agent
        self.thread_pool_manager.shutdown_agent(self.agent_id)
        self.console.print(f"[yellow]Agent {self.agent_id} stopped[/yellow]")

    async def start(self) -> None:
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

    def _show_welcome(self) -> None:
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

    def _show_goodbye(self) -> None:
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

    async def _repl_loop(self) -> None:
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

    def _display_output(self, output: object) -> None:
        """Display output to console."""
        if isinstance(output, str):
            if output.strip():  # Don't display empty strings
                self.console.print(Markdown(output))
        else:
            self.console.print(output)

    async def _handle_user_message(self, message: str) -> None:
        """Handle a user message by sending to agent."""
        if not self.agent:
            self.console.print("[red]No agent initialized![/red]")
            return

        # Update stats
        self.repl_context.session_stats.message_count += 1
        self.context["session_stats"]["message_count"] = self.repl_context.session_stats.message_count

        # Get response from agent via queue - returns AgentResponseData
        response_data = await self._get_agent_response(message)

        # Display activity stream separately (if present and in appropriate mode)
        if response_data.activity:
            self._display_activity_stream(response_data.activity)

        # Display main response content
        self.console.print("\n[bold green]🤖 Assistant[/bold green]")
        self._display_output(response_data.content)

    def _display_activity_stream(self, activity: str) -> None:
        """Display activity stream separately from main response."""
        # Only show activity in debug mode or if explicitly requested
        if self.context.get("mode") == "debug":
            self.console.print("\n[dim]🔄 Activity Stream:[/dim]")
            self.console.print(f"[dim]{activity}[/dim]")

    async def _get_agent_response(self, message: str) -> AgentResponseData:
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
            logger.debug("[REPL] Sending message to agent queue...")
            message_id = await self.thread_pool_manager.send_message(
                self.agent_id, agent_message
            )
            logger.debug(f"[REPL] Message sent to agent queue with ID: {message_id}")

            # Wait for response
            logger.debug("[REPL] Waiting for response from agent...")
            response = await self.thread_pool_manager.wait_for_response(
                self.agent_id, message_id, agent_message.timeout
            )
            logger.debug(f"[REPL] Received response from agent: success={response.success}")

            # Handle response
            if not response.success:
                from flowlib.agent.core.models.messages import AgentResponseData
                return AgentResponseData(content=f"[red]Agent error: {response.error}[/red]")

            # Update stats
            self.repl_context.session_stats.flows_executed += 1
            self.context["session_stats"]["flows_executed"] = self.repl_context.session_stats.flows_executed

            # Return the typed response data directly
            return response.response_data

        except TimeoutError:
            from flowlib.agent.core.models.messages import AgentResponseData
            return AgentResponseData(content="[red]Response timeout[/red]")
        except Exception as e:
            import traceback

            from flowlib.agent.core.models.messages import AgentResponseData
            error_msg = f"Error getting agent response: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
            return AgentResponseData(content=error_msg)


# Main entry point
async def start_agent_repl(agent_id: str, config: AgentConfig, **kwargs: Any) -> None:
    """Start an agent REPL session."""
    repl = AgentREPL(agent_id=agent_id, config=config, **kwargs)
    await repl.start()
