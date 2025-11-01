#!/usr/bin/env python3
"""Unified Flowlib Agent CLI - single entry point for all execution modes."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add project to Python path
apps_dir = os.path.dirname(os.path.abspath(__file__))
flowlib_root = os.path.dirname(apps_dir)
project_root = os.path.dirname(flowlib_root)
sys.path.insert(0, project_root)

from flowlib.agent.execution.strategy import ExecutionMode  # noqa: E402
from flowlib.agent.launcher import AgentLauncher  # noqa: E402

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging based on verbosity."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Quiet noisy libraries
    if level != "DEBUG":
        logging.getLogger("flowlib.providers").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.ERROR)


async def main() -> int:
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        prog="flowlib-agent",
        description="Unified Flowlib Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch REPL for book-writer agent
  flowlib-agent run --project ./projects/book-writer --agent book-writer --mode repl

  # Run autonomous execution
  flowlib-agent run --agent my-agent --mode autonomous --max-cycles 20

  # Start daemon with timer trigger
  flowlib-agent run --agent email-agent --mode daemon --trigger timer:3600

  # Start remote worker
  flowlib-agent run --agent worker --mode remote --queue tasks

Execution Modes:
  repl        - Interactive REPL with Rich UI
  autonomous  - Run N cycles then exit
  daemon      - Continuous execution with triggers
  remote      - Message queue consumer
        """,
    )

    # Global options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run agent")
    run_parser.add_argument("--project", type=str, help="Project path (default: ~/.flowlib/)")
    run_parser.add_argument("--agent", required=True, help="Agent configuration name")
    run_parser.add_argument(
        "--mode",
        required=True,
        choices=["repl", "autonomous", "daemon", "remote"],
        help="Execution mode",
    )

    # Mode-specific options
    run_parser.add_argument("--max-cycles", type=int, help="[autonomous] Maximum execution cycles")
    run_parser.add_argument(
        "--trigger",
        type=str,
        help="[daemon] Trigger config (e.g., 'timer:3600' or 'email:gmail-default:300')",
    )
    run_parser.add_argument("--queue", type=str, help="[remote] Task queue name")
    run_parser.add_argument("--config", type=str, help="Path to mode-specific config file (YAML)")

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        configure_logging("DEBUG")
    elif args.verbose:
        configure_logging("INFO")
    else:
        configure_logging("WARNING")

    if args.command == "run":
        return await run_agent(args)

    return 0


async def run_agent(args: argparse.Namespace) -> int:
    """Run agent in specified mode."""

    try:
        # Parse execution mode
        mode = ExecutionMode(args.mode)

        # Build execution config
        execution_config = await build_execution_config(mode, args)

        # Create launcher
        launcher = AgentLauncher(project_path=args.project)
        await launcher.initialize()

        # Launch agent
        await launcher.launch(
            agent_config_name=args.agent, mode=mode, execution_config=execution_config
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to run agent: {e}", exc_info=True)
        return 1


async def build_execution_config(mode: ExecutionMode, args: argparse.Namespace) -> dict[str, Any]:
    """Build mode-specific execution config from CLI args."""

    # If config file provided, load it
    if args.config:
        import yaml

        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {args.config}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    # Otherwise build from CLI args
    config: dict[str, Any] = {}

    if mode == ExecutionMode.AUTONOMOUS:
        if args.max_cycles:
            config["max_cycles"] = args.max_cycles

    elif mode == ExecutionMode.DAEMON:
        if args.trigger:
            config["message_sources"] = parse_trigger_arg(args.trigger)

    elif mode == ExecutionMode.REMOTE:
        if args.queue:
            config["task_queue_name"] = args.queue

    return config


def parse_trigger_arg(trigger: str) -> list:
    """Parse trigger CLI argument to message source config.

    Examples:
        timer:3600 -> TimerMessageSourceConfig(interval_seconds=3600)
        email:gmail:300 -> EmailMessageSourceConfig(provider=gmail, interval=300)
    """
    from flowlib.agent.core.message_sources import (
        EmailMessageSourceConfig,
        TimerMessageSourceConfig,
    )

    parts = trigger.split(":")
    trigger_type = parts[0]

    if trigger_type == "timer":
        interval = int(parts[1]) if len(parts) > 1 else 3600
        return [
            TimerMessageSourceConfig(name="cli_timer", interval_seconds=interval, run_on_start=True)
        ]

    elif trigger_type == "email":
        provider = parts[1] if len(parts) > 1 else "default-email"
        interval = int(parts[2]) if len(parts) > 2 else 300
        return [
            EmailMessageSourceConfig(
                name="cli_email",
                email_provider_name=provider,
                check_interval_seconds=interval,
            )
        ]

    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
