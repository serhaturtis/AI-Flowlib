#!/usr/bin/env python3
"""
Flowlib Server Runner

Start the FastAPI server with configurable options.

Usage:
    python run_server.py                    # Start with defaults
    python run_server.py --no-reload        # Start without hot-reload
    python run_server.py --port 8080        # Start on custom port
    python run_server.py --host 0.0.0.0     # Bind to all interfaces
"""

import argparse
import os
import subprocess
import sys


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required dependencies are installed."""
    missing = []

    try:
        import server  # noqa: F401
    except ImportError:
        missing.append("server (install with: pip install -e .)")

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        missing.append("uvicorn (install with: pip install uvicorn[standard])")

    try:
        import flowlib  # noqa: F401
    except ImportError:
        print("\033[1;33mWarning: flowlib package not found in Python path\033[0m")
        print("Make sure flowlib is installed: cd ../../flowlib && pip install -e .")
        print()

    return len(missing) == 0, missing


def main() -> int:
    """Run the server with specified configuration."""
    parser = argparse.ArgumentParser(
        description="Start the Flowlib FastAPI server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable hot-reload (enabled by default in dev)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (production only)",
    )

    args = parser.parse_args()

    # Check dependencies
    print("\033[0;32mChecking dependencies...\033[0m")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print("\033[0;31mError: Missing required dependencies:\033[0m")
        for dep in missing:
            print(f"  - {dep}")
        return 1

    # Build uvicorn command
    cmd = [
        "uvicorn",
        "server.main:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.log_level,
    ]

    if not args.no_reload and args.workers is None:
        cmd.append("--reload")
        print("\033[0;32mStarting server with hot-reload enabled...\033[0m")
    else:
        print("\033[0;32mStarting server in production mode...\033[0m")

    if args.workers is not None:
        if not args.no_reload:
            print("\033[1;33mWarning: --workers specified, disabling hot-reload\033[0m")
        cmd.extend(["--workers", str(args.workers)])

    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Hot-reload: {not args.no_reload and args.workers is None}")
    print(f"Log level: {args.log_level}")
    if args.workers:
        print(f"Workers: {args.workers}")
    print()
    print("\033[0;32mStarting Flowlib Server...\033[0m")
    print()

    try:
        subprocess.run(cmd, check=True)
        return 0
    except KeyboardInterrupt:
        print("\n\033[1;33mServer stopped by user\033[0m")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\033[0;31mError: Server exited with code {e.returncode}\033[0m")
        return e.returncode
    except Exception as e:
        print(f"\033[0;31mError: {e}\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
