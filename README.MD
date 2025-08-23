# AI-Flowlib

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

A Python framework for building AI agent systems with provider abstractions, flow-based processing, and knowledge management. Built with async-first design, type safety, and clean architecture patterns.

## Features

- **Agent System**: Multi-tier memory, planning, learning, and reflection capabilities
- **Provider Abstractions**: Unified interface for LLMs, databases, vector stores, and caches
- **Flow Processing**: Type-safe async pipelines with `@flow` and `@pipeline` decorators
- **Knowledge Management**: Plugin system for extensible knowledge sources
- **Tool Calling**: Structured tool execution with automatic schema generation
- **Configuration**: Auto-discovery system with example configurations

## Installation

```bash
git clone https://github.com/your-org/AI-Flowlib.git
cd AI-Flowlib
pip install -e .
```

## Quick Start

```python
from flowlib.flows.decorators import flow
from pydantic import BaseModel

class ProcessRequest(BaseModel):
    text: str

class ProcessResult(BaseModel):
    tokens: list[str]
    count: int

@flow(name="text-processor")
async def process_text(request: ProcessRequest) -> ProcessResult:
    tokens = request.text.split()
    return ProcessResult(tokens=tokens, count=len(tokens))

# Execute flow
result = await process_text(ProcessRequest(text="Hello world"))
```

## Applications

The framework includes example applications in `flowlib/apps/`:

```bash
# Interactive REPL with agent capabilities
python flowlib/apps/run_repl.py

# Qt-based configuration GUI
python flowlib/apps/flowlib_config_gui.py

# Example conversational agent
python flowlib/apps/main_agent.py
```

## Configuration

Flowlib automatically creates a `~/.flowlib/` directory with example configurations:

```bash
~/.flowlib/
├── active_configs/         # Provider configurations
├── knowledge_plugins/      # Custom knowledge plugins
├── logs/                  # Application logs
└── temp/                  # Temporary files
```

Example configurations are copied automatically on first import.

## Testing

```bash
# Run tests
python -m pytest flowlib/tests/

# Run with coverage
python -m pytest flowlib/tests/ --cov=flowlib
```

## Project Structure

```
flowlib/
├── agent/          # Agent system (memory, planning, learning)
├── apps/           # Example applications
├── core/           # Core framework components
├── flows/          # Flow processing engine
├── providers/      # Provider abstractions (LLM, DB, Vector, etc.)
├── resources/      # Configuration and resource management
├── tools/          # Tool calling system
└── tests/          # Test suite
```

## License

MIT License - see LICENSE file for details.
