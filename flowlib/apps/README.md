# Flowlib Applications

This directory contains example applications and tools built with the Flowlib framework.

## Applications

### 🤖 REPL Agent (`run_repl.py`)
Interactive agent development environment with tool calling capabilities.

```bash
python flowlib/apps/run_repl.py
```

Features:
- Interactive agent conversation
- Built-in tool calling (@bash, @read, @write, etc.)
- Memory and learning capabilities
- Development and debugging tools

### 🎛️ Configuration GUI (`flowlib_config_gui.py`)
Qt-based graphical user interface for managing Flowlib configurations.

```bash
python flowlib/apps/flowlib_config_gui.py
```

Features:
- Visual configuration management
- Provider setup and testing
- Knowledge plugin management
- Configuration import/export
- Modern Qt interface with theming
- Improved error handling and logging

### 🤖 Main Agent (`main_agent.py`)
Example conversational agent implementation.

```bash
python flowlib/apps/main_agent.py
```

Features:
- Production-ready agent setup
- Conversation flow handling
- Memory and context management

### 🚀 Service Startup (`start_repl_with_services.sh`)
Development utility to start external services (databases, caches, etc.) before launching the REPL.

```bash
./flowlib/apps/start_repl_with_services.sh
```

**What it does:**
- Checks for and starts Docker services (Neo4j, ChromaDB, Redis, RabbitMQ, Qdrant, Pinecone)
- Waits for services to be healthy
- Launches the REPL with all external services available
- Provides cleanup on exit

**Fallback behavior:**
- If no `docker-compose.yml` is found, runs REPL in standalone mode
- Standalone mode works with file-based providers (SQLite, local storage, etc.)

## Usage

All applications automatically:
- Create the `~/.flowlib/` directory structure
- Copy example configurations to `~/.flowlib/active_configs/`
- Initialize the knowledge plugin system

## Development

These applications serve as:
- **Examples** of how to build with Flowlib
- **Development tools** for framework contributors
- **Reference implementations** for common patterns

## Requirements

Applications may require additional dependencies beyond the core Flowlib library. Check individual application files for specific requirements.