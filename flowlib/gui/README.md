# Flowlib GUI Configuration Manager

A modern graphical interface for managing Flowlib provider configurations, built with PySide6 and following clean MVC architecture principles.

## Quick Start

```bash
cd flowlib/gui
python run_gui.py
```

## Features

### 🎯 **Core Functionality**
- **Visual Configuration Management**: Create, edit, and manage provider configurations with syntax highlighting
- **Environment Management**: Switch between development, staging, and production environments  
- **Role-Based Assignment**: Assign configurations to logical roles (default-llm, primary-db, etc.)
- **Template System**: Generate configurations from pre-built templates
- **Testing Tools**: Validate configurations and test provider connections
- **Import/Export**: Backup and migrate configurations between systems

### 🏗️ **Architecture**
- **MVC Pattern**: Clean separation between UI, business logic, and data
- **Async-First**: Non-blocking operations for better responsiveness
- **Service Factory**: Dependency injection for loose coupling
- **Controller Management**: Centralized controller lifecycle management
- **Pydantic Models**: Strict type validation throughout

### 📦 **Provider Support**
- **LLM Providers**: OpenAI, LlamaCpp, Google AI, and more
- **Databases**: PostgreSQL, MongoDB, SQLite
- **Vector Stores**: ChromaDB, Pinecone, Qdrant
- **Caching**: Redis, Memory
- **Storage**: Local filesystem, S3
- **Graph**: Neo4j, ArangoDB, JanusGraph

## Project Structure

```
flowlib/gui/
├── app.py                     # Main application entry point
├── run_gui.py                 # Application launcher
├── logic/                     # Business logic layer
│   ├── services/              # Core services
│   │   ├── service_factory.py        # Service dependency injection
│   │   ├── flowlib_integration_service.py  # File management
│   │   ├── configuration_service.py  # Configuration CRUD with validation
│   │   └── stub_services.py          # Lightweight service implementations
│   └── config_manager/        # MVC controllers
│       ├── base_controller.py        # Base controller with async operations
│       ├── configuration_controller.py    # Configuration management
│       ├── provider_repository_controller.py  # Environment/role management
│       └── preset_controller.py      # Template management
├── ui/                        # User interface layer
│   ├── main_window.py         # Main application window
│   ├── pages/                 # Main application pages
│   │   ├── configuration_manager/    # Configuration CRUD interface
│   │   ├── provider_repository/      # Environment management
│   │   ├── preset_manager/           # Template system
│   │   ├── knowledge_plugin_manager/ # Plugin management
│   │   ├── import_export/            # Backup/migration tools
│   │   └── testing_tools/            # Validation and testing
│   ├── dialogs/               # Modal dialogs
│   │   ├── configuration_editor_dialog.py  # Configuration editor with syntax highlighting
│   │   ├── file_browser_dialog.py    # File selection and management
│   │   └── settings_dialog.py        # Application settings
│   └── widgets/               # Reusable UI components
│       ├── drag_drop_table.py        # Drag-and-drop configuration table
│       └── template_manager.py       # Template management widget
└── tests/                     # Test scripts
    ├── test_gui_startup.py           # Component instantiation tests
    └── test_minimal_gui.py           # Minimal functionality tests
```

## Technical Implementation

### Service Layer
- **FlowlibIntegrationService**: Core file management and configuration persistence
- **ConfigurationService**: Advanced configuration management with AST validation
- **Stub Services**: Lightweight implementations for Import/Export, Presets, Plugins, Repository management, and Testing

### Controller Layer
- **Async Operations**: All operations run asynchronously to maintain UI responsiveness
- **Operation Management**: Prevents duplicate operations and manages operation state
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Signal/Slot Communication**: Qt-based communication between components

### UI Layer
- **Tabbed Interface**: Six main functional areas organized as tabs
- **Syntax Highlighting**: Python syntax highlighting for configuration editing
- **Real-time Validation**: Immediate feedback on configuration syntax and structure
- **Drag-and-Drop**: Intuitive configuration management with drag-and-drop support

## Configuration Management

### Configuration Types
The GUI supports all Flowlib provider types:
- `@llm_config` - Language model providers
- `@database_config` - Database connections  
- `@vector_config` - Vector databases
- `@cache_config` - Caching systems
- `@storage_config` - File storage
- `@graph_config` - Graph databases

### Environment System
Three-tier environment management:
- **Development**: Local development and testing
- **Staging**: Pre-production validation
- **Production**: Live deployment configurations

### Role-Based Assignment
Logical role assignment instead of hardcoded names:
- `default-llm` - Primary language model
- `primary-db` - Main database
- `vector-store` - Vector database for embeddings
- `cache-provider` - Caching system

## Development

### Requirements
- Python 3.11+
- PySide6
- Pydantic v2
- Async/await support

### Architecture Principles
- **No fallbacks, no workarounds**: Single source of truth with strict contracts
- **Async-first design**: All operations built on asyncio
- **Type safety everywhere**: Strict Pydantic models with `ConfigDict(extra="forbid")`
- **Config-driven provider access**: Switch providers without code changes
- **Clean MVC separation**: UI, business logic, and data layers clearly separated

### Testing
```bash
# Test component creation
python test_minimal_gui.py

# Test full GUI startup  
python test_gui_startup.py

# Launch application
python run_gui.py
```

## Modern Configuration Interface

The GUI provides a modern graphical interface for configuration management while maintaining full compatibility with existing configurations.

### Key Improvements
- **Visual Interface**: Graphical interface instead of terminal-based navigation
- **Real-time Validation**: Immediate syntax checking and error feedback
- **Enhanced Testing**: Comprehensive validation and connection testing tools
- **Better Organization**: Tabbed interface with clear functional separation
- **Improved Workflows**: Streamlined configuration creation and management

### Compatibility
- All existing configuration files work without modification
- Environment and role assignments are preserved
- Configuration syntax remains identical
- Full import/export compatibility with existing systems

For detailed usage instructions, see [USER_GUIDE.md](USER_GUIDE.md).