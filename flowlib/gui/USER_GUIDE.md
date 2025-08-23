# Flowlib GUI Configuration Manager - User Guide

## Quick Start

### Launching the GUI
```bash
cd flowlib/gui
python run_gui.py
```

The application will start with a modern tabbed interface showing six main sections.

## Interface Overview

The GUI provides six main tabs for different configuration tasks:

### 1. Provider Repository 📊
**Purpose**: Manage environments and provider configurations across different deployment contexts.

**Key Features**:
- **Environment Management**: Switch between development, staging, and production
- **Role Assignments**: Assign configurations to specific roles (default-llm, primary-db, etc.)
- **Configuration Overview**: View all configurations and their assignments
- **Backup System**: Create and restore environment backups

**Common Workflows**:
1. **Switch Environment**: Use the environment dropdown to select your target environment
2. **Assign Provider**: Select a configuration and assign it to a role using the "Assign to Role" button
3. **Create Backup**: Use "Create Backup" to save your current environment state

### 2. Configuration Manager 🔧
**Purpose**: Create, edit, and manage individual provider configurations.

**Key Features**:
- **Configuration CRUD**: Create, read, update, delete configurations
- **Syntax Highlighting**: Python syntax highlighting for configuration files
- **Live Validation**: Real-time validation of configuration syntax
- **Configuration Testing**: Test configurations before deployment

**Common Workflows**:
1. **Create Configuration**: Click "Create New" → Select provider type → Write configuration code
2. **Edit Configuration**: Double-click any configuration in the table to open the editor
3. **Test Configuration**: Select a configuration and click "Test" to validate it works
4. **Duplicate Configuration**: Right-click and select "Duplicate" to copy existing configurations

### 3. Preset Manager 📋
**Purpose**: Use templates to quickly generate standard configurations.

**Key Features**:
- **Template Library**: Pre-built templates for common provider types
- **Parameter Forms**: Fill out forms instead of writing code
- **Code Generation**: Automatic generation of valid configuration code
- **Custom Presets**: Create your own reusable templates

**Common Workflows**:
1. **Use Preset**: Select preset type → Fill parameters → Generate configuration
2. **Browse Templates**: Use category filter to find specific preset types
3. **Generate from Template**: Click "Generate" to create configuration from preset

### 4. Knowledge Plugin Manager 🧠
**Purpose**: Manage domain-specific knowledge plugins that extend agent capabilities.

**Key Features**:
- **Plugin Library**: View installed knowledge plugins
- **Plugin Generation**: Create new plugins for specific domains
- **Plugin Testing**: Validate plugin functionality
- **Plugin Management**: Enable, disable, or remove plugins

**Common Workflows**:
1. **Create Plugin**: Click "Generate New Plugin" → Choose domain → Configure extraction
2. **Manage Plugins**: View plugin list, check status, enable/disable as needed
3. **Test Plugin**: Select plugin and run validation tests

### 5. Import/Export 📦
**Purpose**: Backup, share, and migrate configurations between systems.

**Key Features**:
- **Bulk Export**: Export multiple configurations at once
- **Selective Import**: Choose which configurations to import
- **Format Support**: JSON and YAML export formats
- **Backup Management**: Create and restore complete configuration backups

**Common Workflows**:
1. **Export Configurations**: Select configurations → Choose format → Export to file
2. **Import Configurations**: Choose file → Select configurations to import → Import
3. **Backup System**: Export all configurations for system backup

### 6. Testing Tools 🔬
**Purpose**: Validate configurations and test provider connections.

**Key Features**:
- **Configuration Testing**: Syntax and logic validation
- **Connection Testing**: Test actual provider connections
- **Performance Testing**: Measure configuration performance
- **Test History**: View previous test results and trends

**Common Workflows**:
1. **Test Configuration**: Select configuration → Choose test type → Run test
2. **Connection Test**: Test if provider configurations can actually connect
3. **Review History**: Check previous test results to track configuration health

## Key Concepts

### Configuration Types
- **LLM Providers**: Large Language Model configurations (OpenAI, LlamaCpp, etc.)
- **Database Providers**: Database connections (PostgreSQL, MongoDB, SQLite)
- **Vector Providers**: Vector database configurations (ChromaDB, Pinecone, Qdrant)
- **Cache Providers**: Caching systems (Redis, Memory)
- **Storage Providers**: File storage (Local, S3)
- **Graph Providers**: Graph databases (Neo4j, ArangoDB)

### Environment System
The GUI uses a three-tier environment system:
- **Development**: For local development and testing
- **Staging**: For pre-production testing
- **Production**: For live deployments

### Role-Based Assignment
Instead of hardcoding provider names, use roles:
- `default-llm`: Primary language model
- `primary-db`: Main database
- `vector-store`: Vector database for embeddings
- `cache-provider`: Caching system

## Configuration Examples

### Basic LLM Configuration
```python
from flowlib.providers.llm.base import LLMConfigResource
from flowlib.core.decorators import llm_config

@llm_config("my-llm")
class MyLLMConfig(LLMConfigResource):
    provider_type: str = "llamacpp"
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1000
    model_path: str = "/path/to/model"
```

### Database Configuration
```python
from flowlib.providers.db.base import DatabaseConfigResource
from flowlib.core.decorators import database_config

@database_config("my-db")
class MyDBConfig(DatabaseConfigResource):
    provider_type: str = "postgres"
    host: str = "localhost"
    port: int = 5432
    database: str = "flowlib"
    username: str = "user"
    password: str = "password"
```

## Best Practices

### Configuration Management
1. **Use Descriptive Names**: Choose clear, descriptive names for your configurations
2. **Test Before Deploy**: Always test configurations before assigning to production roles
3. **Version Control**: Export configurations regularly for backup
4. **Environment Separation**: Keep development and production configurations separate

### Security
1. **Sensitive Data**: Avoid hardcoding passwords or API keys in configurations
2. **Environment Variables**: Use environment variables for sensitive information
3. **Access Control**: Limit access to production configurations

### Performance
1. **Resource Limits**: Set appropriate resource limits (tokens, timeouts, etc.)
2. **Connection Pooling**: Configure connection pooling for database providers
3. **Caching**: Use cache providers to improve performance

## Troubleshooting

### Common Issues

**Configuration Won't Save**
- Check syntax highlighting for errors (red highlighting indicates issues)
- Ensure all required fields are provided
- Verify decorator syntax is correct

**Provider Connection Fails**
- Use Testing Tools to verify connection parameters
- Check network connectivity and firewall settings
- Verify credentials and permissions

**GUI Performance Issues**
- Close unused tabs to reduce memory usage
- Restart application if experiencing slowdowns
- Check system resources (CPU, memory)

### Getting Help

1. **Validation Errors**: The editor shows real-time validation - red highlighting indicates syntax errors
2. **Test Results**: Use Testing Tools to get detailed error messages
3. **Log Files**: Check console output for detailed error information
4. **Configuration Examples**: Use Preset Manager to see working configuration examples

## Configuration Management

### Key Differences
- **Visual Interface**: Graphical interface instead of terminal-based
- **Tabbed Layout**: Organized into functional tabs instead of menu navigation
- **Real-time Validation**: Immediate feedback on configuration syntax
- **Enhanced Testing**: More comprehensive testing tools

### Migration Steps
1. **Export Old Configurations**: Use the old system to export your configurations
2. **Import to GUI**: Use Import/Export tab to import your configurations
3. **Verify Assignments**: Check Provider Repository to ensure role assignments are correct
4. **Test Configurations**: Use Testing Tools to verify everything works

The GUI maintains full compatibility with existing configurations and provides a more intuitive interface for management tasks.