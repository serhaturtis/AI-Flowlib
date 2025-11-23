"""Example database provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Alias bindings are handled separately in ~/.flowlib/configs/aliases.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import database_config


@database_config("example-database-provider")
class ExampleDatabaseProviderConfig:
    """Example configuration for the default database provider.

    Used for persisting agent state, conversation history, metadata, etc.
    Choose one of the supported providers below.
    """

    # === SQLITE CONFIGURATION (DEFAULT) ===
    provider_type = "sqlite"  # SQLite provider name
    settings = {
        # SQLite file settings
        "database_path": "~/.flowlib/flowlib.db",  # Path to SQLite database file
        "create_if_missing": True,  # Create database file if it doesn't exist
        # SQLite performance settings
        "journal_mode": "WAL",  # WAL mode for better concurrency
        "isolation_level": None,  # None = autocommit mode
        "timeout": 5.0,  # Connection timeout in seconds
        "detect_types": 0,  # SQLite type detection
        # Additional connection arguments
        "connect_args": {},  # Additional connection arguments
        # Provider reliability settings (from ProviderSettings)
        "max_retries": 3,  # Maximum retry attempts
        "retry_delay": 1.0,  # Delay between retries
        "verbose": False,  # Enable verbose logging
    }

    # === ALTERNATIVE CONFIGURATIONS ===
    # Uncomment and modify one of the sections below, then comment out the SQLite config above

    # === POSTGRESQL CONFIGURATION ===
    # super().__init__(
    #     name=name,
    #     type=type,
    #     provider_type="postgresql",  # PostgreSQL provider name
    #     settings={
    #         "host": "localhost",
    #         "port": 5432,
    #         "database": "flowlib",
    #         "username": "flowlib_user",
    #         "password": "your-postgres-password",
    #
    #         # Connection pool settings
    #         "pool_size": 5,
    #         "max_overflow": 10,
    #         "pool_timeout": 30,
    #         "pool_recycle": 3600,
    #
    #         # Provider reliability settings
    #         "max_retries": 3,
    #         "retry_delay": 1.0,
    #         "verbose": False,
    #     }
    # )

    # === MONGODB CONFIGURATION ===
    # super().__init__(
    #     name=name,
    #     type=type,
    #     provider_type="mongodb",  # MongoDB provider name
    #     settings={
    #         "host": "localhost",
    #         "port": 27017,
    #         "database": "flowlib",
    #         "username": "flowlib_user",      # Optional
    #         "password": "your-mongo-password",  # Optional
    #
    #         # Connection pool settings
    #         "max_pool_size": 100,
    #         "min_pool_size": 0,
    #         "max_idle_time_ms": 120000,
    #
    #         # Provider reliability settings
    #         "max_retries": 3,
    #         "retry_delay": 1.0,
    #         "verbose": False,
    #     }
    # )
