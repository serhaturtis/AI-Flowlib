"""Example configuration for default-database role.

This file shows how to configure a database provider for persistence.
Copy this file to ~/.flowlib/active_configs/default_database.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import database_config
from flowlib.resources.models.base import ResourceBase


@database_config("default-database") 
class DefaultDatabaseConfig(ResourceBase):
    """Example configuration for the default database provider.
    
    Used for persisting agent state, conversation history, metadata, etc.
    Choose one of the supported providers below.
    """
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(
            name=name,
            type=type,
            provider_type="sqlite",
            settings={
                # SQLite file settings
                "database_path": "~/.flowlib/flowlib.db",   # Path to SQLite database file
                "create_if_missing": True,                   # Create database file if it doesn't exist
                
                # SQLite performance settings
                "journal_mode": "WAL",                       # WAL mode for better concurrency
                "isolation_level": None,                     # None = autocommit mode
                "timeout": 5.0,                             # Connection timeout in seconds
                "detect_types": 0,                          # SQLite type detection
                
                # Additional connection arguments
                "connect_args": {},                          # Additional connection arguments
                
                # Provider reliability settings (from ProviderSettings)
                "max_retries": 3,                           # Maximum retry attempts
                "retry_delay": 1.0,                         # Delay between retries
                "verbose": False,                           # Enable verbose logging
            }
        )
    
    # === Alternative: POSTGRESQL ===
    # provider_type: str = "postgres"
    # host: str = "localhost"
    # port: int = 5432
    # database: str = "flowlib"
    # username: str = "flowlib_user"
    # password: str = "your-postgres-password"
    # 
    # # Connection settings
    # pool_size: int = 5
    # max_overflow: int = 10
    # pool_timeout: int = 30
    # pool_recycle: int = 3600
    
    # === Alternative: MYSQL ===
    # provider_type: str = "mysql"
    # host: str = "localhost"
    # port: int = 3306
    # database: str = "flowlib"
    # username: str = "flowlib_user" 
    # password: str = "your-mysql-password"
    # charset: str = "utf8mb4"
    
    # === Alternative: MONGODB ===
    # provider_type: str = "mongodb"
    # host: str = "localhost"
    # port: int = 27017
    # database: str = "flowlib"
    # username: str = "flowlib_user"  # Optional
    # password: str = "your-mongo-password"  # Optional
    # 
    # # Connection settings
    # max_pool_size: int = 100
    # min_pool_size: int = 0
    # max_idle_time_ms: int = 120000
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(name=name, type=type)