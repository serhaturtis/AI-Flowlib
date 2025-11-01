"""Example cache provider configuration.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the settings below for your specific setup.
"""

from flowlib.resources.decorators.decorators import cache_config


@cache_config("example-cache-provider")
class ExampleCacheProviderConfig:
    """Example configuration for the default cache provider.

    Used for caching API responses, computed results, temporary data, etc.
    Choose one of the supported providers below.
    """

    # === REDIS (Recommended) ===
    provider_type = "redis"
    settings = {
        # Redis connection settings
        "host": "localhost",
        "port": 6379,
        "db": 0,  # Database number (0-15)
        # Authentication (optional)
        "password": "",  # Set if Redis requires authentication
        "username": "",  # Set if using Redis ACLs
        # Connection settings
        "socket_timeout": 30.0,
        "socket_connect_timeout": 30.0,
        "socket_keepalive": True,
        "socket_keepalive_options": {},
        # Connection pool settings
        "connection_pool_max_connections": 50,
        "retry_on_timeout": True,
        "health_check_interval": 30,
        # Key settings
        "key_prefix": "flowlib:",  # Prefix for all keys
        "default_ttl": 3600,  # Default TTL in seconds (1 hour)
    }

    # === Alternative: MEMCACHED ===
    # provider_type = "memcached"
    # settings = {
    #     "hosts": ["127.0.0.1:11211"],  # List of memcached servers
    #     "key_prefix": "flowlib_",
    #     "default_ttl": 3600,
    #     "timeout": 30.0,
    #     "connect_timeout": 30.0,
    #     "no_delay": True
    # }

    # === Alternative: IN-MEMORY (Development only) ===
    # provider_type = "memory"
    # settings = {
    #     "max_size": 1000,        # Maximum number of cached items
    #     "default_ttl": 3600,     # Default TTL in seconds
    #     "cleanup_interval": 300  # Cleanup interval in seconds
    # }
