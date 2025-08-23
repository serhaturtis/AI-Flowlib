"""Example configuration for default-cache role.

This file shows how to configure a cache provider.
Copy this file to ~/.flowlib/active_configs/default_cache.py and modify as needed.
"""

from flowlib.resources.decorators.decorators import cache_config
from flowlib.resources.models.base import ResourceBase


@cache_config("default-cache")
class DefaultCacheConfig(ResourceBase):
    """Example configuration for the default cache provider.
    
    Used for caching API responses, computed results, temporary data, etc.
    Choose one of the supported providers below.
    """
    
    # === REDIS (Recommended) ===
    provider_type: str = "redis"
    
    # Redis connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0  # Database number (0-15)
    
    # Authentication (optional)
    password: str = ""  # Set if Redis requires authentication
    username: str = ""  # Set if using Redis ACLs
    
    # Connection settings
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 30.0
    socket_keepalive: bool = True
    socket_keepalive_options: dict = {}
    
    # Connection pool settings
    connection_pool_max_connections: int = 50
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Key settings
    key_prefix: str = "flowlib:"  # Prefix for all keys
    default_ttl: int = 3600      # Default TTL in seconds (1 hour)
    
    # === Alternative: MEMCACHED ===
    # provider_type: str = "memcached"
    # hosts: list = ["127.0.0.1:11211"]  # List of memcached servers
    # key_prefix: str = "flowlib_"
    # default_ttl: int = 3600
    # 
    # # Connection settings
    # timeout: float = 30.0
    # connect_timeout: float = 30.0
    # no_delay: bool = True
    
    # === Alternative: IN-MEMORY (Development only) ===
    # provider_type: str = "memory"
    # max_size: int = 1000        # Maximum number of cached items
    # default_ttl: int = 3600     # Default TTL in seconds
    # cleanup_interval: int = 300 # Cleanup interval in seconds
    
    def __init__(self, name: str, type: str, **kwargs):
        super().__init__(name=name, type=type)