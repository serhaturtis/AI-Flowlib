"""Production-ready settings and configuration management.

This module provides comprehensive configuration management for AI-Flowlib,
including environment variable handling, configuration file loading,
secret management, and validation.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import Field, field_validator, ConfigDict, AliasChoices
import json
import yaml

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FlowlibSettings(BaseSettings):
    """Base settings class for AI-Flowlib with environment variable support.
    
    This class provides the foundation for all configuration in AI-Flowlib,
    with automatic environment variable loading, file-based configuration,
    and validation.
    """
    
    # Environment configuration  
    environment: str = Field(default="development", validation_alias=AliasChoices("FLOWLIB_ENV", "env", "environment"))
    debug: bool = Field(default=False, validation_alias=AliasChoices("FLOWLIB_DEBUG", "debug")) 
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("FLOWLIB_LOG_LEVEL", "log_level"))
    
    # Core paths
    data_dir: Path = Field(default=Path.cwd() / "data", validation_alias=AliasChoices("FLOWLIB_DATA_DIR", "data_dir"))
    config_dir: Path = Field(default=Path.cwd() / "config", validation_alias=AliasChoices("FLOWLIB_CONFIG_DIR", "config_dir")) 
    logs_dir: Path = Field(default=Path.cwd() / "logs", validation_alias=AliasChoices("FLOWLIB_LOGS_DIR", "logs_dir"))
    
    # Security settings
    secret_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("FLOWLIB_SECRET_KEY", "secret_key"))
    encryption_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("FLOWLIB_ENCRYPTION_KEY", "encryption_key"))
    
    # Provider settings
    default_llm_provider: str = Field(default="llamacpp", validation_alias=AliasChoices("FLOWLIB_DEFAULT_LLM_PROVIDER", "default_llm_provider"))
    default_db_provider: str = Field(default="sqlite", validation_alias=AliasChoices("FLOWLIB_DEFAULT_DB_PROVIDER", "default_db_provider"))
    default_vector_provider: str = Field(default="chroma", validation_alias=AliasChoices("FLOWLIB_DEFAULT_VECTOR_PROVIDER", "default_vector_provider"))
    
    # Performance settings
    max_concurrent_flows: int = Field(default=10, validation_alias=AliasChoices("FLOWLIB_MAX_CONCURRENT_FLOWS", "max_concurrent_flows"))
    flow_timeout_seconds: int = Field(default=300, validation_alias=AliasChoices("FLOWLIB_FLOW_TIMEOUT", "flow_timeout_seconds"))
    cache_ttl_seconds: int = Field(default=3600, validation_alias=AliasChoices("FLOWLIB_CACHE_TTL", "cache_ttl_seconds"))
    
    # Memory settings
    max_memory_entries: int = Field(default=10000, validation_alias=AliasChoices("FLOWLIB_MAX_MEMORY_ENTRIES", "max_memory_entries"))
    memory_cleanup_interval: int = Field(default=3600, validation_alias=AliasChoices("FLOWLIB_MEMORY_CLEANUP_INTERVAL", "memory_cleanup_interval"))
    
    # Agent settings
    agent_state_persistence: bool = Field(default=True, validation_alias=AliasChoices("FLOWLIB_AGENT_PERSISTENCE", "agent_state_persistence"))
    agent_max_cycles: int = Field(default=100, validation_alias=AliasChoices("FLOWLIB_AGENT_MAX_CYCLES", "agent_max_cycles"))
    agent_reflection_enabled: bool = Field(default=True, validation_alias=AliasChoices("FLOWLIB_AGENT_REFLECTION", "agent_reflection_enabled"))
    
    # Monitoring settings
    metrics_enabled: bool = Field(default=False, validation_alias=AliasChoices("FLOWLIB_METRICS_ENABLED", "metrics_enabled"))
    tracing_enabled: bool = Field(default=False, validation_alias=AliasChoices("FLOWLIB_TRACING_ENABLED", "tracing_enabled"))
    health_check_interval: int = Field(default=60, validation_alias=AliasChoices("FLOWLIB_HEALTH_CHECK_INTERVAL", "health_check_interval"))
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('data_dir', 'config_dir', 'logs_dir')
    @classmethod
    def validate_directories(cls, v):
        """Ensure directories exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('max_concurrent_flows', 'flow_timeout_seconds', 'max_memory_entries')
    @classmethod
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
        

class DatabaseSettings(BaseSettings):
    """Database-specific settings."""
    
    host: str = Field(default="localhost", validation_alias=AliasChoices("DB_HOST", "host"))
    port: int = Field(default=5432, validation_alias=AliasChoices("DB_PORT", "port"))
    name: str = Field(default="flowlib", validation_alias=AliasChoices("DB_NAME", "name"))
    username: str = Field(default="flowlib", validation_alias=AliasChoices("DB_USERNAME", "username"))
    password: Optional[str] = Field(default=None, validation_alias=AliasChoices("DB_PASSWORD", "password"))
    pool_size: int = Field(default=5, validation_alias=AliasChoices("DB_POOL_SIZE", "pool_size"))
    max_overflow: int = Field(default=10, validation_alias=AliasChoices("DB_MAX_OVERFLOW", "max_overflow"))
    pool_timeout: int = Field(default=30, validation_alias=AliasChoices("DB_POOL_TIMEOUT", "pool_timeout"))
    
    model_config = ConfigDict(env_prefix="DB_")
        

class RedisSettings(BaseSettings):
    """Redis cache settings."""
    
    host: str = Field(default="localhost", validation_alias=AliasChoices("REDIS_HOST", "host"))
    port: int = Field(default=6379, validation_alias=AliasChoices("REDIS_PORT", "port"))
    db: int = Field(default=0, validation_alias=AliasChoices("REDIS_DB", "db"))
    password: Optional[str] = Field(default=None, validation_alias=AliasChoices("REDIS_PASSWORD", "password"))
    max_connections: int = Field(default=10, validation_alias=AliasChoices("REDIS_MAX_CONNECTIONS", "max_connections"))
    decode_responses: bool = Field(default=True, validation_alias=AliasChoices("REDIS_DECODE_RESPONSES", "decode_responses"))
    
    model_config = ConfigDict(env_prefix="REDIS_")


class VectorDBSettings(BaseSettings):
    """Vector database settings."""
    
    host: str = Field(default="localhost", validation_alias=AliasChoices("VECTOR_DB_HOST", "host"))
    port: int = Field(default=8000, validation_alias=AliasChoices("VECTOR_DB_PORT", "port"))
    collection_name: str = Field(default="flowlib_vectors", validation_alias=AliasChoices("VECTOR_DB_COLLECTION", "collection_name"))
    dimensions: int = Field(default=384, validation_alias=AliasChoices("VECTOR_DB_DIMENSIONS", "dimensions"))
    distance_metric: str = Field(default="cosine", validation_alias=AliasChoices("VECTOR_DB_DISTANCE_METRIC", "distance_metric"))
    
    model_config = ConfigDict(env_prefix="VECTOR_DB_")


class LLMSettings(BaseSettings):
    """LLM provider settings."""
    
    model_path: Optional[str] = Field(default=None, validation_alias=AliasChoices("LLM_MODEL_PATH", "model_path"))
    model_name: str = Field(default="default", validation_alias=AliasChoices("LLM_MODEL_NAME", "model_name"))
    temperature: float = Field(default=0.7, validation_alias=AliasChoices("LLM_TEMPERATURE", "temperature"))
    max_tokens: int = Field(default=1000, validation_alias=AliasChoices("LLM_MAX_TOKENS", "max_tokens"))
    top_p: float = Field(default=1.0, validation_alias=AliasChoices("LLM_TOP_P", "top_p"))
    context_length: int = Field(default=4096, validation_alias=AliasChoices("LLM_CONTEXT_LENGTH", "context_length"))
    
    model_config = ConfigDict(env_prefix="LLM_")

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature range."""
        if not (0.0 <= v <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v):
        """Validate top_p range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Top-p must be between 0.0 and 1.0")
        return v


class SecuritySettings(BaseSettings):
    """Security-related settings."""
    
    secret_key: str = Field(..., validation_alias=AliasChoices("FLOWLIB_SECRET_KEY", "secret_key"))
    encryption_algorithm: str = Field(default="AES-256-GCM", validation_alias=AliasChoices("FLOWLIB_ENCRYPTION_ALGORITHM", "encryption_algorithm"))
    password_hash_algorithm: str = Field(default="bcrypt", validation_alias=AliasChoices("FLOWLIB_PASSWORD_HASH_ALGORITHM", "password_hash_algorithm"))
    session_timeout: int = Field(default=3600, validation_alias=AliasChoices("FLOWLIB_SESSION_TIMEOUT", "session_timeout"))
    max_login_attempts: int = Field(default=5, validation_alias=AliasChoices("FLOWLIB_MAX_LOGIN_ATTEMPTS", "max_login_attempts"))
    
    # API security
    api_key_required: bool = Field(default=True, validation_alias=AliasChoices("FLOWLIB_API_KEY_REQUIRED", "api_key_required"))
    rate_limit_requests: int = Field(default=1000, validation_alias=AliasChoices("FLOWLIB_RATE_LIMIT_REQUESTS", "rate_limit_requests"))
    rate_limit_window: int = Field(default=3600, validation_alias=AliasChoices("FLOWLIB_RATE_LIMIT_WINDOW", "rate_limit_window"))
    
    model_config = ConfigDict(env_prefix="SECURITY_")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    metrics_endpoint: Optional[str] = Field(default=None, validation_alias=AliasChoices("METRICS_ENDPOINT", "metrics_endpoint"))
    tracing_endpoint: Optional[str] = Field(default=None, validation_alias=AliasChoices("TRACING_ENDPOINT", "tracing_endpoint"))
    service_name: str = Field(default="flowlib", validation_alias=AliasChoices("SERVICE_NAME", "service_name"))
    environment: str = Field(default="development", validation_alias=AliasChoices("ENVIRONMENT", "environment"))
    
    # Health checks
    health_check_enabled: bool = Field(default=True, validation_alias=AliasChoices("HEALTH_CHECK_ENABLED", "health_check_enabled"))
    health_check_interval: int = Field(default=30, validation_alias=AliasChoices("HEALTH_CHECK_INTERVAL", "health_check_interval"))
    
    # Alerting
    alert_webhook_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("ALERT_WEBHOOK_URL", "alert_webhook_url"))
    alert_on_error_threshold: int = Field(default=10, validation_alias=AliasChoices("ALERT_ERROR_THRESHOLD", "alert_on_error_threshold"))
    
    model_config = ConfigDict(env_prefix="MONITORING_")


class ConfigurationManager:
    """Centralized configuration manager for AI-Flowlib.
    
    This class provides a single point of access for all configuration,
    with support for multiple configuration sources and runtime updates.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self._config_file = config_file
        self._settings: Dict[str, BaseSettings] = {}
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from all sources."""
        try:
            # Load base settings
            self._settings['flowlib'] = FlowlibSettings()
            
            # Load component-specific settings
            self._settings['database'] = DatabaseSettings()
            self._settings['redis'] = RedisSettings()
            self._settings['vector_db'] = VectorDBSettings()
            self._settings['llm'] = LLMSettings()
            self._settings['monitoring'] = MonitoringSettings()
            
            # Load security settings if secret key is available
            try:
                self._settings['security'] = SecuritySettings()
            except Exception as e:
                logger.warning(f"Security settings not loaded: {e}")
                
            # Load from configuration file if provided
            if self._config_file and self._config_file.exists():
                self._load_from_file()
                
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_file(self) -> None:
        """Load configuration from file."""
        if not self._config_file:
            return
            
        try:
            with open(self._config_file, 'r') as f:
                if self._config_file.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update settings with file data
            for section, data in config_data.items():
                if section in self._settings:
                    # Create new instance with file data
                    settings_class = type(self._settings[section])
                    self._settings[section] = settings_class(**data)
                    
        except Exception as e:
            logger.error(f"Failed to load configuration from file {self._config_file}: {e}")
    
    def get_settings(self, section: str) -> BaseSettings:
        """Get settings for a specific section.
        
        Args:
            section: Settings section name
            
        Returns:
            Settings instance
            
        Raises:
            KeyError: If section not found
        """
        if section not in self._settings:
            raise KeyError(f"Settings section '{section}' not found")
        return self._settings[section]
    
    def get_flowlib_settings(self) -> FlowlibSettings:
        """Get main flowlib settings."""
        return self.get_settings('flowlib')
    
    def get_database_settings(self) -> DatabaseSettings:
        """Get database settings."""
        return self.get_settings('database')
    
    def get_redis_settings(self) -> RedisSettings:
        """Get Redis settings."""
        return self.get_settings('redis')
    
    def get_vector_db_settings(self) -> VectorDBSettings:
        """Get vector database settings."""
        return self.get_settings('vector_db')
    
    def get_llm_settings(self) -> LLMSettings:
        """Get LLM settings."""
        return self.get_settings('llm')
    
    def get_security_settings(self) -> SecuritySettings:
        """Get security settings."""
        return self.get_settings('security')
    
    def get_monitoring_settings(self) -> MonitoringSettings:
        """Get monitoring settings."""
        return self.get_settings('monitoring')
    
    def reload(self) -> None:
        """Reload configuration from all sources."""
        self._load_configuration()
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration sections.
        
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        for section, settings in self._settings.items():
            try:
                # Trigger validation by accessing all fields
                settings.model_dump()
            except Exception as e:
                errors[section] = [str(e)]
        
        return errors
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export all settings as dictionary.
        
        Returns:
            Nested dictionary of all settings
        """
        result = {}
        for section, settings in self._settings.items():
            result[section] = settings.model_dump()
        return result


# Global configuration manager instance
config_manager = ConfigurationManager()


def create_settings(settings_class: Type[T], **kwargs: Any) -> T:
    """Create settings instance with provided values.
    
    Args:
        settings_class: Settings class to instantiate
        **kwargs: Settings values
        
    Returns:
        Settings instance
    """
    return cast(T, settings_class(**kwargs))


def get_config() -> ConfigurationManager:
    """Get the global configuration manager.
    
    Returns:
        Global configuration manager instance
    """
    return config_manager


def reload_config() -> None:
    """Reload the global configuration."""
    config_manager.reload()


def validate_config() -> Dict[str, List[str]]:
    """Validate the global configuration.
    
    Returns:
        Dictionary of validation errors by section
    """
    return config_manager.validate_all()


# Environment-specific configuration loading
def load_environment_config(env: str = None) -> None:
    """Load environment-specific configuration.
    
    Args:
        env: Environment name (development, staging, production)
    """
    if env is None:
        env = os.getenv('FLOWLIB_ENV', 'development')
    
    config_file = Path(f"config/{env}.yml")
    if config_file.exists():
        global config_manager
        config_manager = ConfigurationManager(config_file)
        logger.info(f"Loaded {env} environment configuration")
    else:
        logger.warning(f"Environment config file not found: {config_file}")


# Initialize configuration on module import
try:
    load_environment_config()
except Exception as e:
    logger.warning(f"Failed to load environment-specific config: {e}")