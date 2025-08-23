"""Event-driven registration system eliminating circular dependencies.

This module provides an event bus for registration that completely eliminates
circular imports. Components register themselves through events, and the
container processes these events without requiring direct imports.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .interfaces import Container

logger = logging.getLogger(__name__)


@dataclass
class RegistrationEvent:
    """Event for registering items without circular imports.
    
    This event carries all information needed for registration without
    requiring any imports between the registering component and the registry.
    """
    item_type: str  # 'provider', 'resource', 'flow', 'config'
    name: str
    factory: Callable[[], Any]  # Factory function to create the item
    metadata: Dict[str, Any]
    category: Optional[str] = None  # Provider category, resource type, etc.


@dataclass  
class ConfigurationEvent:
    """Event for registering configurations."""
    config_name: str
    config_class: type
    provider_type: str
    settings: Dict[str, Any]


class EventBus:
    """Central event bus eliminating circular dependencies.
    
    This event bus allows components to register themselves without importing
    registries, completely eliminating circular dependencies. Registration
    events are buffered until the container is ready to process them.
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._container: Optional['Container'] = None
        self._event_buffer: List[RegistrationEvent] = []
        self._config_buffer: List[ConfigurationEvent] = []
        self._processing = False
        
    def emit_registration(self, event: RegistrationEvent) -> None:
        """Emit a registration event.
        
        Args:
            event: Registration event to emit
        """
        if self._container and not self._processing:
            # Container is ready, process immediately
            try:
                self._container.register(
                    event.item_type,
                    event.name, 
                    event.factory,
                    event.metadata
                )
                logger.debug(f"Registered {event.item_type}: {event.name}")
            except Exception as e:
                logger.error(f"Failed to register {event.item_type} '{event.name}': {e}")
        else:
            # Buffer event until container is ready
            self._event_buffer.append(event)
            logger.debug(f"Buffered registration for {event.item_type}: {event.name}")
    
    def emit_configuration(self, event: ConfigurationEvent) -> None:
        """Emit a configuration event.
        
        Args:
            event: Configuration event to emit
        """
        if self._container and not self._processing:
            # Container is ready, process immediately
            try:
                self._process_configuration_event(event)
                logger.debug(f"Registered configuration: {event.config_name}")
            except Exception as e:
                logger.error(f"Failed to register configuration '{event.config_name}': {e}")
        else:
            # Buffer event until container is ready
            self._config_buffer.append(event)
            logger.debug(f"Buffered configuration: {event.config_name}")
    
    def set_container(self, container: 'Container') -> None:
        """Set the container and process all buffered events.
        
        Args:
            container: Dependency container to receive events
        """
        self._container = container
        self._processing = True
        
        try:
            # Process buffered registration events
            for event in self._event_buffer:
                try:
                    container.register(
                        event.item_type,
                        event.name,
                        event.factory, 
                        event.metadata
                    )
                    logger.debug(f"Processed buffered registration: {event.name}")
                except Exception as e:
                    logger.error(f"Failed to process buffered registration '{event.name}': {e}")
            
            # Process buffered configuration events
            for config_event in self._config_buffer:
                try:
                    self._process_configuration_event(config_event)
                    logger.debug(f"Processed buffered configuration: {config_event.config_name}")
                except Exception as e:
                    logger.error(f"Failed to process buffered configuration '{config_event.config_name}': {e}")
            
            # Clear buffers
            self._event_buffer.clear()
            self._config_buffer.clear()
            
            logger.info(f"EventBus initialized with container, processed {len(self._event_buffer)} events")
            
        finally:
            self._processing = False
    
    def _process_configuration_event(self, event: ConfigurationEvent) -> None:
        """Process a configuration event.
        
        Args:
            event: Configuration event to process
        """
        # Create configuration instance
        config_instance = event.config_class()
        
        # Register as resource
        config_factory = lambda: config_instance
        
        self._container.register(
            'config',
            event.config_name,
            config_factory,
            {
                'provider_type': event.provider_type,
                'settings': event.settings,
                'config_class': event.config_class.__name__
            }
        )
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics.
        
        Returns:
            Statistics about the event bus state
        """
        return {
            'has_container': self._container is not None,
            'buffered_events': len(self._event_buffer),
            'buffered_configs': len(self._config_buffer),
            'handlers': {event_type: len(handlers) for event_type, handlers in self._handlers.items()}
        }


# Global event bus instance
# This is the single global state that eliminates circular dependencies
_global_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance.
    
    Returns:
        Global event bus instance
    """
    return _global_event_bus


def emit_registration(event: RegistrationEvent) -> None:
    """Emit a registration event to the global event bus.
    
    Args:
        event: Registration event to emit
    """
    _global_event_bus.emit_registration(event)


def emit_configuration(event: ConfigurationEvent) -> None:
    """Emit a configuration event to the global event bus.
    
    Args:
        event: Configuration event to emit
    """
    _global_event_bus.emit_configuration(event)


def set_global_container(container: 'Container') -> None:
    """Set the global container for event processing.
    
    Args:
        container: Dependency container to use globally
    """
    _global_event_bus.set_container(container)


def get_event_bus_stats() -> Dict[str, Any]:
    """Get global event bus statistics.
    
    Returns:
        Statistics about the global event bus
    """
    return _global_event_bus.get_stats()