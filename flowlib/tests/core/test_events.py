"""Comprehensive tests for core event system."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from flowlib.core.events.events import (
    RegistrationEvent,
    ConfigurationEvent,
    EventBus,
    get_event_bus,
    emit_registration,
    emit_configuration,
    set_global_container,
    get_event_bus_stats
)


class TestRegistrationEvent:
    """Test RegistrationEvent dataclass."""
    
    def test_registration_event_creation(self):
        """Test creating a registration event."""
        factory = lambda: "test_object"
        metadata = {"test": "metadata"}
        
        event = RegistrationEvent(
            item_type="provider",
            name="test_provider",
            factory=factory,
            metadata=metadata,
            category="llm"
        )
        
        assert event.item_type == "provider"
        assert event.name == "test_provider"
        assert event.factory() == "test_object"
        assert event.metadata == metadata
        assert event.category == "llm"
    
    def test_registration_event_optional_category(self):
        """Test registration event with no category."""
        event = RegistrationEvent(
            item_type="resource",
            name="test_resource", 
            factory=lambda: None,
            metadata={}
        )
        
        assert event.category is None


class TestConfigurationEvent:
    """Test ConfigurationEvent dataclass."""
    
    def test_configuration_event_creation(self):
        """Test creating a configuration event."""
        class TestConfig:
            pass
        
        event = ConfigurationEvent(
            config_name="test_config",
            config_class=TestConfig,
            provider_type="test_provider",
            settings={"param": "value"}
        )
        
        assert event.config_name == "test_config"
        assert event.config_class == TestConfig
        assert event.provider_type == "test_provider"
        assert event.settings == {"param": "value"}


class TestEventBus:
    """Test EventBus functionality."""
    
    def test_eventbus_initialization(self):
        """Test EventBus starts in correct initial state."""
        bus = EventBus()
        
        stats = bus.get_stats()
        assert stats['has_container'] is False
        assert stats['buffered_events'] == 0
        assert stats['buffered_configs'] == 0
        assert stats['handlers'] == {}
    
    def test_emit_registration_without_container(self):
        """Test emitting registration event without container buffers it."""
        bus = EventBus()
        factory = lambda: "test"
        
        event = RegistrationEvent(
            item_type="provider",
            name="test_provider",
            factory=factory,
            metadata={"type": "test"}
        )
        
        bus.emit_registration(event)
        
        stats = bus.get_stats()
        assert stats['buffered_events'] == 1
        assert len(bus._event_buffer) == 1
        assert bus._event_buffer[0] == event
    
    def test_emit_configuration_without_container(self):
        """Test emitting configuration event without container buffers it."""
        bus = EventBus()
        
        class TestConfig:
            pass
        
        event = ConfigurationEvent(
            config_name="test_config",
            config_class=TestConfig,
            provider_type="test",
            settings={}
        )
        
        bus.emit_configuration(event)
        
        stats = bus.get_stats()
        assert stats['buffered_configs'] == 1
        assert len(bus._config_buffer) == 1
        assert bus._config_buffer[0] == event
    
    def test_set_container_processes_buffered_events(self):
        """Test setting container processes all buffered events."""
        bus = EventBus()
        mock_container = Mock()
        
        # Create and buffer events
        reg_event = RegistrationEvent(
            item_type="provider",
            name="test_provider",
            factory=lambda: "test",
            metadata={}
        )
        
        class TestConfig:
            pass
        
        config_event = ConfigurationEvent(
            config_name="test_config",
            config_class=TestConfig,
            provider_type="test",
            settings={}
        )
        
        bus.emit_registration(reg_event)
        bus.emit_configuration(config_event)
        
        # Verify events are buffered
        assert len(bus._event_buffer) == 1
        assert len(bus._config_buffer) == 1
        
        # Set container and verify processing
        bus.set_container(mock_container)
        
        # Verify registration event was processed
        assert mock_container.register.call_count == 2  # reg event + config event
        
        # Verify buffers are cleared
        assert len(bus._event_buffer) == 0
        assert len(bus._config_buffer) == 0
        
        stats = bus.get_stats()
        assert stats['has_container'] is True
        assert stats['buffered_events'] == 0
        assert stats['buffered_configs'] == 0
    
    def test_emit_registration_with_container_processes_immediately(self):
        """Test emitting registration with container processes immediately."""
        bus = EventBus()
        mock_container = Mock()
        bus.set_container(mock_container)
        
        # Reset call count after container setup
        mock_container.reset_mock()
        
        event = RegistrationEvent(
            item_type="provider",
            name="test_provider",
            factory=lambda: "test",
            metadata={"test": "data"}
        )
        
        bus.emit_registration(event)
        
        # Verify immediate processing
        mock_container.register.assert_called_once_with(
            "provider",
            "test_provider",
            event.factory,
            {"test": "data"}
        )
        
        # Verify no buffering
        assert len(bus._event_buffer) == 0
    
    def test_emit_configuration_with_container_processes_immediately(self):
        """Test emitting configuration with container processes immediately."""
        bus = EventBus()
        mock_container = Mock()
        bus.set_container(mock_container)
        
        # Reset call count after container setup
        mock_container.reset_mock()
        
        class TestConfig:
            pass
        
        event = ConfigurationEvent(
            config_name="test_config",
            config_class=TestConfig,
            provider_type="test_provider",
            settings={"param": "value"}
        )
        
        bus.emit_configuration(event)
        
        # Verify immediate processing - config creates resource registration
        mock_container.register.assert_called_once()
        call_args = mock_container.register.call_args
        
        assert call_args[0][0] == 'config'  # item_type
        assert call_args[0][1] == 'test_config'  # name
        assert callable(call_args[0][2])  # factory
        
        metadata = call_args[0][3]
        assert metadata['provider_type'] == 'test_provider'
        assert metadata['settings'] == {"param": "value"}
        assert metadata['config_class'] == 'TestConfig'
        
        # Verify no buffering
        assert len(bus._config_buffer) == 0
    
    def test_registration_handler(self):
        """Test registering event handlers."""
        bus = EventBus()
        
        handler1 = Mock()
        handler2 = Mock()
        
        bus.register_handler("test_event", handler1)
        bus.register_handler("test_event", handler2)
        bus.register_handler("other_event", handler1)
        
        stats = bus.get_stats()
        assert stats['handlers']['test_event'] == 2
        assert stats['handlers']['other_event'] == 1
    
    def test_registration_error_handling(self):
        """Test error handling during registration."""
        bus = EventBus()
        mock_container = Mock()
        
        # Configure container to raise exception
        mock_container.register.side_effect = Exception("Registration failed")
        bus.set_container(mock_container)
        
        # Reset mock to clear setup calls
        mock_container.reset_mock() 
        mock_container.register.side_effect = Exception("Registration failed")
        
        event = RegistrationEvent(
            item_type="provider",
            name="failing_provider",
            factory=lambda: "test",
            metadata={}
        )
        
        # Should not raise exception - errors are logged
        bus.emit_registration(event)
        
        # Verify attempt was made
        mock_container.register.assert_called_once()
    
    def test_configuration_error_handling(self):
        """Test error handling during configuration processing."""
        bus = EventBus()
        mock_container = Mock()
        
        # Configure container to raise exception
        mock_container.register.side_effect = Exception("Configuration failed")
        bus.set_container(mock_container)
        
        # Reset mock to clear setup calls
        mock_container.reset_mock()
        mock_container.register.side_effect = Exception("Configuration failed")
        
        class TestConfig:
            pass
        
        event = ConfigurationEvent(
            config_name="failing_config",
            config_class=TestConfig,
            provider_type="test",
            settings={}
        )
        
        # Should not raise exception - errors are logged
        bus.emit_configuration(event)
        
        # Verify attempt was made
        mock_container.register.assert_called_once()
    
    def test_buffered_event_error_handling(self):
        """Test error handling when processing buffered events."""
        bus = EventBus()
        
        # Buffer events first
        reg_event = RegistrationEvent(
            item_type="provider",
            name="test_provider",
            factory=lambda: "test",
            metadata={}
        )
        
        class TestConfig:
            pass
        
        config_event = ConfigurationEvent(
            config_name="test_config",
            config_class=TestConfig,
            provider_type="test",
            settings={}
        )
        
        bus.emit_registration(reg_event)
        bus.emit_configuration(config_event)
        
        # Create container that fails
        mock_container = Mock()
        mock_container.register.side_effect = Exception("Processing failed")
        
        # Should not raise exception - errors are logged
        bus.set_container(mock_container)
        
        # Verify attempts were made and buffers cleared
        assert mock_container.register.call_count == 2
        assert len(bus._event_buffer) == 0
        assert len(bus._config_buffer) == 0
    
    def test_processing_flag_during_set_container(self):
        """Test processing flag is set during container initialization."""
        bus = EventBus()
        mock_container = Mock()
        
        processing_states = []
        
        # Track processing state when events are emitted
        def track_processing_register(*args, **kwargs):
            processing_states.append(bus._processing)
        
        mock_container.register.side_effect = track_processing_register
        
        initial_event = RegistrationEvent(
            item_type="initial",
            name="initial_provider", 
            factory=lambda: "initial",
            metadata={}
        )
        
        bus.emit_registration(initial_event)  # Buffer it
        bus.set_container(mock_container)  # Trigger processing
        
        # Processing flag should have been True during registration
        assert len(processing_states) == 1
        assert processing_states[0] is True  # Was processing during registration


class TestGlobalEventBusFunctions:
    """Test global event bus functions."""
    
    def test_get_event_bus_returns_singleton(self):
        """Test get_event_bus returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        assert bus1 is bus2
    
    def test_emit_registration_uses_global_bus(self):
        """Test global emit_registration function."""
        # Clear any existing state
        bus = get_event_bus()
        bus._event_buffer.clear()
        bus._config_buffer.clear()
        bus._container = None
        
        event = RegistrationEvent(
            item_type="provider",
            name="global_test_provider",
            factory=lambda: "test",
            metadata={}
        )
        
        emit_registration(event)
        
        stats = get_event_bus_stats()
        assert stats['buffered_events'] == 1
    
    def test_emit_configuration_uses_global_bus(self):
        """Test global emit_configuration function."""
        # Clear any existing state
        bus = get_event_bus()
        bus._event_buffer.clear()
        bus._config_buffer.clear()
        bus._container = None
        
        class TestConfig:
            pass
        
        event = ConfigurationEvent(
            config_name="global_test_config",
            config_class=TestConfig,
            provider_type="test",
            settings={}
        )
        
        emit_configuration(event)
        
        stats = get_event_bus_stats()
        assert stats['buffered_configs'] == 1
    
    def test_set_global_container(self):
        """Test setting global container."""
        mock_container = Mock()
        
        # Clear any existing events
        bus = get_event_bus()
        bus._event_buffer.clear()
        bus._config_buffer.clear()
        
        set_global_container(mock_container)
        
        stats = get_event_bus_stats()
        assert stats['has_container'] is True
    
    def test_get_event_bus_stats(self):
        """Test getting global event bus statistics."""
        stats = get_event_bus_stats()
        
        assert isinstance(stats, dict)
        assert 'has_container' in stats
        assert 'buffered_events' in stats
        assert 'buffered_configs' in stats
        assert 'handlers' in stats


class TestEventBusIntegration:
    """Test event bus integration scenarios."""
    
    def test_complete_event_workflow(self):
        """Test complete workflow: buffer, set container, process."""
        bus = EventBus()
        mock_container = Mock()
        
        # Step 1: Buffer multiple events
        events = [
            RegistrationEvent(
                item_type="provider",
                name=f"provider_{i}",
                factory=lambda i=i: f"provider_object_{i}",
                metadata={"index": i}
            )
            for i in range(3)
        ]
        
        configs = [
            ConfigurationEvent(
                config_name=f"config_{i}",
                config_class=type(f"TestConfig{i}", (), {}),
                provider_type=f"type_{i}",
                settings={"config_param": i}
            )
            for i in range(2)
        ]
        
        for event in events:
            bus.emit_registration(event)
        
        for config in configs:
            bus.emit_configuration(config)
        
        # Verify buffering
        stats = bus.get_stats()
        assert stats['buffered_events'] == 3
        assert stats['buffered_configs'] == 2
        
        # Step 2: Set container
        bus.set_container(mock_container)
        
        # Step 3: Verify processing
        assert mock_container.register.call_count == 5  # 3 providers + 2 configs
        
        # Verify buffers cleared
        stats = bus.get_stats()
        assert stats['buffered_events'] == 0
        assert stats['buffered_configs'] == 0
        assert stats['has_container'] is True
        
        # Step 4: Test immediate processing
        mock_container.reset_mock()
        
        new_event = RegistrationEvent(
            item_type="resource",
            name="immediate_resource",
            factory=lambda: "immediate",
            metadata={}
        )
        
        bus.emit_registration(new_event)
        
        # Should process immediately, not buffer
        mock_container.register.assert_called_once_with(
            "resource",
            "immediate_resource", 
            new_event.factory,
            {}
        )
        
        stats = bus.get_stats()
        assert stats['buffered_events'] == 0  # Still no buffering
    
    def test_configuration_event_creates_proper_factory(self):
        """Test configuration event creates working factory."""
        bus = EventBus()
        mock_container = Mock()
        
        class TestConfig:
            def __init__(self):
                self.value = "test_value"
        
        event = ConfigurationEvent(
            config_name="factory_test_config",
            config_class=TestConfig,
            provider_type="test_provider",
            settings={"test": "setting"}
        )
        
        bus.set_container(mock_container)
        bus.emit_configuration(event)
        
        # Get the factory that was registered
        call_args = mock_container.register.call_args
        factory = call_args[0][2]
        
        # Test the factory creates the right object
        created_instance = factory()
        assert isinstance(created_instance, TestConfig)
        assert created_instance.value == "test_value"
    
    def test_multiple_handlers_per_event_type(self):
        """Test registering multiple handlers for same event type."""
        bus = EventBus()
        
        handlers = [Mock() for _ in range(3)]
        
        for handler in handlers:
            bus.register_handler("test_event", handler)
        
        stats = bus.get_stats()
        assert stats['handlers']['test_event'] == 3
        
        # Verify all handlers are stored
        assert len(bus._handlers['test_event']) == 3
        for handler in handlers:
            assert handler in bus._handlers['test_event']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])