"""Tests for agent core interfaces."""

import pytest
from typing import Protocol, runtime_checkable
from flowlib.agent.core.interfaces import ComponentInterface
from flowlib.agent.core.base import AgentComponent


class MockAgentComponent:
    """Mock component implementing AgentComponent protocol."""
    
    def __init__(self, name: str = "test", initialized: bool = False):
        self._name = name
        self._initialized = initialized
    
    async def initialize(self) -> None:
        """Initialize the component."""
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown the component."""
        self._initialized = False
    
    @property
    def initialized(self) -> bool:
        """Return whether the component is initialized."""
        return self._initialized
    
    @property
    def name(self) -> str:
        """Return the component name."""
        return self._name


class IncompleteComponent:
    """Component that doesn't implement the full protocol."""
    
    def __init__(self, name: str = "incomplete"):
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    # Missing initialize, shutdown, and initialized


class TestComponentInterfaceProtocol:
    """Test ComponentInterface protocol definition."""
    
    def test_protocol_exists(self):
        """Test that ComponentInterface protocol is properly defined."""
        assert ComponentInterface is not None
        assert isinstance(ComponentInterface, type(Protocol))
    
    def test_protocol_has_required_methods(self):
        """Test that protocol defines all required methods."""
        # Check that protocol has the expected methods
        protocol_attrs = dir(ComponentInterface)
        
        assert 'initialize' in protocol_attrs
        assert 'shutdown' in protocol_attrs
        assert 'initialized' in protocol_attrs
        assert 'name' in protocol_attrs
    
    def test_complete_implementation_satisfies_protocol(self):
        """Test that complete implementation satisfies the protocol."""
        component = MockAgentComponent()
        
        # Should satisfy the protocol
        assert isinstance(component, ComponentInterface)
    
    def test_incomplete_implementation_fails_protocol(self):
        """Test that incomplete implementation fails the protocol."""
        component = IncompleteComponent()
        
        # Should not satisfy the protocol
        assert not isinstance(component, ComponentInterface)
    
    @pytest.mark.asyncio
    async def test_protocol_method_signatures(self):
        """Test that protocol methods have correct signatures."""
        component = MockAgentComponent()
        
        # initialize should be async and take no arguments
        await component.initialize()
        assert component.initialized
        
        # shutdown should be async and take no arguments
        await component.shutdown()
        assert not component.initialized
        
        # name should be a property returning string
        name = component.name
        assert isinstance(name, str)
        
        # initialized should be a property returning bool
        initialized = component.initialized
        assert isinstance(initialized, bool)


class TestAgentComponentProtocolCompliance:
    """Test that AgentComponent satisfies AgentComponent protocol."""
    
    def test_base_component_satisfies_protocol(self):
        """Test that AgentComponent implements AgentComponent protocol."""
        
        class TestBase(AgentComponent):
            pass
        
        component = TestBase("test")
        
        # AgentComponent should satisfy the ComponentInterface protocol
        assert isinstance(component, ComponentInterface)
    
    @pytest.mark.asyncio
    async def test_base_component_protocol_methods(self):
        """Test AgentComponent protocol method implementations."""
        
        class TestBase(AgentComponent):
            def __init__(self, name: str = "test"):
                super().__init__(name)
                self.init_called = False
                self.shutdown_called = False
            
            async def _initialize_impl(self):
                self.init_called = True
            
            async def _shutdown_impl(self):
                self.shutdown_called = True
        
        component = TestBase("protocol_test")
        
        # Test initialize
        assert not component.initialized
        await component.initialize()
        assert component.initialized
        assert component.init_called
        
        # Test name property
        assert component.name == "protocol_test"
        
        # Test shutdown
        await component.shutdown()
        assert not component.initialized
        assert component.shutdown_called


class TestProtocolTypeChecking:
    """Test protocol type checking and validation."""
    
    def test_protocol_type_annotation(self):
        """Test using protocol as type annotation."""
        
        def process_component(component: ComponentInterface) -> str:
            """Function that accepts any ComponentInterface."""
            return f"Processing {component.name}"
        
        # Should accept MockAgentComponent
        test_comp = MockAgentComponent("typed_test")
        result = process_component(test_comp)
        assert result == "Processing typed_test"
        
        # Should accept ComponentInterface implementation
        class TestBase(AgentComponent):
            pass
        
        base_comp = TestBase("base_test")
        result = process_component(base_comp)
        assert result == "Processing base_test"
    
    def test_protocol_runtime_checking(self):
        """Test runtime protocol checking with isinstance."""
        
        # Valid implementations
        valid_comp = MockAgentComponent()
        assert isinstance(valid_comp, ComponentInterface)
        
        base_comp = AgentComponent("base")
        assert isinstance(base_comp, ComponentInterface)
        
        # Invalid implementation
        invalid_comp = IncompleteComponent()
        assert not isinstance(invalid_comp, ComponentInterface)
    
    def test_protocol_duck_typing(self):
        """Test that protocol allows duck typing."""
        
        class DuckTypedComponent:
            """Component with correct methods but not explicitly implementing protocol."""
            
            def __init__(self):
                self._name = "duck"
                self._initialized = False
            
            async def initialize(self) -> None:
                self._initialized = True
            
            async def shutdown(self) -> None:
                self._initialized = False
            
            @property
            def initialized(self) -> bool:
                return self._initialized
            
            @property
            def name(self) -> str:
                return self._name
        
        duck_comp = DuckTypedComponent()
        
        # Should satisfy protocol through duck typing
        assert isinstance(duck_comp, ComponentInterface)


class TestProtocolEdgeCases:
    """Test edge cases and error scenarios with the protocol."""
    
    def test_partial_implementation_checking(self):
        """Test checking components with partial implementations."""
        
        class PartialComponent:
            """Component with some but not all required methods."""
            
            def __init__(self):
                self._name = "partial"
            
            @property
            def name(self) -> str:
                return self._name
            
            async def initialize(self) -> None:
                pass
            
            # Missing shutdown and initialized
        
        partial = PartialComponent()
        
        # Should not satisfy the protocol
        assert not isinstance(partial, ComponentInterface)
    
    def test_wrong_method_signatures(self):
        """Test components with wrong method signatures.
        
        Note: Python's runtime protocol checking only verifies method existence,
        not signatures. This test verifies that the component still passes
        isinstance checks but would fail at runtime when methods are called.
        """
        
        class WrongSignatureComponent:
            """Component with incorrect method signatures."""
            
            def __init__(self):
                self._name = "wrong"
                self._initialized = False
            
            def initialize(self, param):  # Wrong signature - not async, has param
                self._initialized = True
            
            async def shutdown(self) -> None:
                self._initialized = False
            
            @property
            def initialized(self) -> bool:
                return self._initialized
            
            @property
            def name(self) -> str:
                return self._name
        
        wrong_comp = WrongSignatureComponent()
        
        # Runtime protocol checking only verifies method existence, not signatures
        # So this will pass isinstance check but fail at runtime
        assert isinstance(wrong_comp, ComponentInterface)
        assert hasattr(wrong_comp, 'initialize')
        assert hasattr(wrong_comp, 'shutdown')
    
    def test_property_vs_method_distinction(self):
        """Test that properties are distinguished from methods.
        
        Note: Python's runtime protocol checking doesn't distinguish between
        properties and methods - it only checks if attributes exist.
        """
        
        class MethodInsteadOfProperty:
            """Component using methods instead of properties."""
            
            def __init__(self):
                self._name = "method_prop"
                self._initialized = False
            
            async def initialize(self) -> None:
                self._initialized = True
            
            async def shutdown(self) -> None:
                self._initialized = False
            
            def initialized(self) -> bool:  # Method instead of property
                return self._initialized
            
            def name(self) -> str:  # Method instead of property
                return self._name
        
        method_comp = MethodInsteadOfProperty()
        
        # Runtime checking passes since the attributes exist
        assert isinstance(method_comp, ComponentInterface)
        # But we can verify they are methods, not properties
        assert callable(getattr(method_comp, 'initialized'))
        assert callable(getattr(method_comp, 'name'))