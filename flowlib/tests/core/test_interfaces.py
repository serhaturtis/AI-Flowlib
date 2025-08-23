"""Tests for core interfaces and protocols."""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Any, Dict, List, Optional

from flowlib.core.interfaces.interfaces import (
    Provider,
    LLMProvider,
    VectorProvider,
    GraphProvider,
    DatabaseProvider,
    CacheProvider,
    Resource,
    Configuration,
    PromptResource,
    ModelResource,
    Flow,
    AgentFlow,
    Stage,
    Memory,
    Planning,
    Factory,
    Container,
    RegistrationEvent
)


class TestProviderProtocol:
    """Test Provider protocol functionality."""
    
    def test_provider_protocol_detection(self):
        """Test that Provider protocol can be detected."""
        class TestProvider:
            def __init__(self):
                self.name = "test_provider"
                self.provider_type = "test"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
        
        provider = TestProvider()
        assert isinstance(provider, Provider)
    
    def test_provider_protocol_missing_methods(self):
        """Test that classes missing required methods are not detected as Provider."""
        class IncompleteProvider:
            def __init__(self):
                self.name = "incomplete"
                self.provider_type = "test"
            # Missing required methods
        
        provider = IncompleteProvider()
        assert not isinstance(provider, Provider)
    
    def test_provider_protocol_missing_attributes(self):
        """Test that classes missing required attributes are not detected as Provider."""
        class NoAttributesProvider:
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            # Missing name and provider_type attributes
        
        provider = NoAttributesProvider()
        assert not isinstance(provider, Provider)


class TestLLMProviderProtocol:
    """Test LLMProvider protocol functionality."""
    
    def test_llm_provider_protocol_detection(self):
        """Test that LLMProvider protocol can be detected."""
        class TestLLMProvider:
            def __init__(self):
                self.name = "test_llm"
                self.provider_type = "llm"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def generate(self, prompt: str, **kwargs) -> str:
                return f"Generated: {prompt}"
            
            async def generate_structured(self, prompt: str, output_type: type, **kwargs) -> Any:
                return output_type()
        
        provider = TestLLMProvider()
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider, Provider)  # Should also be a Provider
    
    @pytest.mark.asyncio
    async def test_llm_provider_methods(self):
        """Test LLMProvider method signatures."""
        class TestLLMProvider:
            def __init__(self):
                self.name = "test_llm"
                self.provider_type = "llm"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def generate(self, prompt: str, **kwargs) -> str:
                return f"Generated: {prompt}"
            
            async def generate_structured(self, prompt: str, output_type: type, **kwargs) -> Any:
                return {"result": "structured"}
        
        provider = TestLLMProvider()
        
        # Test generate method
        result = await provider.generate("test prompt")
        assert result == "Generated: test prompt"
        
        # Test generate_structured method
        structured_result = await provider.generate_structured("test", dict)
        assert structured_result == {"result": "structured"}


class TestVectorProviderProtocol:
    """Test VectorProvider protocol functionality."""
    
    def test_vector_provider_protocol_detection(self):
        """Test that VectorProvider protocol can be detected."""
        class TestVectorProvider:
            def __init__(self):
                self.name = "test_vector"
                self.provider_type = "vector"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def add_vectors(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
                pass
            
            async def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
                return [{"id": "1", "content": "result"}]
        
        provider = TestVectorProvider()
        assert isinstance(provider, VectorProvider)
        assert isinstance(provider, Provider)
    
    @pytest.mark.asyncio
    async def test_vector_provider_methods(self):
        """Test VectorProvider method signatures."""
        class TestVectorProvider:
            def __init__(self):
                self.name = "test_vector"
                self.provider_type = "vector"
                self._vectors = {}
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def add_vectors(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
                for i, doc_id in enumerate(ids):
                    self._vectors[doc_id] = {"document": documents[i], "metadata": metadatas[i]}
            
            async def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
                return [{"id": k, **v} for k, v in self._vectors.items()][:n_results]
        
        provider = TestVectorProvider()
        
        # Test add_vectors method
        await provider.add_vectors(
            ids=["1", "2"],
            documents=["doc1", "doc2"],
            metadatas=[{"meta": "1"}, {"meta": "2"}]
        )
        
        # Test search method
        results = await provider.search("test query", n_results=5)
        assert len(results) == 2
        assert results[0]["document"] == "doc1"


class TestGraphProviderProtocol:
    """Test GraphProvider protocol functionality."""
    
    def test_graph_provider_protocol_detection(self):
        """Test that GraphProvider protocol can be detected."""
        class TestGraphProvider:
            def __init__(self):
                self.name = "test_graph"
                self.provider_type = "graph"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def add_entity(self, entity_data: Dict[str, Any]) -> str:
                return "entity_id"
            
            async def add_relationship(self, source_id: str, target_id: str, relationship_type: str, properties: Dict[str, Any]) -> str:
                return "relationship_id"
        
        provider = TestGraphProvider()
        assert isinstance(provider, GraphProvider)
        assert isinstance(provider, Provider)


class TestDatabaseProviderProtocol:
    """Test DatabaseProvider protocol functionality."""
    
    def test_database_provider_protocol_detection(self):
        """Test that DatabaseProvider protocol can be detected."""
        class TestDatabaseProvider:
            def __init__(self):
                self.name = "test_db"
                self.provider_type = "database"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
                return {"result": "success"}
            
            async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
                return {"id": 1, "name": "test"}
        
        provider = TestDatabaseProvider()
        assert isinstance(provider, DatabaseProvider)
        assert isinstance(provider, Provider)


class TestCacheProviderProtocol:
    """Test CacheProvider protocol functionality."""
    
    def test_cache_provider_protocol_detection(self):
        """Test that CacheProvider protocol can be detected."""
        class TestCacheProvider:
            def __init__(self):
                self.name = "test_cache"
                self.provider_type = "cache"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def get(self, key: str) -> Optional[Any]:
                return "cached_value"
            
            async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
                pass
        
        provider = TestCacheProvider()
        assert isinstance(provider, CacheProvider)
        assert isinstance(provider, Provider)


class TestResourceProtocol:
    """Test Resource protocol functionality."""
    
    def test_resource_protocol_detection(self):
        """Test that Resource protocol can be detected."""
        class TestResource:
            def __init__(self):
                self.name = "test_resource"
                self.resource_type = "test"
            
            def get_data(self) -> Any:
                return {"data": "value"}
            
            def get_metadata(self) -> Dict[str, Any]:
                return {"meta": "data"}
        
        resource = TestResource()
        assert isinstance(resource, Resource)
    
    def test_resource_protocol_missing_methods(self):
        """Test that classes missing required methods are not detected as Resource."""
        class IncompleteResource:
            def __init__(self):
                self.name = "incomplete"
                self.resource_type = "test"
            # Missing required methods
        
        resource = IncompleteResource()
        assert not isinstance(resource, Resource)


class TestConfigurationProtocol:
    """Test Configuration protocol functionality."""
    
    def test_configuration_protocol_detection(self):
        """Test that Configuration protocol can be detected."""
        class TestConfiguration:
            def __init__(self):
                self.name = "test_config"
                self.resource_type = "configuration"
            
            def get_data(self) -> Any:
                return {"config": "data"}
            
            def get_metadata(self) -> Dict[str, Any]:
                return {"meta": "config"}
            
            def get_settings(self) -> Dict[str, Any]:
                return {"setting": "value"}
            
            def get_provider_type(self) -> str:
                return "test_provider"
        
        config = TestConfiguration()
        assert isinstance(config, Configuration)
        assert isinstance(config, Resource)  # Should also be a Resource


class TestPromptResourceProtocol:
    """Test PromptResource protocol functionality."""
    
    def test_prompt_resource_protocol_detection(self):
        """Test that PromptResource protocol can be detected."""
        class TestPromptResource:
            def __init__(self):
                self.name = "test_prompt"
                self.resource_type = "prompt"
            
            def get_data(self) -> Any:
                return "Hello {name}!"
            
            def get_metadata(self) -> Dict[str, Any]:
                return {"author": "test"}
            
            def get_template(self) -> str:
                return "Hello {name}!"
            
            def format(self, **kwargs) -> str:
                template = self.get_template()
                return template.format(**kwargs)
        
        prompt = TestPromptResource()
        assert isinstance(prompt, PromptResource)
        assert isinstance(prompt, Resource)
        
        # Test format method
        result = prompt.format(name="World")
        assert result == "Hello World!"


class TestModelResourceProtocol:
    """Test ModelResource protocol functionality."""
    
    def test_model_resource_protocol_detection(self):
        """Test that ModelResource protocol can be detected."""
        class TestModel:
            def __init__(self, value: str):
                self.value = value
        
        class TestModelResource:
            def __init__(self):
                self.name = "test_model"
                self.resource_type = "model"
            
            def get_data(self) -> Any:
                return TestModel
            
            def get_metadata(self) -> Dict[str, Any]:
                return {"type": "pydantic"}
            
            def get_model_class(self) -> type:
                return TestModel
            
            def create_instance(self, **kwargs) -> Any:
                return TestModel(**kwargs)
        
        model_resource = TestModelResource()
        assert isinstance(model_resource, ModelResource)
        assert isinstance(model_resource, Resource)
        
        # Test create_instance method
        instance = model_resource.create_instance(value="test")
        assert isinstance(instance, TestModel)
        assert instance.value == "test"


class TestFlowProtocol:
    """Test Flow protocol functionality."""
    
    def test_flow_protocol_detection(self):
        """Test that Flow protocol can be detected."""
        class TestFlow:
            def __init__(self):
                self.name = "test_flow"
                self.description = "Test flow"
            
            async def execute(self, input_data: Any, **kwargs) -> Any:
                return {"result": input_data}
            
            def get_input_schema(self) -> Dict[str, Any]:
                return {"type": "object"}
            
            def get_output_schema(self) -> Dict[str, Any]:
                return {"type": "object"}
        
        flow = TestFlow()
        assert isinstance(flow, Flow)
    
    @pytest.mark.asyncio
    async def test_flow_execution(self):
        """Test Flow execution."""
        class TestFlow:
            def __init__(self):
                self.name = "test_flow"
                self.description = "Test flow"
            
            async def execute(self, input_data: Any, **kwargs) -> Any:
                return {"processed": input_data, "kwargs": kwargs}
            
            def get_input_schema(self) -> Dict[str, Any]:
                return {"type": "object"}
            
            def get_output_schema(self) -> Dict[str, Any]:
                return {"type": "object"}
        
        flow = TestFlow()
        result = await flow.execute("test_input", extra_param="value")
        
        assert result["processed"] == "test_input"
        assert result["kwargs"]["extra_param"] == "value"


class TestAgentFlowProtocol:
    """Test AgentFlow protocol functionality."""
    
    def test_agent_flow_protocol_detection(self):
        """Test that AgentFlow protocol can be detected."""
        class TestAgentFlow:
            def __init__(self):
                self.name = "test_agent_flow"
                self.description = "Test agent flow"
            
            async def execute(self, input_data: Any, **kwargs) -> Any:
                return await self.run_pipeline(input_data)
            
            def get_input_schema(self) -> Dict[str, Any]:
                return {"type": "object"}
            
            def get_output_schema(self) -> Dict[str, Any]:
                return {"type": "object"}
            
            async def run_pipeline(self, input_data: Any) -> Any:
                return {"pipeline_result": input_data}
        
        flow = TestAgentFlow()
        assert isinstance(flow, AgentFlow)
        assert isinstance(flow, Flow)  # Should also be a Flow


class TestStageProtocol:
    """Test Stage protocol functionality."""
    
    def test_stage_protocol_detection(self):
        """Test that Stage protocol can be detected."""
        class TestStage:
            def __init__(self):
                self.name = "test_stage"
            
            async def execute(self, context: Any) -> Any:
                return {"stage_result": context}
        
        stage = TestStage()
        assert isinstance(stage, Stage)


class TestMemoryProtocol:
    """Test Memory protocol functionality."""
    
    def test_memory_protocol_detection(self):
        """Test that Memory protocol can be detected."""
        class TestMemory:
            async def store(self, key: str, value: Any, **kwargs) -> None:
                pass
            
            async def retrieve(self, key: str, **kwargs) -> Optional[Any]:
                return "retrieved_value"
            
            async def search(self, query: str, **kwargs) -> List[Any]:
                return ["result1", "result2"]
        
        memory = TestMemory()
        assert isinstance(memory, Memory)


class TestPlanningProtocol:
    """Test Planning protocol functionality."""
    
    def test_planning_protocol_detection(self):
        """Test that Planning protocol can be detected."""
        class TestPlanning:
            async def create_plan(self, objective: str, context: Dict[str, Any]) -> Any:
                return {"plan": "test_plan", "objective": objective}
            
            async def execute_plan(self, plan: Any) -> Any:
                return {"executed": plan}
        
        planning = TestPlanning()
        assert isinstance(planning, Planning)


class TestFactoryProtocol:
    """Test Factory protocol functionality."""
    
    def test_factory_protocol_detection(self):
        """Test that Factory protocol can be detected."""
        class TestFactory:
            def create(self, type_name: str, config: Dict[str, Any]) -> Any:
                return {"type": type_name, "config": config}
            
            def supports(self, type_name: str) -> bool:
                return type_name in ["test_type", "another_type"]
        
        factory = TestFactory()
        assert isinstance(factory, Factory)
        
        # Test methods
        obj = factory.create("test_type", {"setting": "value"})
        assert obj["type"] == "test_type"
        assert obj["config"]["setting"] == "value"
        
        assert factory.supports("test_type") is True
        assert factory.supports("unsupported_type") is False


class TestContainerProtocol:
    """Test Container protocol functionality."""
    
    def test_container_protocol_detection(self):
        """Test that Container protocol can be detected."""
        class MockProvider:
            def __init__(self):
                self.name = "mock"
                self.provider_type = "test"
        
        class MockResource:
            def __init__(self):
                self.name = "mock"
                self.resource_type = "test"
        
        class MockFlow:
            def __init__(self):
                self.name = "mock"
        
        class TestContainer:
            async def get_provider(self, config_name: str) -> Provider:
                return MockProvider()
            
            def get_resource(self, name: str, resource_type: Optional[str] = None) -> Resource:
                return MockResource()
            
            def get_flow(self, name: str) -> Flow:
                return MockFlow()
            
            def register(self, item_type: str, name: str, factory: Any, metadata: Dict[str, Any]) -> None:
                pass
        
        container = TestContainer()
        assert isinstance(container, Container)


class TestRegistrationEventProtocol:
    """Test RegistrationEvent protocol functionality."""
    
    def test_registration_event_protocol_detection(self):
        """Test that RegistrationEvent protocol can be detected."""
        class TestRegistrationEvent:
            def __init__(self):
                self.item_type = "provider"
                self.name = "test_provider"
                self.factory = lambda: "factory_result"
                self.metadata = {"version": "1.0"}
        
        event = TestRegistrationEvent()
        assert isinstance(event, RegistrationEvent)
        
        # Test attributes
        assert event.item_type == "provider"
        assert event.name == "test_provider"
        assert event.factory() == "factory_result"
        assert event.metadata["version"] == "1.0"


class TestProtocolInheritance:
    """Test protocol inheritance relationships."""
    
    def test_llm_provider_is_provider(self):
        """Test that LLMProvider inherits from Provider."""
        class TestLLMProvider:
            def __init__(self):
                self.name = "test"
                self.provider_type = "llm"
            
            async def initialize(self) -> None:
                pass
            
            async def shutdown(self) -> None:
                pass
            
            def is_available(self) -> bool:
                return True
            
            async def generate(self, prompt: str, **kwargs) -> str:
                return "generated"
            
            async def generate_structured(self, prompt: str, output_type: type, **kwargs) -> Any:
                return {}
        
        provider = TestLLMProvider()
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider, Provider)
    
    def test_configuration_is_resource(self):
        """Test that Configuration inherits from Resource."""
        class TestConfiguration:
            def __init__(self):
                self.name = "test_config"
                self.resource_type = "configuration"
            
            def get_data(self) -> Any:
                return {}
            
            def get_metadata(self) -> Dict[str, Any]:
                return {}
            
            def get_settings(self) -> Dict[str, Any]:
                return {}
            
            def get_provider_type(self) -> str:
                return "test"
        
        config = TestConfiguration()
        assert isinstance(config, Configuration)
        assert isinstance(config, Resource)
    
    def test_agent_flow_is_flow(self):
        """Test that AgentFlow inherits from Flow."""
        class TestAgentFlow:
            def __init__(self):
                self.name = "test"
                self.description = "test"
            
            async def execute(self, input_data: Any, **kwargs) -> Any:
                return {}
            
            def get_input_schema(self) -> Dict[str, Any]:
                return {}
            
            def get_output_schema(self) -> Dict[str, Any]:
                return {}
            
            async def run_pipeline(self, input_data: Any) -> Any:
                return {}
        
        flow = TestAgentFlow()
        assert isinstance(flow, AgentFlow)
        assert isinstance(flow, Flow)