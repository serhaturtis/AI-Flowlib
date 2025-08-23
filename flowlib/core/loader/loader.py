"""Dynamic loading system eliminating circular dependencies.

This module provides dynamic loading capabilities that completely eliminate
the need for imports in factory patterns. All provider and component
creation is done through dynamic loading based on string identifiers.
"""

import importlib
import inspect
from typing import Type, Any, Dict, Optional, Callable, List
import logging
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel

from flowlib.core.interfaces.interfaces import Provider, Resource, Flow

logger = logging.getLogger(__name__)


class ModuleClassInfo(StrictBaseModel):
    """Module class information model."""
    # Inherits strict configuration from StrictBaseModel
    
    module_name: str = Field(description="Module name to import")
    class_name: str = Field(description="Class name to load from module")


class DynamicLoader:
    """Dynamic loader eliminating circular imports.
    
    This loader dynamically imports and creates objects based on string
    identifiers, completely eliminating the need for import statements
    in factory patterns.
    """
    
    # Provider module mapping (no imports required)
    PROVIDER_MODULES = {
        'llamacpp': 'flowlib.providers.llm.llama_cpp.provider:LlamaCppProvider',
        'google_ai': 'flowlib.providers.llm.google_ai.provider:GoogleAIProvider',
        'openai': 'flowlib.providers.llm.openai_provider:OpenAIProvider',
        'postgres': 'flowlib.providers.db.postgres.provider:PostgreSQLProvider',
        'mongodb': 'flowlib.providers.db.mongodb.provider:MongoDBProvider',
        'sqlite': 'flowlib.providers.db.sqlite.provider:SQLiteProvider',
        'chroma': 'flowlib.providers.vector.chroma.provider:ChromaProvider',
        'pinecone': 'flowlib.providers.vector.pinecone.provider:PineconeProvider',
        'qdrant': 'flowlib.providers.vector.qdrant.provider:QdrantProvider',
        'neo4j': 'flowlib.providers.graph.neo4j_provider:Neo4jProvider',
        'arango': 'flowlib.providers.graph.arango_provider:ArangoProvider',
        'redis': 'flowlib.providers.cache.redis.provider:RedisProvider',
        's3': 'flowlib.providers.storage.s3.provider:S3Provider',
        'local': 'flowlib.providers.storage.local.provider:LocalProvider',
        'llamacpp_embedding': 'flowlib.providers.embedding.llama_cpp.provider:LlamaCppEmbeddingProvider',
        'rabbitmq': 'flowlib.providers.mq.rabbitmq.provider:RabbitMQProvider',
        'kafka': 'flowlib.providers.mq.kafka.provider:KafkaProvider',
    }
    
    # Resource module mapping 
    RESOURCE_MODULES = {
        'prompt': 'flowlib.resources.prompt_resource:PromptResource',
        'model': 'flowlib.resources.model_resource:ModelResource', 
        'config': 'flowlib.resources.config_resource:ConfigurationResource',
        'template': 'flowlib.resources.template_resource:TemplateResource',
    }
    
    # Flow module mapping
    FLOW_MODULES = {
        # Agent flows
        'entity-extraction': 'flowlib.agent.learn.entity_extraction.flow:EntityExtractionFlow',
        'concept-formation': 'flowlib.agent.learn.concept_formation.flow:ConceptFormationFlow',
        'pattern-recognition': 'flowlib.agent.learn.pattern_recognition.flow:PatternRecognitionFlow',
        'relationship-learning': 'flowlib.agent.learn.relationship_learning.flow:RelationshipLearningFlow',
        'knowledge-integration': 'flowlib.agent.learn.knowledge_integration.flow:KnowledgeIntegrationFlow',
        'conversation': 'flowlib.agent.components.conversation.flow:ConversationFlow',
        'classification': 'flowlib.agent.components.classification.flow:MessageClassifierFlow',
        'planning': 'flowlib.agent.components.planning.planner:AgentPlanner',
        'reflection': 'flowlib.agent.components.reflection.base:BaseReflection',
        
        # Knowledge flows
        'knowledge-extraction': 'flowlib.agent.components.knowledge_flows.knowledge_extraction:KnowledgeExtractionFlow',
        'knowledge-retrieval': 'flowlib.agent.components.knowledge_flows.knowledge_retrieval:KnowledgeRetrievalFlow',
        
        # Music generation flows
        'track-generation': 'music_generation.track.generation.flow:TrackGenerationFlow',
        'song-structure': 'music_generation.song.structure.flow:SongStructureFlow',
    }
    
    @classmethod
    def _parse_module_path(cls, module_path: str) -> ModuleClassInfo:
        """Parse module path into structured info.
        
        Args:
            module_path: Module path in format 'module.path:ClassName'
            
        Returns:
            Parsed module class information
            
        Raises:
            ValueError: If module path format is invalid
        """
        if ':' not in module_path:
            raise ValueError(f"Invalid module path format: {module_path}. Expected 'module:class'")
        
        module_name, class_name = module_path.split(':', 1)
        return ModuleClassInfo(module_name=module_name, class_name=class_name)
    
    @classmethod
    def load_class(cls, module_path: str) -> Type[Any]:
        """Dynamically load a class from module path.
        
        Args:
            module_path: Module path in format 'module.path:ClassName'
            
        Returns:
            Loaded class
            
        Raises:
            ImportError: If module or class cannot be loaded
        """
        try:
            # Parse module path strictly
            info = cls._parse_module_path(module_path)
            
            # Import module
            module = importlib.import_module(info.module_name)
            
            # Verify class exists before accessing
            if not hasattr(module, info.class_name):
                raise AttributeError(
                    f"Module '{info.module_name}' has no class '{info.class_name}'. "
                    f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}"
                )
            
            # Load class with explicit error handling
            class_obj = getattr(module, info.class_name)
            if not inspect.isclass(class_obj):
                raise TypeError(f"'{info.class_name}' in module '{info.module_name}' is not a class")
                
            return class_obj
            
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to load class from {module_path}: {e}")
            raise ImportError(f"Cannot load class from {module_path}") from e
    
    @classmethod
    def load_provider_class(cls, provider_type: str) -> Type[Provider]:
        """Load provider class by provider type.
        
        Args:
            provider_type: Provider type identifier
            
        Returns:
            Provider class
            
        Raises:
            ValueError: If provider type is unknown
        """
        # Strict lookup - no fallbacks
        try:
            module_path = cls.PROVIDER_MODULES[provider_type]
        except KeyError:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available types: {list(cls.PROVIDER_MODULES.keys())}"
            )
        
        return cls.load_class(module_path)
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> Provider:
        """Create provider instance with configuration.
        
        Args:
            provider_type: Provider type identifier
            config: Configuration dictionary
            
        Returns:
            Configured provider instance
        """
        provider_class = cls.load_provider_class(provider_type)
        
        # Filter config to only include constructor parameters
        init_signature = inspect.signature(provider_class.__init__)
        filtered_config = {}
        
        for param_name, param in init_signature.parameters.items():
            if param_name in config and param_name != 'self':
                filtered_config[param_name] = config[param_name]
        
        logger.debug(f"Creating {provider_type} provider with config: {filtered_config}")
        return provider_class(**filtered_config)
    
    @classmethod
    def load_resource_class(cls, resource_type: str) -> Type[Resource]:
        """Load resource class by resource type.
        
        Args:
            resource_type: Resource type identifier
            
        Returns:
            Resource class
        """
        # Strict lookup - no fallbacks
        try:
            module_path = cls.RESOURCE_MODULES[resource_type]
        except KeyError:
            raise ValueError(
                f"Unknown resource type: {resource_type}. "
                f"Available types: {list(cls.RESOURCE_MODULES.keys())}"
            )
        
        return cls.load_class(module_path)
    
    @classmethod
    def create_resource(cls, resource_type: str, config: Dict[str, Any]) -> Resource:
        """Create resource instance with configuration.
        
        Args:
            resource_type: Resource type identifier  
            config: Configuration dictionary
            
        Returns:
            Configured resource instance
        """
        resource_class = cls.load_resource_class(resource_type)
        
        # Filter config to constructor parameters
        init_signature = inspect.signature(resource_class.__init__)
        filtered_config = {}
        
        for param_name, param in init_signature.parameters.items():
            if param_name in config and param_name != 'self':
                filtered_config[param_name] = config[param_name]
        
        logger.debug(f"Creating {resource_type} resource with config: {filtered_config}")
        return resource_class(**filtered_config)
    
    @classmethod
    def load_flow_class(cls, flow_name: str) -> Type[Flow]:
        """Load flow class by flow name.
        
        Args:
            flow_name: Flow name identifier
            
        Returns:
            Flow class
        """
        # Strict lookup - no fallbacks
        try:
            module_path = cls.FLOW_MODULES[flow_name]
        except KeyError:
            raise ValueError(
                f"Unknown flow: {flow_name}. "
                f"Available flows: {list(cls.FLOW_MODULES.keys())}"
            )
        
        return cls.load_class(module_path)
    
    @classmethod
    def create_flow(cls, flow_name: str, config: Optional[Dict[str, Any]] = None) -> Flow:
        """Create flow instance with optional configuration.
        
        Args:
            flow_name: Flow name identifier
            config: Optional configuration dictionary
            
        Returns:
            Flow instance
        """
        flow_class = cls.load_flow_class(flow_name)
        
        if config:
            # Filter config to constructor parameters
            init_signature = inspect.signature(flow_class.__init__)
            filtered_config = {}
            
            for param_name, param in init_signature.parameters.items():
                if param_name in config and param_name != 'self':
                    filtered_config[param_name] = config[param_name]
            
            logger.debug(f"Creating {flow_name} flow with config: {filtered_config}")
            return flow_class(**filtered_config)
        else:
            logger.debug(f"Creating {flow_name} flow with no config")
            return flow_class()
    
    @classmethod
    def register_provider_module(cls, provider_type: str, module_path: str) -> None:
        """Register a new provider module mapping.
        
        Args:
            provider_type: Provider type identifier
            module_path: Module path in format 'module.path:ClassName'
        """
        cls.PROVIDER_MODULES[provider_type] = module_path
        logger.info(f"Registered provider module: {provider_type} -> {module_path}")
    
    @classmethod
    def register_resource_module(cls, resource_type: str, module_path: str) -> None:
        """Register a new resource module mapping.
        
        Args:
            resource_type: Resource type identifier
            module_path: Module path in format 'module.path:ClassName'
        """
        cls.RESOURCE_MODULES[resource_type] = module_path
        logger.info(f"Registered resource module: {resource_type} -> {module_path}")
    
    @classmethod
    def register_flow_module(cls, flow_name: str, module_path: str) -> None:
        """Register a new flow module mapping.
        
        Args:
            flow_name: Flow name identifier
            module_path: Module path in format 'module.path:ClassName'
        """
        cls.FLOW_MODULES[flow_name] = module_path
        logger.info(f"Registered flow module: {flow_name} -> {module_path}")
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider types.
        
        Returns:
            List of provider type identifiers
        """
        return list(cls.PROVIDER_MODULES.keys())
    
    @classmethod
    def get_available_resources(cls) -> List[str]:
        """Get list of available resource types.
        
        Returns:
            List of resource type identifiers
        """
        return list(cls.RESOURCE_MODULES.keys())
    
    @classmethod
    def get_available_flows(cls) -> List[str]:
        """Get list of available flows.
        
        Returns:
            List of flow name identifiers
        """
        return list(cls.FLOW_MODULES.keys())