"""Base classes for knowledge providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
import logging
from ...providers.vector.chroma.provider import ChromaDBProvider, ChromaDBProviderSettings

logger = logging.getLogger(__name__)


class Knowledge(BaseModel):
    """Base model for knowledge items returned by providers."""
    model_config = ConfigDict(
        frozen=True, 
        extra="forbid",
        json_schema_extra={
            "example": {
                "content": "Caffeine has the molecular formula C8H10N4O2",
                "source": "vector", 
                "domain": "chemistry",
                "confidence": 0.95,
                "metadata": {
                    "cas_number": "58-08-2",
                    "pubchem_id": "2519"
                }
            }
        }
    )
    
    content: str = Field(..., description="The knowledge content or answer")
    source: str = Field(..., description="Source of knowledge: 'vector', 'graph', 'hybrid'")
    domain: str = Field(..., description="Knowledge domain this belongs to")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KnowledgeProvider(ABC):
    """Abstract base class for knowledge providers.
    
    Knowledge providers connect to domain-specific databases and provide
    query capabilities for the agent's learning and memory system.
    """
    
    # Subclasses must define the domains they handle
    domains: List[str] = []
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]):
        """Initialize the knowledge provider.
        
        Args:
            config: Configuration dictionary containing database connections
                   and other provider-specific settings
        
        Raises:
            ConnectionError: If unable to connect to knowledge databases
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def query(self, domain: str, query: str, limit: int = 10) -> List[Knowledge]:
        """Query knowledge in the specified domain.
        
        Args:
            domain: Knowledge domain to search in
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of Knowledge objects matching the query
            
        Raises:
            ValueError: If domain is not supported by this provider
            QueryError: If the query fails
        """
        pass
    
    async def shutdown(self):
        """Clean up resources and close connections.
        
        Called when the provider is being shut down. Subclasses should
        override this to properly close database connections and clean up.
        """
        pass
    
    def supports_domain(self, domain: str) -> bool:
        """Check if this provider supports the given domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is supported, False otherwise
        """
        return domain in self.domains


class MultiDatabaseKnowledgeProvider(KnowledgeProvider):
    """Knowledge provider that can use multiple database types.
    
    This class provides a framework for providers that use both vector
    databases (like ChromaDB) and graph databases (like Neo4j) to provide
    comprehensive knowledge retrieval.
    """
    
    def __init__(self):
        """Initialize the multi-database provider."""
        self.vector_db: Optional[Any] = None
        self.graph_db: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        
    async def initialize(self, config: Dict[str, Any]):
        """Initialize database connections.
        
        Args:
            config: Configuration with 'chromadb' and/or 'neo4j' sections
        """
        self._config = config
        
        # Initialize vector database if configured
        if config.get("chromadb"):
            await self._initialize_vector_db(config["chromadb"])
            
        # Initialize graph database if configured  
        if config.get("neo4j"):
            await self._initialize_graph_db(config["neo4j"])
            
        logger.info(f"Initialized {self.__class__.__name__} with "
                   f"vector={'chromadb' in config}, graph={'neo4j' in config}")
    
    async def _initialize_vector_db(self, config: Dict[str, Any]):
        """Initialize vector database connection.
        
        Args:
            config: ChromaDB configuration
        """
        # Removed ProviderType import - using config-driven provider access
        # Convert config to proper settings object
        chroma_settings = ChromaDBProviderSettings(**config["connection"])
        
        self.vector_db = ChromaDBProvider(
            name=f"{self.__class__.__name__}-vectors",
            provider_type="vector_db",
            settings=chroma_settings
        )
        await self.vector_db.initialize()
        logger.debug("Vector database initialized")
    
    async def _initialize_graph_db(self, config: Dict[str, Any]):
        """Initialize graph database connection.
        
        Args:
            config: Neo4j configuration
        """
        from ...providers.graph.neo4j.provider import Neo4jProvider, Neo4jProviderSettings
        
        # Convert config to proper settings object
        neo4j_settings = Neo4jProviderSettings(**config["connection"])
        
        self.graph_db = Neo4jProvider(
            name=f"{self.__class__.__name__}-graph",
            settings=neo4j_settings
        )
        await self.graph_db.initialize()
        logger.debug("Graph database initialized")
    
    async def query(self, domain: str, query: str, limit: int = 10) -> List[Knowledge]:
        """Query across available databases and fuse results.
        
        Args:
            domain: Knowledge domain to search
            query: Search query
            limit: Maximum results to return
            
        Returns:
            Fused and ranked knowledge results
        """
        if not self.supports_domain(domain):
            raise ValueError(f"Domain '{domain}' not supported by {self.__class__.__name__}")
            
        results = []
        
        # Query vector database if available
        if self.vector_db:
            try:
                vector_results = await self._query_vector(domain, query, limit // 2)
                results.extend(vector_results)
                logger.debug(f"Vector search returned {len(vector_results)} results")
            except Exception as e:
                logger.error(f"Vector query failed: {e}")
        
        # Query graph database if available
        if self.graph_db:
            try:
                graph_results = await self._query_graph(domain, query, limit // 2)
                results.extend(graph_results)
                logger.debug(f"Graph search returned {len(graph_results)} results")
            except Exception as e:
                logger.error(f"Graph query failed: {e}")
        
        # Fuse and rank results
        fused_results = self._fuse_and_rank_results(results, limit)
        logger.info(f"Knowledge query returned {len(fused_results)} fused results for domain '{domain}'")
        
        return fused_results
    
    @abstractmethod
    async def _query_vector(self, domain: str, query: str, limit: int) -> List[Knowledge]:
        """Query vector database for semantic similarity.
        
        Args:
            domain: Knowledge domain
            query: Search query  
            limit: Maximum results
            
        Returns:
            Knowledge results from vector search
        """
        pass
    
    @abstractmethod
    async def _query_graph(self, domain: str, query: str, limit: int) -> List[Knowledge]:
        """Query graph database for relationship-based knowledge.
        
        Args:
            domain: Knowledge domain
            query: Search query
            limit: Maximum results
            
        Returns:
            Knowledge results from graph traversal
        """
        pass
        
    def _fuse_and_rank_results(self, results: List[Knowledge], limit: int) -> List[Knowledge]:
        """Combine and rank results from multiple sources.
        
        Args:
            results: Results from different database sources
            limit: Maximum results to return
            
        Returns:
            Deduplicated and ranked knowledge results
        """
        if not results:
            return []
            
        # Remove duplicates based on content similarity
        unique_results = {}
        for result in results:
            # Create a key based on first 100 chars and domain
            key = f"{result.content[:100].strip()}_{result.domain}"
            
            # Keep the result with higher confidence
            if key not in unique_results or result.confidence > unique_results[key].confidence:
                unique_results[key] = result
        
        # Sort by confidence score (descending) and return top results
        sorted_results = sorted(
            unique_results.values(), 
            key=lambda x: x.confidence, 
            reverse=True
        )
        
        return sorted_results[:limit]
    
    async def shutdown(self):
        """Shutdown all database connections."""
        if self.vector_db:
            try:
                await self.vector_db.shutdown()
                logger.debug("Vector database connection closed")
            except Exception as e:
                logger.error(f"Error shutting down vector database: {e}")
                
        if self.graph_db:
            try:
                await self.graph_db.shutdown()
                logger.debug("Graph database connection closed")
            except Exception as e:
                logger.error(f"Error shutting down graph database: {e}")