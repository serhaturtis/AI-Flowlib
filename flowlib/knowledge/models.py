"""Models for knowledge base creation flows."""

from enum import Enum
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from typing import List, Optional, Dict, Any, Set
from datetime import datetime


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    TXT = "txt"
    EPUB = "epub"
    MOBI = "mobi"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "md"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    TOPIC = "topic"
    DATE = "date"
    EVENT = "event"
    TECHNOLOGY = "technology"
    METHODOLOGY = "methodology"
    THEORY = "theory"
    TERM = "term"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    MENTIONS = "mentions"
    DEFINES = "defines"
    RELATES_TO = "relates_to"
    IS_PART_OF = "is_part_of"
    CREATED_BY = "created_by"
    OCCURS_IN = "occurs_in"
    ASSOCIATED_WITH = "associated_with"
    INFLUENCES = "influences"
    DERIVED_FROM = "derived_from"
    IMPLEMENTS = "implements"
    CITES = "cites"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"


# ========== DOCUMENT MODELS ==========

class DocumentMetadata(StrictBaseModel):
    """Metadata for a processed document."""
    file_path: str
    file_name: str
    file_size: int
    file_type: DocumentType
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None


class TextChunk(StrictBaseModel):
    """A chunk of text with metadata."""
    chunk_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    document_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    
    # Computed properties
    word_count: int = 0
    char_count: int = 0
    
    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


class DocumentContent(StrictBaseModel):
    """Extracted content from a document."""
    document_id: str
    metadata: DocumentMetadata
    full_text: str
    chunks: List[TextChunk]
    status: ProcessingStatus
    error_message: Optional[str] = None
    
    # Analysis results
    summary: Optional[str] = None
    key_topics: List[str] = Field(default_factory=list)
    language_detected: Optional[str] = None
    reading_time_minutes: Optional[int] = None


# ========== ENTITY AND RELATIONSHIP MODELS ==========

class Entity(StrictBaseModel):
    """Extracted entity with metadata."""
    entity_id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    
    # Source information
    documents: List[str] = Field(default_factory=list)  # Document IDs where entity appears
    mentions: List[Dict[str, Any]] = Field(default_factory=list)  # Specific mentions with context
    
    # Properties
    frequency: int = 0
    confidence: float = 0.0
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(StrictBaseModel):
    """Relationship between two entities."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationType
    description: Optional[str] = None
    
    # Evidence
    documents: List[str] = Field(default_factory=list)
    context_sentences: List[str] = Field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    frequency: int = 0
    properties: Dict[str, Any] = Field(default_factory=dict)


class TopicModel(StrictBaseModel):
    """Topic discovered in the documents."""
    topic_id: str
    name: str
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Statistics
    document_count: int = 0
    prevalence_score: float = 0.0
    related_entities: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)


# ========== VECTOR DB MODELS ==========

class VectorEmbedding(StrictBaseModel):
    """Vector embedding for a text chunk."""
    embedding_id: str
    chunk_id: str
    document_id: str
    vector: List[float]
    
    # Metadata for search
    text_preview: str = Field(..., max_length=200)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(StrictBaseModel):
    """Result from vector similarity search."""
    chunk_id: str
    document_id: str
    similarity_score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ========== GRAPH DB MODELS ==========

class GraphNode(StrictBaseModel):
    """Node in the knowledge graph."""
    node_id: str
    node_type: str  # document, entity, topic, etc.
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)


class GraphEdge(StrictBaseModel):
    """Edge in the knowledge graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class GraphStatistics(StrictBaseModel):
    """Statistics about the knowledge graph."""
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int] = Field(default_factory=dict)
    relationship_types: Dict[str, int] = Field(default_factory=dict)
    
    # Network metrics
    average_degree: float = 0.0
    density: float = 0.0
    connected_components: int = 0
    largest_component_size: int = 0
    
    # Top entities by centrality
    top_entities_by_degree: List[Dict[str, Any]] = Field(default_factory=list)
    top_entities_by_betweenness: List[Dict[str, Any]] = Field(default_factory=list)
    top_entities_by_pagerank: List[Dict[str, Any]] = Field(default_factory=list)


# ========== OUTPUT MODELS ==========

class DocumentProcessingResult(StrictBaseModel):
    """Result of processing a single document."""
    document_id: str
    status: ProcessingStatus
    content: Optional[DocumentContent] = None
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    vector_count: int = 0
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class KnowledgeBaseStats(StrictBaseModel):
    """Statistics about the created knowledge base."""
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    total_vectors: int
    
    # Processing stats
    successful_documents: int
    failed_documents: int
    processing_time_seconds: float
    
    # Content stats
    total_words: int
    total_characters: int
    average_document_length: float
    
    # Entity and relationship stats
    entity_types: Dict[EntityType, int] = Field(default_factory=dict)
    relationship_types: Dict[RelationType, int] = Field(default_factory=dict)
    
    # Topics
    topics_discovered: int
    top_topics: List[Dict[str, Any]] = Field(default_factory=list)


class KnowledgeBaseResult(StrictBaseModel):
    """Final result of knowledge base creation."""
    status: str
    message: str
    
    # Outputs
    vector_collection_name: str
    graph_database_name: str
    output_directory: str
    
    # Statistics
    stats: KnowledgeBaseStats
    graph_stats: GraphStatistics
    
    # Processing details
    processed_documents: List[DocumentProcessingResult] = Field(default_factory=list)
    failed_files: List[Dict[str, str]] = Field(default_factory=list)
    
    # Metadata
    creation_timestamp: str
    processing_config: Dict[str, Any] = Field(default_factory=dict)


# ========== FLOW INPUT/OUTPUT MODELS ==========

class DocumentExtractionOutput(StrictBaseModel):
    """Output from document extraction stage."""
    documents: List[DocumentContent]
    failed_files: List[Dict[str, str]] = Field(default_factory=list)


class EntityExtractionInput(StrictBaseModel):
    """Input for entity extraction stage."""
    documents: List[DocumentContent]
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # LLM extraction configuration
    extraction_domain: str = Field("general", description="Technical domain for extraction")
    llm_model_name: str = Field("music-album-model", description="LLM model to use")
    min_entity_frequency: int = Field(2, description="Minimum frequency for entity inclusion")
    min_relationship_confidence: float = Field(0.7, description="Minimum confidence for relationships")


class EntityExtractionOutput(StrictBaseModel):
    """Output from entity extraction stage."""
    entities: List[Entity]
    relationships: List[Relationship]
    entity_document_map: Dict[str, List[str]] = Field(default_factory=dict)


class VectorStoreInput(StrictBaseModel):
    """Input for vector store creation."""
    documents: List[DocumentContent]
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Vector storage configuration
    collection_name: str = Field("knowledge_base", description="Name for the vector collection")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model to use")
    vector_dimensions: int = Field(384, description="Vector embedding dimensions")
    vector_provider_name: str = Field("chroma", description="Vector database provider")
    embedding_provider_name: Optional[str] = Field(None, description="Embedding provider name")


class VectorDocument(StrictBaseModel):
    """Document indexed in vector store."""
    document_id: str
    chunk_count: int
    status: str
    indexed_at: datetime


class VectorStoreOutput(StrictBaseModel):
    """Output from vector store operations."""
    collection_name: str
    total_vectors: int
    embeddings_created: List[VectorEmbedding] = Field(default_factory=list)

    # Optional fields for search operations
    search_results: Optional[List[VectorSearchResult]] = Field(default=None)
    query_text: Optional[str] = Field(default=None)

    # Optional fields for index/update operations
    documents_indexed: Optional[List[VectorDocument]] = Field(default=None)
    status: Optional[str] = Field(default=None)
    timestamp: Optional[datetime] = Field(default=None)


class GraphStoreInput(StrictBaseModel):
    """Input for graph store creation and queries."""
    documents: List[DocumentContent]
    entities: List[Entity]
    relationships: List[Relationship]
    config: Dict[str, Any] = Field(default_factory=dict)

    # Graph storage configuration
    graph_name: str = Field("knowledge_graph", description="Name for the graph database")
    graph_provider_name: str = Field("neo4j", description="Graph database provider")
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field("neo4j", description="Neo4j username")
    neo4j_password: str = Field("password", description="Neo4j password")

    # Query fields for graph operations
    query_entity_id: str = Field("", description="Entity ID for queries")
    query_entity_type: str = Field("", description="Entity type for queries")
    query_source_id: str = Field("", description="Source entity ID for path queries")
    query_target_id: str = Field("", description="Target entity ID for path queries")
    query_limit: int = Field(100, description="Limit for query results")


class GraphStoreOutput(StrictBaseModel):
    """Output from graph store creation."""
    graph_name: str
    nodes_created: int
    edges_created: int
    graph_stats: GraphStatistics


# ========== STREAMING MODELS ==========

class ChunkingStrategy(str, Enum):
    """Chunking strategies for smart text processing."""
    FIXED_SIZE = "fixed_size"
    PARAGRAPH_AWARE = "paragraph_aware"
    SENTENCE_AWARE = "sentence_aware"
    SEMANTIC_AWARE = "semantic_aware"


class ExtractionConfig(StrictBaseModel):
    """Configuration for streaming knowledge extraction."""
    batch_size: int = Field(5, description="Documents processed per batch")
    checkpoint_interval: int = Field(10, description="Save checkpoint every N documents")
    memory_limit_gb: int = Field(8, description="Memory limit for processing")
    enable_resumption: bool = Field(True, description="Enable checkpoint resumption")
    
    # Chunking configuration
    chunking_strategy: ChunkingStrategy = Field(ChunkingStrategy.PARAGRAPH_AWARE, description="Text chunking strategy")
    max_chunk_size: int = Field(1000, description="Maximum chunk size in characters")
    overlap_size: int = Field(200, description="Overlap between chunks")
    min_chunk_size: int = Field(100, description="Minimum chunk size")
    preserve_structure: bool = Field(True, description="Preserve document structure")


class ExtractionProgress(StrictBaseModel):
    """Progress tracking for streaming extraction."""
    total_documents: int = Field(0, description="Total documents to process")
    processed_documents: int = Field(0, description="Documents processed so far")
    current_document: Optional[str] = Field(None, description="Currently processing document")
    
    # Extraction statistics
    entities_extracted: int = Field(0, description="Total entities extracted")
    relationships_extracted: int = Field(0, description="Total relationships extracted")
    chunks_created: int = Field(0, description="Total chunks created")
    
    # Timing information
    start_time: datetime = Field(default_factory=datetime.now, description="Processing start time")
    last_update: datetime = Field(default_factory=datetime.now, description="Last progress update")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Progress calculation
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100.0
    
    @property
    def documents_per_minute(self) -> float:
        """Calculate processing rate."""
        if self.processed_documents == 0:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return (self.processed_documents / elapsed) * 60 if elapsed > 0 else 0.0


class ExtractionState(StrictBaseModel):
    """Complete state for streaming extraction."""
    # Document tracking
    processed_docs: Set[str] = Field(default_factory=set, description="Set of processed document IDs")
    failed_docs: Dict[str, str] = Field(default_factory=dict, description="Failed documents with error messages")
    
    # Progress tracking
    progress: ExtractionProgress = Field(default_factory=lambda: ExtractionProgress(
        total_documents=0,
        processed_documents=0,
        current_document=None,
        entities_extracted=0,
        relationships_extracted=0,
        chunks_created=0,
        estimated_completion=None
    ), description="Progress information")
    
    # Knowledge accumulation
    detected_domains: Set[str] = Field(default_factory=set, description="Detected knowledge domains")
    accumulated_entities: List[Entity] = Field(default_factory=list, description="All extracted entities")
    accumulated_relationships: List[Relationship] = Field(default_factory=list, description="All extracted relationships")
    
    # Database paths (persistent streaming instances)
    streaming_vector_db_path: str = Field("", description="Path to streaming vector database")
    streaming_graph_db_path: str = Field("", description="Path to streaming graph database")
    
    # Checkpoint information
    last_checkpoint_at: int = Field(0, description="Document count at last checkpoint")
    checkpoint_plugins: List[str] = Field(default_factory=list, description="Paths to exported checkpoint plugins")
    
    # Configuration
    extraction_config: ExtractionConfig = Field(default_factory=lambda: ExtractionConfig(
        batch_size=5,
        checkpoint_interval=10,
        memory_limit_gb=8,
        enable_resumption=True,
        chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE,
        max_chunk_size=1000,
        overlap_size=200,
        min_chunk_size=100,
        preserve_structure=True
    ), description="Extraction configuration")


class CheckpointData(StrictBaseModel):
    """Checkpoint state data for persistence."""
    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Checkpoint creation time")
    
    # Processing state
    streaming_state: ExtractionState = Field(..., description="Complete extraction state")
    
    # Metadata
    resumable: bool = Field(True, description="Whether this checkpoint can be resumed")
    plugin_exported: bool = Field(False, description="Whether incremental plugin was exported")
    plugin_path: Optional[str] = Field(None, description="Path to exported plugin")
    
    # Validation
    checksum: Optional[str] = Field(None, description="Data integrity checksum")


class PluginManifest(StrictBaseModel):
    """Manifest for exported knowledge plugins."""
    name: str = Field(..., description="Plugin name")
    version: str = Field("1.0.0", description="Plugin version")
    description: str = Field(..., description="Plugin description")
    
    # Plugin type and metadata
    plugin_type: str = Field("knowledge_plugin", description="Type of plugin")
    provider_class: str = Field(..., description="Provider class name")
    
    # Knowledge statistics
    entities_count: int = Field(0, description="Number of entities in plugin")
    relationships_count: int = Field(0, description="Number of relationships in plugin")
    documents_processed: int = Field(0, description="Number of documents processed")
    
    # Domain information
    domains: List[str] = Field(default_factory=list, description="Knowledge domains covered")
    
    # Database configuration
    databases: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Database configurations")
    
    # Creation metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Plugin creation time")
    
    # Checkpoint-specific fields
    checkpoint: Optional[Dict[str, Any]] = Field(None, description="Checkpoint metadata if applicable")


# ========== STREAMING INPUT/OUTPUT MODELS ==========

class KnowledgeExtractionRequest(StrictBaseModel):
    """Request for knowledge extraction with streaming support."""
    input_directory: str = Field(..., description="Directory containing documents to process")
    output_directory: str = Field(..., description="Directory to store knowledge base and checkpoints")
    collection_name: str = Field("knowledge_base", description="Name for the vector collection")
    graph_name: str = Field("knowledge_graph", description="Name for the graph database")
    
    # Processing options
    chunk_size: int = Field(1000, description="Size of text chunks for embedding")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    max_files: Optional[int] = Field(None, description="Maximum number of files to process")
    supported_formats: List[DocumentType] = Field(
        default_factory=lambda: [DocumentType.PDF, DocumentType.TXT, DocumentType.EPUB],
        description="Document formats to process"
    )
    
    # Analysis options
    extract_entities: bool = Field(True, description="Extract named entities")
    extract_relationships: bool = Field(True, description="Extract entity relationships")
    create_summaries: bool = Field(True, description="Create document summaries")
    detect_topics: bool = Field(True, description="Detect topics in documents")
    
    # LLM extraction configuration
    extraction_domain: str = Field("general", description="Technical domain for extraction")
    llm_model_name: str = Field("music-album-model", description="LLM model to use")
    
    # Vector DB options
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model to use")
    vector_dimensions: int = Field(384, description="Vector embedding dimensions")
    vector_provider_name: str = Field("chroma", description="Vector database provider")
    embedding_provider_name: Optional[str] = Field(None, description="Embedding provider name")
    
    # Graph DB options
    enable_graph_analysis: bool = Field(True, description="Enable graph analysis and metrics")
    min_entity_frequency: int = Field(2, description="Minimum frequency for entity inclusion")
    min_relationship_confidence: float = Field(0.7, description="Minimum confidence for relationships")
    graph_provider_name: str = Field("neo4j", description="Graph database provider")
    neo4j_uri: str = Field("bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field("neo4j", description="Neo4j username")
    neo4j_password: str = Field("password", description="Neo4j password")
    
    # Database configuration
    use_vector_db: bool = Field(True, description="Enable ChromaDB vector storage")
    use_graph_db: bool = Field(True, description="Enable Neo4j graph storage")
    
    # Streaming configuration
    extraction_config: ExtractionConfig = Field(default_factory=lambda: ExtractionConfig(
        batch_size=5,
        checkpoint_interval=10,
        memory_limit_gb=8,
        enable_resumption=True,
        chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE,
        max_chunk_size=1000,
        overlap_size=200,
        min_chunk_size=100,
        preserve_structure=True
    ), description="Extraction settings")
    
    # Resume configuration
    resume_from_checkpoint: bool = Field(True, description="Resume from existing checkpoint if available")
    
    # Plugin configuration
    plugin_domains: List[str] = Field(default_factory=list, description="Expected knowledge domains")
    plugin_name_prefix: str = Field("knowledge_extraction", description="Prefix for generated plugins")


class KnowledgeExtractionResult(StrictBaseModel):
    """Result from streaming knowledge extraction."""
    status: str = Field(..., description="Completion status")
    message: str = Field(..., description="Result message")
    
    # Processing statistics
    final_stats: KnowledgeBaseStats = Field(..., description="Final processing statistics")
    
    # Output information
    final_plugin_path: str = Field(..., description="Path to final complete plugin")
    checkpoint_plugins: List[str] = Field(default_factory=list, description="Paths to incremental plugins")
    
    # Performance metrics
    total_processing_time_seconds: float = Field(0.0, description="Total processing time")
    average_documents_per_minute: float = Field(0.0, description="Average processing rate")
    
    # Streaming state
    final_extraction_state: ExtractionState = Field(..., description="Final extraction state")
    
    # Convenience properties for easier access
    @property
    def total_documents_processed(self) -> int:
        """Total documents processed."""
        return self.final_stats.total_documents
    
    @property
    def total_entities_extracted(self) -> int:
        """Total entities extracted."""
        return self.final_stats.total_entities
    
    @property
    def total_relationships_extracted(self) -> int:
        """Total relationships extracted."""
        return self.final_stats.total_relationships
    
    @property
    def total_chunks_created(self) -> int:
        """Total chunks created."""
        return self.final_stats.total_chunks


class DocumentExtractionInput(StrictBaseModel):
    """Input for document extraction stage."""
    request: KnowledgeExtractionRequest
    file_paths: List[str]


class ChunkingInput(StrictBaseModel):
    """Input for smart chunking flow."""
    document: DocumentContent = Field(..., description="Document to chunk")
    config: ExtractionConfig = Field(default_factory=lambda: ExtractionConfig(
        batch_size=5,
        checkpoint_interval=10,
        memory_limit_gb=8,
        enable_resumption=True,
        chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE,
        max_chunk_size=1000,
        overlap_size=200,
        min_chunk_size=100,
        preserve_structure=True
    ), description="Chunking configuration")


class ChunkingOutput(StrictBaseModel):
    """Output from smart chunking flow."""
    document: DocumentContent = Field(..., description="Document with enhanced chunks")
    chunking_stats: Dict[str, Any] = Field(default_factory=dict, description="Chunking statistics")


class StreamingDocumentBatch(StrictBaseModel):
    """Batch of documents for streaming processing."""
    batch_id: str = Field(..., description="Unique batch identifier")
    documents: List[DocumentContent] = Field(..., description="Documents in this batch")
    batch_size: int = Field(..., description="Number of documents in batch")
    
    # Batch metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Batch creation time")
    processing_domain: str = Field("general", description="Processing domain for this batch")