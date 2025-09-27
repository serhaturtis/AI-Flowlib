"""Models for knowledge plugin generation."""

from typing import Dict, List, Any, Optional
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel
from enum import Enum


class DomainStrategy(str, Enum):
    """Available domain strategies for plugin generation."""
    GENERIC = "generic"
    SOFTWARE_ENGINEERING = "software_engineering"
    SCIENTIFIC_RESEARCH = "scientific_research"
    BUSINESS_PROCESS = "business_process"
    LEGAL_COMPLIANCE = "legal_compliance"


class DomainStrategyConfig(StrictBaseModel):
    """Configuration for domain-specific generation strategy."""
    # Inherits strict configuration from StrictBaseModel
    
    strategy: DomainStrategy = Field(description="Domain strategy to use")
    custom_entity_types: Optional[List[str]] = Field(default=None, description="Custom entity types for this domain")
    custom_relationship_types: Optional[List[str]] = Field(default=None, description="Custom relationship types")
    extraction_focus: Optional[List[str]] = Field(default=None, description="Areas to focus extraction on")
    validation_rules: Optional[Dict[str, Any]] = Field(default=None, description="Domain-specific validation rules")


class DatabaseConfig(StrictBaseModel):
    """Database configuration settings."""
    # Inherits strict configuration from StrictBaseModel
    
    enabled: bool = Field(default=True, description="Whether this database is enabled")
    config_file: str = Field(description="Configuration file path")


class DatabaseSettings(StrictBaseModel):
    """Database settings configuration."""
    # Inherits strict configuration from StrictBaseModel
    
    chromadb: Optional[DatabaseConfig] = Field(default=None, description="ChromaDB configuration")
    neo4j: Optional[DatabaseConfig] = Field(default=None, description="Neo4j configuration")


class ExtractionStats(StrictBaseModel):
    """Knowledge extraction statistics."""
    # Inherits strict configuration from StrictBaseModel
    
    total_documents: int = Field(default=0, description="Total documents processed")
    successful_documents: int = Field(default=0, description="Successfully processed documents")
    failed_documents: int = Field(default=0, description="Failed document processing")
    total_entities: int = Field(default=0, description="Total entities extracted")
    total_relationships: int = Field(default=0, description="Total relationships found")
    total_chunks: int = Field(default=0, description="Total text chunks created")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    extraction_error: Optional[str] = Field(default=None, description="Error message if extraction failed")


class ProcessedDataStats(StrictBaseModel):
    """Statistics for processed data."""
    # Inherits strict configuration from StrictBaseModel
    
    documents: int = Field(default=0, description="Number of documents")
    entities: int = Field(default=0, description="Number of entities")
    relationships: int = Field(default=0, description="Number of relationships")
    chunks: int = Field(default=0, description="Number of chunks")


class ProcessedData(StrictBaseModel):
    """Processed knowledge extraction data."""
    model_config = ConfigDict(extra="forbid")
    
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Processed documents")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Found relationships")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Text chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class PluginGenerationSummary(StrictBaseModel):
    """Summary of plugin generation results."""
    model_config = ConfigDict(extra="forbid")
    
    plugin_name: str = Field(description="Generated plugin name")
    plugin_directory: str = Field(description="Plugin directory path")
    domains: List[str] = Field(description="Knowledge domains covered")
    generation_time: str = Field(description="Generation timestamp")
    extraction_stats: ExtractionStats = Field(description="Extraction statistics")
    processed_data_stats: ProcessedDataStats = Field(description="Processed data statistics")
    files_created: List[str] = Field(description="List of created files")


class PluginGenerationRequest(StrictBaseModel):
    """Request model for plugin generation."""
    model_config = ConfigDict(extra="forbid")
    
    input_directory: str = Field(description="Directory containing documents to process")
    output_directory: str = Field(description="Where to create the plugin folder")
    plugin_name: str = Field(description="Name for the generated plugin")
    domains: List[str] = Field(description="List of knowledge domains this plugin handles")
    description: Optional[str] = Field(default=None, description="Plugin description")
    author: str = Field(default="Knowledge Plugin Generator", description="Plugin author")
    version: str = Field(default="1.0.0", description="Plugin version")
    use_vector_db: bool = Field(default=True, description="Whether to include ChromaDB support")
    use_graph_db: bool = Field(default=True, description="Whether to include Neo4j support")
    max_files: Optional[int] = Field(default=None, description="Maximum files to process")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    domain_strategy: DomainStrategy = Field(default=DomainStrategy.GENERIC, description="Domain-specific generation strategy")
    domain_config: Optional[DomainStrategyConfig] = Field(default=None, description="Domain strategy configuration")


class PluginGenerationResult(StrictBaseModel):
    """Result model for plugin generation."""
    model_config = ConfigDict(extra="forbid")
    
    success: bool = Field(description="Whether generation was successful")
    plugin_path: str = Field(description="Path to generated plugin")
    summary: PluginGenerationSummary = Field(description="Generation summary")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
