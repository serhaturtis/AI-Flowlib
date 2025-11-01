"""Models for knowledge orchestration flow."""

from typing import Any

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.knowledge.models import (
    DocumentContent,
    DocumentType,
    Entity,
    Relationship,
)


class OrchestrationRequest(StrictBaseModel):
    """Request for orchestrating knowledge extraction pipeline."""

    input_directory: str = Field(..., description="Directory containing documents to process")
    output_directory: str = Field(..., description="Directory for output files")
    collection_name: str = Field(..., description="Name for vector collection")
    graph_name: str = Field(..., description="Name for graph database")

    # Processing configuration
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap size")
    max_files: int | None = Field(default=None, description="Maximum files to process")

    # Document types to process
    supported_formats: list[DocumentType] = Field(
        default_factory=lambda: [DocumentType.PDF, DocumentType.TXT, DocumentType.MARKDOWN],
        description="Supported document formats",
    )

    # Processing options
    extract_entities: bool = Field(default=True, description="Extract named entities")
    extract_relationships: bool = Field(default=True, description="Extract relationships")
    create_summaries: bool = Field(default=True, description="Create document summaries")
    detect_topics: bool = Field(default=True, description="Detect topics")

    # Database configuration (config-driven)
    vector_provider_config: str | None = Field(
        default="default-vector-db", description="Vector provider configuration name"
    )
    graph_provider_config: str | None = Field(
        default="default-graph-db", description="Graph provider configuration name"
    )
    vector_config: dict[str, Any] | None = Field(
        default=None, description="Vector database additional config"
    )
    graph_config: dict[str, Any] | None = Field(
        default=None, description="Graph database additional config"
    )


class OrchestrationProgress(StrictBaseModel):
    """Progress tracking for orchestration."""

    current_stage: str = Field(default="initialization", description="Current processing stage")
    total_documents: int = Field(default=0, description="Total documents to process")
    processed_documents: int = Field(default=0, description="Documents processed")
    current_document: str | None = Field(
        default=None, description="Current document being processed"
    )

    # Stage progress
    extraction_complete: bool = Field(default=False, description="Document extraction complete")
    analysis_complete: bool = Field(default=False, description="Entity analysis complete")
    vector_storage_complete: bool = Field(default=False, description="Vector storage complete")
    graph_storage_complete: bool = Field(default=False, description="Graph storage complete")

    # Statistics
    total_entities: int = Field(default=0, description="Total entities extracted")
    total_relationships: int = Field(default=0, description="Total relationships found")
    total_chunks: int = Field(default=0, description="Total text chunks created")

    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100.0


class OrchestrationResult(StrictBaseModel):
    """Result of knowledge orchestration pipeline."""

    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Result message")
    progress: OrchestrationProgress = Field(..., description="Final progress state")

    # Processing results
    documents: list[DocumentContent] = Field(
        default_factory=list, description="Processed documents"
    )
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: list[Relationship] = Field(
        default_factory=list, description="Extracted relationships"
    )

    # Database information
    vector_collection: str | None = Field(default=None, description="Vector collection name")
    graph_database: str | None = Field(default=None, description="Graph database name")

    # File outputs
    output_files: list[str] = Field(default_factory=list, description="Generated output files")
    export_directory: str | None = Field(default=None, description="Export directory path")

    # Statistics
    processing_time_seconds: float = Field(default=0.0, description="Total processing time")
    total_size_bytes: int = Field(default=0, description="Total processed data size")


__all__ = ["OrchestrationRequest", "OrchestrationProgress", "OrchestrationResult"]
