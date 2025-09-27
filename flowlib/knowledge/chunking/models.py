"""Models for smart chunking flow."""

from pydantic import BaseModel, Field
from typing import Dict, Any
from flowlib.knowledge.models import DocumentContent, ExtractionConfig, ChunkingStrategy


class ChunkingInput(BaseModel):
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


class ChunkingOutput(BaseModel):
    """Output from smart chunking flow."""
    document: DocumentContent = Field(..., description="Document with enhanced chunks")
    chunking_stats: Dict[str, Any] = Field(default_factory=dict, description="Chunking statistics")