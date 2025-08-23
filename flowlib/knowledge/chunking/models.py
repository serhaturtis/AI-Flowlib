"""Models for smart chunking flow."""

from pydantic import BaseModel, Field
from typing import Dict, Any
from flowlib.knowledge.models.models import DocumentContent, ExtractionConfig

# Re-export for convenience
ChunkingInput = DocumentContent.__class__.__dict__.copy()
ChunkingOutput = DocumentContent.__class__.__dict__.copy()

class ChunkingInput(BaseModel):
    """Input for smart chunking flow."""
    document: DocumentContent = Field(..., description="Document to chunk")
    config: ExtractionConfig = Field(default_factory=ExtractionConfig, description="Chunking configuration")


class ChunkingOutput(BaseModel):
    """Output from smart chunking flow."""
    document: DocumentContent = Field(..., description="Document with enhanced chunks")
    chunking_stats: Dict[str, Any] = Field(default_factory=dict, description="Chunking statistics")