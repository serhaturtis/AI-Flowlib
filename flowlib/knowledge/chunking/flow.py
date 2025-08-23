"""Smart chunking flow for document processing."""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.knowledge.models.models import (
    TextChunk, DocumentContent, ChunkingStrategy, 
    ChunkingInput, ChunkingOutput
)

logger = logging.getLogger(__name__)


@flow(name="smart-chunking", description="Intelligent text chunking with boundary awareness")
class SmartChunkingFlow:
    """Smart chunking flow that respects natural text boundaries."""

    @pipeline(input_model=ChunkingInput, output_model=ChunkingOutput)
    async def run_pipeline(self, input_data: ChunkingInput) -> ChunkingOutput:
        """Process document with smart chunking strategy."""
        
        document = input_data.document
        config = input_data.config
        
        logger.info(f"Smart chunking document: {document.document_id}")
        logger.info(f"Strategy: {config.chunking_strategy}, Max size: {config.max_chunk_size}")
        
        # Apply smart chunking based on strategy
        if config.chunking_strategy == ChunkingStrategy.PARAGRAPH_AWARE:
            new_chunks = await self._paragraph_aware_chunking(document, config)
        elif config.chunking_strategy == ChunkingStrategy.SENTENCE_AWARE:
            new_chunks = await self._sentence_aware_chunking(document, config)
        elif config.chunking_strategy == ChunkingStrategy.SEMANTIC_AWARE:
            new_chunks = await self._semantic_aware_chunking(document, config)
        else:
            new_chunks = await self._fixed_size_chunking(document, config)
        
        # Update document with new chunks
        enhanced_document = DocumentContent(
            document_id=document.document_id,
            metadata=document.metadata,
            full_text=document.full_text,
            chunks=new_chunks,
            status=document.status,
            error_message=document.error_message,
            summary=document.summary,
            key_topics=document.key_topics,
            language_detected=document.language_detected,
            reading_time_minutes=document.reading_time_minutes
        )
        
        # Generate chunking statistics
        stats = self._generate_chunking_stats(document.chunks, new_chunks, config)
        
        logger.info(f"Chunking complete: {len(new_chunks)} chunks created")
        
        return ChunkingOutput(
            document=enhanced_document,
            chunking_stats=stats
        )

    async def _paragraph_aware_chunking(
        self, 
        document: DocumentContent, 
        config
    ) -> List[TextChunk]:
        """Create chunks respecting paragraph boundaries."""
        
        text = document.full_text
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        current_start_char = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Calculate potential chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            # Check if adding paragraph exceeds limit
            if len(potential_chunk) <= config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Finalize current chunk if it has content
                if current_chunk:
                    chunk = self._create_text_chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start_char,
                        document_id=document.document_id
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_start_char += len(current_chunk)
                
                # Handle oversized paragraph
                if len(paragraph) > config.max_chunk_size:
                    # Split large paragraphs by sentences
                    paragraph_chunks = await self._split_paragraph_by_sentences(
                        paragraph, config, chunk_index, 
                        current_start_char, document.document_id
                    )
                    chunks.extend(paragraph_chunks)
                    chunk_index += len(paragraph_chunks)
                    current_start_char += len(paragraph)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_text_chunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start_char,
                document_id=document.document_id
            )
            chunks.append(chunk)
        
        return chunks

    async def _sentence_aware_chunking(
        self, 
        document: DocumentContent, 
        config
    ) -> List[TextChunk]:
        """Create chunks respecting sentence boundaries."""
        
        text = document.full_text
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_start_char = 0
        chunk_index = 0
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Finalize current chunk
                if current_chunk:
                    chunk = self._create_text_chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start_char,
                        document_id=document.document_id
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_start_char += len(current_chunk)
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_text_chunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start_char,
                document_id=document.document_id
            )
            chunks.append(chunk)
        
        return chunks

    async def _semantic_aware_chunking(
        self, 
        document: DocumentContent, 
        config
    ) -> List[TextChunk]:
        """Create chunks based on semantic boundaries (future implementation)."""
        
        # For now, fall back to paragraph-aware chunking
        logger.info("Semantic chunking not yet implemented, falling back to paragraph-aware")
        return await self._paragraph_aware_chunking(document, config)

    async def _fixed_size_chunking(
        self, 
        document: DocumentContent, 
        config
    ) -> List[TextChunk]:
        """Create fixed-size chunks (existing behavior)."""
        
        text = document.full_text
        chunks = []
        
        chunk_size = config.max_chunk_size
        overlap = config.overlap_size
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk = self._create_text_chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start,
                document_id=document.document_id
            )
            chunks.append(chunk)
            
            chunk_index += 1
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks

    async def _split_paragraph_by_sentences(
        self, 
        paragraph: str, 
        config, 
        start_chunk_index: int, 
        start_char: int,
        document_id: str
    ) -> List[TextChunk]:
        """Split large paragraphs by sentences."""
        
        sentences = self._split_into_sentences(paragraph)
        chunks = []
        
        current_chunk = ""
        chunk_index = start_chunk_index
        current_start = start_char
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Finalize current chunk
                if current_chunk:
                    chunk = self._create_text_chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        document_id=document_id
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_start += len(current_chunk)
                
                # Handle very long sentences by character splitting
                if len(sentence) > config.max_chunk_size:
                    char_chunks = self._split_by_characters(
                        sentence, config.max_chunk_size, 
                        chunk_index, current_start, document_id
                    )
                    chunks.extend(char_chunks)
                    chunk_index += len(char_chunks)
                    current_start += len(sentence)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_text_chunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start,
                document_id=document_id
            )
            chunks.append(chunk)
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        
        # Simple sentence splitting (can be enhanced with spaCy/NLTK)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _split_by_characters(
        self, 
        text: str, 
        chunk_size: int, 
        start_index: int, 
        start_char: int,
        document_id: str
    ) -> List[TextChunk]:
        """Split text by characters as last resort."""
        
        chunks = []
        chunk_index = start_index
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            
            chunk = self._create_text_chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char + i,
                document_id=document_id
            )
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks

    def _create_text_chunk(
        self, 
        text: str, 
        chunk_index: int, 
        start_char: int,
        document_id: str
    ) -> TextChunk:
        """Create a TextChunk instance with proper metadata."""
        
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        
        return TextChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=start_char + len(text),
            document_id=document_id,
            word_count=len(text.split()),
            char_count=len(text)
        )

    def _generate_chunking_stats(
        self, 
        original_chunks: List[TextChunk], 
        new_chunks: List[TextChunk],
        config
    ) -> Dict[str, Any]:
        """Generate statistics about the chunking process."""
        
        return {
            "strategy_used": config.chunking_strategy,
            "original_chunk_count": len(original_chunks),
            "new_chunk_count": len(new_chunks),
            "average_chunk_size": sum(chunk.char_count for chunk in new_chunks) / len(new_chunks) if new_chunks else 0,
            "max_chunk_size_configured": config.max_chunk_size,
            "min_chunk_size_configured": config.min_chunk_size,
            "overlap_size": config.overlap_size,
            "total_characters": sum(chunk.char_count for chunk in new_chunks),
            "total_words": sum(chunk.word_count for chunk in new_chunks),
            "chunks_over_limit": len([c for c in new_chunks if c.char_count > config.max_chunk_size]),
            "chunks_under_minimum": len([c for c in new_chunks if c.char_count < config.min_chunk_size])
        }