"""Vector storage flow for knowledge base."""

import logging
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry

from flowlib.knowledge.models import DocumentContent, TextChunk
from flowlib.knowledge.vector.models import (
    VectorStoreInput,
    VectorStoreOutput,
    VectorDocument,
    SearchQuery,
    SearchResult
)

logger = logging.getLogger(__name__)


# Removed SimpleEmbeddingProvider - now using real embedding providers


@flow(name="vector-storage-flow", description="Store document chunks in vector database")
class VectorStorageFlow:
    """Flow for creating vector embeddings and storing in ChromaDB."""
    
    async def _get_providers(self, config: VectorStoreInput):
        """Get initialized vector and embedding providers."""
        # Get real vector provider using config-driven access
        from flowlib.providers.core.registry import provider_registry
        
        logger.info(f"Getting vector provider from registry: default-vector-db")
        vector_provider = await provider_registry.get_by_config("default-vector-db")
        
        # Initialize the provider
        await vector_provider.initialize()
        
        # Get real embedding provider using config-driven access
        logger.info(f"Getting embedding provider from registry: default-embedding")
        embedding_provider = await provider_registry.get_by_config("default-embedding")
        
        # Initialize the embedding provider
        await embedding_provider.initialize()
        
        return vector_provider, embedding_provider
    
    def _prepare_chunk_metadata(self, chunk: TextChunk, doc: DocumentContent) -> Dict:
        """Prepare metadata for a chunk."""
        return {
            "document_id": doc.document_id,
            "document_name": doc.metadata.file_name,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "section": chunk.section_title or "",
            "file_type": doc.metadata.file_type,
            "language": doc.language_detected,
            "created_date": doc.metadata.created_date or ""
        }
    
    async def stream_upsert(self, doc_content: "DocumentContent", db_path: str) -> None:
        """Stream upsert document chunks to persistent vector database."""
        
        logger.info(f"Streaming upsert document: {doc_content.document_id}")
        
        # Get providers (reuse connection if available)
        if not hasattr(self, '_streaming_vector_provider'):
            self._streaming_vector_provider, self._streaming_embedding_provider = await self._get_streaming_providers(db_path)
        
        try:
            # Process chunks from the document
            if not doc_content.chunks:
                logger.warning(f"No chunks found in document: {doc_content.document_id}")
                return
            
            # Extract texts and prepare metadata
            texts = []
            metadatas = []
            chunk_ids = []
            
            for chunk in doc_content.chunks:
                chunk_id = f"{doc_content.document_id}_{chunk.chunk_index}"
                texts.append(chunk.text)
                chunk_ids.append(chunk_id)
                
                metadata = self._prepare_chunk_metadata(chunk, doc_content)
                metadatas.append(metadata)
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} chunks")
            embeddings = await self._streaming_embedding_provider.embed_texts(texts)
            
            # Insert to vector database using standard interface
            await self._streaming_vector_provider.insert_vectors(
                index_name="streaming_knowledge",
                vectors=embeddings,
                metadata=metadatas,
                ids=chunk_ids
            )
            
            logger.debug(f"Successfully streamed {len(texts)} chunks to vector DB")
            
        except Exception as e:
            logger.error(f"Failed to stream upsert document {doc_content.document_id}: {e}")
            raise

    async def finalize_streaming_collection(self, db_path: str) -> dict:
        """Finalize streaming vector collection."""
        
        logger.info("Finalizing streaming vector collection")
        
        try:
            # Cleanup streaming providers
            if hasattr(self, '_streaming_vector_provider'):
                await self._streaming_vector_provider.shutdown()
                delattr(self, '_streaming_vector_provider')
                delattr(self, '_streaming_embedding_provider')
            
            return {"status": "finalized", "db_path": db_path}
            
        except Exception as e:
            logger.error(f"Failed to finalize streaming collection: {e}")
            raise

    async def _get_streaming_providers(self, db_path: str):
        """Get real providers configured for streaming operations."""
        from flowlib.providers.core.registry import provider_registry
        
        logger.debug(f"Initializing streaming vector providers at: {db_path}")
        
        # Use the same real providers as regular processing
        logger.debug(f"Streaming vector provider initialized (connection #1)")
        vector_provider = await provider_registry.get_by_config("default-vector-db")
        await vector_provider.initialize()
        
        logger.debug(f"Getting streaming embedding provider from registry: default-embedding") 
        embedding_provider = await provider_registry.get_by_config("default-embedding")
        await embedding_provider.initialize()
        
        return vector_provider, embedding_provider

    async def _index_documents(self, input_data: VectorStoreInput) -> VectorStoreOutput:
        """Create vector embeddings and store document chunks."""
        logger.info(f"Indexing {len(input_data.documents)} documents into collection: {input_data.collection_name}")
        
        # Get providers
        vector_provider, embedding_provider = await self._get_providers(input_data)
        
        try:
            indexed_docs = []
            total_chunks = 0
            
            # Process documents in batches for efficiency
            batch_size = 10  # Process 10 documents at a time
            
            for i in range(0, len(input_data.documents), batch_size):
                batch_docs = input_data.documents[i:i+batch_size]
                
                # Collect all chunks from batch
                all_chunks = []
                chunk_to_doc_map = {}  # chunk_id -> document
                
                for doc in batch_docs:
                    for chunk in doc.chunks:
                        chunk_id = f"{doc.document_id}_{chunk.chunk_index}"
                        all_chunks.append((chunk_id, chunk))
                        chunk_to_doc_map[chunk_id] = doc
                
                if not all_chunks:
                    continue
                
                # Extract texts from chunks
                texts = [chunk.text for _, chunk in all_chunks]
                chunk_ids = [chunk_id for chunk_id, _ in all_chunks]
                
                # Generate embeddings
                logger.info(f"Generating embeddings for {len(texts)} chunks")
                embeddings = await embedding_provider.embed_texts(texts)
                
                # Prepare metadata for each chunk
                metadatas = []
                for chunk_id, chunk in all_chunks:
                    doc = chunk_to_doc_map[chunk_id]
                    metadata = self._prepare_chunk_metadata(chunk, doc)
                    metadatas.append(metadata)
                
                # Store in vector database
                await vector_provider.insert_vectors(
                    index_name=input_data.collection_name,
                    vectors=embeddings,
                    metadata=metadatas,
                    ids=chunk_ids
                )
                
                # Track indexed documents
                for doc in batch_docs:
                    indexed_docs.append(VectorDocument(
                        document_id=doc.document_id,
                        chunk_count=len(doc.chunks),
                        status="indexed",
                        indexed_at=datetime.now()
                    ))
                    total_chunks += len(doc.chunks)
            
            logger.info(f"Successfully indexed {total_chunks} chunks from {len(indexed_docs)} documents")
            
            return VectorStoreOutput(
                collection_name=input_data.collection_name,
                documents_indexed=indexed_docs,
                total_chunks=total_chunks,
                vector_dimensions=input_data.vector_dimensions,
                status="completed",
                timestamp=datetime.now()
            )
            
        finally:
            # Always shutdown providers
            await vector_provider.shutdown()
            if hasattr(embedding_provider, 'shutdown'):
                await embedding_provider.shutdown()
    
    async def _search_documents(self, query: SearchQuery) -> VectorStoreOutput:
        """Search for similar documents using vector similarity."""
        logger.info(f"Searching for: {query.query_text[:100]}...")
        
        # Create minimal config for providers
        config = VectorStoreInput(
            collection_name=query.collection_name,
            vector_provider_name=query.vector_provider_name or "chromadb",
            embedding_provider_name=query.embedding_provider_name or "simple",
            vector_dimensions=query.vector_dimensions or 384,
            documents=[]
        )
        
        # Get providers
        vector_provider, embedding_provider = await self._get_providers(config)
        
        try:
            # Generate query embedding
            query_embedding = await embedding_provider.embed_texts([query.query_text])
            
            # Search vector database
            results = await vector_provider.search_vectors(
                index_name=query.collection_name,
                query_vector=query_embedding[0],
                top_k=query.top_k,
                filter_conditions=query.filter_metadata
            )
            
            # Convert results to SearchResult objects
            search_results = []
            for i, result in enumerate(results):
                # Fail-fast approach - all required metadata must be present
                if "document_id" not in result.metadata:
                    raise ValueError(f"Search result {i} missing required 'document_id' in metadata")
                if "chunk_index" not in result.metadata:
                    raise ValueError(f"Search result {i} missing required 'chunk_index' in metadata")
                if "text" not in result.metadata:
                    raise ValueError(f"Search result {i} missing required 'text' in metadata")
                
                search_results.append(SearchResult(
                    document_id=result.metadata["document_id"],
                    chunk_index=result.metadata["chunk_index"], 
                    score=result.score,
                    text=result.metadata["text"],
                    metadata=result.metadata,
                    rank=i + 1
                ))
            
            return VectorStoreOutput(
                collection_name=query.collection_name,
                search_results=search_results,
                query_text=query.query_text,
                status="completed",
                timestamp=datetime.now()
            )
            
        finally:
            # Always shutdown providers
            await vector_provider.shutdown()
            if hasattr(embedding_provider, 'shutdown'):
                await embedding_provider.shutdown()
    
    async def _update_collection(self, input_data: VectorStoreInput) -> VectorStoreOutput:
        """Update existing collection with new or modified documents."""
        logger.info(f"Updating collection: {input_data.collection_name}")
        
        # Get providers
        vector_provider, embedding_provider = await self._get_providers(input_data)
        
        try:
            updated_docs = []
            
            for doc in input_data.documents:
                # Delete existing chunks for this document
                doc_filter = {"document_id": doc.document_id}
                await vector_provider.delete(filter=doc_filter)
                
                # Add new chunks
                if doc.chunks:
                    texts = [chunk.text for chunk in doc.chunks]
                    chunk_ids = [f"{doc.document_id}_{chunk.chunk_index}" for chunk in doc.chunks]
                    
                    # Generate embeddings
                    embeddings = await embedding_provider.embed_texts(texts)
                    
                    # Prepare metadata
                    metadatas = [self._prepare_chunk_metadata(chunk, doc) for chunk in doc.chunks]
                    
                    # Store in vector database
                    await vector_provider.insert_vectors(
                        index_name=input_data.collection_name,
                        vectors=embeddings,
                        metadata=metadatas,
                        ids=chunk_ids
                    )
                    
                    updated_docs.append(VectorDocument(
                        document_id=doc.document_id,
                        chunk_count=len(doc.chunks),
                        status="updated",
                        indexed_at=datetime.now()
                    ))
            
            return VectorStoreOutput(
                collection_name=input_data.collection_name,
                documents_indexed=updated_docs,
                total_chunks=sum(d.chunk_count for d in updated_docs),
                vector_dimensions=input_data.vector_dimensions,
                status="completed",
                timestamp=datetime.now()
            )
            
        finally:
            # Always shutdown providers
            await vector_provider.shutdown()
            if hasattr(embedding_provider, 'shutdown'):
                await embedding_provider.shutdown()
    
    @pipeline(input_model=VectorStoreInput, output_model=VectorStoreOutput)
    async def run_pipeline(self, input_data: VectorStoreInput) -> VectorStoreOutput:
        """Execute vector storage pipeline."""
        # For basic input, just index documents
        return await self._index_documents(input_data)