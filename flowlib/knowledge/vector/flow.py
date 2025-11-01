"""Vector storage flow for knowledge base."""

import logging
from datetime import datetime
from typing import cast

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.knowledge.models import (
    DocumentContent,
    TextChunk,
    VectorDocument,
    VectorSearchResult,
)
from flowlib.knowledge.vector.models import (
    SearchQuery,
    VectorStoreInput,
    VectorStoreOutput,
)
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.embedding.base import EmbeddingProvider
from flowlib.providers.vector.base import VectorDBProvider

logger = logging.getLogger(__name__)


# Removed SimpleEmbeddingProvider - now using real embedding providers


@flow(name="vector-storage-flow", description="Store document chunks in vector database")  # type: ignore[arg-type]
class VectorStorageFlow:
    """Flow for creating vector embeddings and storing in ChromaDB."""

    async def _get_providers(
        self, config: VectorStoreInput
    ) -> tuple[VectorDBProvider, EmbeddingProvider]:
        """Get initialized vector and embedding providers."""
        # Get real vector provider using config-driven access

        logger.info("Getting vector provider from registry: default-vector-db")
        vector_provider = await provider_registry.get_by_config("default-vector-db")

        # Initialize the provider
        await vector_provider.initialize()

        # Get real embedding provider using config-driven access
        logger.info("Getting embedding provider from registry: default-embedding")
        embedding_provider = await provider_registry.get_by_config("default-embedding")

        # Initialize the embedding provider
        await embedding_provider.initialize()

        # Cast to specific types for type safety
        return cast(VectorDBProvider, vector_provider), cast(EmbeddingProvider, embedding_provider)

    def _prepare_chunk_metadata(self, chunk: TextChunk, doc: DocumentContent) -> dict:
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
            "created_date": doc.metadata.created_date or "",
        }

    async def stream_upsert(self, doc_content: DocumentContent, db_path: str) -> None:
        """Stream upsert document chunks to persistent vector database."""

        logger.info(f"Streaming upsert document: {doc_content.document_id}")

        # Get providers (reuse connection if available)
        if not hasattr(self, "_streaming_vector_provider"):
            (
                self._streaming_vector_provider,
                self._streaming_embedding_provider,
            ) = await self._get_streaming_providers(db_path)

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
            embeddings = await self._streaming_embedding_provider.embed(texts)

            # Insert to vector database using standard interface
            await self._streaming_vector_provider.insert_vectors(
                index_name="streaming_knowledge",
                vectors=embeddings,
                metadata=metadatas,
                ids=chunk_ids,
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
            if hasattr(self, "_streaming_vector_provider"):
                await self._streaming_vector_provider.shutdown()
                delattr(self, "_streaming_vector_provider")
                delattr(self, "_streaming_embedding_provider")

            return {"status": "finalized", "db_path": db_path}

        except Exception as e:
            logger.error(f"Failed to finalize streaming collection: {e}")
            raise

    async def _get_streaming_providers(
        self, db_path: str
    ) -> tuple[VectorDBProvider, EmbeddingProvider]:
        """Get real providers configured for streaming operations."""

        logger.debug(f"Initializing streaming vector providers at: {db_path}")

        # Use the same real providers as regular processing
        logger.debug("Streaming vector provider initialized (connection #1)")
        vector_provider = await provider_registry.get_by_config("default-vector-db")
        await vector_provider.initialize()

        logger.debug("Getting streaming embedding provider from registry: default-embedding")
        embedding_provider = await provider_registry.get_by_config("default-embedding")
        await embedding_provider.initialize()

        # Cast to specific types for type safety
        return cast(VectorDBProvider, vector_provider), cast(EmbeddingProvider, embedding_provider)

    async def _index_documents(self, input_data: VectorStoreInput) -> VectorStoreOutput:
        """Create vector embeddings and store document chunks."""
        logger.info(
            f"Indexing {len(input_data.documents)} documents into collection: {input_data.collection_name}"
        )

        # Get providers
        vector_provider, embedding_provider = await self._get_providers(input_data)

        try:
            indexed_docs = []
            total_chunks = 0

            # Process documents in batches for efficiency
            batch_size = 10  # Process 10 documents at a time

            for i in range(0, len(input_data.documents), batch_size):
                batch_docs = input_data.documents[i : i + batch_size]

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
                embeddings = await embedding_provider.embed(texts)

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
                    ids=chunk_ids,
                )

                # Track indexed documents
                for doc in batch_docs:
                    indexed_docs.append(
                        VectorDocument(
                            document_id=doc.document_id,
                            chunk_count=len(doc.chunks),
                            status="indexed",
                            indexed_at=datetime.now(),
                        )
                    )
                    total_chunks += len(doc.chunks)

            logger.info(
                f"Successfully indexed {total_chunks} chunks from {len(indexed_docs)} documents"
            )

            return VectorStoreOutput(
                collection_name=input_data.collection_name,
                total_vectors=total_chunks,
                embeddings_created=[],
                documents_indexed=indexed_docs,
                status="completed",
                timestamp=datetime.now(),
            )

        finally:
            # Always shutdown providers
            await vector_provider.shutdown()
            if hasattr(embedding_provider, "shutdown"):
                await embedding_provider.shutdown()

    async def _search_documents(self, query: SearchQuery) -> VectorStoreOutput:
        """Search for similar documents using vector similarity."""
        logger.info(f"Searching for: {query.query_text[:100]}...")

        # Get embedding provider first to retrieve model info
        logger.info("Getting embedding provider from registry: default-embedding")
        embedding_provider_temp = await provider_registry.get_by_config(
            query.embedding_provider_config or "default-embedding"
        )
        await embedding_provider_temp.initialize()
        embedding_provider_cast = cast(EmbeddingProvider, embedding_provider_temp)

        # Get model config from provider settings
        embedding_model = embedding_provider_cast.settings.model_name
        vector_dimensions = embedding_provider_cast.settings.embedding_dim

        # Create minimal config for providers
        config = VectorStoreInput(
            collection_name=query.collection_name,
            vector_provider_config=query.vector_provider_config or "default-vector-db",
            embedding_provider_config=query.embedding_provider_config or "default-embedding",
            embedding_model=embedding_model,
            vector_dimensions=vector_dimensions,
            documents=[],
        )

        # Get providers (reuse embedding provider)
        vector_provider, embedding_provider = await self._get_providers(config)

        try:
            # Generate query embedding
            query_embedding = await embedding_provider.embed([query.query_text])

            # Search vector database
            results = await vector_provider.search_vectors(
                index_name=query.collection_name,
                query_vector=query_embedding[0],
                top_k=query.top_k,
                filter_conditions=query.filter_metadata,
            )

            # Convert results to SearchResult objects
            search_results = []
            for i, result in enumerate(results):
                # Fail-fast approach - all required metadata must be present
                if "document_id" not in result.metadata:
                    raise ValueError(
                        f"Search result {i} missing required 'document_id' in metadata"
                    )
                if "chunk_index" not in result.metadata:
                    raise ValueError(
                        f"Search result {i} missing required 'chunk_index' in metadata"
                    )
                if "text" not in result.metadata:
                    raise ValueError(f"Search result {i} missing required 'text' in metadata")

                chunk_id = f"{result.metadata['document_id']}_{result.metadata['chunk_index']}"
                search_results.append(
                    VectorSearchResult(
                        chunk_id=chunk_id,
                        document_id=result.metadata["document_id"],
                        similarity_score=result.score,
                        text=result.metadata["text"],
                        metadata=result.metadata,
                    )
                )

            return VectorStoreOutput(
                collection_name=query.collection_name,
                total_vectors=len(search_results),
                embeddings_created=[],
                search_results=search_results,
                query_text=query.query_text,
            )

        finally:
            # Always shutdown providers
            await vector_provider.shutdown()
            if hasattr(embedding_provider, "shutdown"):
                await embedding_provider.shutdown()

    async def _update_collection(self, input_data: VectorStoreInput) -> VectorStoreOutput:
        """Update existing collection with new or modified documents."""
        logger.info(f"Updating collection: {input_data.collection_name}")

        # Get providers
        vector_provider, embedding_provider = await self._get_providers(input_data)

        try:
            updated_docs = []

            for doc in input_data.documents:
                # Note: Cannot delete by filter in current vector provider interface
                # Would need to query existing chunks first then delete by ID
                # For now, we'll just add new chunks (overwriting if same ID)

                # Add new chunks
                if doc.chunks:
                    texts = [chunk.text for chunk in doc.chunks]
                    chunk_ids = [f"{doc.document_id}_{chunk.chunk_index}" for chunk in doc.chunks]

                    # Generate embeddings
                    embeddings = await embedding_provider.embed(texts)

                    # Prepare metadata
                    metadatas = [self._prepare_chunk_metadata(chunk, doc) for chunk in doc.chunks]

                    # Store in vector database
                    await vector_provider.insert_vectors(
                        index_name=input_data.collection_name,
                        vectors=embeddings,
                        metadata=metadatas,
                        ids=chunk_ids,
                    )

                    updated_docs.append(
                        VectorDocument(
                            document_id=doc.document_id,
                            chunk_count=len(doc.chunks),
                            status="updated",
                            indexed_at=datetime.now(),
                        )
                    )

            return VectorStoreOutput(
                collection_name=input_data.collection_name,
                total_vectors=sum(d.chunk_count for d in updated_docs),
                embeddings_created=[],
                documents_indexed=updated_docs,
                status="completed",
                timestamp=datetime.now(),
            )

        finally:
            # Always shutdown providers
            await vector_provider.shutdown()
            if hasattr(embedding_provider, "shutdown"):
                await embedding_provider.shutdown()

    @pipeline(input_model=VectorStoreInput, output_model=VectorStoreOutput)
    async def run_pipeline(self, input_data: VectorStoreInput) -> VectorStoreOutput:
        """Execute vector storage pipeline."""
        # For basic input, just index documents
        return await self._index_documents(input_data)
