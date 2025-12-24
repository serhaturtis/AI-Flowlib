"""Company knowledge provider for smart email sales agent.

Provides access to company information including:
- Product catalog and specifications
- Pricing information
- Company policies (returns, shipping, warranty)
- FAQs and common responses
- General company information
"""

import logging
from typing import Any

from flowlib.providers.knowledge.base import Knowledge, KnowledgeProvider
from flowlib.providers.vector.chroma.provider import (
    ChromaDBProvider,
    ChromaDBProviderSettings,
)

logger = logging.getLogger(__name__)


class CompanyKnowledgeProvider(KnowledgeProvider):
    """Knowledge provider for company-specific information.

    Uses ChromaDB for semantic search across company knowledge base.
    Supports multiple domains for different types of company information.
    """

    domains = ["products", "pricing", "policies", "faq", "company"]

    def __init__(self) -> None:
        """Initialize the provider."""
        self.vector_db: ChromaDBProvider | None = None
        self._config: dict[str, Any] = {}
        self._collection_name: str = "company_knowledge"

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the knowledge provider.

        Args:
            config: Configuration dictionary with chromadb settings
        """
        self._config = config

        if "chromadb" not in config:
            raise ValueError("Company knowledge plugin requires chromadb configuration")

        chromadb_config = config["chromadb"]

        # Get connection settings
        connection = chromadb_config.get("connection", {})
        if not connection:
            raise ValueError("Missing chromadb connection configuration")

        # Store collection name for queries
        self._collection_name = connection.get("collection_name", "company_knowledge")

        # Create provider settings
        settings = ChromaDBProviderSettings(
            persist_directory=connection.get("persist_directory", "~/.flowlib/knowledge/company"),
            collection_name=self._collection_name,
        )

        # Initialize vector database
        self.vector_db = ChromaDBProvider(
            name="company-knowledge-vectors",
            provider_type="vector_db",
            settings=settings,
        )
        await self.vector_db.initialize()

        logger.info(f"Company knowledge provider initialized with collection: {self._collection_name}")

    async def query(self, domain: str, query: str, limit: int = 10) -> list[Knowledge]:
        """Query company knowledge base.

        Args:
            domain: Knowledge domain (products, pricing, policies, faq, company)
            query: Search query
            limit: Maximum results to return

        Returns:
            List of relevant knowledge items
        """
        if not self.supports_domain(domain):
            raise ValueError(f"Domain '{domain}' not supported by CompanyKnowledgeProvider")

        if not self.vector_db:
            raise RuntimeError("Provider not initialized - call initialize() first")

        try:
            # Query vector database with domain filter
            results = await self.vector_db.query(
                collection_name=self._collection_name,
                query_text=query,
                n_results=limit,
                where={"domain": domain} if domain != "company" else None,
            )

            # Convert to Knowledge objects
            knowledge_items = []
            for result in results:
                # Extract metadata
                metadata = result.get("metadata", {})
                content = result.get("document", result.get("content", ""))
                distance = result.get("distance", 0.5)

                # Convert distance to confidence (lower distance = higher confidence)
                # Assuming cosine distance where 0 = identical, 2 = opposite
                confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                knowledge_items.append(
                    Knowledge(
                        content=content,
                        source="vector",
                        domain=metadata.get("domain", domain),
                        confidence=confidence,
                        metadata=metadata,
                    )
                )

            logger.debug(f"Company knowledge query returned {len(knowledge_items)} results for domain '{domain}'")
            return knowledge_items

        except Exception as e:
            logger.error(f"Error querying company knowledge: {e}")
            raise

    async def add_knowledge(
        self,
        content: str,
        domain: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add knowledge to the company knowledge base.

        Args:
            content: The knowledge content to add
            domain: Knowledge domain (products, pricing, policies, faq, company)
            metadata: Additional metadata for the knowledge item

        Returns:
            ID of the added knowledge item
        """
        if not self.supports_domain(domain):
            raise ValueError(f"Domain '{domain}' not supported")

        if not self.vector_db:
            raise RuntimeError("Provider not initialized")

        # Build metadata
        doc_metadata = metadata or {}
        doc_metadata["domain"] = domain

        # Add to vector database
        doc_id = await self.vector_db.add(
            collection_name=self._collection_name,
            documents=[content],
            metadatas=[doc_metadata],
        )

        logger.info(f"Added knowledge to domain '{domain}': {content[:50]}...")
        return doc_id[0] if isinstance(doc_id, list) else doc_id

    async def add_knowledge_batch(
        self,
        items: list[dict[str, Any]],
    ) -> list[str]:
        """Add multiple knowledge items in batch.

        Args:
            items: List of dicts with 'content', 'domain', and optional 'metadata'

        Returns:
            List of IDs for added items
        """
        if not self.vector_db:
            raise RuntimeError("Provider not initialized")

        documents = []
        metadatas = []

        for item in items:
            content = item.get("content")
            domain = item.get("domain")
            metadata = item.get("metadata", {})

            if not content or not domain:
                raise ValueError("Each item must have 'content' and 'domain'")

            if not self.supports_domain(domain):
                raise ValueError(f"Domain '{domain}' not supported")

            metadata["domain"] = domain
            documents.append(content)
            metadatas.append(metadata)

        # Add batch to vector database
        ids = await self.vector_db.add(
            collection_name=self._collection_name,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(documents)} knowledge items in batch")
        return ids

    async def delete_knowledge(self, doc_id: str) -> bool:
        """Delete a knowledge item by ID.

        Args:
            doc_id: ID of the knowledge item to delete

        Returns:
            True if deleted successfully
        """
        if not self.vector_db:
            raise RuntimeError("Provider not initialized")

        await self.vector_db.delete(
            collection_name=self._collection_name,
            ids=[doc_id],
        )

        logger.info(f"Deleted knowledge item: {doc_id}")
        return True

    async def get_domain_stats(self) -> dict[str, int]:
        """Get count of knowledge items per domain.

        Returns:
            Dictionary mapping domain names to item counts
        """
        if not self.vector_db:
            raise RuntimeError("Provider not initialized")

        stats = {}
        for domain in self.domains:
            try:
                count = await self.vector_db.count(
                    collection_name=self._collection_name,
                    where={"domain": domain},
                )
                stats[domain] = count
            except Exception as e:
                logger.warning(f"Could not get count for domain '{domain}': {e}")
                stats[domain] = 0

        return stats

    async def shutdown(self) -> None:
        """Shutdown the provider and close connections."""
        if self.vector_db:
            try:
                await self.vector_db.shutdown()
                logger.info("Company knowledge provider shut down")
            except Exception as e:
                logger.error(f"Error shutting down vector database: {e}")

        self.vector_db = None
