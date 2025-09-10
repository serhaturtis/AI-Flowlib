"""Entity and relationship analysis flow."""

import logging
import hashlib
from typing import List, Dict, Optional, Any

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry

from flowlib.knowledge.models import (
    DocumentContent,
    Entity,
    Relationship,
    EntityType,
    RelationType,
    EntityExtractionInput,
    EntityExtractionOutput,
)
from flowlib.knowledge.analysis.models import (
    LLMEntityExtractionResult,
    LLMRelationshipExtractionResult,
    LLMConceptExtractionResult,
)

logger = logging.getLogger(__name__)


@flow(name="entity-analysis-flow", description="Extract entities, relationships, and topics from documents")
class EntityAnalysisFlow:
    """Flow for analyzing entities and relationships in documents."""
    
    async def extract_from_single_document(self, doc_content: "DocumentContent") -> "EntityExtractionOutput":
        """Extract entities and relationships from a single document (for streaming)."""
        
        logger.info(f"Analyzing single document: {doc_content.document_id}")
        
        # Create minimal input for single document
        from flowlib.knowledge.models import EntityExtractionInput
        
        single_doc_input = EntityExtractionInput(
            documents=[doc_content],
            extraction_domain="general"
        )
        
        # Extract using existing logic but optimized for single document
        extraction_result = await self._extract_from_document_llm(doc_content, single_doc_input)
        
        # Convert to entities and relationships
        entities = []
        relationships = []
        entity_doc_map = {}
        
        # Convert LLM entities
        for llm_entity in extraction_result['entities']:
            entity_id = hashlib.md5(llm_entity['name'].encode()).hexdigest()
            
            entity = Entity(
                entity_id=entity_id,
                name=llm_entity['name'],
                entity_type=self._map_entity_type(llm_entity['type']),
                description=llm_entity['description'] if 'description' in llm_entity else None,
                documents=[doc_content.document_id],
                frequency=1,
                confidence=self._importance_to_confidence(llm_entity['importance'] if 'importance' in llm_entity else 'medium'),
                properties=llm_entity['attributes'] if 'attributes' in llm_entity else {}
            )
            entities.append(entity)
            
            if entity_id not in entity_doc_map:
                entity_doc_map[entity_id] = []
            entity_doc_map[entity_id].append(doc_content.document_id)
        
        # Convert LLM relationships
        for llm_rel in extraction_result['relationships']:
            # Find entity IDs
            source_id = None
            target_id = None
            
            for entity in entities:
                if entity.name == llm_rel['source']:
                    source_id = entity.entity_id
                if entity.name == llm_rel['target']:
                    target_id = entity.entity_id
            
            if source_id and target_id:
                rel_id = hashlib.md5(
                    f"{source_id}_{llm_rel['type']}_{target_id}".encode()
                ).hexdigest()
                
                relationship = Relationship(
                    relationship_id=rel_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=self._map_relationship_type(llm_rel['type']),
                    description=llm_rel['description'] if 'description' in llm_rel else None,
                    documents=[doc_content.document_id],
                    context_sentences=[llm_rel['description']] if 'description' in llm_rel else [],
                    confidence=llm_rel['confidence'] if 'confidence' in llm_rel else 0.8,
                    frequency=1
                )
                relationships.append(relationship)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from single document")
        
        from flowlib.knowledge.models import EntityExtractionOutput
        return EntityExtractionOutput(
            entities=entities,
            relationships=relationships,
            entity_document_map=entity_doc_map
        )

    async def process_document_batch(self, documents: List["DocumentContent"], batch_size: int = 5) -> List["EntityExtractionOutput"]:
        """Process multiple documents in efficient batches for streaming."""
        
        logger.info(f"Processing batch of {len(documents)} documents")
        
        results = []
        
        # Process documents in batches to optimize LLM usage
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            logger.debug(f"Processing document batch {i//batch_size + 1}: {len(batch_docs)} documents")
            
            # Process each document in the batch
            batch_results = []
            for doc in batch_docs:
                try:
                    doc_result = await self.extract_from_single_document(doc)
                    batch_results.append(doc_result)
                except Exception as e:
                    logger.error(f"Failed to process document {doc.document_id}: {e}")
                    # Create empty result for failed document
                    from flowlib.knowledge.models import EntityExtractionOutput
                    empty_result = EntityExtractionOutput(
                        entities=[],
                        relationships=[],
                        entity_document_map={}
                    )
                    batch_results.append(empty_result)
            
            results.extend(batch_results)
            
            # Small delay between batches to prevent LLM overload
            import asyncio
            await asyncio.sleep(0.1)
        
        logger.info(f"Completed batch processing: {len(results)} document results")
        return results

    async def _analyze_entities(self, input_data: EntityExtractionInput) -> EntityExtractionOutput:
        """Extract entities and relationships from documents using LLM."""
        logger.info("Starting entity and relationship analysis")
        
        all_entities = []
        all_relationships = []
        entity_doc_map = {}
        
        for doc in input_data.documents:
            if not doc.full_text:
                continue
            
            logger.info(f"Analyzing document: {doc.metadata.file_name}")
            
            # Extract using LLM
            extraction_result = await self._extract_from_document_llm(doc, input_data)
            
            # Convert LLM results to our Entity models
            for llm_entity in extraction_result['entities']:
                entity_id = hashlib.md5(llm_entity['name'].encode()).hexdigest()
                
                entity = Entity(
                    entity_id=entity_id,
                    name=llm_entity['name'],
                    entity_type=self._map_entity_type(llm_entity['type']),
                    description=llm_entity['description'] if 'description' in llm_entity else None,
                    documents=[doc.document_id],
                    frequency=1,  # Will be aggregated later
                    confidence=self._importance_to_confidence(llm_entity['importance'] if 'importance' in llm_entity else 'medium'),
                    properties=llm_entity['attributes'] if 'attributes' in llm_entity else {}
                )
                all_entities.append(entity)
                
                if entity_id not in entity_doc_map:
                    entity_doc_map[entity_id] = []
                entity_doc_map[entity_id].append(doc.document_id)
            
            # Convert LLM relationships
            for llm_rel in extraction_result['relationships']:
                # Find entity IDs
                source_id = None
                target_id = None
                
                for entity in all_entities:
                    if entity.name == llm_rel['source']:
                        source_id = entity.entity_id
                    if entity.name == llm_rel['target']:
                        target_id = entity.entity_id
                
                if source_id and target_id:
                    rel_id = hashlib.md5(
                        f"{source_id}_{llm_rel['type']}_{target_id}".encode()
                    ).hexdigest()
                    
                    relationship = Relationship(
                        relationship_id=rel_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relationship_type=self._map_relationship_type(llm_rel['type']),
                        description=llm_rel['description'] if 'description' in llm_rel else None,
                        documents=[doc.document_id],
                        context_sentences=[llm_rel['description']] if 'description' in llm_rel else [],
                        confidence=llm_rel['confidence'] if 'confidence' in llm_rel else 0.8,
                        frequency=1
                    )
                    all_relationships.append(relationship)
        
        # Deduplicate and aggregate entities
        entities = self._deduplicate_entities(all_entities)
        relationships = self._deduplicate_relationships(all_relationships)
        
        logger.info(f"Analysis complete: {len(entities)} entities, {len(relationships)} relationships")
        
        return EntityExtractionOutput(
            entities=entities,
            relationships=relationships,
            entity_document_map=entity_doc_map
        )
    
    def _map_entity_type(self, llm_type: str) -> EntityType:
        """Map LLM entity type to our EntityType enum."""
        type_mapping = {
            'person': EntityType.PERSON,
            'organization': EntityType.ORGANIZATION,
            'location': EntityType.LOCATION,
            'technology': EntityType.TECHNOLOGY,
            'concept': EntityType.CONCEPT,
            'system': EntityType.TECHNOLOGY,
            'component': EntityType.TECHNOLOGY,
            'standard': EntityType.TERM,
            'protocol': EntityType.TECHNOLOGY,
            'framework': EntityType.METHODOLOGY,
            'tool': EntityType.TECHNOLOGY,
            'methodology': EntityType.METHODOLOGY,
            'theory': EntityType.THEORY,
        }
        return type_mapping[llm_type.lower()] if llm_type.lower() in type_mapping else EntityType.CONCEPT
    
    def _map_relationship_type(self, llm_type: str) -> RelationType:
        """Map LLM relationship type to our RelationType enum."""
        type_mapping = {
            'implements': RelationType.IMPLEMENTS,
            'extends': RelationType.DERIVED_FROM,
            'uses': RelationType.IMPLEMENTS,
            'depends_on': RelationType.RELATES_TO,
            'interfaces_with': RelationType.RELATES_TO,
            'complies_with': RelationType.IMPLEMENTS,
            'replaces': RelationType.DERIVED_FROM,
            'based_on': RelationType.DERIVED_FROM,
            'defines': RelationType.DEFINES,
            'part_of': RelationType.IS_PART_OF,
            'influences': RelationType.INFLUENCES,
            'cites': RelationType.CITES,
            'contradicts': RelationType.CONTRADICTS,
            'supports': RelationType.SUPPORTS,
        }
        return type_mapping[llm_type.lower()] if llm_type.lower() in type_mapping else RelationType.RELATES_TO
    
    def _importance_to_confidence(self, importance: str) -> float:
        """Convert importance to confidence score."""
        mapping = {
            'critical': 1.0,
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        return mapping[importance.lower()] if importance.lower() in mapping else 0.7
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities by name and aggregate properties."""
        entity_map = {}
        
        for entity in entities:
            key = entity.name.lower()
            
            if key in entity_map:
                # Merge with existing
                existing = entity_map[key]
                existing.documents.extend(entity.documents)
                existing.documents = list(set(existing.documents))
                existing.frequency += entity.frequency
                existing.confidence = max(existing.confidence, entity.confidence)
                
                # Merge properties
                for k, v in entity.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
            else:
                entity_map[key] = entity
        
        return list(entity_map.values())
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Deduplicate relationships."""
        rel_map = {}
        
        for rel in relationships:
            key = (rel.source_entity_id, rel.target_entity_id, rel.relationship_type)
            
            if key in rel_map:
                # Merge with existing
                existing = rel_map[key]
                existing.documents.extend(rel.documents)
                existing.documents = list(set(existing.documents))
                existing.frequency += rel.frequency
                existing.confidence = max(existing.confidence, rel.confidence)
                existing.context_sentences.extend(rel.context_sentences)
            else:
                rel_map[key] = rel
        
        return list(rel_map.values())
    
    async def _extract_from_document_llm(self, document: DocumentContent, config: EntityExtractionInput) -> Dict[str, Any]:
        """Extract entities and relationships from document using LLM."""
        llm = await provider_registry.get_by_config("default-llm")
        
        # Process all chunks - no artificial limits
        chunks = document.chunks
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            # Extract entities
            entity_prompt = resource_registry.get("entity-extraction-llm")
            entity_result = await llm.generate_structured(
                prompt=entity_prompt,
                output_type=LLMEntityExtractionResult,
                model_name="default-model",
                prompt_variables={
                    "domain": config.extraction_domain,
                    "context": f"Document: {document.metadata.file_name}",
                    "text": chunk.text
                }
            )
            
            for entity in entity_result.entities:
                all_entities.append({
                    'name': entity.name,
                    'type': entity.type,
                    'description': entity.description,
                    'importance': entity.importance,
                    'attributes': entity.attributes
                })
        
        # Extract relationships if we have entities
        if all_entities:
            entity_names = list(set(e['name'] for e in all_entities))[:20]
            rel_prompt = resource_registry.get("relationship-extraction-llm")
            
            for chunk in chunks:  # Process all chunks for relationships
                rel_result = await llm.generate_structured(
                    prompt=rel_prompt,
                    output_type=LLMRelationshipExtractionResult,
                    model_name="default-model",
                    prompt_variables={
                        "domain": config.extraction_domain,
                        "entity_list": ', '.join(entity_names),
                        "text": chunk.text
                    }
                )
                
                for rel in rel_result.relationships:
                    all_relationships.append({
                        'source': rel.source,
                        'target': rel.target,
                        'type': rel.type,
                        'description': rel.description,
                        'confidence': rel.confidence
                    })
        
        # Extract key concepts
        concept_prompt = resource_registry.get("concept-extraction-llm")
        concept_result = await llm.generate_structured(
            prompt=concept_prompt,
            output_type=LLMConceptExtractionResult,
            model_name="knowledge-extraction",
            prompt_variables={
                "domain": config.extraction_domain,
                "max_concepts": 10,
                "text": document.full_text[:2000]
            }
        )
        
        # Add concepts as special entities
        for concept in concept_result.concepts:
            all_entities.append({
                'name': concept.concept,
                'type': 'concept',
                'description': concept.explanation,
                'importance': concept.importance,
                'attributes': {
                    'abbreviation': concept.abbreviation,
                    'related_concepts': concept.related_concepts
                }
            })
        
        return {
            'entities': all_entities,
            'relationships': all_relationships,
            'concepts': [c.model_dump() for c in concept_result.concepts]
        }
    
    @pipeline(input_model=EntityExtractionInput, output_model=EntityExtractionOutput)
    async def run_pipeline(self, input_data: EntityExtractionInput) -> EntityExtractionOutput:
        """Run complete entity and relationship analysis."""
        return await self._analyze_entities(input_data)