"""Knowledge extraction flow for learning from conversations and research."""

import logging
from typing import List, Optional

from ....flows.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
# Removed ProviderType import - using config-driven provider access
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType

from .models import (
    KnowledgeExtractionInput,
    KnowledgeExtractionOutput,
    ExtractedKnowledge,
    KnowledgeType
)

logger = logging.getLogger(__name__)


@flow(
    name="knowledge-extraction",
    description="Extract knowledge from conversations and research for long-term learning",
    is_infrastructure=False
)
class KnowledgeExtractionFlow:
    """Flow that extracts structured knowledge from text for agent learning.
    
    This flow analyzes conversations, web research, and other text sources to
    identify valuable knowledge that should be stored for future reference.
    It categorizes knowledge by type and domain, making it searchable and
    retrievable by the knowledge plugin system.
    """

    @pipeline(
        input_model=KnowledgeExtractionInput,
        output_model=KnowledgeExtractionOutput
    )
    async def run_pipeline(self, input_data: KnowledgeExtractionInput) -> KnowledgeExtractionOutput:
        """Extract knowledge from the provided text.
        
        Args:
            input_data: Text and context to extract knowledge from
            
        Returns:
            Extracted knowledge items with categorization and metadata
        """
        logger.info(f"Starting knowledge extraction for {len(input_data.text)} characters of text")
        
        try:
            # Get LLM provider for extraction
            llm_provider = await provider_registry.get_by_config("default-llm")
            if not llm_provider:
                raise RuntimeError("LLM provider not available for knowledge extraction")
            
            # Get extraction prompt
            extraction_prompt = resource_registry.get(
                name="knowledge_extraction",
                resource_type=ResourceType.PROMPT
            )
            
            # Prepare prompt variables
            prompt_vars = {
                "text": input_data.text,
                "context": input_data.context,
                "domain_hint": input_data.domain_hint or "general",
                "extract_personal": input_data.extract_personal
            }
            
            # Perform knowledge extraction
            logger.debug("Performing LLM-based knowledge extraction")
            extraction_result = await llm_provider.generate_structured(
                prompt=extraction_prompt,
                prompt_variables=prompt_vars,
                output_type=KnowledgeExtractionOutput,
                model_name="agent-model-large"  # Use large model for knowledge extraction
            )
            
            # Validate and enhance extracted knowledge
            validated_knowledge = []
            for knowledge in extraction_result.extracted_knowledge:
                # Apply validation and enhancement
                enhanced_knowledge = await self._enhance_knowledge(knowledge, input_data)
                if enhanced_knowledge:
                    validated_knowledge.append(enhanced_knowledge)
            
            # Detect additional domains using secondary analysis
            detected_domains = await self._detect_domains(input_data.text)
            
            result = KnowledgeExtractionOutput(
                extracted_knowledge=validated_knowledge,
                processing_notes=f"Extracted {len(validated_knowledge)} knowledge items from text. "
                                f"Domains detected: {', '.join(detected_domains)}",
                domains_detected=detected_domains
            )
            
            logger.info(f"Knowledge extraction completed: {len(validated_knowledge)} items extracted")
            return result
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            # Return empty result rather than failing
            return KnowledgeExtractionOutput(
                extracted_knowledge=[],
                processing_notes=f"Knowledge extraction failed: {str(e)}",
                domains_detected=[]
            )
    
    async def _enhance_knowledge(
        self, 
        knowledge: ExtractedKnowledge, 
        input_data: KnowledgeExtractionInput
    ) -> Optional[ExtractedKnowledge]:
        """Enhance and validate extracted knowledge.
        
        Args:
            knowledge: Raw extracted knowledge
            input_data: Original input for context
            
        Returns:
            Enhanced knowledge or None if invalid
        """
        try:
            # Skip if confidence is too low
            if knowledge.confidence < 0.3:
                logger.debug(f"Skipping low-confidence knowledge: {knowledge.confidence}")
                return None
            
            # Skip if content is too short or generic
            if len(knowledge.content.strip()) < 10:
                logger.debug("Skipping too-short knowledge content")
                return None
            
            # Enhance metadata
            enhanced_metadata = {
                **knowledge.metadata,
                "extraction_context": input_data.context,
                "source_length": len(input_data.text),
                "extraction_timestamp": logger.handlers[0].formatter.formatTime(logger.handlers[0], None) if logger.handlers else "unknown"
            }
            
            # Create enhanced knowledge
            return ExtractedKnowledge(
                content=knowledge.content,
                knowledge_type=knowledge.knowledge_type,
                domain=knowledge.domain,
                confidence=knowledge.confidence,
                source_context=knowledge.source_context,
                entities=knowledge.entities,
                metadata=enhanced_metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to enhance knowledge item: {e}")
            return knowledge  # Return original if enhancement fails
    
    async def _detect_domains(self, text: str) -> List[str]:
        """Detect knowledge domains in the text using secondary analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected domains
        """
        try:
            # Get LLM provider and domain detection prompt
            llm_provider = await provider_registry.get_by_config("default-llm")
            domain_prompt = resource_registry.get(
                name="domain_detection",
                resource_type=ResourceType.PROMPT
            )
            
            # Implement proper structured generation with domain detection model
            from flowlib.providers.core.registry import provider_registry
            
            try:
                llm_provider = await provider_registry.get_by_config("default-llm")
                if not llm_provider:
                    raise RuntimeError("No default-llm provider configured for domain detection")
                
                # Use the LLM provider to generate structured domain detection
                domain_prompt_text = domain_prompt.render(text=input.text)
                
                response = await llm_provider.generate(
                    prompt=domain_prompt_text,
                    max_tokens=500,
                    temperature=0.1  # Low temperature for consistent classification
                )
                
                if not response or not hasattr(response, 'text'):
                    raise RuntimeError("LLM provider returned invalid response for domain detection")
                    
                domain_result = response.text
                
            except Exception as e:
                logger.error(f"Domain detection failed: {str(e)}")
                raise RuntimeError(f"Domain detection provider access failed: {str(e)}") from e
            
            # Parse JSON result
            import json
            try:
                domains = json.loads(domain_result.strip())
                if isinstance(domains, list):
                    return [d for d in domains if isinstance(d, str)]
            except json.JSONDecodeError:
                logger.warning("Failed to parse domain detection result as JSON")
            
            return []
            
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            return []