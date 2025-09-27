"""Integration helpers for domain-aware plugin generation."""

import logging
from typing import Dict, Any, Tuple

from .domain_strategies import domain_strategy_registry, BaseDomainStrategy
from .models import PluginGenerationRequest
from ..models import ExtractionConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class DomainAwarePluginGenerator:
    """Helper class to integrate domain strategies into plugin generation."""
    
    def __init__(self) -> None:
        self.registry = domain_strategy_registry
    
    def get_domain_strategy(self, request: PluginGenerationRequest) -> BaseDomainStrategy:
        """Get the appropriate domain strategy for a request."""
        return self.registry.get_strategy(
            request.domain_strategy,
            request.domain_config
        )
    
    def create_domain_aware_extraction_config(
        self, 
        request: PluginGenerationRequest,
        domain_strategy: BaseDomainStrategy
    ) -> Tuple[ExtractionConfig, Dict[str, Any]]:
        """Create extraction config enhanced with domain-specific settings."""
        
        # Get domain-specific extraction configuration
        domain_config = domain_strategy.get_extraction_config()
        
        # Create extraction config with domain enhancements
        extraction_config = ExtractionConfig(
            max_chunk_size=request.chunk_size,
            overlap_size=request.chunk_overlap,
            chunking_strategy=ChunkingStrategy.SEMANTIC_AWARE,
            batch_size=5,
            checkpoint_interval=10,
            memory_limit_gb=8,
            enable_resumption=True,
            min_chunk_size=100,
            preserve_structure=True
        )
        
        # Create domain-specific metadata for later use
        domain_metadata = {
            "entity_extraction_prompt": domain_strategy.get_entity_extraction_prompt(),
            "relationship_extraction_prompt": domain_strategy.get_relationship_extraction_prompt(),
            "concept_extraction_prompt": domain_strategy.get_concept_extraction_prompt(),
            "expected_entity_types": domain_config.entity_types,
            "expected_relationship_types": domain_config.relationship_types,
            "domain_strategy": domain_strategy.strategy_name
        }
        
        logger.info(f"Created domain-aware extraction config for {domain_strategy.strategy_name}")
        logger.debug(f"Entity types: {len(domain_config.entity_types)}")
        logger.debug(f"Relationship types: {len(domain_config.relationship_types)}")
        
        return extraction_config, domain_metadata
    
    def validate_extraction_results(
        self, 
        domain_strategy: BaseDomainStrategy,
        extraction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate extraction results using domain-specific rules."""
        
        validation_errors = domain_strategy.validate_extracted_data(extraction_data)
        
        validation_result = {
            "is_valid": len(validation_errors) == 0,
            "error_count": len(validation_errors),
            "errors": validation_errors,
            "warnings": []
        }
        
        if validation_errors:
            logger.warning(f"Domain validation found {len(validation_errors)} issues")
            for i, error in enumerate(validation_errors[:10]):  # Log first 10 errors
                logger.warning(f"  {i+1}. {error}")
                
            if len(validation_errors) > 10:
                logger.warning(f"  ... and {len(validation_errors) - 10} more issues")
        else:
            logger.info("✅ Domain validation passed - all extracted data conforms to domain expectations")
        
        return validation_result
    
    def enhance_plugin_metadata(
        self,
        base_metadata: Dict[str, Any],
        domain_strategy: BaseDomainStrategy,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance plugin metadata with domain-specific information."""
        
        enhanced_metadata = base_metadata.copy()
        
        # Add domain strategy information
        enhanced_metadata.update({
            "domain_strategy": {
                "name": domain_strategy.strategy_name,
                "description": domain_strategy.description,
                "strategy_type": type(domain_strategy).__name__
            },
            "domain_extraction_config": domain_strategy.get_extraction_config().model_dump(),
            "domain_validation": validation_result
        })
        
        # Add domain-specific quality metrics
        if validation_result["is_valid"]:
            enhanced_metadata["quality_score"] = 1.0
        else:
            # Calculate quality score based on error ratio
            entities = base_metadata["entities"] if "entities" in base_metadata else []
            total_entities = len(entities)
            error_ratio = validation_result["error_count"] / max(total_entities, 1)
            enhanced_metadata["quality_score"] = max(0.0, 1.0 - error_ratio)
        
        logger.info(f"Enhanced plugin metadata with domain strategy: {domain_strategy.strategy_name}")
        logger.info(f"Quality score: {enhanced_metadata['quality_score']:.2f}")
        
        return enhanced_metadata
    
    def get_available_strategies(self) -> Dict[str, Dict[str, str]]:
        """Get list of available domain strategies for UI/CLI."""
        strategies = self.registry.list_available_strategies()
        return {
            str(domain): metadata
            for domain, metadata in strategies.items()
        }


# Global instance for easy access
domain_plugin_generator = DomainAwarePluginGenerator()