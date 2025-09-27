"""Entity formatting utilities.

This module provides utilities for formatting entity data for display
and for use in prompts or other contexts.
"""

from typing import List, Dict
# Importing Entity type for type hints
from flowlib.providers.graph.models import Entity, EntityRelationship


def format_entity_for_display(entity: Entity, detailed: bool = False) -> str:
    """Format an entity for human-readable display.
    
    Args:
        entity: The entity to format
        detailed: Whether to include all details
        
    Returns:
        Formatted string representation
    """
    lines = [f"Entity: {entity.type.title()} - {entity.id}"]
    
    # Add attributes
    if entity.attributes:
        lines.append("Attributes:")
        for name, attr in sorted(entity.attributes.items()):
            attr_line = f"  {name}: {attr.value}"
            if detailed and attr.source:
                attr_line += f" (from {attr.source}, confidence: {attr.confidence:.2f})"
            lines.append(attr_line)
    
    # Add relationships
    if entity.relationships:
        lines.append("Relationships:")
        # Group by relation type for cleaner display
        rel_by_type: Dict[str, List[EntityRelationship]] = {}
        for rel in entity.relationships:
            if rel.relation_type not in rel_by_type:
                rel_by_type[rel.relation_type] = []
            rel_by_type[rel.relation_type].append(rel)
            
        for rel_type, rels in sorted(rel_by_type.items()):
            targets = [rel.target_entity for rel in rels]
            rel_line = f"  {rel_type}: {', '.join(targets)}"
            lines.append(rel_line)
    
    # Add metadata if detailed
    if detailed:
        if entity.tags:
            lines.append(f"Tags: {', '.join(entity.tags)}")
            
        if entity.importance is not None:
            lines.append(f"Importance: {entity.importance:.2f}")
            
        if hasattr(entity, 'source') and entity.source:
            lines.append(f"Source: {entity.source}")
            
        if entity.last_updated:
            lines.append(f"Last Updated: {entity.last_updated}")
    
    return "\n".join(lines)


def format_entities_as_context(entities: List['Entity'], include_relationships: bool = True) -> str:
    """Format multiple entities as context for prompt injection.
    
    Args:
        entities: List of entities to format
        include_relationships: Whether to include relationship information
        
    Returns:
        Formatted context string
    """
    if not entities:
        return ""
        
    parts = ["Relevant memory information:"]
    
    for entity in entities:
        entity_part = format_entity_for_display(
            entity,
            detailed=False  # Less verbose for context
        )
        parts.append(entity_part)
    
    return "\n\n".join(parts)


def format_entity_list(entities: List['Entity'], compact: bool = False) -> str:
    """Format a list of entities in a compact or detailed format.
    
    Args:
        entities: List of entities to format
        compact: Whether to use a compact single-line format
        
    Returns:
        Formatted entity list
    """
    if not entities:
        return "No entities"
        
    if compact:
        entity_strs = [f"{e.type}:{e.id}" for e in entities]
        return ", ".join(entity_strs)
    else:
        return "\n\n".join([format_entity_for_display(e) for e in entities]) 