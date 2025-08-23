"""Knowledge provider system for domain-specific databases."""

from .base import (
    Knowledge,
    KnowledgeProvider,
    MultiDatabaseKnowledgeProvider
)
from .plugin_manager import KnowledgePluginManager

__all__ = [
    "Knowledge",
    "KnowledgeProvider", 
    "MultiDatabaseKnowledgeProvider",
    "KnowledgePluginManager"
]