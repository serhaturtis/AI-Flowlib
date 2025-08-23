"""
Memory interface definitions.

This module defines the protocols and interfaces for memory operations.
"""

from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from .models import (
    MemoryItem,
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext
)


@runtime_checkable
class MemoryInterface(Protocol):
    """Interface for memory operations.
    
    Defines the methods for storing, retrieving, and searching memory items.
    """
    
    async def store_with_model(self, request: MemoryStoreRequest) -> None:
        """Store a value in memory using a structured request.
        
        Args:
            request: Memory store request with parameters
        """
        ...
    
    
    async def retrieve_with_model(self, request: MemoryRetrieveRequest) -> Any:
        """Retrieve a value from memory using a structured request.
        
        Args:
            request: Memory retrieve request with parameters
            
        Returns:
            Retrieved value or default if not found
        """
        ...
    
    async def search_with_model(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search memory using a structured request.
        
        Args:
            request: Memory search request with parameters
            
        Returns:
            Memory search result with matching items
        """
        ...    
    
    def create_context(
        self,
        context_name: str,
        parent: Optional[Union[str, MemoryContext]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new memory context.
        
        Args:
            context_name: Name for the new context
            parent: Optional parent context or context path
            metadata: Optional metadata for the context
            
        Returns:
            Full context path
        """
        ...
    
    def get_context_model(self, context_path: str) -> Optional[MemoryContext]:
        """Get a memory context model by path.
        
        Args:
            context_path: Context path
            
        Returns:
            Memory context model or None if not found
        """
        ...
    
    async def wipe(
        self,
        context: Optional[str] = None
    ) -> None:
        """Wipe memory contents.
        
        Args:
            context: Optional context to wipe (wipes all if None)
        """
        ... 