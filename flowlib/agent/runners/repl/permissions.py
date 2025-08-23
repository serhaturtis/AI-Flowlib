"""Permission system for REPL tool operations.

This module implements opencode-style permissions for file operations,
adapted to flowlib's architecture and CLAUDE.md principles.
"""

import asyncio
from enum import Enum
from typing import Dict, Optional, Any, Protocol
from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel


class PermissionLevel(Enum):
    """Permission levels for operations."""
    ALLOW = "allow"
    ASK = "ask" 
    DENY = "deny"


class PermissionRequest(BaseModel):
    """Permission request model."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)
    
    operation_type: str = Field(..., description="Type of operation (edit, write, bash, etc.)")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    message_id: Optional[str] = Field(default=None, description="Message ID")
    call_id: Optional[str] = Field(default=None, description="Tool call ID")
    title: str = Field(..., description="Human-readable operation description")
    details: str = Field(default="", description="Additional operation details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PermissionResponse(BaseModel):
    """Permission response model."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True, strict=True)
    
    granted: bool = Field(..., description="Whether permission was granted")
    reason: Optional[str] = Field(default=None, description="Reason for decision")


class PermissionHandler(Protocol):
    """Protocol for permission handlers."""
    
    async def request_permission(self, request: PermissionRequest) -> PermissionResponse:
        """Request permission for an operation."""
        ...


class InteractivePermissionHandler:
    """Interactive permission handler using rich console."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
    
    async def request_permission(self, request: PermissionRequest) -> PermissionResponse:
        """Request permission interactively from user."""
        # Create formatted request panel
        content = f"[bold yellow]Operation:[/bold yellow] {request.operation_type}\n"
        content += f"[bold yellow]Action:[/bold yellow] {request.title}\n"
        
        if request.details:
            content += f"[bold yellow]Details:[/bold yellow] {request.details}\n"
        
        # Add metadata if present
        if request.metadata:
            content += f"[bold yellow]Metadata:[/bold yellow]\n"
            for key, value in request.metadata.items():
                content += f"  {key}: {value}\n"
        
        panel = Panel(
            content,
            title="🔐 Permission Required",
            border_style="yellow",
            expand=False
        )
        
        self.console.print(panel)
        
        # Ask for permission
        try:
            granted = Confirm.ask("Do you want to allow this operation?", console=self.console)
            
            return PermissionResponse(
                granted=granted,
                reason="User interactive decision"
            )
        except KeyboardInterrupt:
            return PermissionResponse(
                granted=False,
                reason="User interrupted permission request"
            )


class AutoPermissionHandler:
    """Automatic permission handler for non-interactive environments."""
    
    def __init__(self, default_allow: bool = False):
        self.default_allow = default_allow
    
    async def request_permission(self, request: PermissionRequest) -> PermissionResponse:
        """Automatically grant or deny permission."""
        return PermissionResponse(
            granted=self.default_allow,
            reason=f"Auto-handler: {'allowed' if self.default_allow else 'denied'}"
        )


class PermissionManager:
    """Manages permissions for REPL tool operations."""
    
    def __init__(self, handler: Optional[PermissionHandler] = None):
        self.handler = handler or InteractivePermissionHandler()
        self.permissions: Dict[str, PermissionLevel] = {
            "edit": PermissionLevel.ASK,
            "write": PermissionLevel.ASK, 
            "bash": PermissionLevel.ASK,
            "read": PermissionLevel.ALLOW,
            "grep": PermissionLevel.ALLOW,
            "glob": PermissionLevel.ALLOW,
            "ls": PermissionLevel.ALLOW,
        }
    
    def set_permission(self, operation: str, level: PermissionLevel):
        """Set permission level for an operation type."""
        self.permissions[operation] = level
    
    def get_permission(self, operation: str) -> PermissionLevel:
        """Get permission level for an operation type."""
        return self.permissions.get(operation, PermissionLevel.ASK)
    
    async def check_permission(self, request: PermissionRequest) -> PermissionResponse:
        """Check if operation is permitted."""
        permission_level = self.get_permission(request.operation_type)
        
        if permission_level == PermissionLevel.ALLOW:
            return PermissionResponse(
                granted=True,
                reason="Operation automatically allowed"
            )
        
        if permission_level == PermissionLevel.DENY:
            return PermissionResponse(
                granted=False,
                reason="Operation denied by permission policy"
            )
        
        # ASK level - request permission from handler
        return await self.handler.request_permission(request)
    
    def set_handler(self, handler: PermissionHandler):
        """Set the permission handler."""
        self.handler = handler


# Global permission manager instance
permission_manager = PermissionManager()


async def request_permission(
    operation_type: str,
    title: str,
    details: str = "",
    session_id: Optional[str] = None,
    message_id: Optional[str] = None,
    call_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Request permission for an operation.
    
    Args:
        operation_type: Type of operation (edit, write, bash, etc.)
        title: Human-readable description
        details: Additional details about the operation
        session_id: Optional session ID
        message_id: Optional message ID  
        call_id: Optional tool call ID
        metadata: Optional additional metadata
        
    Returns:
        bool: True if permission granted, False otherwise
    """
    request = PermissionRequest(
        operation_type=operation_type,
        session_id=session_id,
        message_id=message_id,
        call_id=call_id,
        title=title,
        details=details,
        metadata=metadata or {}
    )
    
    response = await permission_manager.check_permission(request)
    return response.granted


def set_permission_level(operation: str, level: PermissionLevel):
    """Set permission level for an operation type."""
    permission_manager.set_permission(operation, level)


def set_permission_handler(handler: PermissionHandler):
    """Set the global permission handler."""
    permission_manager.set_handler(handler)