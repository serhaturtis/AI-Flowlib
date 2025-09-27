"""Strict Pydantic models for S3 responses and operations.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from datetime import datetime
from typing import Dict, Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel


class S3ErrorResponse(StrictBaseModel):
    """Strict S3 error response model."""
    # Inherits strict configuration from StrictBaseModel
    
    Code: str = Field(..., description="S3 error code")
    Message: Optional[str] = Field(None, description="Error message")
    RequestId: Optional[str] = Field(None, description="Request ID")


class S3ErrorContainer(StrictBaseModel):
    """Container for S3 error responses."""
    # Inherits strict configuration from StrictBaseModel
    
    Error: S3ErrorResponse = Field(..., description="S3 error details")


class S3ClientErrorResponse(StrictBaseModel):
    """Complete S3 client error response."""
    # Inherits strict configuration from StrictBaseModel
    
    response: S3ErrorContainer = Field(..., description="Error response container")


class S3ObjectResponse(StrictBaseModel):
    """Strict S3 object response model."""
    # Inherits strict configuration from StrictBaseModel
    
    ContentLength: int = Field(..., description="Size of object in bytes")
    ETag: str = Field(..., description="ETag of the object")
    ContentType: str = Field(..., description="MIME type of the object")
    LastModified: datetime = Field(..., description="Last modification time")
    Metadata: Dict[str, str] = Field(..., description="User metadata")


class S3ListObjectResponse(StrictBaseModel):
    """Strict S3 list objects response."""
    # Inherits strict configuration from StrictBaseModel
    
    Key: str = Field(..., description="Object key")
    Size: int = Field(..., description="Size in bytes")
    ETag: str = Field(..., description="ETag")
    LastModified: datetime = Field(..., description="Last modified time")


class S3ListResponse(StrictBaseModel):
    """Strict S3 list response."""
    # Inherits strict configuration from StrictBaseModel
    
    Contents: list[S3ListObjectResponse] = Field(..., description="List of objects")
    IsTruncated: bool = Field(..., description="Whether response is truncated")
    NextContinuationToken: Optional[str] = Field(None, description="Continuation token")


class S3UploadResponse(StrictBaseModel):
    """Strict S3 upload response."""
    # Inherits strict configuration from StrictBaseModel
    
    ETag: str = Field(..., description="ETag of uploaded object")
    VersionId: Optional[str] = Field(None, description="Version ID if versioning enabled")


class S3DeleteResponse(StrictBaseModel):
    """Strict S3 delete response."""
    # Inherits strict configuration from StrictBaseModel
    
    DeleteMarker: Optional[bool] = Field(None, description="Whether delete marker was created")
    VersionId: Optional[str] = Field(None, description="Version ID of deleted object")


class S3CopyResponse(StrictBaseModel):
    """Strict S3 copy response."""
    # Inherits strict configuration from StrictBaseModel
    
    ETag: str = Field(..., description="ETag of copied object")
    LastModified: datetime = Field(..., description="Last modified time")


class S3CopyResponseContainer(StrictBaseModel):
    """Container for S3 copy response."""
    # Inherits strict configuration from StrictBaseModel
    
    CopyObjectResult: S3CopyResponse = Field(..., description="Copy result")


class S3PresignedUrlResponse(StrictBaseModel):
    """Strict presigned URL response."""
    # Inherits strict configuration from StrictBaseModel
    
    url: str = Field(..., description="Presigned URL")
    expires_in: int = Field(..., description="Expiration time in seconds")