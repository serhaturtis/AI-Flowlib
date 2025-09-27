"""Storage provider base class and related functionality.

This module provides the base class for implementing storage providers
that share common functionality for storing, retrieving, and managing
files and binary data in object storage systems.
"""

import logging
import os
from typing import Dict, List, Optional, TypeVar, Union, BinaryIO, Tuple, Generic
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from flowlib.providers.core.base import ProviderSettings
from flowlib.providers.core.base import Provider

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
SettingsT = TypeVar('SettingsT', bound='StorageProviderSettings')


class StorageProviderSettings(ProviderSettings):
    """Base settings for storage providers.
    
    Attributes:
        endpoint: Storage service endpoint
        region: Storage service region
        bucket: Default bucket/container name
        access_key: Access key ID
        secret_key: Secret access key
        use_ssl: Whether to use SSL for connections
        create_bucket: Whether to create bucket if it doesn't exist
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    # Connection settings
    endpoint: Optional[str] = Field(default=None, description="Storage service endpoint URL")
    region: Optional[str] = Field(default=None, description="Storage service region")
    bucket: str = Field(default="default", description="Default storage bucket/container name")
    access_key: Optional[str] = Field(default=None, description="Access key ID for authentication")
    secret_key: Optional[str] = Field(default=None, description="Secret access key for authentication")
    
    # Security settings
    use_ssl: bool = Field(default=True, description="Use SSL/TLS for secure connections")
    
    # Behavior settings
    create_bucket: bool = Field(default=True, description="Create bucket/container if it doesn't exist")
    auto_content_type: bool = Field(default=True, description="Automatically detect content type from file extension")
    
    # Performance settings
    part_size: int = Field(default=5 * 1024 * 1024, description="Part size for multipart uploads in bytes (5MB)")
    max_concurrency: int = Field(default=10, description="Maximum concurrent upload/download threads")
    timeout: float = Field(default=60.0, description="Operation timeout in seconds")


class FileMetadata(BaseModel):
    """Metadata for stored files.
    
    Attributes:
        key: Object key/path
        size: File size in bytes
        etag: Entity tag for the object
        content_type: MIME content type
        modified: Last modified timestamp
        metadata: Custom user metadata
    """
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    key: str
    size: int
    etag: Optional[str] = None
    content_type: str = "application/octet-stream"
    modified: Optional[datetime] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class StorageProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for storage providers.
    
    This class provides:
    1. File upload and download
    2. Metadata management
    3. Bucket/container operations
    4. Presigned URL generation
    """
    
    def __init__(self, name: str = "storage", settings: Optional[SettingsT] = None):
        """Initialize storage provider.
        
        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Pass provider_type="storage" to the parent class
        super().__init__(name=name, settings=settings, provider_type="storage")
        self._initialized = False
        self._client = None
        self._settings = settings or StorageProviderSettings()
        
    @property
    def initialized(self) -> bool:
        """Return whether provider has been initialized."""
        return self._initialized
        
    async def initialize(self) -> None:
        """Initialize the storage provider.
        
        This method should be implemented by subclasses to establish
        connections to the storage service and create the default bucket
        if needed.
        """
        self._initialized = True
        
    async def shutdown(self) -> None:
        """Close all connections and release resources.
        
        This method should be implemented by subclasses to properly
        close connections and clean up resources.
        """
        self._initialized = False
        self._client = None
        
    async def create_bucket(self, bucket: Optional[str] = None) -> bool:
        """Create a new bucket/container.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket was created successfully
            
        Raises:
            ProviderError: If bucket creation fails
        """
        raise NotImplementedError("Subclasses must implement create_bucket()")
        
    async def delete_bucket(self, bucket: Optional[str] = None, force: bool = False) -> bool:
        """Delete a bucket/container.
        
        Args:
            bucket: Bucket name (default from settings if None)
            force: Whether to delete all objects in the bucket first
            
        Returns:
            True if bucket was deleted successfully
            
        Raises:
            ProviderError: If bucket deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_bucket()")
        
    async def bucket_exists(self, bucket: Optional[str] = None) -> bool:
        """Check if a bucket/container exists.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket exists
            
        Raises:
            ProviderError: If check fails
        """
        raise NotImplementedError("Subclasses must implement bucket_exists()")
        
    async def upload_file(self, file_path: str, object_key: str, bucket: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload a file to storage.
        
        Args:
            file_path: Path to local file
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            metadata: Optional object metadata
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If upload fails
        """
        raise NotImplementedError("Subclasses must implement upload_file()")
        
    async def upload_data(self, data: Union[bytes, BinaryIO], object_key: str, bucket: Optional[str] = None,
                        content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload binary data to storage.
        
        Args:
            data: Binary data or file-like object
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            content_type: Optional content type
            metadata: Optional object metadata
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If upload fails
        """
        raise NotImplementedError("Subclasses must implement upload_data()")
        
    async def download_file(self, object_key: str, file_path: str, bucket: Optional[str] = None) -> FileMetadata:
        """Download a file from storage.
        
        Args:
            object_key: Storage object key/path
            file_path: Path to save file
            bucket: Bucket name (default from settings if None)
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If download fails
        """
        raise NotImplementedError("Subclasses must implement download_file()")
        
    async def download_data(self, object_key: str, bucket: Optional[str] = None) -> Tuple[bytes, FileMetadata]:
        """Download binary data from storage.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Tuple of (data bytes, file metadata)
            
        Raises:
            ProviderError: If download fails
        """
        raise NotImplementedError("Subclasses must implement download_data()")
        
    async def get_metadata(self, object_key: str, bucket: Optional[str] = None) -> FileMetadata:
        """Get metadata for a storage object.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If metadata retrieval fails
        """
        raise NotImplementedError("Subclasses must implement get_metadata()")
        
    async def delete_object(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """Delete a storage object.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if object was deleted successfully
            
        Raises:
            ProviderError: If deletion fails
        """
        raise NotImplementedError("Subclasses must implement delete_object()")
        
    async def list_objects(self, prefix: Optional[str] = None, bucket: Optional[str] = None) -> List[FileMetadata]:
        """List objects in a bucket/container.
        
        Args:
            prefix: Optional prefix to filter objects
            bucket: Bucket name (default from settings if None)
            
        Returns:
            List of file metadata
            
        Raises:
            ProviderError: If listing fails
        """
        raise NotImplementedError("Subclasses must implement list_objects()")
        
    async def generate_presigned_url(self, object_key: str, expiration: int = 3600, bucket: Optional[str] = None,
                                  operation: str = "get") -> str:
        """Generate a presigned URL for a storage object.
        
        Args:
            object_key: Storage object key/path
            expiration: URL expiration in seconds
            bucket: Bucket name (default from settings if None)
            operation: Operation type ('get', 'put', 'delete')
            
        Returns:
            Presigned URL
            
        Raises:
            ProviderError: If URL generation fails
        """
        raise NotImplementedError("Subclasses must implement generate_presigned_url()")
        
    async def check_connection(self) -> bool:
        """Check if storage connection is active.
        
        Returns:
            True if connection is active, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_connection()")
        
    def get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME content type
        """
        # Basic mapping of common extensions to content types
        extension_map = {
            ".txt": "text/plain",
            ".html": "text/html",
            ".htm": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".xml": "application/xml",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".zip": "application/zip",
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
            ".mp3": "audio/mpeg",
            ".mp4": "video/mp4",
            ".wav": "audio/wav",
            ".avi": "video/x-msvideo",
            ".csv": "text/csv",
        }
        
        _, ext = os.path.splitext(file_path.lower())
        if ext not in extension_map:
            return "application/octet-stream"
        return extension_map[ext] 