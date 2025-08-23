"""Local file storage provider implementation.

This module provides a concrete implementation of the StorageProvider
for local filesystem storage operations.
"""

import logging
import os
import shutil
import mimetypes
import hashlib
import pathlib
from typing import Dict, List, Optional, BinaryIO, Tuple, Union
from datetime import datetime
from pydantic import field_validator

from flowlib.core.errors.errors import ProviderError, ErrorContext
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.storage.base import StorageProvider, FileMetadata
from flowlib.providers.core.base import ProviderSettings
from pydantic import Field

logger = logging.getLogger(__name__)


class LocalStorageProviderSettings(ProviderSettings):
    """Local storage provider settings - direct inheritance, only local filesystem-specific fields.
    
    Local storage requires:
    1. Base directory path for operations
    2. File system permissions and behavior settings
    
    Note: Local storage is filesystem-based, no network connection needed.
    """
    
    # Local filesystem settings
    base_path: str = Field(default="./storage", description="Base directory for file operations (e.g., './storage', '/data/files')")
    create_dirs: bool = Field(default=True, description="Whether to create directories if they don't exist")
    use_relative_paths: bool = Field(default=True, description="Whether to interpret object keys as relative paths")
    permissions: Optional[int] = Field(default=None, description="Default file permissions (Unix style, e.g. 0o644)")
    
    @field_validator('base_path')
    def validate_base_path(cls, v):
        """Validate that base_path is absolute and normalized."""
        base_path = os.path.abspath(os.path.normpath(v))
        # Ensure base path ends with a trailing slash
        if not base_path.endswith(os.path.sep):
            base_path += os.path.sep
        return base_path


from flowlib.providers.storage.base import StorageProvider

@provider(provider_type="storage", name="local-storage", settings_class=LocalStorageProviderSettings)
class LocalStorageProvider(StorageProvider):
    """Local filesystem implementation of the StorageProvider.
    
    This provider implements storage operations using the local filesystem,
    useful for development, testing, or deployments without cloud storage.
    """
    
    def __init__(self, name: str = "local-storage", settings: Optional[LocalStorageProviderSettings] = None):
        """Initialize local file storage provider.
        
        Args:
            name: Provider instance name
            settings: Optional provider settings
        """
        if settings is None:
            raise ValueError("LocalStorageProvider requires settings with base_path")
        super().__init__(name=name, settings=settings)
    
    def _create_provider_error(
        self,
        message: str,
        operation: str,
        cause: Exception = None
    ) -> ProviderError:
        """Create a provider error with strict context.
        
        Args:
            message: Error message
            operation: Operation being performed
            cause: Original exception
            
        Returns:
            ProviderError with strict context
        """
        error_context = ErrorContext.create(
            flow_name="storage_operation",
            error_type="ProviderError",
            error_location=f"{self.__class__.__name__}.{operation}",
            component=self.name,
            operation=operation
        )
        
        provider_context = ProviderErrorContext(
            provider_name=self.name,
            provider_type="storage",
            operation=operation,
            retry_count=0
        )
        
        return ProviderError(
            message=message,
            context=error_context,
            provider_context=provider_context,
            cause=cause
        )
        
    async def initialize(self):
        """Initialize the local file storage provider."""
        if self._initialized:
            return
            
        try:
            # Create base directory if it doesn't exist
            if self.settings.create_dirs and not os.path.exists(self.settings.base_path):
                os.makedirs(self.settings.base_path, exist_ok=True)
                logger.debug(f"Created base directory: {self.settings.base_path}")
                
            # Check if base directory is writable
            if not os.access(self.settings.base_path, os.W_OK):
                raise self._create_provider_error(
                    message=f"Base directory {self.settings.base_path} is not writable",
                    operation="initialize"
                )
                
            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise self._create_provider_error(
                message=f"Failed to initialize local file storage provider: {str(e)}",
                operation="initialize",
                cause=e
            )
            
    async def shutdown(self):
        """Shut down local file storage provider."""
        self._initialized = False
        logger.debug(f"{self.name} provider shut down successfully")
            
    def _get_bucket_path(self, bucket: Optional[str] = None) -> str:
        """Get the absolute path for a bucket.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Absolute path to the bucket directory
        """
        bucket = bucket or "default"
        return os.path.join(self.settings.base_path, bucket)
        
    def _get_object_path(self, object_key: str, bucket: Optional[str] = None) -> str:
        """Get the absolute path for an object.
        
        Args:
            object_key: Object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Absolute path to the object file
        """
        bucket_path = self._get_bucket_path(bucket)
        
        # Normalize object key to handle both Windows and Unix paths
        if self.settings.use_relative_paths:
            # Convert to posix path and remove leading slashes to avoid escaping the bucket
            object_key = object_key.lstrip('/').lstrip('\\')
            
        return os.path.join(bucket_path, object_key)
        
    async def create_bucket(self, bucket: Optional[str] = None) -> bool:
        """Create a new bucket (directory).
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket was created successfully
            
        Raises:
            ProviderError: If bucket creation fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="create_bucket"
            )
            
        bucket = bucket or "default"
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            # Check if bucket already exists
            if os.path.exists(bucket_path):
                return True
                
            # Create bucket directory
            os.makedirs(bucket_path, exist_ok=True)
            logger.debug(f"Created bucket directory: {bucket_path}")
            
            return True
            
        except Exception as e:
            raise self._create_provider_error(
                message=f"Failed to create bucket directory {bucket}: {str(e)}",
                operation="create_bucket",
                cause=e
            )
        
    async def delete_bucket(self, bucket: Optional[str] = None, force: bool = False) -> bool:
        """Delete a bucket (directory).
        
        Args:
            bucket: Bucket name (default from settings if None)
            force: Whether to delete all objects in the bucket first
            
        Returns:
            True if bucket was deleted successfully
            
        Raises:
            ProviderError: If bucket deletion fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="delete_bucket"
            )
            
        bucket = bucket or "default"
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            # Check if bucket exists
            if not os.path.exists(bucket_path):
                return True
                
            # Check if bucket is empty or force flag is set
            if not force and os.listdir(bucket_path):
                raise self._create_provider_error(
                    message=f"Bucket {bucket} is not empty. Use force=True to delete it anyway.",
                    operation="delete_bucket"
                )
                
            # Delete bucket directory
            shutil.rmtree(bucket_path)
            logger.debug(f"Deleted bucket directory: {bucket_path}")
            
            return True
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise self._create_provider_error(
                message=f"Failed to delete bucket directory {bucket}: {str(e)}",
                operation="delete_bucket",
                cause=e
            )
        
    async def bucket_exists(self, bucket: Optional[str] = None) -> bool:
        """Check if a bucket (directory) exists.
        
        Args:
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if bucket exists
            
        Raises:
            ProviderError: If check fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="exists"
            )
            
        bucket = bucket or "default"
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            return os.path.exists(bucket_path) and os.path.isdir(bucket_path)
            
        except Exception as e:
            raise self._create_provider_error(
                message=f"Failed to check if bucket {bucket} exists: {str(e)}",
                operation="exists",
                cause=e
            )
        
    async def upload_file(self, file_path: str, object_key: str, bucket: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload a file to local storage.
        
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
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="upload_file"
            )
            
        bucket = bucket or "default"
        file_path = os.path.abspath(file_path)
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if source file exists
            if not os.path.exists(file_path):
                raise self._create_provider_error(
                    message=f"Source file {file_path} does not exist",
                    operation="upload_file"
                )
                
            # Create directory for the object if it doesn't exist
            os.makedirs(os.path.dirname(object_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, object_path)
            
            # Set permissions if specified
            if self.settings.permissions is not None:
                os.chmod(object_path, self.settings.permissions)
                
            # Get file metadata
            stat = os.stat(object_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate MD5 hash for etag
            md5_hash = hashlib.md5()
            with open(object_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            etag = md5_hash.hexdigest()
            
            # Determine content type
            content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            
            # Create metadata object
            file_metadata = FileMetadata(
                key=object_key,
                size=size,
                etag=etag,
                content_type=content_type,
                modified=modified,
                metadata=metadata or {}
            )
            
            logger.debug(f"Uploaded file to {object_key} in bucket {bucket}")
            
            return file_metadata
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise self._create_provider_error(
                message=f"Failed to upload file to {object_key}: {str(e)}",
                operation="upload_file",
                cause=e
            )
        
    async def upload_data(self, data: Union[bytes, BinaryIO], object_key: str, bucket: Optional[str] = None,
                        content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Upload binary data to local storage.
        
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
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="upload_data"
            )
            
        bucket = bucket or "default"
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Create directory for the object if it doesn't exist
            os.makedirs(os.path.dirname(object_path), exist_ok=True)
            
            # Write data to file
            md5_hash = hashlib.md5()
            
            with open(object_path, 'wb') as f:
                if isinstance(data, bytes):
                    f.write(data)
                    md5_hash.update(data)
                else:
                    # File-like object
                    chunk = data.read(4096)
                    while chunk:
                        f.write(chunk)
                        md5_hash.update(chunk)
                        chunk = data.read(4096)
            
            # Set permissions if specified
            if self.settings.permissions is not None:
                os.chmod(object_path, self.settings.permissions)
                
            # Get file metadata
            stat = os.stat(object_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate etag
            etag = md5_hash.hexdigest()
            
            # Determine content type
            if content_type is None:
                # Default behavior: auto-detect content type
                content_type = mimetypes.guess_type(object_key)[0] or "application/octet-stream"
            else:
                content_type = content_type or "application/octet-stream"
                
            # Create metadata object
            file_metadata = FileMetadata(
                key=object_key,
                size=size,
                etag=etag,
                content_type=content_type,
                modified=modified,
                metadata=metadata or {}
            )
            
            logger.debug(f"Uploaded data to {object_key} in bucket {bucket}")
            
            return file_metadata
            
        except Exception as e:
            raise self._create_provider_error(
                message=f"Failed to upload data to {object_key}: {str(e)}",
                operation="upload_data",
                cause=e
            )
        
    async def download_file(self, object_key: str, file_path: str, bucket: Optional[str] = None) -> FileMetadata:
        """Download a file from local storage.
        
        Args:
            object_key: Storage object key/path
            file_path: Path to save file
            bucket: Bucket name (default from settings if None)
            
        Returns:
            File metadata
            
        Raises:
            ProviderError: If download fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="download_file"
            )
            
        bucket = bucket or "default"
        object_path = self._get_object_path(object_key, bucket)
        file_path = os.path.abspath(file_path)
        
        try:
            # Check if source object exists
            if not os.path.exists(object_path):
                raise self._create_provider_error(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    operation="download_file"
                )
                
            # Create directory for the destination file if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(object_path, file_path)
            
            # Get file metadata
            return await self.get_metadata(object_key, bucket)
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise self._create_provider_error(
                message=f"Failed to download object {object_key}: {str(e)}",
                operation="download_file",
                cause=e
            )
        
    async def download_data(self, object_key: str, bucket: Optional[str] = None) -> Tuple[bytes, FileMetadata]:
        """Download binary data from local storage.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            Tuple of (data bytes, file metadata)
            
        Raises:
            ProviderError: If download fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="download_data"
            )
            
        bucket = bucket or "default"
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists
            if not os.path.exists(object_path):
                raise self._create_provider_error(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    operation="download_data"
                )
                
            # Read file data
            with open(object_path, 'rb') as f:
                data = f.read()
                
            # Get file metadata
            metadata = await self.get_metadata(object_key, bucket)
            
            return data, metadata
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise self._create_provider_error(
                message=f"Failed to download object data {object_key}: {str(e)}",
                operation="download_data",
                cause=e
            )
        
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
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="get_metadata"
            )
            
        bucket = bucket or "default"
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists
            if not os.path.exists(object_path):
                raise self._create_provider_error(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    operation="get_metadata"
                )
                
            # Get file stats
            stat = os.stat(object_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Calculate MD5 hash for etag
            md5_hash = hashlib.md5()
            with open(object_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            etag = md5_hash.hexdigest()
            
            # Determine content type
            content_type = mimetypes.guess_type(object_path)[0] or "application/octet-stream"
            
            # Create metadata object
            file_metadata = FileMetadata(
                key=object_key,
                size=size,
                etag=etag,
                content_type=content_type,
                modified=modified,
                metadata={}  # Local storage doesn't support object metadata
            )
            
            return file_metadata
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise self._create_provider_error(
                message=f"Failed to get metadata for {object_key}: {str(e)}",
                operation="get_metadata",
                cause=e
            )
        
    async def delete_object(self, object_key: str, bucket: Optional[str] = None) -> bool:
        """Delete an object from local storage.
        
        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)
            
        Returns:
            True if object was deleted successfully
            
        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="delete_file"
            )
            
        bucket = bucket or "default"
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists
            if not os.path.exists(object_path):
                return True
                
            # Delete object
            if os.path.isdir(object_path):
                shutil.rmtree(object_path)
            else:
                os.remove(object_path)
                
            logger.debug(f"Deleted object {object_key} from bucket {bucket}")
            
            return True
            
        except Exception as e:
            raise self._create_provider_error(
                message=f"Failed to delete object {object_key}: {str(e)}",
                operation="delete_file",
                cause=e
            )
        
    async def list_objects(self, prefix: Optional[str] = None, bucket: Optional[str] = None) -> List[FileMetadata]:
        """List objects in local storage.
        
        Args:
            prefix: Optional prefix to filter objects
            bucket: Bucket name (default from settings if None)
            
        Returns:
            List of file metadata
            
        Raises:
            ProviderError: If listing fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="list_files"
            )
            
        bucket = bucket or "default"
        bucket_path = self._get_bucket_path(bucket)
        
        try:
            # Check if bucket exists
            if not os.path.exists(bucket_path):
                return []
                
            results = []
            prefix_path = ''
            
            if prefix:
                # Normalize prefix path
                prefix = prefix.replace('/', os.path.sep)
                prefix_path = os.path.join(bucket_path, prefix)
                
                # If prefix is a directory, list contents
                if os.path.isdir(prefix_path):
                    base_dir = prefix_path
                    prefix = ''
                else:
                    base_dir = bucket_path
            else:
                base_dir = bucket_path
                prefix = ''
                
            # Walk directory tree
            for root, dirs, files in os.walk(base_dir):
                # Skip files not under prefix
                if prefix and not root.startswith(prefix_path):
                    continue
                    
                # Process files
                for file in files:
                    file_path = os.path.join(root, file)
                    # Generate object key relative to bucket path
                    rel_path = os.path.relpath(file_path, bucket_path)
                    object_key = rel_path.replace(os.path.sep, '/')
                    
                    # Get file metadata
                    try:
                        metadata = await self.get_metadata(object_key, bucket)
                        results.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error getting metadata for {object_key}: {str(e)}")
            
            return results
            
        except Exception as e:
            raise self._create_provider_error(
                message=f"Failed to list objects in bucket {bucket}: {str(e)}",
                operation="list_files",
                cause=e
            )
        
    async def generate_presigned_url(self, object_key: str, expiration: int = 3600, bucket: Optional[str] = None,
                                  operation: str = "get") -> str:
        """Generate a presigned URL for an object.
        
        For local storage, this returns a file:// URL to the object.
        
        Args:
            object_key: Storage object key/path
            expiration: Expiration time in seconds (ignored for local storage)
            bucket: Bucket name (default from settings if None)
            operation: Operation type ('get', 'put', etc.) (ignored for local storage)
            
        Returns:
            File URL
            
        Raises:
            ProviderError: If URL generation fails
        """
        if not self._initialized:
            raise self._create_provider_error(
                message="Provider not initialized",
                operation="generate_presigned_url"
            )
            
        bucket = bucket or "default"
        object_path = self._get_object_path(object_key, bucket)
        
        try:
            # Check if object exists for GET operation
            if operation.lower() == "get" and not os.path.exists(object_path):
                raise self._create_provider_error(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    operation="generate_presigned_url"
                )
                
            # Create file:// URL
            url = pathlib.Path(object_path).as_uri()
            
            return url
            
        except Exception as e:
            if isinstance(e, ProviderError):
                raise e
                
            raise self._create_provider_error(
                message=f"Failed to generate URL for {object_key}: {str(e)}",
                operation="generate_presigned_url",
                cause=e
            )
        
    async def check_connection(self) -> bool:
        """Check if local storage is accessible.
        
        Returns:
            True if local storage is accessible
        """
        try:
            if not self._initialized:
                await self.initialize()
                
            # Check if base directory exists and is writable
            base_path = self.settings.base_path
            return os.path.exists(base_path) and os.access(base_path, os.W_OK)
        except Exception as e:
            logger.error(f"Local storage connection check failed: {str(e)}")
            return False 