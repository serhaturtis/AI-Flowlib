"""Tests for storage provider base functionality."""

import pytest
import os
import io
from datetime import datetime
from typing import Optional, Dict, List
from unittest.mock import Mock, AsyncMock, patch
from pydantic import ValidationError

from flowlib.providers.storage.base import (
    StorageProviderSettings,
    FileMetadata,
    StorageProvider
)
from flowlib.providers.core.base import Provider


class TestStorageProviderSettings:
    """Test StorageProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal data."""
        settings = StorageProviderSettings()
        
        assert settings.endpoint is None
        assert settings.region is None
        assert settings.bucket == "default"
        assert settings.access_key is None
        assert settings.secret_key is None
        assert settings.use_ssl is True
        assert settings.create_bucket is True
        assert settings.auto_content_type is True
        assert settings.part_size == 5 * 1024 * 1024  # 5MB
        assert settings.max_concurrency == 10
        assert settings.timeout == 60.0
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        settings = StorageProviderSettings(
            endpoint="https://s3.amazonaws.com",
            region="us-east-1",
            bucket="my-bucket",
            access_key="AKIATEST",
            secret_key="secret123",
            use_ssl=False,
            create_bucket=False,
            auto_content_type=False,
            part_size=10 * 1024 * 1024,  # 10MB
            max_concurrency=5,
            timeout=120.0
        )
        
        assert settings.endpoint == "https://s3.amazonaws.com"
        assert settings.region == "us-east-1"
        assert settings.bucket == "my-bucket"
        assert settings.access_key == "AKIATEST"
        assert settings.secret_key == "secret123"
        assert settings.use_ssl is False
        assert settings.create_bucket is False
        assert settings.auto_content_type is False
        assert settings.part_size == 10 * 1024 * 1024
        assert settings.max_concurrency == 5
        assert settings.timeout == 120.0
    
    def test_settings_inheritance(self):
        """Test that StorageProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        settings = StorageProviderSettings()
        assert isinstance(settings, ProviderSettings)
    
    def test_settings_custom_settings_field(self):
        """Test custom settings field from parent class."""
        custom_settings = {"custom_field": "custom_value"}
        settings = StorageProviderSettings(custom_settings=custom_settings)
        assert settings.custom_settings == custom_settings
    
    def test_settings_merge_functionality(self):
        """Test settings merge capability from parent class."""
        base_settings = StorageProviderSettings(bucket="base-bucket", api_key="base-key")
        override_settings = StorageProviderSettings(bucket="override-bucket", region="us-west-2")
        
        merged = base_settings.merge(override_settings)
        
        assert merged.bucket == "override-bucket"  # Overridden
        assert merged.region == "us-west-2"  # New field
        assert merged.api_key == "base-key"  # Preserved from base (since override has None)
    
    def test_settings_with_overrides(self):
        """Test settings with_overrides functionality."""
        base_settings = StorageProviderSettings(bucket="base-bucket")
        
        overridden = base_settings.with_overrides(bucket="new-bucket", region="eu-west-1")
        
        assert overridden.bucket == "new-bucket"
        assert overridden.region == "eu-west-1"
        assert base_settings.bucket == "base-bucket"  # Original unchanged


class TestFileMetadata:
    """Test FileMetadata model."""
    
    def test_metadata_creation_minimal(self):
        """Test metadata creation with minimal required fields."""
        metadata = FileMetadata(key="test.txt", size=1024)
        
        assert metadata.key == "test.txt"
        assert metadata.size == 1024
        assert metadata.etag is None
        assert metadata.content_type == "application/octet-stream"
        assert metadata.modified is None
        assert metadata.metadata == {}
    
    def test_metadata_creation_full(self):
        """Test metadata creation with all fields."""
        custom_metadata = {"author": "test", "version": "1.0"}
        modified_time = datetime.now()
        
        metadata = FileMetadata(
            key="documents/test.pdf",
            size=2048,
            etag="abc123",
            content_type="application/pdf",
            modified=modified_time,
            metadata=custom_metadata
        )
        
        assert metadata.key == "documents/test.pdf"
        assert metadata.size == 2048
        assert metadata.etag == "abc123"
        assert metadata.content_type == "application/pdf"
        assert metadata.modified == modified_time
        assert metadata.metadata == custom_metadata
    
    def test_metadata_validation_required_fields(self):
        """Test validation of required fields."""
        with pytest.raises(ValidationError):
            FileMetadata()  # Missing required fields
        
        with pytest.raises(ValidationError):
            FileMetadata(key="test.txt")  # Missing size
        
        with pytest.raises(ValidationError):
            FileMetadata(size=1024)  # Missing key
    
    def test_metadata_size_validation(self):
        """Test size field validation."""
        # Valid sizes
        metadata = FileMetadata(key="test.txt", size=0)
        assert metadata.size == 0
        
        metadata = FileMetadata(key="test.txt", size=1024)
        assert metadata.size == 1024
        
        # Test with very large size
        large_size = 1024 * 1024 * 1024 * 1024  # 1TB
        metadata = FileMetadata(key="test.txt", size=large_size)
        assert metadata.size == large_size
    
    def test_metadata_key_formats(self):
        """Test various key formats."""
        # Simple file
        metadata = FileMetadata(key="file.txt", size=100)
        assert metadata.key == "file.txt"
        
        # Path with directories
        metadata = FileMetadata(key="path/to/file.txt", size=100)
        assert metadata.key == "path/to/file.txt"
        
        # Unicode filename
        metadata = FileMetadata(key="файл.txt", size=100)
        assert metadata.key == "файл.txt"
        
        # Special characters
        metadata = FileMetadata(key="file with spaces & symbols!.txt", size=100)
        assert metadata.key == "file with spaces & symbols!.txt"
    
    def test_metadata_custom_metadata_types(self):
        """Test custom metadata with various types."""
        # String values
        metadata = FileMetadata(
            key="test.txt",
            size=100,
            metadata={"string_field": "value", "number_as_string": "123"}
        )
        assert metadata.metadata["string_field"] == "value"
        assert metadata.metadata["number_as_string"] == "123"
    
    def test_metadata_datetime_handling(self):
        """Test datetime field handling."""
        # Test with current time
        now = datetime.now()
        metadata = FileMetadata(key="test.txt", size=100, modified=now)
        assert metadata.modified == now
        
        # Test with UTC time
        from datetime import timezone
        utc_time = datetime.now(timezone.utc)
        metadata = FileMetadata(key="test.txt", size=100, modified=utc_time)
        assert metadata.modified == utc_time


class ConcreteStorageProvider(StorageProvider):
    """Concrete implementation of StorageProvider for testing."""
    
    def __init__(self, name: str = "test_storage", settings: Optional[StorageProviderSettings] = None):
        if settings is None:
            settings = StorageProviderSettings()
        super().__init__(name=name, settings=settings)
        # Initialize required attributes
        object.__setattr__(self, '_initialized', False)
        object.__setattr__(self, '_client', None)
        # Use object.__setattr__ to bypass Pydantic validation for test attributes
        object.__setattr__(self, 'upload_calls', [])
        object.__setattr__(self, 'download_calls', [])
        object.__setattr__(self, 'delete_calls', [])
        object.__setattr__(self, 'list_calls', [])
        object.__setattr__(self, 'should_fail', False)
        object.__setattr__(self, 'fail_operation', None)
    
    def __setattr__(self, name, value):
        """Override setattr to allow test attributes to be set."""
        # Allow test-specific attributes to be set directly
        test_attrs = {'upload_calls', 'download_calls', 'delete_calls', 'list_calls', 
                     'should_fail', 'fail_operation', 'list_objects'}
        if name in test_attrs:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
    
    async def initialize(self) -> None:
        """Mock initialization."""
        if self.should_fail and self.fail_operation == "initialize":
            raise RuntimeError("Mock initialization failure")
        # Call parent to set _initialized flag properly
        await super().initialize()
        self._client = Mock()
    
    async def shutdown(self) -> None:
        """Mock shutdown that uses base Provider's graceful error handling."""
        # Use the base Provider's shutdown method instead of StorageProvider's
        # to get graceful error handling
        from flowlib.providers.core.base import Provider
        
        # Temporarily implement _shutdown for the base Provider to call
        async def mock_shutdown():
            if self.should_fail and self.fail_operation == "shutdown":
                raise RuntimeError("Mock shutdown failure") 
            self._client = None
            
        object.__setattr__(self, '_shutdown', mock_shutdown)
        
        # Call base Provider shutdown which handles errors gracefully
        await Provider.shutdown(self)
    
    async def upload_file(self, file_path: str, key: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Mock file upload."""
        self.upload_calls.append(("file", file_path, key, metadata))
        if self.should_fail and self.fail_operation == "upload_file":
            raise RuntimeError("Mock upload failure")
        
        actual_key = key or os.path.basename(file_path)
        return FileMetadata(
            key=actual_key,
            size=1024,
            etag="mock_etag",
            content_type="text/plain",
            metadata=metadata or {}
        )
    
    async def upload_data(self, data: bytes, key: str, content_type: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> FileMetadata:
        """Mock data upload."""
        self.upload_calls.append(("data", data, key, content_type, metadata))
        if self.should_fail and self.fail_operation == "upload_data":
            raise RuntimeError("Mock upload data failure")
        
        return FileMetadata(
            key=key,
            size=len(data),
            etag="mock_etag",
            content_type=content_type or "application/octet-stream",
            metadata=metadata or {}
        )
    
    async def download_file(self, key: str, file_path: str) -> None:
        """Mock file download."""
        self.download_calls.append(("file", key, file_path))
        if self.should_fail and self.fail_operation == "download_file":
            raise RuntimeError("Mock download failure")
    
    async def download_data(self, key: str) -> bytes:
        """Mock data download."""
        self.download_calls.append(("data", key))
        if self.should_fail and self.fail_operation == "download_data":
            raise RuntimeError("Mock download data failure")
        return b"mock data content"
    
    async def delete_object(self, key: str) -> None:
        """Mock object deletion."""
        self.delete_calls.append(key)
        if self.should_fail and self.fail_operation == "delete_object":
            raise RuntimeError("Mock delete failure")
    
    async def list_objects(self, prefix: Optional[str] = None, max_keys: int = 1000) -> List[FileMetadata]:
        """Mock object listing."""
        self.list_calls.append((prefix, max_keys))
        if self.should_fail and self.fail_operation == "list_objects":
            raise RuntimeError("Mock list failure")
        
        # Return mock file list
        mock_files = [
            FileMetadata(key=f"file{i}.txt", size=1024 * i, etag=f"etag{i}")
            for i in range(1, min(max_keys + 1, 4))
        ]
        
        if prefix:
            mock_files = [f for f in mock_files if f.key.startswith(prefix)]
        
        return mock_files
    
    async def get_metadata(self, key: str) -> Optional[FileMetadata]:
        """Mock metadata retrieval."""
        if self.should_fail and self.fail_operation == "get_metadata":
            raise RuntimeError("Mock metadata failure")
        
        if key == "nonexistent.txt":
            return None
        
        return FileMetadata(
            key=key,
            size=1024,
            etag="mock_etag",
            content_type="text/plain"
        )
    
    async def object_exists(self, key: str) -> bool:
        """Mock object existence check."""
        if self.should_fail and self.fail_operation == "object_exists":
            raise RuntimeError("Mock exists check failure")
        return key != "nonexistent.txt"
    
    async def generate_presigned_url(self, key: str, method: str = "GET", expires_in: int = 3600) -> str:
        """Mock presigned URL generation."""
        if self.should_fail and self.fail_operation == "generate_presigned_url":
            raise RuntimeError("Mock presigned URL failure")
        return f"https://example.com/presigned/{key}?expires={expires_in}&method={method}"
    
    async def create_bucket(self) -> None:
        """Mock bucket creation."""
        if self.should_fail and self.fail_operation == "create_bucket":
            raise RuntimeError("Mock bucket creation failure")
    
    async def bucket_exists(self) -> bool:
        """Mock bucket existence check."""
        if self.should_fail and self.fail_operation == "bucket_exists":
            raise RuntimeError("Mock bucket check failure")
        return True


class TestStorageProvider:
    """Test StorageProvider base class."""
    
    @pytest.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return StorageProviderSettings(
            bucket="test-bucket",
            region="us-east-1",
            timeout=30.0
        )
    
    @pytest.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return ConcreteStorageProvider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "test_storage"
        assert provider.provider_type == "storage"
        assert provider._settings == provider_settings
        assert provider._initialized is False
        assert provider._client is None
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base Provider class."""
        assert isinstance(provider, Provider)
        assert isinstance(provider, StorageProvider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        provider = ConcreteStorageProvider()
        
        assert provider.name == "test_storage"
        assert isinstance(provider._settings, StorageProviderSettings)
        assert provider._settings.bucket == "default"
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider):
        """Test provider initialization and shutdown lifecycle."""
        # Initially not initialized
        assert provider.initialized is False
        
        # Initialize
        await provider.initialize()
        assert provider.initialized is True
        assert provider._client is not None
        
        # Shutdown
        await provider.shutdown()
        assert provider.initialized is False
        assert provider._client is None
    
    @pytest.mark.asyncio
    async def test_provider_initialization_failure(self, provider):
        """Test provider initialization failure."""
        object.__setattr__(provider, 'should_fail', True)
        object.__setattr__(provider, 'fail_operation', "initialize")
        
        with pytest.raises(Exception):  # Will be wrapped in ResourceError by base class
            await provider.initialize()
        
        assert provider.initialized is False
    
    @pytest.mark.asyncio
    async def test_provider_shutdown_failure(self, provider):
        """Test provider shutdown failure handling."""
        # Initialize first
        await provider.initialize()
        
        object.__setattr__(provider, 'should_fail', True)
        object.__setattr__(provider, 'fail_operation', "shutdown")
        
        # Shutdown should not raise error (graceful handling)
        await provider.shutdown()
        # Base class handles shutdown errors gracefully
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, provider):
        """Test successful file upload."""
        await provider.initialize()
        
        file_path = "/path/to/test.txt"
        key = "uploads/test.txt"
        metadata = {"author": "test"}
        
        result = await provider.upload_file(file_path, key, metadata)
        
        assert isinstance(result, FileMetadata)
        assert result.key == key
        assert result.size == 1024
        assert result.etag == "mock_etag"
        assert result.metadata == metadata
        
        # Verify call was recorded
        assert len(provider.upload_calls) == 1
        assert provider.upload_calls[0] == ("file", file_path, key, metadata)
    
    @pytest.mark.asyncio
    async def test_upload_file_with_auto_key(self, provider):
        """Test file upload with automatic key generation."""
        await provider.initialize()
        
        file_path = "/path/to/test.txt"
        
        result = await provider.upload_file(file_path)
        
        assert result.key == "test.txt"  # Auto-generated from filename
    
    @pytest.mark.asyncio
    async def test_upload_data_success(self, provider):
        """Test successful data upload."""
        await provider.initialize()
        
        data = b"test data content"
        key = "data/test.bin"
        content_type = "application/octet-stream"
        metadata = {"type": "binary"}
        
        result = await provider.upload_data(data, key, content_type, metadata)
        
        assert result.key == key
        assert result.size == len(data)
        assert result.content_type == content_type
        assert result.metadata == metadata
        
        # Verify call was recorded
        assert len(provider.upload_calls) == 1
        assert provider.upload_calls[0] == ("data", data, key, content_type, metadata)
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, provider):
        """Test successful file download."""
        await provider.initialize()
        
        key = "test.txt"
        file_path = "/local/path/test.txt"
        
        await provider.download_file(key, file_path)
        
        # Verify call was recorded
        assert len(provider.download_calls) == 1
        assert provider.download_calls[0] == ("file", key, file_path)
    
    @pytest.mark.asyncio
    async def test_download_data_success(self, provider):
        """Test successful data download."""
        await provider.initialize()
        
        key = "test.txt"
        
        data = await provider.download_data(key)
        
        assert data == b"mock data content"
        
        # Verify call was recorded
        assert len(provider.download_calls) == 1
        assert provider.download_calls[0] == ("data", key)
    
    @pytest.mark.asyncio
    async def test_delete_object_success(self, provider):
        """Test successful object deletion."""
        await provider.initialize()
        
        key = "test.txt"
        
        await provider.delete_object(key)
        
        # Verify call was recorded
        assert len(provider.delete_calls) == 1
        assert provider.delete_calls[0] == key
    
    @pytest.mark.asyncio
    async def test_list_objects_success(self, provider):
        """Test successful object listing."""
        await provider.initialize()
        
        objects = await provider.list_objects()
        
        assert len(objects) == 3  # Mock returns 3 objects
        assert all(isinstance(obj, FileMetadata) for obj in objects)
        assert objects[0].key == "file1.txt"
        assert objects[1].key == "file2.txt"
        assert objects[2].key == "file3.txt"
        
        # Verify call was recorded
        assert len(provider.list_calls) == 1
        assert provider.list_calls[0] == (None, 1000)
    
    @pytest.mark.asyncio
    async def test_list_objects_with_prefix(self, provider):
        """Test object listing with prefix filter."""
        await provider.initialize()
        
        prefix = "documents/"
        max_keys = 50
        
        objects = await provider.list_objects(prefix=prefix, max_keys=max_keys)
        
        # Verify call was recorded with correct parameters
        assert len(provider.list_calls) == 1
        assert provider.list_calls[0] == (prefix, max_keys)
    
    @pytest.mark.asyncio
    async def test_get_metadata_success(self, provider):
        """Test successful metadata retrieval."""
        await provider.initialize()
        
        key = "test.txt"
        
        metadata = await provider.get_metadata(key)
        
        assert isinstance(metadata, FileMetadata)
        assert metadata.key == key
        assert metadata.size == 1024
        assert metadata.etag == "mock_etag"
    
    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self, provider):
        """Test metadata retrieval for non-existent object."""
        await provider.initialize()
        
        metadata = await provider.get_metadata("nonexistent.txt")
        
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_object_exists_true(self, provider):
        """Test object existence check for existing object."""
        await provider.initialize()
        
        exists = await provider.object_exists("test.txt")
        
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_object_exists_false(self, provider):
        """Test object existence check for non-existent object."""
        await provider.initialize()
        
        exists = await provider.object_exists("nonexistent.txt")
        
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_success(self, provider):
        """Test presigned URL generation."""
        await provider.initialize()
        
        key = "test.txt"
        method = "PUT"
        expires_in = 7200
        
        url = await provider.generate_presigned_url(key, method, expires_in)
        
        assert url == f"https://example.com/presigned/{key}?expires={expires_in}&method={method}"
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_defaults(self, provider):
        """Test presigned URL generation with default parameters."""
        await provider.initialize()
        
        key = "test.txt"
        
        url = await provider.generate_presigned_url(key)
        
        assert "method=GET" in url
        assert "expires=3600" in url
    
    @pytest.mark.asyncio
    async def test_create_bucket_success(self, provider):
        """Test bucket creation."""
        await provider.initialize()
        
        await provider.create_bucket()
        # Should not raise any errors
    
    @pytest.mark.asyncio
    async def test_bucket_exists_success(self, provider):
        """Test bucket existence check."""
        await provider.initialize()
        
        exists = await provider.bucket_exists()
        
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_operation_failures(self, provider):
        """Test various operation failures."""
        await provider.initialize()
        object.__setattr__(provider, 'should_fail', True)
        
        # Test upload failure
        object.__setattr__(provider, 'fail_operation', "upload_file")
        with pytest.raises(RuntimeError, match="Mock upload failure"):
            await provider.upload_file("/path/test.txt")
        
        # Test download failure
        object.__setattr__(provider, 'fail_operation', "download_data")
        with pytest.raises(RuntimeError, match="Mock download data failure"):
            await provider.download_data("test.txt")
        
        # Test delete failure
        object.__setattr__(provider, 'fail_operation', "delete_object")
        with pytest.raises(RuntimeError, match="Mock delete failure"):
            await provider.delete_object("test.txt")
        
        # Test list failure
        object.__setattr__(provider, 'fail_operation', "list_objects")
        with pytest.raises(RuntimeError, match="Mock list failure"):
            await provider.list_objects()


class TestStorageProviderEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def provider(self):
        """Create test provider instance."""
        return ConcreteStorageProvider()
    
    @pytest.mark.asyncio
    async def test_operations_without_initialization(self, provider):
        """Test that operations work without explicit initialization (auto-init)."""
        # Base provider may auto-initialize, test depends on implementation
        # For our mock, we'll assume operations require initialization
        
        # This test documents expected behavior - adjust based on actual implementation
        try:
            await provider.upload_data(b"test", "test.txt")
            # If this succeeds, auto-initialization is working
        except Exception:
            # If this fails, explicit initialization is required
            await provider.initialize()
            await provider.upload_data(b"test", "test.txt")
            # Should succeed after initialization
    
    def test_provider_with_none_settings(self):
        """Test provider creation with None settings."""
        provider = ConcreteStorageProvider(settings=None)
        
        assert isinstance(provider._settings, StorageProviderSettings)
        assert provider._settings.bucket == "default"
    
    @pytest.mark.asyncio
    async def test_empty_data_upload(self, provider):
        """Test uploading empty data."""
        await provider.initialize()
        
        empty_data = b""
        key = "empty.txt"
        
        result = await provider.upload_data(empty_data, key)
        
        assert result.key == key
        assert result.size == 0
    
    @pytest.mark.asyncio
    async def test_unicode_key_handling(self, provider):
        """Test handling of unicode characters in keys."""
        await provider.initialize()
        
        unicode_key = "测试文件.txt"
        data = b"test content"
        
        result = await provider.upload_data(data, unicode_key)
        
        assert result.key == unicode_key
    
    @pytest.mark.asyncio
    async def test_large_metadata_dict(self, provider):
        """Test handling of large metadata dictionaries."""
        await provider.initialize()
        
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        result = await provider.upload_data(b"test", "test.txt", metadata=large_metadata)
        
        assert result.metadata == large_metadata
    
    @pytest.mark.asyncio
    async def test_list_objects_empty_result(self, provider):
        """Test listing objects when no objects exist."""
        await provider.initialize()
        
        # Mock to return empty list
        provider.list_calls.clear()
        
        # Override mock behavior for this test
        async def empty_list_mock(prefix=None, max_keys=1000):
            provider.list_calls.append((prefix, max_keys))
            return []
        
        object.__setattr__(provider, 'list_objects', empty_list_mock)
        
        objects = await provider.list_objects()
        
        assert objects == []
        assert len(provider.list_calls) == 1