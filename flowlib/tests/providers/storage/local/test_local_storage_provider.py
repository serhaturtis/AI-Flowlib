"""Tests for local storage provider implementation."""

import pytest
import pytest_asyncio
import os
import tempfile
import shutil
import io
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import patch, Mock

from flowlib.providers.storage.local.provider import (
    LocalStorageProvider,
    LocalStorageProviderSettings
)
from flowlib.providers.storage.base import FileMetadata
from flowlib.core.errors.errors import ProviderError


class TestLocalStorageProviderSettings:
    """Test LocalStorageProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal required data."""
        settings = LocalStorageProviderSettings(base_path="/tmp/test")
        
        assert settings.base_path.endswith(os.path.sep)  # Should normalize with trailing separator
        assert settings.create_dirs is True
        assert settings.use_relative_paths is True
        assert settings.permissions is None
        # Inherited provider settings
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
        assert settings.api_key is None
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        settings = LocalStorageProviderSettings(
            base_path="/tmp/storage",
            create_dirs=False,
            use_relative_paths=False,
            permissions=0o644,
            timeout_seconds=30.0
        )
        
        assert settings.base_path == "/tmp/storage" + os.path.sep
        assert settings.create_dirs is False
        assert settings.use_relative_paths is False
        assert settings.permissions == 0o644
        assert settings.timeout_seconds == 30.0
    
    def test_base_path_normalization(self):
        """Test base_path normalization and validation."""
        # Test relative path normalization
        settings = LocalStorageProviderSettings(base_path="./test/path")
        assert os.path.isabs(settings.base_path)
        assert settings.base_path.endswith(os.path.sep)
        
        # Test path with extra separators
        settings = LocalStorageProviderSettings(base_path="/tmp//test///path//")
        assert settings.base_path == "/tmp/test/path" + os.path.sep
        
        # Test Windows-style path on Unix (should be normalized)
        settings = LocalStorageProviderSettings(base_path="/tmp\\test\\path")
        expected = os.path.normpath("/tmp\\test\\path") + os.path.sep
        assert settings.base_path == expected
    
    def test_settings_inheritance(self):
        """Test that LocalStorageProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        settings = LocalStorageProviderSettings(base_path="/tmp/test")
        assert isinstance(settings, ProviderSettings)


class TestLocalStorageProvider:
    """Test LocalStorageProvider implementation."""
    
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    
    @pytest_asyncio.fixture
    def provider_settings(self, temp_dir):
        """Create test provider settings."""
        return LocalStorageProviderSettings(
            base_path=temp_dir,
            create_dirs=True,
            permissions=0o644
        )
    
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return LocalStorageProvider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "local-storage"
        assert provider.provider_type == "storage"
        assert provider.settings == provider_settings
        assert not provider._initialized
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base Provider class."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
        # LocalStorageProvider doesn't inherit from StorageProvider directly but implements its interface
    
    def test_provider_default_settings(self, temp_dir):
        """Test provider with default settings."""
        # Should raise ValidationError because base_path is required
        with pytest.raises(Exception):  # Pydantic ValidationError
            LocalStorageProvider()
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider, temp_dir):
        """Test provider initialization and shutdown lifecycle."""
        # Initially not initialized
        assert not provider._initialized
        
        # Initialize
        await provider.initialize()
        assert provider._initialized
        # Base directory should exist, bucket is created separately
        assert os.path.exists(temp_dir)
        
        # Shutdown
        await provider.shutdown()
        assert not provider._initialized
    
    @pytest.mark.asyncio
    async def test_initialization_creates_base_directory(self, temp_dir):
        """Test initialization creates base directory if it doesn't exist."""
        # Remove the temp directory
        shutil.rmtree(temp_dir)
        assert not os.path.exists(temp_dir)
        
        settings = LocalStorageProviderSettings(
            base_path=temp_dir,
            create_dirs=True
        )
        provider = LocalStorageProvider(settings=settings)
        
        await provider.initialize()
        assert os.path.exists(temp_dir)
    
    @pytest.mark.asyncio
    async def test_initialization_fails_with_unwritable_directory(self, temp_dir):
        """Test initialization fails with unwritable directory."""
        # Make directory read-only
        os.chmod(temp_dir, 0o444)
        
        try:
            settings = LocalStorageProviderSettings(base_path=temp_dir)
            provider = LocalStorageProvider(settings=settings)
            
            with pytest.raises(ProviderError, match="not writable"):
                await provider.initialize()
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_dir, 0o755)
    
    @pytest.mark.asyncio
    async def test_initialization_skip_directory_creation(self, temp_dir):
        """Test initialization with create_dirs=False."""
        non_existent_dir = os.path.join(temp_dir, "nonexistent")
        
        settings = LocalStorageProviderSettings(
            base_path=non_existent_dir,
            create_dirs=False
        )
        provider = LocalStorageProvider(settings=settings)
        
        # Should fail because directory doesn't exist
        with pytest.raises(ProviderError):
            await provider.initialize()


class TestLocalStorageProviderBucketOperations:
    """Test bucket operations."""
    
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest_asyncio.fixture
    async def provider(self, temp_dir):
        """Create and initialize test provider."""
        settings = LocalStorageProviderSettings(base_path=temp_dir)
        provider = LocalStorageProvider(settings=settings)
        await provider.initialize()
        return provider
    
    @pytest.mark.asyncio
    async def test_create_bucket_success(self, provider, temp_dir):
        """Test successful bucket creation."""
        bucket_name = "new-bucket"
        result = await provider.create_bucket(bucket_name)
        
        assert result is True
        bucket_path = os.path.join(temp_dir, bucket_name)
        assert os.path.exists(bucket_path)
        assert os.path.isdir(bucket_path)
    
    @pytest.mark.asyncio
    async def test_create_bucket_already_exists(self, provider, temp_dir):
        """Test creating bucket that already exists."""
        bucket_name = "existing-bucket"
        bucket_path = os.path.join(temp_dir, bucket_name)
        os.makedirs(bucket_path)
        
        result = await provider.create_bucket(bucket_name)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_create_bucket_default(self, provider, temp_dir):
        """Test creating default bucket."""
        result = await provider.create_bucket()
        
        assert result is True
        bucket_path = os.path.join(temp_dir, "default")
        assert os.path.exists(bucket_path)
    
    @pytest.mark.asyncio
    async def test_bucket_exists_true(self, provider, temp_dir):
        """Test bucket existence check for existing bucket."""
        bucket_name = "existing-bucket"
        bucket_path = os.path.join(temp_dir, bucket_name)
        os.makedirs(bucket_path)
        
        exists = await provider.bucket_exists(bucket_name)
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_bucket_exists_false(self, provider):
        """Test bucket existence check for non-existent bucket."""
        exists = await provider.bucket_exists("nonexistent-bucket")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_bucket_exists_default(self, provider, temp_dir):
        """Test bucket existence check for default bucket."""
        bucket_path = os.path.join(temp_dir, "default")
        os.makedirs(bucket_path)
        
        exists = await provider.bucket_exists()
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_delete_bucket_success(self, provider, temp_dir):
        """Test successful bucket deletion."""
        bucket_name = "delete-bucket"
        bucket_path = os.path.join(temp_dir, bucket_name)
        os.makedirs(bucket_path)
        
        result = await provider.delete_bucket(bucket_name)
        
        assert result is True
        assert not os.path.exists(bucket_path)
    
    @pytest.mark.asyncio
    async def test_delete_bucket_not_empty_without_force(self, provider, temp_dir):
        """Test deleting non-empty bucket without force flag."""
        bucket_name = "nonempty-bucket"
        bucket_path = os.path.join(temp_dir, bucket_name)
        os.makedirs(bucket_path)
        
        # Create a file in the bucket
        file_path = os.path.join(bucket_path, "test.txt")
        with open(file_path, 'w') as f:
            f.write("test content")
        
        with pytest.raises(ProviderError, match="is not empty"):
            await provider.delete_bucket(bucket_name, force=False)
    
    @pytest.mark.asyncio
    async def test_delete_bucket_not_empty_with_force(self, provider, temp_dir):
        """Test deleting non-empty bucket with force flag."""
        bucket_name = "nonempty-bucket"
        bucket_path = os.path.join(temp_dir, bucket_name)
        os.makedirs(bucket_path)
        
        # Create a file in the bucket
        file_path = os.path.join(bucket_path, "test.txt")
        with open(file_path, 'w') as f:
            f.write("test content")
        
        result = await provider.delete_bucket(bucket_name, force=True)
        
        assert result is True
        assert not os.path.exists(bucket_path)
    
    @pytest.mark.asyncio
    async def test_delete_bucket_nonexistent(self, provider):
        """Test deleting non-existent bucket."""
        result = await provider.delete_bucket("nonexistent-bucket")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_operations_require_initialization(self, temp_dir):
        """Test that bucket operations require initialization."""
        settings = LocalStorageProviderSettings(base_path=temp_dir)
        provider = LocalStorageProvider(settings=settings)
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.create_bucket("test")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.bucket_exists("test")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.delete_bucket("test")


class TestLocalStorageProviderFileOperations:
    """Test file upload and download operations."""
    
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    
    @pytest_asyncio.fixture
    async def provider(self, temp_dir):
        """Create and initialize test provider."""
        settings = LocalStorageProviderSettings(
            base_path=temp_dir,
            permissions=0o644
        )
        provider = LocalStorageProvider(settings=settings)
        await provider.initialize()
        await provider.create_bucket()
        return provider
    
    
    @pytest_asyncio.fixture
    def test_file(self, temp_dir):
        """Create a test file."""
        file_path = os.path.join(temp_dir, "source_test.txt")
        with open(file_path, 'w') as f:
            f.write("Hello, World!")
        return file_path
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, provider, test_file):
        """Test successful file upload."""
        object_key = "uploads/test.txt"
        metadata = {"author": "test", "version": "1.0"}
        
        result = await provider.upload_file(test_file, object_key, metadata=metadata)
        
        assert isinstance(result, FileMetadata)
        assert result.key == object_key
        assert result.size == 13  # Length of "Hello, World!"
        assert result.content_type == "text/plain"
        assert result.metadata == metadata
        assert result.etag is not None
        assert isinstance(result.modified, datetime)
        
        # Verify file was copied to the right location
        expected_path = os.path.join(provider._get_bucket_path(), object_key)
        assert os.path.exists(expected_path)
        
        with open(expected_path, 'r') as f:
            assert f.read() == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_upload_file_with_default_bucket(self, provider, test_file):
        """Test file upload with default bucket."""
        object_key = "test.txt"
        
        result = await provider.upload_file(test_file, object_key)
        
        assert result.key == object_key
        expected_path = os.path.join(provider._get_bucket_path(), object_key)
        assert os.path.exists(expected_path)
    
    @pytest.mark.asyncio
    async def test_upload_file_creates_subdirectories(self, provider, test_file):
        """Test that file upload creates necessary subdirectories."""
        object_key = "deep/nested/path/test.txt"
        
        result = await provider.upload_file(test_file, object_key)
        
        assert result.key == object_key
        expected_path = os.path.join(provider._get_bucket_path(), object_key)
        assert os.path.exists(expected_path)
        assert os.path.exists(os.path.dirname(expected_path))
    
    @pytest.mark.asyncio
    async def test_upload_file_nonexistent_source(self, provider):
        """Test uploading non-existent source file."""
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.upload_file("/nonexistent/file.txt", "test.txt")
    
    @pytest.mark.asyncio
    async def test_upload_file_with_permissions(self, provider, test_file):
        """Test file upload with custom permissions."""
        object_key = "permissions_test.txt"
        
        await provider.upload_file(test_file, object_key)
        
        expected_path = os.path.join(provider._get_bucket_path(), object_key)
        file_mode = os.stat(expected_path).st_mode & 0o777
        assert file_mode == 0o644
    
    @pytest.mark.asyncio
    async def test_upload_data_bytes(self, provider):
        """Test uploading binary data."""
        data = b"Binary test data \x00\x01\x02"
        object_key = "binary/test.bin"
        content_type = "application/octet-stream"
        metadata = {"type": "binary"}
        
        result = await provider.upload_data(data, object_key, content_type=content_type, metadata=metadata)
        
        assert result.key == object_key
        assert result.size == len(data)
        assert result.content_type == content_type
        assert result.metadata == metadata
        
        # Verify data was written correctly
        expected_path = os.path.join(provider._get_bucket_path(), object_key)
        with open(expected_path, 'rb') as f:
            assert f.read() == data
    
    @pytest.mark.asyncio
    async def test_upload_data_file_like_object(self, provider):
        """Test uploading data from file-like object."""
        data = b"File-like object data"
        file_obj = io.BytesIO(data)
        object_key = "fileobj/test.bin"
        
        result = await provider.upload_data(file_obj, object_key)
        
        assert result.size == len(data)
        
        # Verify data was written correctly
        expected_path = os.path.join(provider._get_bucket_path(), object_key)
        with open(expected_path, 'rb') as f:
            assert f.read() == data
    
    @pytest.mark.asyncio
    async def test_upload_data_auto_content_type(self, provider):
        """Test automatic content type detection."""
        data = b"JSON data"
        object_key = "data.json"
        
        result = await provider.upload_data(data, object_key)
        
        assert result.content_type == "application/json"
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, provider, test_file, temp_dir):
        """Test successful file download."""
        object_key = "download_test.txt"
        download_path = os.path.join(temp_dir, "downloaded.txt")
        
        # First upload the file
        await provider.upload_file(test_file, object_key)
        
        # Then download it
        result = await provider.download_file(object_key, download_path)
        
        assert isinstance(result, FileMetadata)
        assert result.key == object_key
        assert os.path.exists(download_path)
        
        with open(download_path, 'r') as f:
            assert f.read() == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_download_file_creates_directories(self, provider, test_file, temp_dir):
        """Test that file download creates necessary directories."""
        object_key = "test.txt"
        download_path = os.path.join(temp_dir, "deep", "nested", "downloaded.txt")
        
        # First upload the file
        await provider.upload_file(test_file, object_key)
        
        # Then download it
        await provider.download_file(object_key, download_path)
        
        assert os.path.exists(download_path)
        assert os.path.exists(os.path.dirname(download_path))
    
    @pytest.mark.asyncio
    async def test_download_file_nonexistent(self, provider, temp_dir):
        """Test downloading non-existent file."""
        download_path = os.path.join(temp_dir, "downloaded.txt")
        
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.download_file("nonexistent.txt", download_path)
    
    @pytest.mark.asyncio
    async def test_download_data_success(self, provider):
        """Test successful data download."""
        data = b"Download test data"
        object_key = "download_data_test.bin"
        
        # First upload the data
        await provider.upload_data(data, object_key)
        
        # Then download it
        downloaded_data, metadata = await provider.download_data(object_key)
        
        assert downloaded_data == data
        assert isinstance(metadata, FileMetadata)
        assert metadata.key == object_key
        assert metadata.size == len(data)
    
    @pytest.mark.asyncio
    async def test_download_data_nonexistent(self, provider):
        """Test downloading data for non-existent object."""
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.download_data("nonexistent.bin")


class TestLocalStorageProviderMetadataOperations:
    """Test metadata and object operations."""
    
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    
    @pytest_asyncio.fixture
    async def provider(self, temp_dir):
        """Create and initialize test provider."""
        settings = LocalStorageProviderSettings(base_path=temp_dir)
        provider = LocalStorageProvider(settings=settings)
        await provider.initialize()
        await provider.create_bucket()
        return provider
    
    @pytest.mark.asyncio
    async def test_get_metadata_success(self, provider):
        """Test successful metadata retrieval."""
        data = b"Metadata test data"
        object_key = "metadata_test.txt"
        
        # Upload data first
        await provider.upload_data(data, object_key)
        
        # Get metadata
        metadata = await provider.get_metadata(object_key)
        
        assert isinstance(metadata, FileMetadata)
        assert metadata.key == object_key
        assert metadata.size == len(data)
        assert metadata.content_type == "text/plain"
        assert metadata.etag is not None
        assert isinstance(metadata.modified, datetime)
        assert metadata.metadata == {}  # Local storage doesn't support object metadata
    
    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self, provider):
        """Test metadata retrieval for non-existent object."""
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.get_metadata("nonexistent.txt")
    
    @pytest.mark.asyncio
    async def test_delete_object_success(self, provider):
        """Test successful object deletion."""
        data = b"Delete test data"
        object_key = "delete_test.txt"
        
        # Upload data first
        await provider.upload_data(data, object_key)
        object_path = os.path.join(provider._get_bucket_path(), object_key)
        assert os.path.exists(object_path)
        
        # Delete object
        result = await provider.delete_object(object_key)
        
        assert result is True
        assert not os.path.exists(object_path)
    
    @pytest.mark.asyncio
    async def test_delete_object_nonexistent(self, provider):
        """Test deleting non-existent object."""
        result = await provider.delete_object("nonexistent.txt")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_object_directory(self, provider, temp_dir):
        """Test deleting a directory object."""
        object_key = "test_dir"
        object_path = os.path.join(provider._get_bucket_path(), object_key)
        os.makedirs(object_path)
        
        # Create a file in the directory
        file_path = os.path.join(object_path, "test.txt")
        with open(file_path, 'w') as f:
            f.write("test")
        
        result = await provider.delete_object(object_key)
        
        assert result is True
        assert not os.path.exists(object_path)
    
    @pytest.mark.asyncio
    async def test_list_objects_empty_bucket(self, provider):
        """Test listing objects in empty bucket."""
        objects = await provider.list_objects()
        assert objects == []
    
    @pytest.mark.asyncio
    async def test_list_objects_with_files(self, provider):
        """Test listing objects with files."""
        # Upload some test files
        test_files = [
            ("file1.txt", b"content1"),
            ("dir/file2.txt", b"content2"),
            ("dir/subdir/file3.txt", b"content3"),
            ("another_file.bin", b"binary_content")
        ]
        
        for key, data in test_files:
            await provider.upload_data(data, key)
        
        # List all objects
        objects = await provider.list_objects()
        
        assert len(objects) == 4
        object_keys = [obj.key for obj in objects]
        expected_keys = [key for key, _ in test_files]
        
        for expected_key in expected_keys:
            assert expected_key in object_keys
    
    @pytest.mark.asyncio
    async def test_list_objects_with_prefix(self, provider):
        """Test listing objects with prefix filter."""
        # Upload test files with different prefixes
        test_files = [
            ("docs/readme.txt", b"readme"),
            ("docs/manual.pdf", b"manual"),
            ("images/photo.jpg", b"photo"),
            ("config.json", b"config")
        ]
        
        for key, data in test_files:
            await provider.upload_data(data, key)
        
        # List objects with "docs/" prefix
        objects = await provider.list_objects(prefix="docs/")
        
        assert len(objects) == 2
        object_keys = [obj.key for obj in objects]
        assert "docs/readme.txt" in object_keys
        assert "docs/manual.pdf" in object_keys
    
    @pytest.mark.asyncio
    async def test_list_objects_nonexistent_bucket(self, provider):
        """Test listing objects in non-existent bucket."""
        objects = await provider.list_objects(bucket="nonexistent-bucket")
        assert objects == []


class TestLocalStorageProviderAdvancedFeatures:
    """Test advanced provider features."""
    
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    
    @pytest_asyncio.fixture
    async def provider(self, temp_dir):
        """Create and initialize test provider."""
        settings = LocalStorageProviderSettings(base_path=temp_dir)
        provider = LocalStorageProvider(settings=settings)
        await provider.initialize()
        await provider.create_bucket()
        return provider
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_get(self, provider):
        """Test generating presigned URL for GET operation."""
        data = b"URL test data"
        object_key = "url_test.txt"
        
        # Upload data first
        await provider.upload_data(data, object_key)
        
        # Generate URL
        url = await provider.generate_presigned_url(object_key, operation="get")
        
        assert url.startswith("file://")
        assert object_key in url
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_put(self, provider):
        """Test generating presigned URL for PUT operation."""
        object_key = "put_test.txt"
        
        # Generate URL for PUT (object doesn't need to exist)
        url = await provider.generate_presigned_url(object_key, operation="put")
        
        assert url.startswith("file://")
        assert object_key in url
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_nonexistent_get(self, provider):
        """Test generating presigned URL for non-existent object with GET."""
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.generate_presigned_url("nonexistent.txt", operation="get")
    
    @pytest.mark.asyncio
    async def test_check_connection_success(self, provider):
        """Test successful connection check."""
        result = await provider.check_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_connection_auto_initialize(self, temp_dir):
        """Test connection check with auto-initialization."""
        settings = LocalStorageProviderSettings(base_path=temp_dir)
        provider = LocalStorageProvider(settings=settings)
        
        # Should auto-initialize and succeed
        result = await provider.check_connection()
        assert result is True
        assert provider._initialized
    
    @pytest.mark.asyncio
    async def test_check_connection_unwritable_directory(self, temp_dir):
        """Test connection check with unwritable directory."""
        # Make directory read-only
        os.chmod(temp_dir, 0o444)
        
        try:
            settings = LocalStorageProviderSettings(base_path=temp_dir)
            provider = LocalStorageProvider(settings=settings)
            
            result = await provider.check_connection()
            assert result is False
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_dir, 0o755)


class TestLocalStorageProviderErrorHandling:
    """Test error handling and edge cases."""
    
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_operations_without_initialization(self, temp_dir):
        """Test that operations fail without initialization."""
        settings = LocalStorageProviderSettings(base_path=temp_dir)
        provider = LocalStorageProvider(settings=settings)
        
        # All operations should fail
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.upload_data(b"test", "test.txt")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.download_data("test.txt")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.get_metadata("test.txt")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.delete_object("test.txt")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.list_objects()
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.generate_presigned_url("test.txt")
    
    @pytest.mark.asyncio
    async def test_provider_decorator_registration(self):
        """Test that provider is properly registered with decorator."""
        # This tests the @provider decorator functionality
        assert hasattr(LocalStorageProvider, '__provider_name__')
        assert hasattr(LocalStorageProvider, '__provider_type__')
        assert LocalStorageProvider.__provider_name__ == "local-storage"
        assert LocalStorageProvider.__provider_type__ == "storage"
    
    @pytest.mark.asyncio
    async def test_permissions_error_handling(self, temp_dir):
        """Test handling of permission errors."""
        settings = LocalStorageProviderSettings(
            base_path=temp_dir,
            permissions=0o644
        )
        provider = LocalStorageProvider(settings=settings)
        await provider.initialize()
        await provider.create_bucket()
        
        # Upload a file
        await provider.upload_data(b"test", "test.txt")
        
        # Make the bucket read-only
        bucket_path = provider._get_bucket_path()
        os.chmod(bucket_path, 0o444)
        
        try:
            # Should fail to upload new files
            with pytest.raises(ProviderError):
                await provider.upload_data(b"test2", "test2.txt")
        finally:
            # Restore permissions for cleanup
            os.chmod(bucket_path, 0o755)