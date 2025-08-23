"""Tests for S3 storage provider implementation."""

import pytest
import pytest_asyncio
import os
import tempfile
import shutil
import io
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from botocore.exceptions import ClientError

from flowlib.providers.storage.s3.provider import (
    S3Provider,
    S3ProviderSettings
)
from flowlib.providers.storage.base import FileMetadata
from flowlib.core.errors.errors import ProviderError


class TestS3ProviderSettings:
    """Test S3ProviderSettings model."""
    
    def test_settings_creation_minimal(self):
        """Test settings creation with minimal required data."""
        settings = S3ProviderSettings()
        
        assert settings.bucket == "default"
        assert settings.endpoint_url is None
        assert settings.region_name is None
        assert settings.access_key_id is None
        assert settings.secret_access_key is None
        assert settings.session_token is None
        assert settings.profile_name is None
        assert settings.path_style is False
        assert settings.signature_version == "s3v4"
        assert settings.validate_checksums is True
        assert settings.max_pool_connections == 10
        assert settings.multipart_threshold == 8 * 1024 * 1024
        assert settings.multipart_chunksize == 8 * 1024 * 1024
        assert settings.acl is None
        assert settings.cache_control is None
        # Inherited fields
        assert settings.timeout_seconds == 60.0
    
    def test_settings_creation_full(self):
        """Test settings creation with all fields."""
        settings = S3ProviderSettings(
            endpoint_url="https://s3.amazonaws.com",
            region_name="us-west-2",
            bucket="my-bucket",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            session_token="session-token",
            profile_name="my-profile",
            path_style=True,
            signature_version="s3v2",
            validate_checksums=False,
            max_pool_connections=20,
            multipart_threshold=16 * 1024 * 1024,
            multipart_chunksize=16 * 1024 * 1024,
            acl="public-read",
            cache_control="max-age=3600",
            timeout_seconds=30.0
        )
        
        assert settings.endpoint_url == "https://s3.amazonaws.com"
        assert settings.region_name == "us-west-2"
        assert settings.bucket == "my-bucket"
        assert settings.access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert settings.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert settings.session_token == "session-token"
        assert settings.profile_name == "my-profile"
        assert settings.path_style is True
        assert settings.signature_version == "s3v2"
        assert settings.validate_checksums is False
        assert settings.max_pool_connections == 20
        assert settings.multipart_threshold == 16 * 1024 * 1024
        assert settings.multipart_chunksize == 16 * 1024 * 1024
        assert settings.acl == "public-read"
        assert settings.cache_control == "max-age=3600"
        assert settings.timeout_seconds == 30.0
    
    def test_settings_inheritance(self):
        """Test that S3ProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        settings = S3ProviderSettings()
        assert isinstance(settings, ProviderSettings)


class TestS3Provider:
    """Test S3Provider implementation."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1",
            endpoint_url="http://localhost:9000",  # MinIO-style endpoint
            path_style=True  # Typically needed for MinIO
        )
    
    @pytest_asyncio.fixture
    def provider(self, provider_settings):
        """Create test provider instance."""
        return S3Provider(settings=provider_settings)
    
    def test_provider_initialization(self, provider, provider_settings):
        """Test provider initialization."""
        assert provider.name == "s3"
        assert provider.settings == provider_settings
        assert not provider._initialized
    
    def test_provider_inheritance(self, provider):
        """Test provider inheritance from base Provider class."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    def test_provider_default_settings(self):
        """Test provider with default settings."""
        provider = S3Provider()
        assert isinstance(provider.settings, S3ProviderSettings)
        assert provider.settings.bucket == "default"
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle_without_boto3(self, provider):
        """Test provider initialization fails without boto3."""
        with patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', False):
            with pytest.raises(ProviderError, match="boto3 package not installed"):
                await provider.initialize()


@pytest.mark.skipif(
    os.getenv("SKIP_S3_TESTS") == "true",
    reason="S3 tests require boto3 and moto dependencies"
)
class TestS3ProviderWithMocks:
    """Test S3Provider with mocked AWS services."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1"
        )
    
    @pytest_asyncio.fixture
    def mock_boto3_session(self):
        """Create mock boto3 session."""
        session = Mock()
        client = Mock()
        resource = Mock()
        
        # Setup session mocks
        session.client.return_value = client
        session.resource.return_value = resource
        
        # Setup basic client operations
        client.head_bucket.return_value = {}
        client.create_bucket.return_value = {}
        client.delete_bucket.return_value = {}
        client.list_buckets.return_value = {"Buckets": []}
        client.head_object.return_value = {
            "ContentLength": 100,
            "ETag": '"abc123"',
            "ContentType": "text/plain",
            "LastModified": datetime.now(),
            "Metadata": {}
        }
        client.upload_file.return_value = None
        client.download_file.return_value = None
        client.delete_object.return_value = {}
        client.generate_presigned_url.return_value = "https://example.com/presigned"
        
        # Setup paginator
        paginator = Mock()
        paginator.paginate.return_value = [{"Contents": []}]
        client.get_paginator.return_value = paginator
        
        return session, client, resource
    
    @pytest_asyncio.fixture
    def mock_async_session(self):
        """Create mock aiobotocore session."""
        session = Mock()
        client = AsyncMock()
        
        # Setup async client
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.put_object = AsyncMock()
        client.get_object = AsyncMock(return_value={
            "Body": AsyncMock(read=AsyncMock(return_value=b"test data"), close=AsyncMock())
        })
        client.delete_object = AsyncMock()
        client.close = AsyncMock()
        
        session.create_client.return_value = client
        
        return session, client
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_boto3_session, mock_async_session):
        """Create and initialize test provider with mocks."""
        session, client, resource = mock_boto3_session
        async_session, async_client = mock_async_session
        
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock(return_value=session))), \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.get_session', return_value=async_session), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True):
            
            provider = S3Provider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_provider_lifecycle(self, provider_settings, mock_boto3_session, mock_async_session):
        """Test provider initialization and shutdown lifecycle."""
        session, client, resource = mock_boto3_session
        async_session, async_client = mock_async_session
        
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock(return_value=session))), \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.get_session', return_value=async_session), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True):
            
            provider = S3Provider(settings=provider_settings)
            
            # Initially not initialized
            assert not provider._initialized
            
            # Initialize
            await provider.initialize()
            assert provider._initialized
            
            # Shutdown
            await provider.shutdown()
            assert not provider._initialized
    
    @pytest.mark.asyncio
    async def test_initialization_creates_bucket(self, provider_settings, mock_boto3_session, mock_async_session):
        """Test initialization creates bucket if create_bucket is True."""
        session, client, resource = mock_boto3_session
        async_session, async_client = mock_async_session
        
        # Make bucket_exists return False initially
        client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock(return_value=session))), \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.get_session', return_value=async_session), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True):
            
            provider = S3Provider(settings=provider_settings)
            await provider.initialize()
            
            # Should have called create_bucket
            client.create_bucket.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_fails_with_invalid_credentials(self, provider_settings, mock_boto3_session, mock_async_session):
        """Test initialization fails with invalid credentials."""
        session, client, resource = mock_boto3_session
        async_session, async_client = mock_async_session
        
        # Make head_bucket fail with access denied
        client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "HeadBucket"
        )
        client.create_bucket.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "CreateBucket"
        )
        
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock(return_value=session))), \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.get_session', return_value=async_session), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True):
            
            provider = S3Provider(settings=provider_settings)
            
            with pytest.raises(ProviderError):
                await provider.initialize()


class TestS3ProviderBucketOperations:
    """Test bucket operations with mocks."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1"
        )
    
    @pytest_asyncio.fixture
    def mock_client(self):
        """Create mock S3 client."""
        client = Mock()
        client.head_bucket.return_value = {}
        client.create_bucket.return_value = {}
        client.delete_bucket.return_value = {}
        client.list_buckets.return_value = {"Buckets": []}
        
        # Setup paginator for delete_bucket with force
        paginator = Mock()
        paginator.paginate.return_value = [{"Contents": [{"Key": "test.txt"}]}]
        client.get_paginator.return_value = paginator
        client.delete_objects.return_value = {}
        
        return client
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_client):
        """Create and initialize test provider."""
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock())) as mock_boto3, \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True), \
             patch('flowlib.providers.storage.s3.provider.get_session'):
            
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            
            provider = S3Provider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_create_bucket_success(self, provider, mock_client):
        """Test successful bucket creation."""
        # Make bucket not exist initially
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        
        result = await provider.create_bucket("new-bucket")
        
        assert result is True
        mock_client.create_bucket.assert_called_with(Bucket="new-bucket")
    
    @pytest.mark.asyncio
    async def test_create_bucket_already_exists(self, provider, mock_client):
        """Test creating bucket that already exists."""
        # Make bucket exist
        mock_client.head_bucket.return_value = {}
        
        result = await provider.create_bucket("existing-bucket")
        assert result is True
        # Should not call create_bucket
        mock_client.create_bucket.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_bucket_with_region(self, provider_settings, mock_client):
        """Test creating bucket with region constraint."""
        # Create new settings with us-west-2 region
        regional_settings = provider_settings.model_copy(update={"region_name": "us-west-2"})
        
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock())) as mock_boto3, \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True), \
             patch('flowlib.providers.storage.s3.provider.get_session'):
            
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            
            provider = S3Provider(settings=regional_settings)
            await provider.initialize()
            
            mock_client.head_bucket.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "HeadBucket"
            )
            
            result = await provider.create_bucket("regional-bucket")
            
            assert result is True
            mock_client.create_bucket.assert_called_with(
                Bucket="regional-bucket",
                CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
            )
    
    @pytest.mark.asyncio
    async def test_bucket_exists_true(self, provider, mock_client):
        """Test bucket existence check for existing bucket."""
        mock_client.head_bucket.return_value = {}
        
        exists = await provider.bucket_exists("existing-bucket")
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_bucket_exists_false(self, provider, mock_client):
        """Test bucket existence check for non-existent bucket."""
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        
        exists = await provider.bucket_exists("nonexistent-bucket")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_bucket_exists_access_denied(self, provider, mock_client):
        """Test bucket existence check with access denied (bucket exists but no permission)."""
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403"}}, "HeadBucket"
        )
        
        exists = await provider.bucket_exists("restricted-bucket")
        assert exists is True  # Bucket exists but we don't have permission
    
    @pytest.mark.asyncio
    async def test_delete_bucket_success(self, provider, mock_client):
        """Test successful bucket deletion."""
        mock_client.head_bucket.return_value = {}
        
        result = await provider.delete_bucket("delete-bucket")
        
        assert result is True
        mock_client.delete_bucket.assert_called_with(Bucket="delete-bucket")
    
    @pytest.mark.asyncio
    async def test_delete_bucket_with_force(self, provider, mock_client):
        """Test deleting bucket with force flag."""
        mock_client.head_bucket.return_value = {}
        
        result = await provider.delete_bucket("nonempty-bucket", force=True)
        
        assert result is True
        # Should delete objects first
        mock_client.get_paginator.assert_called_with("list_objects_v2")
        mock_client.delete_objects.assert_called_once()
        mock_client.delete_bucket.assert_called_with(Bucket="nonempty-bucket")
    
    @pytest.mark.asyncio
    async def test_delete_bucket_nonexistent(self, provider, mock_client):
        """Test deleting non-existent bucket."""
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        
        result = await provider.delete_bucket("nonexistent-bucket")
        assert result is True
        mock_client.delete_bucket.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_operations_require_initialization(self, provider_settings):
        """Test that bucket operations require initialization."""
        provider = S3Provider(settings=provider_settings)
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.create_bucket("test")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.bucket_exists("test")
        
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.delete_bucket("test")


class TestS3ProviderFileOperations:
    """Test file upload and download operations with mocks."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1",
            acl="private",
            cache_control="max-age=3600"
        )
    
    @pytest_asyncio.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest_asyncio.fixture
    def test_file(self, temp_dir):
        """Create a test file."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, 'w') as f:
            f.write("Hello, S3!")
        return file_path
    
    @pytest_asyncio.fixture
    def mock_client(self):
        """Create mock S3 client."""
        client = Mock()
        client.head_bucket.return_value = {}
        client.upload_file.return_value = None
        client.download_file.return_value = None
        client.head_object.return_value = {
            "ContentLength": 10,
            "ETag": '"abc123"',
            "ContentType": "text/plain",
            "LastModified": datetime.now(),
            "Metadata": {}
        }
        return client
    
    @pytest_asyncio.fixture
    def mock_async_client(self):
        """Create mock async S3 client."""
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.put_object = AsyncMock()
        client.get_object = AsyncMock(return_value={
            "Body": AsyncMock(read=AsyncMock(return_value=b"Hello, S3!"), close=AsyncMock())
        })
        client.delete_object = AsyncMock()
        return client
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_client, mock_async_client):
        """Create and initialize test provider."""
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock())) as mock_boto3, \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True), \
             patch('flowlib.providers.storage.s3.provider.get_session') as mock_get_session:
            
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            mock_get_session.return_value.create_client.return_value = mock_async_client
            
            provider = S3Provider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, provider, test_file, mock_client):
        """Test successful file upload."""
        object_key = "uploads/test.txt"
        metadata = {"author": "test", "version": "1.0"}
        
        result = await provider.upload_file(test_file, object_key, metadata=metadata)
        
        assert isinstance(result, FileMetadata)
        assert result.key == object_key
        
        # Verify upload_file was called with correct parameters
        mock_client.upload_file.assert_called_once()
        call_args = mock_client.upload_file.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert call_args[1]["Key"] == object_key
        assert call_args[1]["Filename"] == test_file
        
        # Check ExtraArgs
        extra_args = call_args[1]["ExtraArgs"]
        assert extra_args["Metadata"] == metadata
        assert extra_args["ACL"] == "private"
        assert extra_args["CacheControl"] == "max-age=3600"
    
    @pytest.mark.asyncio
    async def test_upload_file_nonexistent_source(self, provider):
        """Test uploading non-existent source file."""
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.upload_file("/nonexistent/file.txt", "test.txt")
    
    @pytest.mark.asyncio
    async def test_upload_file_bucket_not_exists(self, provider_settings, test_file, mock_client):
        """Test uploading to non-existent bucket."""
        # Use the original settings
        no_create_settings = provider_settings
        
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock())) as mock_boto3, \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True), \
             patch('flowlib.providers.storage.s3.provider.get_session'):
            
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            
            provider = S3Provider(settings=no_create_settings)
            await provider.initialize()
            
            mock_client.head_bucket.side_effect = ClientError(
                {"Error": {"Code": "404"}}, "HeadBucket"
            )
        
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.upload_file(test_file, "test.txt")
    
    @pytest.mark.asyncio
    async def test_upload_data_bytes(self, provider, mock_async_client):
        """Test uploading binary data."""
        data = b"Binary test data \x00\x01\x02"
        object_key = "binary/test.bin"
        content_type = "application/octet-stream"
        metadata = {"type": "binary"}
        
        result = await provider.upload_data(data, object_key, content_type=content_type, metadata=metadata)
        
        assert result.key == object_key
        
        # Verify put_object was called
        mock_async_client.put_object.assert_called_once()
        call_args = mock_async_client.put_object.call_args[1]
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == object_key
        assert call_args["Body"] == data
        assert call_args["ContentType"] == content_type
        assert call_args["Metadata"] == metadata
    
    @pytest.mark.asyncio
    async def test_upload_data_file_like_object(self, provider, mock_async_client):
        """Test uploading data from file-like object."""
        data = b"File-like object data"
        file_obj = io.BytesIO(data)
        object_key = "fileobj/test.bin"
        
        result = await provider.upload_data(file_obj, object_key)
        
        assert result.key == object_key
        
        # Verify put_object was called with file object
        mock_async_client.put_object.assert_called_once()
        call_args = mock_async_client.put_object.call_args[1]
        assert call_args["Body"] == file_obj
    
    @pytest.mark.asyncio
    async def test_download_file_success(self, provider, temp_dir, mock_client):
        """Test successful file download."""
        object_key = "download_test.txt"
        download_path = os.path.join(temp_dir, "downloaded.txt")
        
        result = await provider.download_file(object_key, download_path)
        
        assert isinstance(result, FileMetadata)
        assert result.key == object_key
        
        # Verify download_file was called
        mock_client.download_file.assert_called_with(
            Bucket="test-bucket",
            Key=object_key,
            Filename=download_path
        )
    
    @pytest.mark.asyncio
    async def test_download_file_nonexistent(self, provider, temp_dir, mock_client):
        """Test downloading non-existent file."""
        mock_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        
        download_path = os.path.join(temp_dir, "downloaded.txt")
        
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.download_file("nonexistent.txt", download_path)
    
    @pytest.mark.asyncio
    async def test_download_data_success(self, provider, mock_async_client):
        """Test successful data download."""
        object_key = "download_data_test.bin"
        expected_data = b"Download test data"
        
        # Setup mock response
        mock_async_client.get_object.return_value = {
            "Body": AsyncMock(read=AsyncMock(return_value=expected_data), close=AsyncMock())
        }
        
        downloaded_data, metadata = await provider.download_data(object_key)
        
        assert downloaded_data == expected_data
        assert isinstance(metadata, FileMetadata)
        assert metadata.key == object_key
        
        # Verify get_object was called
        mock_async_client.get_object.assert_called_with(
            Bucket="test-bucket",
            Key=object_key
        )
    
    @pytest.mark.asyncio
    async def test_download_data_nonexistent(self, provider, mock_async_client):
        """Test downloading data for non-existent object."""
        mock_async_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.download_data("nonexistent.bin")


class TestS3ProviderMetadataOperations:
    """Test metadata and object operations with mocks."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1"
        )
    
    @pytest_asyncio.fixture
    def mock_client(self):
        """Create mock S3 client."""
        client = Mock()
        client.head_bucket.return_value = {}
        client.head_object.return_value = {
            "ContentLength": 100,
            "ETag": '"abc123def456"',
            "ContentType": "text/plain",
            "LastModified": datetime(2023, 1, 1, 12, 0, 0),
            "Metadata": {"author": "test"}
        }
        
        # Setup list_objects_v2 paginator
        paginator = Mock()
        paginator.paginate.return_value = [{
            "Contents": [
                {
                    "Key": "file1.txt",
                    "Size": 100,
                    "ETag": '"abc123"',
                    "LastModified": datetime(2023, 1, 1, 12, 0, 0)
                },
                {
                    "Key": "file2.txt", 
                    "Size": 200,
                    "ETag": '"def456"',
                    "LastModified": datetime(2023, 1, 2, 12, 0, 0)
                }
            ]
        }]
        client.get_paginator.return_value = paginator
        client.generate_presigned_url.return_value = "https://example.com/presigned"
        
        return client
    
    @pytest_asyncio.fixture
    def mock_async_client(self):
        """Create mock async S3 client."""
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.delete_object = AsyncMock()
        return client
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_client, mock_async_client):
        """Create and initialize test provider."""
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock())) as mock_boto3, \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True), \
             patch('flowlib.providers.storage.s3.provider.get_session') as mock_get_session:
            
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            mock_get_session.return_value.create_client.return_value = mock_async_client
            
            provider = S3Provider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_get_metadata_success(self, provider, mock_client):
        """Test successful metadata retrieval."""
        object_key = "metadata_test.txt"
        
        metadata = await provider.get_metadata(object_key)
        
        assert isinstance(metadata, FileMetadata)
        assert metadata.key == object_key
        assert metadata.size == 100
        assert metadata.etag == "abc123def456"
        assert metadata.content_type == "text/plain"
        assert metadata.modified == datetime(2023, 1, 1, 12, 0, 0)
        assert metadata.metadata == {"author": "test"}
        
        mock_client.head_object.assert_called_with(
            Bucket="test-bucket",
            Key=object_key
        )
    
    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self, provider, mock_client):
        """Test metadata retrieval for non-existent object."""
        mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "HeadObject"
        )
        
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.get_metadata("nonexistent.txt")
    
    @pytest.mark.asyncio
    async def test_delete_object_success(self, provider, mock_async_client):
        """Test successful object deletion."""
        object_key = "delete_test.txt"
        
        result = await provider.delete_object(object_key)
        
        assert result is True
        mock_async_client.delete_object.assert_called_with(
            Bucket="test-bucket",
            Key=object_key
        )
    
    @pytest.mark.asyncio
    async def test_delete_object_nonexistent(self, provider, mock_async_client):
        """Test deleting non-existent object."""
        mock_async_client.delete_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "DeleteObject"
        )
        
        result = await provider.delete_object("nonexistent.txt")
        assert result is True  # Should still return True
    
    @pytest.mark.asyncio
    async def test_list_objects_success(self, provider, mock_client):
        """Test listing objects."""
        objects = await provider.list_objects()
        
        assert len(objects) == 2
        
        # Check first object
        obj1 = objects[0]
        assert obj1.key == "file1.txt"
        assert obj1.size == 100
        assert obj1.etag == "abc123"
        assert obj1.modified == datetime(2023, 1, 1, 12, 0, 0)
        assert obj1.content_type == "application/octet-stream"  # Default
        
        # Check second object
        obj2 = objects[1]
        assert obj2.key == "file2.txt"
        assert obj2.size == 200
        assert obj2.etag == "def456"
        assert obj2.modified == datetime(2023, 1, 2, 12, 0, 0)
    
    @pytest.mark.asyncio
    async def test_list_objects_with_prefix(self, provider, mock_client):
        """Test listing objects with prefix."""
        await provider.list_objects(prefix="docs/")
        
        # Verify paginate was called with prefix
        mock_client.get_paginator.assert_called_with("list_objects_v2")
        paginator = mock_client.get_paginator.return_value
        paginator.paginate.assert_called_with(Bucket="test-bucket", Prefix="docs/")
    
    @pytest.mark.asyncio
    async def test_list_objects_empty_bucket(self, provider, mock_client):
        """Test listing objects in empty bucket."""
        # Setup empty response
        paginator = Mock()
        paginator.paginate.return_value = [{}]  # No Contents key
        mock_client.get_paginator.return_value = paginator
        
        objects = await provider.list_objects()
        assert objects == []
    
    @pytest.mark.asyncio
    async def test_list_objects_nonexistent_bucket(self, provider, mock_client):
        """Test listing objects in non-existent bucket."""
        mock_client.get_paginator.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket"}}, "ListObjects"
        )
        
        with pytest.raises(ProviderError, match="does not exist"):
            await provider.list_objects()


class TestS3ProviderAdvancedFeatures:
    """Test advanced provider features with mocks."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1"
        )
    
    @pytest_asyncio.fixture
    def mock_client(self):
        """Create mock S3 client."""
        client = Mock()
        client.head_bucket.return_value = {}
        client.list_buckets.return_value = {"Buckets": []}
        client.generate_presigned_url.return_value = "https://example.com/presigned?signature=abc123"
        return client
    
    @pytest_asyncio.fixture
    async def provider(self, provider_settings, mock_client):
        """Create and initialize test provider."""
        with patch('flowlib.providers.storage.s3.provider.boto3', Mock(Session=Mock())) as mock_boto3, \
             patch('flowlib.providers.storage.s3.provider.botocore', Mock(config=Mock(Config=Mock(return_value=Mock())))), \
             patch('flowlib.providers.storage.s3.provider.BOTO3_AVAILABLE', True), \
             patch('flowlib.providers.storage.s3.provider.get_session'):
            
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            
            provider = S3Provider(settings=provider_settings)
            await provider.initialize()
            return provider
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_get(self, provider, mock_client):
        """Test generating presigned URL for GET operation."""
        object_key = "url_test.txt"
        expiration = 7200
        
        url = await provider.generate_presigned_url(object_key, expiration=expiration, operation="get")
        
        assert url == "https://example.com/presigned?signature=abc123"
        
        mock_client.generate_presigned_url.assert_called_with(
            ClientMethod="get_object",
            Params={"Bucket": "test-bucket", "Key": object_key},
            ExpiresIn=expiration
        )
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_put(self, provider, mock_client):
        """Test generating presigned URL for PUT operation."""
        object_key = "put_test.txt"
        
        url = await provider.generate_presigned_url(object_key, operation="put")
        
        assert url == "https://example.com/presigned?signature=abc123"
        
        mock_client.generate_presigned_url.assert_called_with(
            ClientMethod="put_object",
            Params={"Bucket": "test-bucket", "Key": object_key},
            ExpiresIn=3600  # Default expiration
        )
    
    @pytest.mark.asyncio
    async def test_generate_presigned_url_invalid_operation(self, provider):
        """Test generating presigned URL with invalid operation."""
        with pytest.raises(ProviderError, match="Invalid operation"):
            await provider.generate_presigned_url("test.txt", operation="invalid")
    
    @pytest.mark.asyncio
    async def test_check_connection_success(self, provider, mock_client):
        """Test successful connection check."""
        result = await provider.check_connection()
        assert result is True
        mock_client.list_buckets.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_connection_failure(self, provider, mock_client):
        """Test connection check failure."""
        mock_client.list_buckets.side_effect = Exception("Connection failed")
        
        result = await provider.check_connection()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_connection_uninitialized(self, provider_settings, mock_client):
        """Test connection check with uninitialized provider."""
        with patch('flowlib.providers.storage.s3.provider.boto3.Session') as mock_session:
            mock_boto3.Session.return_value.client.return_value = mock_client
            mock_boto3.Session.return_value.resource.return_value = Mock()
            
            provider = S3Provider(settings=provider_settings)
            
            result = await provider.check_connection()
            assert result is False


class TestS3ProviderErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest_asyncio.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return S3ProviderSettings(
            bucket="test-bucket",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1"
        )
    
    @pytest.mark.asyncio
    async def test_operations_without_initialization(self, provider_settings):
        """Test that operations fail without initialization."""
        provider = S3Provider(settings=provider_settings)
        
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
    async def test_provider_inheritance_check(self):
        """Test that S3Provider properly inherits from Provider."""
        from flowlib.providers.core.base import Provider
        provider = S3Provider()
        assert isinstance(provider, Provider)
        assert provider.provider_type == "storage"  # Inferred from base class
    
    @pytest.mark.asyncio
    async def test_credentials_handling(self, provider_settings):
        """Test credential configuration."""
        # Test with explicit credentials
        settings = S3ProviderSettings(
            access_key_id="AKIATEST",
            secret_access_key="secret123",
            session_token="token456"
        )
        provider = S3Provider(settings=settings)
        
        assert provider._settings.access_key_id == "AKIATEST"
        assert provider._settings.secret_access_key == "secret123"
        assert provider._settings.session_token == "token456"
    
    @pytest.mark.asyncio
    async def test_multipart_settings(self):
        """Test multipart upload settings."""
        settings = S3ProviderSettings(
            multipart_threshold=16 * 1024 * 1024,
            multipart_chunksize=8 * 1024 * 1024
        )
        provider = S3Provider(settings=settings)
        
        assert provider._settings.multipart_threshold == 16 * 1024 * 1024
        assert provider._settings.multipart_chunksize == 8 * 1024 * 1024
    
    @pytest.mark.asyncio
    async def test_s3_compatible_endpoint(self):
        """Test S3-compatible service configuration."""
        settings = S3ProviderSettings(
            endpoint_url="http://localhost:9000",  # MinIO
            path_style=True
        )
        provider = S3Provider(settings=settings)
        
        assert provider._settings.endpoint_url == "http://localhost:9000"
        assert provider._settings.path_style is True