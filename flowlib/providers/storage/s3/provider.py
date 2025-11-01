"""S3 storage provider implementation.

This module provides a concrete implementation of the StorageProvider
for AWS S3 and S3-compatible storage services using boto3.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, BinaryIO

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.storage.base import (
    FileMetadata,
    StorageProvider,
    StorageProviderSettings,
)

from .models import S3ErrorContainer, S3ListObjectResponse, S3ObjectResponse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import boto3
    import botocore  # type: ignore[import-untyped]
    from aiobotocore.session import get_session  # type: ignore[import-not-found]
    from botocore.exceptions import ClientError  # type: ignore[import-untyped]

    BOTO3_AVAILABLE = True
else:
    try:
        import boto3
        import botocore  # type: ignore[import-untyped]
        from aiobotocore.session import get_session  # type: ignore[import-not-found]
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]

        BOTO3_AVAILABLE = True
    except ImportError:
        logger.warning(
            "boto3 or aiobotocore packages not found. Install with 'pip install boto3 aiobotocore'"
        )
        boto3 = None
        botocore = None
        ClientError = Exception
        get_session = None
        BOTO3_AVAILABLE = False


class S3ProviderSettings(StorageProviderSettings):
    """S3 storage provider settings - direct inheritance, only S3-specific fields.

    S3 requires:
    1. AWS credentials (access key, secret key)
    2. Bucket and region configuration
    3. S3-specific settings (endpoint, path style, etc.)

    Note: S3 is cloud-based, no traditional host/port needed.
    """

    # AWS credentials
    access_key_id: str | None = Field(default=None, description="AWS access key ID")
    secret_access_key: str | None = Field(default=None, description="AWS secret access key")
    session_token: str | None = Field(default=None, description="AWS session token (optional)")
    profile_name: str | None = Field(
        default=None, description="AWS profile name (optional, for local credentials)"
    )

    # S3 connection settings
    endpoint_url: str | None = Field(
        default=None, description="Custom endpoint URL for S3-compatible services"
    )
    region_name: str | None = Field(default=None, description="AWS region name")
    bucket: str = Field(default="default", description="S3 bucket name")
    create_bucket: bool = Field(
        default=True, description="Whether to create bucket if it doesn't exist"
    )
    path_style: bool = Field(default=False, description="Whether to use path-style addressing")
    signature_version: str = Field(default="s3v4", description="AWS signature version")

    # Performance settings
    validate_checksums: bool = Field(
        default=True, description="Whether to validate checksums on uploads/downloads"
    )
    max_pool_connections: int = Field(
        default=10, description="Maximum number of connections to keep in the pool"
    )

    # Multipart upload settings
    multipart_threshold: int = Field(
        default=8 * 1024 * 1024, description="Multipart upload threshold (8MB)"
    )
    multipart_chunksize: int = Field(
        default=8 * 1024 * 1024, description="Multipart upload chunk size (8MB)"
    )

    # S3 metadata settings
    acl: str | None = Field(
        default=None, description="S3 ACL setting ('private', 'public-read', etc.)"
    )
    cache_control: str | None = Field(
        default=None, description="Cache-Control header for objects"
    )
    auto_content_type: bool = Field(
        default=True,
        description="Whether to automatically detect content type from file extensions",
    )


# Decorator already imported above


@provider(provider_type="storage", name="s3", settings_class=S3ProviderSettings)
class S3Provider(StorageProvider[S3ProviderSettings]):
    """S3 implementation of the StorageProvider.

    This provider implements storage operations using boto3,
    supporting AWS S3 and S3-compatible storage services.
    """

    def __init__(self, name: str = "s3", settings: S3ProviderSettings | None = None):
        """Initialize S3 provider.

        Args:
            name: Unique provider name
            settings: Optional provider settings
        """
        # Create settings first to avoid issues with _default_settings() method
        settings = settings or S3ProviderSettings()

        # Pass explicit settings to parent class
        super().__init__(name=name, settings=settings)

        # Store settings for local use with proper type annotation
        self._settings: S3ProviderSettings = settings
        self._client: Any | None = None
        self._resource: Any | None = None
        self._async_session: Any | None = None

    def _parse_s3_error(self, client_error: ClientError) -> str:
        """Parse S3 error response strictly.

        Args:
            client_error: ClientError from boto3

        Returns:
            Error code string

        Raises:
            ValueError: If error response is malformed
        """
        try:
            error_container = S3ErrorContainer(**client_error.response)
            return error_container.Error.Code
        except Exception as e:
            raise ValueError(f"S3 error response malformed: {str(e)}") from e

    def _create_provider_error(
        self, message: str, operation: str, cause: Exception | None = None
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
            operation=operation,
        )

        provider_context = ProviderErrorContext(
            provider_name=self.name, provider_type="storage", operation=operation, retry_count=0
        )

        return ProviderError(
            message=message, context=error_context, provider_context=provider_context, cause=cause
        )

    async def initialize(self) -> None:
        """Initialize the S3 client and check or create the bucket."""
        if self._initialized:
            return

        try:
            # Check if boto3 is installed
            if not BOTO3_AVAILABLE:
                raise ProviderError(
                    message="boto3 package not installed. Install with 'pip install boto3 aiobotocore'",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="DependencyError",
                        error_location="initialize",
                        component=self.name,
                        operation="check_boto3_dependency",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="initialize",
                        retry_count=0,
                    ),
                )

            # Create session
            session_kwargs = {}
            if self._settings.profile_name:
                session_kwargs["profile_name"] = self._settings.profile_name

            session = boto3.Session(**session_kwargs)

            # Create client config
            client_kwargs = {
                "region_name": self._settings.region_name,
                "config": botocore.config.Config(
                    signature_version=self._settings.signature_version,
                    s3={"addressing_style": "path" if self._settings.path_style else "auto"},
                    max_pool_connections=self._settings.max_pool_connections,
                ),
            }

            # Add endpoint_url for S3-compatible services
            if self._settings.endpoint_url:
                client_kwargs["endpoint_url"] = self._settings.endpoint_url

            # Add credentials if provided
            if self._settings.access_key_id and self._settings.secret_access_key:
                client_kwargs["aws_access_key_id"] = self._settings.access_key_id
                client_kwargs["aws_secret_access_key"] = self._settings.secret_access_key

                if self._settings.session_token:
                    client_kwargs["aws_session_token"] = self._settings.session_token

            # Create S3 client
            self._client = session.client("s3", **client_kwargs)

            # Create S3 resource for some operations
            self._resource = session.resource("s3", **client_kwargs)

            # Create aiobotocore session for async operations
            self._async_session = get_session()

            # Check if bucket exists or create it
            if self._settings.create_bucket:
                await self._ensure_bucket_exists()

            self._initialized = True
            logger.debug(f"{self.name} provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize {self.name} provider: {str(e)}")
            raise ProviderError(
                message=f"Failed to initialize S3 provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="InitializationError",
                    error_location="initialize",
                    component=self.name,
                    operation="initialize_s3_client",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="initialize",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def shutdown(self) -> None:
        """Close S3 client and release resources."""
        if not self._initialized:
            return

        try:
            # No need to close boto3 clients, they handle their own lifecycle
            self._client = None
            self._resource = None
            self._async_session = None
            self._initialized = False
            logger.debug(f"{self.name} provider shut down successfully")

        except Exception as e:
            logger.error(f"Error during {self.name} provider shutdown: {str(e)}")
            raise ProviderError(
                message=f"Failed to shut down S3 provider: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="ShutdownError",
                    error_location="shutdown",
                    component=self.name,
                    operation="shutdown_s3_client",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="shutdown",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def create_bucket(self, bucket: str | None = None) -> bool:
        """Create a new bucket/container.

        Args:
            bucket: Bucket name (default from settings if None)

        Returns:
            True if bucket was created successfully

        Raises:
            ProviderError: If bucket creation fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Check if bucket already exists
            if await self.bucket_exists(bucket):
                logger.debug(f"Bucket {bucket} already exists")
                return True

            # Create bucket
            create_bucket_kwargs: dict[str, str | dict[str, str]] = {"Bucket": bucket}

            # Add location constraint if region is specified
            if self._settings.region_name and self._settings.region_name != "us-east-1":
                create_bucket_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self._settings.region_name
                }

            # Create bucket using sync client (boto3)
            self._client.create_bucket(**create_bucket_kwargs)
            logger.debug(f"Created bucket: {bucket}")

            return True

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code == "BucketAlreadyOwnedByYou":
                logger.debug(f"Bucket {bucket} already owned by you")
                return True
            elif error_code == "BucketAlreadyExists":
                logger.debug(f"Bucket {bucket} already exists but is owned by another account")
                return True

            # Wrap and re-raise other errors
            raise self._create_provider_error(
                message=f"Failed to create bucket {bucket}: {str(e)}",
                operation="create_bucket",
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise self._create_provider_error(
                message=f"Failed to create bucket {bucket}: {str(e)}",
                operation="create_bucket",
                cause=e,
) from e

    async def delete_bucket(self, bucket: str | None = None, force: bool = False) -> bool:
        """Delete a bucket/container.

        Args:
            bucket: Bucket name (default from settings if None)
            force: Whether to delete all objects in the bucket first

        Returns:
            True if bucket was deleted successfully

        Raises:
            ProviderError: If bucket deletion fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Check if bucket exists
            if not await self.bucket_exists(bucket):
                logger.debug(f"Bucket {bucket} doesn't exist, nothing to delete")
                return True

            # If force is True, delete all objects in the bucket first
            if force:
                # List all objects in the bucket
                paginator = self._client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket):
                    if "Contents" in page:
                        # Collect keys to delete
                        keys = [{"Key": obj["Key"]} for obj in page["Contents"]]

                        # Delete objects in a batch
                        if keys:
                            self._client.delete_objects(Bucket=bucket, Delete={"Objects": keys})

            # Delete bucket using sync client (boto3)
            self._client.delete_bucket(Bucket=bucket)
            logger.debug(f"Deleted bucket: {bucket}")

            return True

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code == "NoSuchBucket":
                logger.debug(f"Bucket {bucket} doesn't exist, nothing to delete")
                return True
            elif error_code == "BucketNotEmpty" and not force:
                raise ProviderError(
                    message=f"Cannot delete non-empty bucket {bucket}. Use force=True to delete all objects first.",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="BucketNotEmptyError",
                        error_location="delete_bucket",
                        component=self.name,
                        operation="delete_bucket",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="delete_bucket",
                        retry_count=0,
                    ),
) from e

            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete bucket {bucket}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DeleteBucketError",
                    error_location="delete_bucket",
                    component=self.name,
                    operation="delete_bucket",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="delete_bucket",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete bucket {bucket}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DeleteBucketError",
                    error_location="delete_bucket",
                    component=self.name,
                    operation="delete_bucket",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="delete_bucket",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def bucket_exists(self, bucket: str | None = None) -> bool:
        """Check if a bucket/container exists.

        Args:
            bucket: Bucket name (default from settings if None)

        Returns:
            True if bucket exists

        Raises:
            ProviderError: If check fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Check if bucket exists using head_bucket
            self._client.head_bucket(Bucket=bucket)
            return True
        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code in ("404", "NoSuchBucket"):
                return False
            elif error_code in ("403", "AccessDenied"):
                # Bucket exists but we don't have permission to access it
                return True

            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to check if bucket {bucket} exists: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="BucketExistsCheckError",
                    error_location="bucket_exists",
                    component=self.name,
                    operation="bucket_exists",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="bucket_exists",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to check if bucket {bucket} exists: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="BucketExistsCheckError",
                    error_location="bucket_exists",
                    component=self.name,
                    operation="bucket_exists",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="bucket_exists",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def upload_file(
        self,
        file_path: str,
        object_key: str,
        bucket: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> FileMetadata:
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
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Ensure bucket exists
            if not await self.bucket_exists(bucket):
                if self._settings.create_bucket:
                    await self.create_bucket(bucket)
                else:
                    raise ProviderError(
                        message=f"Bucket {bucket} does not exist",
                        context=ErrorContext.create(
                            flow_name="s3_provider",
                            error_type="BucketNotFoundError",
                            error_location="upload_file",
                            component=self.name,
                            operation="upload_file",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="storage",
                            operation="upload_file",
                            retry_count=0,
                        ),
                    )

            # Check if file exists
            if not os.path.isfile(file_path):
                raise ProviderError(
                    message=f"File {file_path} does not exist",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="FileNotFoundError",
                        error_location="upload_file",
                        component=self.name,
                        operation="upload_file",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="upload_file",
                        retry_count=0,
                    ),
                )

            # Determine content type based on file extension
            content_type = None
            if self._settings.auto_content_type:
                content_type = self.get_content_type(file_path)

            # Prepare upload parameters
            upload_args: dict[str, str | dict[str, str]] = {
                "Bucket": bucket,
                "Key": object_key,
                "Filename": file_path,
            }

            # Add extra parameters if provided
            if content_type:
                upload_args["ContentType"] = content_type

            if metadata:
                upload_args["Metadata"] = metadata

            if self._settings.acl:
                upload_args["ACL"] = self._settings.acl

            if self._settings.cache_control:
                upload_args["CacheControl"] = self._settings.cache_control

            # Upload file using boto3's upload_file which supports multipart uploads
            self._client.upload_file(
                Filename=file_path,
                Bucket=bucket,
                Key=object_key,
                ExtraArgs={
                    k: v for k, v in upload_args.items() if k not in ["Bucket", "Key", "Filename"]
                },
            )

            logger.debug(f"Uploaded file {file_path} to {bucket}/{object_key}")

            # Get metadata for the uploaded file
            return await self.get_metadata(object_key, bucket)

        except ClientError as e:
            # Wrap and re-raise boto3 errors
            raise ProviderError(
                message=f"Failed to upload file {file_path} to {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="UploadFileError",
                    error_location="upload_file",
                    component=self.name,
                    operation="upload_file",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="upload_file",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to upload file {file_path} to {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="UploadFileError",
                    error_location="upload_file",
                    component=self.name,
                    operation="upload_file",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="upload_file",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def upload_data(
        self,
        data: bytes | BinaryIO,
        object_key: str,
        bucket: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> FileMetadata:
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
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Ensure bucket exists
            if not await self.bucket_exists(bucket):
                if self._settings.create_bucket:
                    await self.create_bucket(bucket)
                else:
                    raise ProviderError(
                        message=f"Bucket {bucket} does not exist",
                        context=ErrorContext.create(
                            flow_name="s3_provider",
                            error_type="BucketNotFoundError",
                            error_location="upload_data",
                            component=self.name,
                            operation="upload_data",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="storage",
                            operation="upload_data",
                            retry_count=0,
                        ),
                    )

            # Determine content type if not provided
            if not content_type and self._settings.auto_content_type:
                content_type = self.get_content_type(object_key)

            # Prepare upload parameters
            upload_args: dict[str, str | bytes | BinaryIO | dict[str, str]] = {
                "Bucket": bucket,
                "Key": object_key,
            }

            # Check if data is a file-like object or bytes
            if hasattr(data, "read"):
                upload_args["Body"] = data
            else:
                # Data must be bytes based on type annotation
                upload_args["Body"] = data

            # Add extra parameters if provided
            if content_type:
                upload_args["ContentType"] = content_type

            if metadata:
                upload_args["Metadata"] = metadata

            if self._settings.acl:
                upload_args["ACL"] = self._settings.acl

            if self._settings.cache_control:
                upload_args["CacheControl"] = self._settings.cache_control

            # Upload data using boto3's put_object
            async with self._get_async_client() as client:
                await client.put_object(**upload_args)

            logger.debug(f"Uploaded data to {bucket}/{object_key}")

            # Get metadata for the uploaded file
            return await self.get_metadata(object_key, bucket)

        except ClientError as e:
            # Wrap and re-raise boto3 errors
            raise ProviderError(
                message=f"Failed to upload data to {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="UploadDataError",
                    error_location="upload_data",
                    component=self.name,
                    operation="upload_data",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="upload_data",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to upload data to {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="UploadDataError",
                    error_location="upload_data",
                    component=self.name,
                    operation="upload_data",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="upload_data",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def download_file(
        self, object_key: str, file_path: str, bucket: str | None = None
    ) -> FileMetadata:
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
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Download file using boto3's download_file
            self._client.download_file(Bucket=bucket, Key=object_key, Filename=file_path)

            logger.debug(f"Downloaded {bucket}/{object_key} to {file_path}")

            # Get metadata for the downloaded file
            return await self.get_metadata(object_key, bucket)

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="ObjectNotFoundError",
                        error_location="download_file",
                        component=self.name,
                        operation="download_file",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="download_file",
                        retry_count=0,
                    ),
                    cause=e,
) from e

            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download {bucket}/{object_key} to {file_path}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DownloadFileError",
                    error_location="download_file",
                    component=self.name,
                    operation="download_file",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="download_file",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download {bucket}/{object_key} to {file_path}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DownloadFileError",
                    error_location="download_file",
                    component=self.name,
                    operation="download_file",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="download_file",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def download_data(
        self, object_key: str, bucket: str | None = None
    ) -> tuple[bytes, FileMetadata]:
        """Download binary data from storage.

        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)

        Returns:
            Tuple of (data bytes, file metadata)

        Raises:
            ProviderError: If download fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Get object using async client
            async with self._get_async_client() as client:
                response = await client.get_object(Bucket=bucket, Key=object_key)

                # Read all data from stream
                data = await response["Body"].read()
                await response["Body"].close()

            logger.debug(f"Downloaded data from {bucket}/{object_key}")

            # Get metadata
            metadata = await self.get_metadata(object_key, bucket)

            return data, metadata

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                raise ProviderError(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="ObjectNotFoundError",
                        error_location="download_data",
                        component=self.name,
                        operation="download_data",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="download_data",
                        retry_count=0,
                    ),
                    cause=e,
) from e

            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download data from {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DownloadDataError",
                    error_location="download_data",
                    component=self.name,
                    operation="download_data",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="download_data",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to download data from {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DownloadDataError",
                    error_location="download_data",
                    component=self.name,
                    operation="download_data",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="download_data",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def get_metadata(self, object_key: str, bucket: str | None = None) -> FileMetadata:
        """Get metadata for a storage object.

        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)

        Returns:
            File metadata

        Raises:
            ProviderError: If metadata retrieval fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Get object metadata using head_object
            response = self._client.head_object(Bucket=bucket, Key=object_key)

            # Extract metadata from response
            # Parse S3 response strictly
            s3_response = S3ObjectResponse(**response)

            metadata = FileMetadata(
                key=object_key,
                size=s3_response.ContentLength,
                etag=s3_response.ETag.strip('"'),
                content_type=s3_response.ContentType,
                modified=s3_response.LastModified,
                metadata=s3_response.Metadata,
            )

            return metadata

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                raise self._create_provider_error(
                    message=f"Object {object_key} does not exist in bucket {bucket}",
                    operation="get_metadata",
                    cause=e,
) from e

            # Wrap and re-raise other errors
            raise self._create_provider_error(
                message=f"Failed to get metadata for {bucket}/{object_key}: {str(e)}",
                operation="get_metadata",
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to get metadata for {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="GetMetadataError",
                    error_location="get_metadata",
                    component=self.name,
                    operation="get_metadata",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="get_metadata",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def delete_object(self, object_key: str, bucket: str | None = None) -> bool:
        """Delete a storage object.

        Args:
            object_key: Storage object key/path
            bucket: Bucket name (default from settings if None)

        Returns:
            True if object was deleted successfully

        Raises:
            ProviderError: If deletion fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # Delete object using async client
            async with self._get_async_client() as client:
                await client.delete_object(Bucket=bucket, Key=object_key)

            logger.debug(f"Deleted object {bucket}/{object_key}")
            return True

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code in ("404", "NoSuchKey"):
                logger.debug(
                    f"Object {object_key} does not exist in bucket {bucket}, nothing to delete"
                )
                return True

            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete object {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DeleteObjectError",
                    error_location="delete_object",
                    component=self.name,
                    operation="delete_object",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="delete_object",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to delete object {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="DeleteObjectError",
                    error_location="delete_object",
                    component=self.name,
                    operation="delete_object",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="delete_object",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def list_objects(
        self, prefix: str | None = None, bucket: str | None = None
    ) -> list[FileMetadata]:
        """List objects in a bucket/container.

        Args:
            prefix: Optional prefix to filter objects
            bucket: Bucket name (default from settings if None)

        Returns:
            List of file metadata

        Raises:
            ProviderError: If listing fails
        """
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        try:
            # List objects using boto3's list_objects_v2
            result = []
            paginator = self._client.get_paginator("list_objects_v2")

            # Prepare list parameters
            list_args = {"Bucket": bucket}
            if prefix:
                list_args["Prefix"] = prefix

            # Paginate through results
            for page in paginator.paginate(**list_args):
                if "Contents" in page:
                    for obj_data in page["Contents"]:
                        # Parse using strict Pydantic model
                        s3_obj = S3ListObjectResponse(**obj_data)

                        metadata = FileMetadata(
                            key=s3_obj.Key,
                            size=s3_obj.Size,
                            etag=s3_obj.ETag.strip('"'),
                            modified=s3_obj.LastModified,
                            content_type="application/octet-stream",  # Default, we'd need a head_object call to get the real content type
                        )
                        result.append(metadata)

            return result

        except ClientError as e:
            error_code = self._parse_s3_error(e)

            # Handle specific error codes
            if error_code in ("404", "NoSuchBucket"):
                raise ProviderError(
                    message=f"Bucket {bucket} does not exist",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="BucketNotFoundError",
                        error_location="list_objects",
                        component=self.name,
                        operation="list_objects",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="list_objects",
                        retry_count=0,
                    ),
                    cause=e,
) from e

            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to list objects in {bucket} with prefix {prefix}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="ListObjectsError",
                    error_location="list_objects",
                    component=self.name,
                    operation="list_objects",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="list_objects",
                    retry_count=0,
                ),
                cause=e,
) from e
        except Exception as e:
            # Wrap and re-raise other errors
            raise ProviderError(
                message=f"Failed to list objects in {bucket} with prefix {prefix}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="ListObjectsError",
                    error_location="list_objects",
                    component=self.name,
                    operation="list_objects",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="list_objects",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def generate_presigned_url(
        self,
        object_key: str,
        expiration: int = 3600,
        bucket: str | None = None,
        operation: str = "get",
    ) -> str:
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
        if not self._initialized or not self._client:
            raise self._create_provider_error(
                message="Provider not initialized", operation="operation"
            )

        bucket = bucket or self._settings.bucket

        # Map operation to S3 client method
        operation_map = {
            "get": "get_object",
            "put": "put_object",
            "delete": "delete_object",
            "head": "head_object",
        }

        if operation.lower() not in operation_map:
            raise ProviderError(
                message=f"Invalid operation: {operation}. Must be one of: {', '.join(operation_map.keys())}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="InvalidOperationError",
                    error_location="generate_presigned_url",
                    component=self.name,
                    operation="generate_presigned_url",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="generate_presigned_url",
                    retry_count=0,
                ),
            )
        client_method = operation_map[operation.lower()]

        try:
            # Generate presigned URL
            url = self._client.generate_presigned_url(
                ClientMethod=client_method,
                Params={"Bucket": bucket, "Key": object_key},
                ExpiresIn=expiration,
            )

            return str(url)

        except Exception as e:
            # Wrap and re-raise errors
            raise ProviderError(
                message=f"Failed to generate presigned URL for {bucket}/{object_key}: {str(e)}",
                context=ErrorContext.create(
                    flow_name="s3_provider",
                    error_type="PresignedUrlError",
                    error_location="generate_presigned_url",
                    component=self.name,
                    operation="generate_presigned_url",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="storage",
                    operation="generate_presigned_url",
                    retry_count=0,
                ),
                cause=e,
) from e

    async def check_connection(self) -> bool:
        """Check if storage connection is active.

        Returns:
            True if connection is active, False otherwise
        """
        if not self._initialized or not self._client:
            return False

        try:
            # Try to list buckets
            self._client.list_buckets()
            return True
        except Exception:
            return False

    async def _ensure_bucket_exists(self) -> None:
        """Ensure that the configured bucket exists, creating it if necessary."""
        bucket = self._settings.bucket

        try:
            # Check if bucket exists using direct client call (skip initialization check)
            assert self._client is not None  # Method only called after initialization
            self._client.head_bucket(Bucket=bucket)
        except Exception as bucket_check_error:
            # Bucket doesn't exist, create it if requested
            if self._settings.create_bucket:
                try:
                    # Create bucket
                    create_bucket_kwargs: dict[str, str | dict[str, str]] = {"Bucket": bucket}

                    # Add location constraint if region is specified
                    if self._settings.region_name and self._settings.region_name != "us-east-1":
                        create_bucket_kwargs["CreateBucketConfiguration"] = {
                            "LocationConstraint": self._settings.region_name
                        }

                    # Create bucket using sync client (boto3)
                    assert self._client is not None  # Method only called after initialization
                    self._client.create_bucket(**create_bucket_kwargs)
                    logger.debug(f"Created bucket: {bucket}")
                except ClientError as e:
                    # Use structured error parsing for S3 client errors
                    error_code = self._parse_s3_error(e)
                    if error_code in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
                        logger.debug(f"Bucket {bucket} already exists")
                    else:
                        # Actual error creating bucket
                        raise ProviderError(
                            message=f"Failed to create bucket {bucket}: {str(e)}",
                            context=ErrorContext.create(
                                flow_name="s3_provider",
                                error_type="CreateBucketError",
                                error_location="_ensure_bucket_exists",
                                component=self.name,
                                operation="create_bucket",
                            ),
                            provider_context=ProviderErrorContext(
                                provider_name=self.name,
                                provider_type="storage",
                                operation="create_bucket",
                                retry_count=0,
                            ),
                            cause=e,
) from e
                except Exception as e:
                    # Handle non-S3 errors
                    raise ProviderError(
                        message=f"Failed to create bucket {bucket}: {str(e)}",
                        context=ErrorContext.create(
                            flow_name="s3_provider",
                            error_type="CreateBucketError",
                            error_location="_ensure_bucket_exists",
                            component=self.name,
                            operation="create_bucket",
                        ),
                        provider_context=ProviderErrorContext(
                            provider_name=self.name,
                            provider_type="storage",
                            operation="create_bucket",
                            retry_count=0,
                        ),
                        cause=e,
) from e
            else:
                raise ProviderError(
                    message=f"Bucket {bucket} does not exist and create_bucket is set to False",
                    context=ErrorContext.create(
                        flow_name="s3_provider",
                        error_type="BucketNotFoundError",
                        error_location="_ensure_bucket_exists",
                        component=self.name,
                        operation="ensure_bucket_exists",
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=self.name,
                        provider_type="storage",
                        operation="ensure_bucket_exists",
                        retry_count=0,
                    ),
) from bucket_check_error

    def _get_async_client(self) -> Any:
        """Get or create an async S3 client context manager."""
        if not self._async_session:
            self._async_session = get_session()

        # Create client config for the session
        client_kwargs = {}

        # Add region if specified
        if self._settings.region_name:
            client_kwargs["region_name"] = self._settings.region_name

        # Add endpoint URL for S3-compatible services
        if self._settings.endpoint_url:
            client_kwargs["endpoint_url"] = self._settings.endpoint_url

        # Add credentials if provided
        if self._settings.access_key_id and self._settings.secret_access_key:
            client_kwargs["aws_access_key_id"] = self._settings.access_key_id
            client_kwargs["aws_secret_access_key"] = self._settings.secret_access_key

            if self._settings.session_token:
                client_kwargs["aws_session_token"] = self._settings.session_token

        # Add config
        client_kwargs["config"] = botocore.config.Config(
            signature_version=self._settings.signature_version,
            s3={"addressing_style": "path" if self._settings.path_style else "auto"},
            max_pool_connections=self._settings.max_pool_connections,
        )

        # Create and return the async client context manager
        assert self._async_session is not None  # Help mypy understand this can't be None here
        return self._async_session.create_client("s3", **client_kwargs)
