"""Web search provider base class and models.

This module provides the base class for implementing web search providers
such as DuckDuckGo, SerpAPI, Brave Search, etc.
"""

import logging
from datetime import datetime
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.base import Provider, ProviderSettings

logger = logging.getLogger(__name__)


class WebSearchResult(StrictBaseModel):
    """Individual web search result."""

    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet/description")
    source: str | None = Field(default=None, description="Source domain")
    published_date: str | None = Field(default=None, description="Published date if available")


class WebSearchResponse(StrictBaseModel):
    """Response from a web search operation."""

    results: list[WebSearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(default=0, description="Total number of results")
    query: str = Field(..., description="Query that was executed")


@runtime_checkable
class NewsSearchCapable(Protocol):
    """Protocol for providers that support news search.

    Providers implementing this protocol can search for news articles
    in addition to standard web search.
    """

    async def search_news(
        self,
        query: str,
        num_results: int = 10,
        region: str | None = None,
    ) -> "WebSearchResponse":
        """Search for news articles.

        Args:
            query: Search query
            num_results: Maximum results
            region: Region filter

        Returns:
            WebSearchResponse with news results
        """
        ...


class WebSearchProviderSettings(ProviderSettings):
    """Base settings for web search providers.

    Common settings that apply to all web search provider implementations.
    Individual providers can extend this with provider-specific settings.
    """

    # Request settings
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_results: int = Field(default=10, ge=1, le=100, description="Default max results per search")

    # Rate limiting
    rate_limit_per_minute: int = Field(
        default=10, ge=1, description="Maximum requests per minute"
    )

    # Retry settings
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries in seconds")

    # Default search parameters
    default_country: str | None = Field(default=None, description="Default country filter")
    safe_search: bool = Field(default=True, description="Enable safe search")


SettingsT = TypeVar("SettingsT", bound=WebSearchProviderSettings)


class WebSearchProvider(Provider[SettingsT], Generic[SettingsT]):
    """Base class for web search providers.

    This class provides the interface for:
    1. Performing web searches with various filters
    2. Rate limiting and retry logic
    3. Result normalization

    Subclasses must implement the `_execute_search` method.
    """

    def __init__(
        self,
        name: str,
        provider_type: str = "web_search",
        settings: SettingsT | None = None,
        **kwargs: Any,
    ):
        """Initialize web search provider.

        Args:
            name: Unique provider name
            provider_type: Provider type (default: 'web_search')
            settings: Optional provider settings
            **kwargs: Additional keyword arguments
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._request_count = 0

    async def _initialize(self) -> None:
        """Initialize the web search provider.

        Subclasses should override this to perform provider-specific initialization.
        """
        pass

    async def _shutdown(self) -> None:
        """Clean up resources.

        Subclasses should override this to perform provider-specific cleanup.
        """
        pass

    async def search(
        self,
        query: str,
        num_results: int | None = None,
        country: str | None = None,
        date_range: Literal["day", "week", "month", "year", "any"] | None = None,
    ) -> WebSearchResponse:
        """Perform a web search.

        Args:
            query: Search query string
            num_results: Maximum number of results to return
            country: Filter results by country
            date_range: Filter results by date range

        Returns:
            WebSearchResponse with search results

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        if num_results is None:
            num_results = self.settings.max_results

        if country is None:
            country = self.settings.default_country

        return await self._execute_search(
            query=query,
            num_results=num_results,
            country=country,
            date_range=date_range,
        )

    async def _execute_search(
        self,
        query: str,
        num_results: int,
        country: str | None,
        date_range: str | None,
    ) -> WebSearchResponse:
        """Execute the actual search operation.

        Subclasses must implement this method.

        Args:
            query: Search query string
            num_results: Maximum number of results
            country: Country filter
            date_range: Date range filter

        Returns:
            WebSearchResponse with search results

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _execute_search()")

    async def search_events(
        self,
        country: str,
        event_type: str | None = None,
        industry: str | None = None,
        num_results: int = 20,
    ) -> WebSearchResponse:
        """Search for business events in a specific country.

        Convenience method that builds an event-focused search query.

        Args:
            country: Country to search for events
            event_type: Type of event (trade_show, conference, exhibition)
            industry: Industry focus
            num_results: Maximum results

        Returns:
            WebSearchResponse with event search results
        """
        # Build query
        query_parts = [country]

        if event_type:
            event_type_map = {
                "trade_show": "trade show",
                "conference": "conference",
                "exhibition": "exhibition",
                "seminar": "seminar",
            }
            query_parts.append(event_type_map.get(event_type, event_type))
        else:
            query_parts.append("business events trade show conference exhibition")

        if industry:
            query_parts.append(industry)

        # Add dynamic year filter for upcoming events
        current_year = datetime.now().year
        query_parts.append(f"{current_year} {current_year + 1}")

        query = " ".join(query_parts)

        return await self.search(
            query=query,
            num_results=num_results,
            country=country,
            date_range="year",
        )

    async def check_connection(self) -> bool:
        """Check if the provider is working.

        Performs a simple test search to verify connectivity.
        A successful connection means the search executed without errors,
        regardless of whether results were found.

        Returns:
            True if connection is working, False if an error occurred
        """
        try:
            await self.search("test connection check", num_results=1)
            return True
        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            return False
