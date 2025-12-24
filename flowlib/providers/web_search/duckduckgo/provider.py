"""DuckDuckGo web search provider implementation.

This module implements web search using DuckDuckGo's search engine
via the duckduckgo-search library. No API key required.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from pydantic import Field

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.core.decorators import provider
from flowlib.providers.web_search.base import (
    WebSearchProvider,
    WebSearchProviderSettings,
    WebSearchResponse,
    WebSearchResult,
)

if TYPE_CHECKING:
    from duckduckgo_search import DDGS as DDGSType

logger = logging.getLogger(__name__)

# Runtime import of duckduckgo-search
try:
    from duckduckgo_search import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS = None  # type: ignore[misc, assignment]


class DuckDuckGoSettings(WebSearchProviderSettings):
    """Settings for DuckDuckGo web search provider.

    DuckDuckGo doesn't require an API key but has rate limits.
    """

    # DuckDuckGo-specific settings
    region: str = Field(
        default="wt-wt",
        description="Region for search results (wt-wt for worldwide)",
    )
    backend: str = Field(
        default="api",
        description="Search backend: api, html, lite",
    )


# Country name to DuckDuckGo region code mapping
COUNTRY_TO_REGION: dict[str, str] = {
    "turkey": "tr-tr",
    "united states": "us-en",
    "usa": "us-en",
    "germany": "de-de",
    "france": "fr-fr",
    "united kingdom": "uk-en",
    "uk": "uk-en",
    "spain": "es-es",
    "italy": "it-it",
    "netherlands": "nl-nl",
    "japan": "jp-jp",
    "china": "cn-zh",
    "brazil": "br-pt",
    "india": "in-en",
    "australia": "au-en",
    "canada": "ca-en",
    "russia": "ru-ru",
    "mexico": "mx-es",
    "south korea": "kr-kr",
    "indonesia": "id-en",
    "poland": "pl-pl",
    "sweden": "se-sv",
    "belgium": "be-nl",
    "austria": "at-de",
    "switzerland": "ch-de",
    "portugal": "pt-pt",
    "greece": "gr-el",
    "czech republic": "cz-cs",
    "denmark": "dk-da",
    "finland": "fi-fi",
    "norway": "no-no",
    "ireland": "ie-en",
    "new zealand": "nz-en",
    "singapore": "sg-en",
    "hong kong": "hk-tzh",
    "taiwan": "tw-tzh",
    "malaysia": "my-en",
    "philippines": "ph-en",
    "thailand": "th-th",
    "vietnam": "vn-vi",
    "argentina": "ar-es",
    "chile": "cl-es",
    "colombia": "co-es",
    "peru": "pe-es",
    "south africa": "za-en",
    "egypt": "eg-ar",
    "saudi arabia": "sa-ar",
    "uae": "ae-ar",
    "united arab emirates": "ae-ar",
    "israel": "il-he",
}


@provider(provider_type="web_search", name="duckduckgo", settings_class=DuckDuckGoSettings)
class DuckDuckGoProvider(WebSearchProvider[DuckDuckGoSettings]):
    """DuckDuckGo web search provider.

    Free web search without requiring an API key.
    Uses the duckduckgo-search library.
    """

    def __init__(
        self,
        name: str,
        provider_type: str = "web_search",
        settings: DuckDuckGoSettings | None = None,
        **kwargs: Any,
    ):
        """Initialize DuckDuckGo provider.

        Args:
            name: Provider name
            provider_type: Provider type
            settings: Provider settings
            **kwargs: Additional arguments
        """
        super().__init__(name=name, provider_type=provider_type, settings=settings, **kwargs)
        self._ddgs: "DDGSType | None" = None

    async def _initialize(self) -> None:
        """Initialize DuckDuckGo client."""
        if not DDGS_AVAILABLE:
            raise ProviderError(
                message="duckduckgo-search package not available. Install with: pip install duckduckgo-search",
                context=ErrorContext.create(
                    flow_name="web_search_provider",
                    error_type="DependencyError",
                    error_location="DuckDuckGoProvider._initialize",
                    component=self.name,
                    operation="initialize",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="initialize",
                    retry_count=0,
                ),
            )

        self._ddgs = DDGS()
        logger.info(f"DuckDuckGo search provider initialized: {self.name}")

    async def _shutdown(self) -> None:
        """Shutdown DuckDuckGo client."""
        self._ddgs = None
        logger.info(f"DuckDuckGo search provider shutdown: {self.name}")

    def _execute_ddgs_search(
        self,
        query: str,
        region: str,
        timelimit: str | None,
        num_results: int,
    ) -> list[dict]:
        """Execute synchronous DuckDuckGo text search.

        This method runs in a thread pool via asyncio.to_thread.

        Args:
            query: Search query
            region: Region code
            timelimit: Time limit filter
            num_results: Maximum results

        Returns:
            List of raw result dictionaries
        """
        assert self._ddgs is not None
        return list(
            self._ddgs.text(
                keywords=query,
                region=region,
                safesearch="moderate" if self.settings.safe_search else "off",
                timelimit=timelimit,
                backend=self.settings.backend,
                max_results=num_results,
            )
        )

    def _execute_ddgs_news(
        self,
        query: str,
        region: str,
        num_results: int,
    ) -> list[dict]:
        """Execute synchronous DuckDuckGo news search.

        This method runs in a thread pool via asyncio.to_thread.

        Args:
            query: Search query
            region: Region code
            num_results: Maximum results

        Returns:
            List of raw result dictionaries
        """
        assert self._ddgs is not None
        return list(
            self._ddgs.news(
                keywords=query,
                region=region,
                safesearch="moderate" if self.settings.safe_search else "off",
                max_results=num_results,
            )
        )

    async def _execute_search(
        self,
        query: str,
        num_results: int,
        country: str | None,
        date_range: str | None,
    ) -> WebSearchResponse:
        """Execute search using DuckDuckGo.

        Args:
            query: Search query
            num_results: Maximum results
            country: Country filter
            date_range: Date range filter

        Returns:
            WebSearchResponse with results
        """
        if not self._ddgs:
            raise ProviderError(
                message="DuckDuckGo client not initialized",
                context=ErrorContext.create(
                    flow_name="web_search_provider",
                    error_type="StateError",
                    error_location="DuckDuckGoProvider._execute_search",
                    component=self.name,
                    operation="search",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="search",
                    retry_count=0,
                ),
            )

        try:
            # Map date_range to DuckDuckGo timelimit
            timelimit = self._get_timelimit(date_range)

            # Build region code from country if provided
            region = self._get_region(country)

            # Execute search in thread pool (DDGS is synchronous)
            raw_results: list[dict] = await asyncio.to_thread(
                self._execute_ddgs_search,
                query,
                region,
                timelimit,
                num_results,
            )

            # Convert to WebSearchResult objects
            results = [
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("href") or item.get("link", ""),
                    snippet=item.get("body") or item.get("snippet", ""),
                    source=self._extract_domain(item.get("href", "")),
                    published_date=None,  # DuckDuckGo text search doesn't provide dates
                )
                for item in raw_results
            ]

            logger.debug(f"DuckDuckGo search returned {len(results)} results for: {query}")

            return WebSearchResponse(
                results=results,
                total_results=len(results),
                query=query,
            )

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise ProviderError(
                message=f"DuckDuckGo search failed: {str(e)}",
                context=ErrorContext.create(
                    flow_name="web_search_provider",
                    error_type="OperationError",
                    error_location="DuckDuckGoProvider._execute_search",
                    component=self.name,
                    operation="search",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="search",
                    retry_count=0,
                ),
                cause=e,
            ) from e

    def _extract_domain(self, url: str) -> str | None:
        """Extract domain from URL.

        Args:
            url: Full URL

        Returns:
            Domain name or None
        """
        if not url:
            return None
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return None

    def _get_timelimit(self, date_range: str | None) -> str | None:
        """Convert date_range to DuckDuckGo timelimit parameter.

        Args:
            date_range: Date range filter (day, week, month, year, any)

        Returns:
            DuckDuckGo timelimit string or None
        """
        if not date_range:
            return None

        timelimit_map = {
            "day": "d",
            "week": "w",
            "month": "m",
            "year": "y",
            "any": None,
        }
        return timelimit_map.get(date_range)

    def _get_region(self, country: str | None) -> str:
        """Get DuckDuckGo region code from country name.

        Args:
            country: Country name

        Returns:
            DuckDuckGo region code
        """
        if not country:
            return self.settings.region

        return COUNTRY_TO_REGION.get(country.lower(), self.settings.region)

    async def search_news(
        self,
        query: str,
        num_results: int = 10,
        region: str | None = None,
    ) -> WebSearchResponse:
        """Search DuckDuckGo news.

        Args:
            query: Search query
            num_results: Maximum results
            region: Region filter

        Returns:
            WebSearchResponse with news results
        """
        if not self._ddgs:
            raise ProviderError(
                message="DuckDuckGo client not initialized",
                context=ErrorContext.create(
                    flow_name="web_search_provider",
                    error_type="StateError",
                    error_location="DuckDuckGoProvider.search_news",
                    component=self.name,
                    operation="search_news",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="search_news",
                    retry_count=0,
                ),
            )

        try:
            # Execute news search in thread pool (DDGS is synchronous)
            raw_results: list[dict] = await asyncio.to_thread(
                self._execute_ddgs_news,
                query,
                region or self.settings.region,
                num_results,
            )

            # Convert to WebSearchResult objects
            results = [
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("body", ""),
                    source=item.get("source") or self._extract_domain(item.get("url", "")),
                    published_date=item.get("date"),
                )
                for item in raw_results
            ]

            return WebSearchResponse(
                results=results,
                total_results=len(results),
                query=query,
            )

        except Exception as e:
            logger.error(f"DuckDuckGo news search failed: {e}")
            raise ProviderError(
                message=f"DuckDuckGo news search failed: {str(e)}",
                context=ErrorContext.create(
                    flow_name="web_search_provider",
                    error_type="OperationError",
                    error_location="DuckDuckGoProvider.search_news",
                    component=self.name,
                    operation="search_news",
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type=self.provider_type,
                    operation="search_news",
                    retry_count=0,
                ),
                cause=e,
            ) from e
