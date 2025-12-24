"""Web search providers for AI-Flowlib.

This module provides pluggable web search capabilities through:
- WebSearchProvider: Base class for all web search implementations
- DuckDuckGoProvider: Free search using DuckDuckGo
- SerpAPIProvider: Paid search using SerpAPI (Google/Bing results)
- BraveProvider: Search using Brave Search API
"""

from flowlib.providers.web_search.base import (
    WebSearchProvider,
    WebSearchProviderSettings,
    WebSearchResponse,
    WebSearchResult,
)

__all__ = [
    "WebSearchProvider",
    "WebSearchProviderSettings",
    "WebSearchResponse",
    "WebSearchResult",
]
