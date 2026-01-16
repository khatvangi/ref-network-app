"""
composite provider - combines multiple providers with fallback.
tries primary provider first, falls back to secondary on failure.
"""

import logging
from typing import List, Optional, Dict, Any

from .base import PaperProvider, AuthorInfo
from ..core.models import Paper


logger = logging.getLogger("refnet.composite")


class CompositeProvider(PaperProvider):
    """
    combines multiple paper providers with automatic fallback.

    tries providers in order, falling back to the next on failure.
    useful for combining OpenAlex (comprehensive) with S2 (citation-focused).
    """

    def __init__(self, providers: List[PaperProvider]):
        """
        args:
            providers: list of providers in priority order
        """
        if not providers:
            raise ValueError("at least one provider required")

        self.providers = providers
        self._primary = providers[0]

        # stats
        self._fallback_count = 0
        self._provider_usage: Dict[str, int] = {p.name: 0 for p in providers}

    @property
    def name(self) -> str:
        return f"composite({','.join(p.name for p in self.providers)})"

    def _try_providers(
        self,
        method_name: str,
        *args,
        accept_empty: bool = False,
        **kwargs
    ) -> Any:
        """
        try method on each provider until one succeeds.

        args:
            method_name: name of method to call
            accept_empty: if True, accept empty list as valid result
            *args, **kwargs: arguments to pass to method
        """
        last_error = None

        for i, provider in enumerate(self.providers):
            method = getattr(provider, method_name, None)
            if not method:
                continue

            try:
                result = method(*args, **kwargs)

                # check if result is valid
                if result is not None:
                    if accept_empty or (not isinstance(result, list) or len(result) > 0):
                        self._provider_usage[provider.name] += 1
                        if i > 0:
                            self._fallback_count += 1
                            logger.info(
                                f"[composite] {method_name} succeeded with fallback "
                                f"provider {provider.name}"
                            )
                        return result

                # result was None or empty list, try next provider
                logger.debug(
                    f"[composite] {provider.name}.{method_name} returned empty, "
                    f"trying next provider"
                )

            except Exception as e:
                logger.warning(
                    f"[composite] {provider.name}.{method_name} failed: {e}"
                )
                last_error = e
                continue

        # all providers failed
        if last_error:
            logger.warning(f"[composite] all providers failed for {method_name}")

        return [] if method_name in ("search_papers", "get_references", "get_citations", "get_author_works") else None

    # paper methods

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[Paper]:
        """search papers across providers."""
        return self._try_providers(
            "search_papers",
            query,
            year_min=year_min,
            year_max=year_max,
            limit=limit
        )

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """get paper from any provider."""
        return self._try_providers("get_paper", paper_id)

    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """get references from any provider."""
        return self._try_providers("get_references", paper_id, limit=limit)

    def get_citations(self, paper_id: str, limit: int = 30) -> List[Paper]:
        """get citations from any provider."""
        return self._try_providers("get_citations", paper_id, limit=limit)

    def get_count_estimate(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> int:
        """get count estimate from primary provider."""
        result = self._try_providers(
            "get_count_estimate",
            query,
            year_min=year_min,
            year_max=year_max
        )
        return result if result is not None else -1

    # author methods

    def supports_authors(self) -> bool:
        return any(p.supports_authors() for p in self.providers)

    def get_author(self, author_id: str) -> Optional[AuthorInfo]:
        """get author from any provider."""
        return self._try_providers("get_author", author_id)

    def get_author_works(
        self,
        author_id: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 50
    ) -> List[Paper]:
        """get author works from any provider."""
        return self._try_providers(
            "get_author_works",
            author_id,
            year_min=year_min,
            year_max=year_max,
            limit=limit
        )

    def resolve_author_id(
        self,
        name: str,
        affiliation: Optional[str] = None,
        coauthor_names: Optional[List[str]] = None
    ) -> Optional[AuthorInfo]:
        """resolve author from any provider."""
        return self._try_providers(
            "resolve_author_id",
            name,
            affiliation=affiliation,
            coauthor_names=coauthor_names
        )

    def close(self):
        """close all provider sessions."""
        for provider in self.providers:
            if hasattr(provider, "close"):
                provider.close()

    def __del__(self):
        """cleanup on garbage collection."""
        self.close()

    def stats(self) -> Dict:
        """get composite provider statistics."""
        provider_stats = {}
        for p in self.providers:
            if hasattr(p, "stats"):
                provider_stats[p.name] = p.stats()

        return {
            "provider": self.name,
            "fallback_count": self._fallback_count,
            "provider_usage": self._provider_usage,
            "providers": provider_stats
        }


def create_default_provider(
    email: str = "user@example.com",
    s2_api_key: Optional[str] = None
) -> CompositeProvider:
    """
    create default composite provider with OpenAlex primary, S2 fallback.
    """
    from .openalex import OpenAlexProvider
    from .semantic_scholar import SemanticScholarProvider

    providers = [
        OpenAlexProvider(email=email),
        SemanticScholarProvider(api_key=s2_api_key)
    ]

    return CompositeProvider(providers)
