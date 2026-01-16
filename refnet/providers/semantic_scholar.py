"""
semantic scholar provider.
https://api.semanticscholar.org/
"""

import httpx
import time
import os
import logging
from typing import List, Optional, Dict, Any

from .base import PaperProvider, AuthorInfo
from ..core.models import Paper, PaperStatus
from ..core.resilience import (
    ResilientAPIClient, RetryConfig, CircuitBreakerConfig
)

logger = logging.getLogger("refnet.s2")


class SemanticScholarProvider(PaperProvider):
    """
    semantic scholar API client.
    secondary provider with good citation data.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self._session = None  # lazy init
        self._last_request = 0
        # with key: 100 req/sec, without: 100 req/5 min
        self._min_delay = 0.01 if self.api_key else 3.0

        # resilience components
        self._resilient = ResilientAPIClient(
            name="semantic_scholar",
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=2.0 if not self.api_key else 0.5,  # longer delay without key
                max_delay=60.0,
                retryable_exceptions=(
                    ConnectionError,
                    TimeoutError,
                    OSError,
                    httpx.ConnectError,
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                )
            ),
            circuit_config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=120.0  # longer recovery for S2
            )
        )

    @property
    def session(self) -> httpx.Client:
        """lazy session initialization."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.Client(timeout=30.0)
        return self._session

    def close(self):
        """close the http session."""
        if self._session and not self._session.is_closed:
            self._session.close()
            self._session = None

    def __del__(self):
        """cleanup on garbage collection."""
        self.close()

    @property
    def name(self) -> str:
        return "semantic_scholar"

    def _request(self, endpoint: str, params: Optional[Dict] = None,
                 method: str = "GET") -> Optional[Dict]:
        """make rate-limited request with retry and circuit breaker."""
        # rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        def do_request():
            if method == "GET":
                resp = self.session.get(url, params=params, headers=headers)
            else:
                resp = self.session.post(url, json=params, headers=headers)

            self._last_request = time.time()

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # rate limited - raise to trigger retry with longer delay
                logger.warning(f"[s2] rate limited on {endpoint}")
                time.sleep(5)  # extra delay for S2
                raise ConnectionError("rate limited")
            elif resp.status_code >= 500:
                # server error - raise to trigger retry
                logger.warning(f"[s2] server error {resp.status_code} on {endpoint}")
                raise ConnectionError(f"server error {resp.status_code}")
            elif resp.status_code == 404:
                # not found - valid response
                logger.debug(f"[s2] 404 for {endpoint}")
                return None
            else:
                logger.warning(f"[s2] {resp.status_code} for {endpoint}")
                return None

        # execute with resilience
        result = self._resilient.execute(
            operation=do_request,
            operation_name=f"{method} {endpoint}"
        )

        return result

    # paper methods

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[Paper]:
        """search papers by query."""
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,externalIds,title,year,venue,authors,citationCount,referenceCount,abstract,isOpenAccess,openAccessPdf,s2FieldsOfStudy"
        }

        if year_min or year_max:
            year_filter = ""
            if year_min:
                year_filter = f"{year_min}-"
            if year_max:
                year_filter += str(year_max)
            elif year_min:
                year_filter += "2030"
            params["year"] = year_filter

        data = self._request("paper/search", params)
        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        return papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """get paper by id (DOI, S2 ID, etc)."""
        # normalize id
        if paper_id.startswith("10."):
            lookup_id = f"DOI:{paper_id}"
        elif paper_id.startswith("doi:"):
            lookup_id = f"DOI:{paper_id[4:]}"
        else:
            lookup_id = paper_id

        data = self._request(f"paper/{lookup_id}", {
            "fields": "paperId,externalIds,title,year,venue,authors,citationCount,referenceCount,abstract,isOpenAccess,openAccessPdf,s2FieldsOfStudy"
        })

        if not data:
            return None

        return self._parse_paper(data)

    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """get papers cited by this paper."""
        if paper_id.startswith("10."):
            lookup_id = f"DOI:{paper_id}"
        else:
            lookup_id = paper_id

        data = self._request(f"paper/{lookup_id}/references", {
            "fields": "paperId,externalIds,title,year,venue,authors,citationCount,referenceCount,isOpenAccess",
            "limit": min(limit, 1000)
        })

        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            cited_paper = item.get("citedPaper", {})
            if cited_paper:
                paper = self._parse_paper(cited_paper)
                if paper:
                    papers.append(paper)

        return papers[:limit]

    def get_citations(self, paper_id: str, limit: int = 30) -> List[Paper]:
        """get papers that cite this paper."""
        if paper_id.startswith("10."):
            lookup_id = f"DOI:{paper_id}"
        else:
            lookup_id = paper_id

        data = self._request(f"paper/{lookup_id}/citations", {
            "fields": "paperId,externalIds,title,year,venue,authors,citationCount,referenceCount,isOpenAccess",
            "limit": min(limit, 1000)
        })

        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            citing_paper = item.get("citingPaper", {})
            if citing_paper:
                paper = self._parse_paper(citing_paper)
                if paper:
                    papers.append(paper)

        return papers[:limit]

    def get_count_estimate(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> int:
        """estimate count for query."""
        params = {
            "query": query,
            "limit": 1,
            "fields": "paperId"
        }
        if year_min or year_max:
            year_filter = f"{year_min or ''}-{year_max or '2030'}"
            params["year"] = year_filter

        data = self._request("paper/search", params)
        if data:
            return data.get("total", 0)
        return -1

    # author methods

    def supports_authors(self) -> bool:
        return True

    def get_author(self, author_id: str) -> Optional[AuthorInfo]:
        """get author by S2 author ID."""
        data = self._request(f"author/{author_id}", {
            "fields": "authorId,externalIds,name,affiliations,paperCount,citationCount,hIndex"
        })

        if not data:
            return None

        return self._parse_author(data)

    def get_author_works(
        self,
        author_id: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 50
    ) -> List[Paper]:
        """get papers by author."""
        data = self._request(f"author/{author_id}/papers", {
            "fields": "paperId,externalIds,title,year,venue,authors,citationCount,referenceCount,isOpenAccess",
            "limit": min(limit, 1000)
        })

        if not data:
            return []

        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                # filter by year if specified
                if year_min and paper.year and paper.year < year_min:
                    continue
                if year_max and paper.year and paper.year > year_max:
                    continue
                papers.append(paper)

        return papers[:limit]

    def resolve_author_id(
        self,
        name: str,
        affiliation: Optional[str] = None,
        coauthor_names: Optional[List[str]] = None
    ) -> Optional[AuthorInfo]:
        """try to resolve author by name."""
        data = self._request("author/search", {
            "query": name,
            "limit": 5,
            "fields": "authorId,externalIds,name,affiliations,paperCount,citationCount"
        })

        if not data:
            return None

        results = data.get("data", [])
        if not results:
            return None

        # if affiliation provided, try to match
        if affiliation:
            for r in results:
                affs = r.get("affiliations", [])
                for aff in affs:
                    if affiliation.lower() in aff.lower():
                        return self._parse_author(r)

        # return first (highest relevance)
        return self._parse_author(results[0])

    # helpers

    def _parse_paper(self, data: Dict) -> Optional[Paper]:
        """parse S2 paper to Paper."""
        if not data or not data.get("paperId"):
            return None

        # external ids
        ext_ids = data.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI")

        # authors
        authors = []
        author_ids = []
        for auth in data.get("authors", [])[:10]:
            name = auth.get("name", "")
            if name:
                authors.append(name)
            aid = auth.get("authorId")
            if aid:
                author_ids.append(aid)

        # fields/concepts
        concepts = []
        for field in data.get("s2FieldsOfStudy", [])[:10]:
            concepts.append({
                "name": field.get("category", ""),
                "score": 1.0
            })

        # check if review
        title = data.get("title", "") or ""
        is_review = "review" in title.lower()[:50]

        # OA
        is_oa = data.get("isOpenAccess", False)
        oa_pdf = data.get("openAccessPdf", {})
        oa_url = oa_pdf.get("url") if oa_pdf else None

        return Paper(
            doi=doi,
            s2_id=data.get("paperId"),
            title=title,
            year=data.get("year"),
            venue=data.get("venue", ""),
            authors=authors,
            author_ids=author_ids,
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            is_review=is_review,
            is_oa=is_oa,
            oa_pdf_url=oa_url,
            concepts=concepts,
            abstract=data.get("abstract"),
            status=PaperStatus.CANDIDATE
        )

    def _parse_author(self, data: Dict) -> AuthorInfo:
        """parse S2 author to AuthorInfo."""
        ext_ids = data.get("externalIds", {}) or {}
        orcid = ext_ids.get("ORCID")

        affiliations = data.get("affiliations", []) or []

        return AuthorInfo(
            name=data.get("name", ""),
            orcid=orcid,
            s2_id=data.get("authorId"),
            affiliations=affiliations[:3],
            paper_count=data.get("paperCount"),
            citation_count=data.get("citationCount")
        )

    def stats(self) -> Dict:
        """get provider statistics including resilience metrics."""
        return {
            "provider": self.name,
            "has_api_key": self.api_key is not None,
            **self._resilient.stats()
        }
