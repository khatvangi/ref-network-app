"""
openalex provider - primary source for papers and authors.
https://docs.openalex.org/
"""

import httpx
import time
import logging
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base import PaperProvider, AuthorInfo
from ..core.models import Paper, PaperStatus
from ..core.resilience import (
    ResilientAPIClient, RetryConfig, CircuitBreakerConfig,
    validate_paper_data, safe_execute
)

logger = logging.getLogger("refnet.openalex")


class OpenAlexProvider(PaperProvider):
    """
    openalex.org API client.
    primary provider for paper and author data.
    """

    def __init__(self, email: str = "user@example.com"):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self._session = None  # lazy init
        self._last_request = 0
        self._min_delay = 0.1  # 10 req/sec polite
        self._rate_lock = threading.Lock()  # thread-safe rate limiting

        # resilience components
        self._resilient = ResilientAPIClient(
            name="openalex",
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=30.0,
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
                recovery_timeout=60.0
            )
        )

    @property
    def session(self) -> httpx.Client:
        """lazy session initialization with proper cleanup."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.Client(timeout=30.0)
        return self._session

    def close(self):
        """close the http session."""
        if self._session and not self._session.is_closed:
            self._session.close()
            self._session = None

    def __enter__(self):
        """context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """context manager exit - ensures cleanup."""
        self.close()
        return False

    def __del__(self):
        """cleanup on garbage collection (fallback)."""
        self.close()

    @property
    def name(self) -> str:
        return "openalex"

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """make rate-limited request with retry and circuit breaker."""
        # thread-safe rate limiting
        with self._rate_lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_delay:
                time.sleep(self._min_delay - elapsed)
            self._last_request = time.time()

        if params is None:
            params = {}
        params["mailto"] = self.email

        url = f"{self.base_url}/{endpoint}"

        def do_request():
            resp = self.session.get(url, params=params)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # rate limited - raise to trigger retry
                logger.warning(f"[openalex] rate limited on {endpoint}")
                raise ConnectionError("rate limited")
            elif resp.status_code >= 500:
                # server error - raise to trigger retry
                logger.warning(f"[openalex] server error {resp.status_code} on {endpoint}")
                raise ConnectionError(f"server error {resp.status_code}")
            elif resp.status_code == 400:
                # bad request - log and return None (don't retry)
                filter_val = params.get("filter", "")[:100]
                logger.warning(f"[openalex] 400 for {endpoint} (filter: {filter_val}...)")
                return None
            elif resp.status_code == 404:
                # not found - valid response, don't log as error
                logger.debug(f"[openalex] 404 for {endpoint}")
                return None
            else:
                logger.warning(f"[openalex] {resp.status_code} for {endpoint}")
                return None

        # execute with resilience
        result = self._resilient.execute(
            operation=do_request,
            operation_name=f"GET {endpoint}"
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
        filters = []
        if year_min:
            filters.append(f"from_publication_date:{year_min}-01-01")
        if year_max:
            filters.append(f"to_publication_date:{year_max}-12-31")

        params = {
            "search": query,
            "per_page": min(limit, 200),
            "sort": "cited_by_count:desc"
        }
        if filters:
            params["filter"] = ",".join(filters)

        data = self._request("works", params)
        if not data:
            return []

        papers = []
        for work in data.get("results", []):
            paper = self._parse_work(work)
            if paper:
                papers.append(paper)

        return papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """get paper by id (DOI or OpenAlex ID)."""
        # normalize id
        if paper_id.startswith("10."):
            endpoint = f"works/doi:{paper_id}"
        elif paper_id.startswith("https://doi.org/"):
            endpoint = f"works/doi:{paper_id[16:]}"
        elif paper_id.startswith("W"):
            endpoint = f"works/{paper_id}"
        elif paper_id.startswith("https://openalex.org/"):
            endpoint = f"works/{paper_id.split('/')[-1]}"
        else:
            endpoint = f"works/{paper_id}"

        data = self._request(endpoint)
        if not data:
            return None

        return self._parse_work(data)

    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """get papers cited by this paper."""
        # get the work first to get referenced_works
        if paper_id.startswith("10."):
            endpoint = f"works/doi:{paper_id}"
        elif paper_id.startswith("W"):
            endpoint = f"works/{paper_id}"
        else:
            endpoint = f"works/{paper_id}"

        data = self._request(endpoint)
        if not data:
            return []

        ref_ids = data.get("referenced_works", [])
        if not ref_ids:
            return []

        # fetch referenced works in batch
        ref_ids = ref_ids[:limit]
        papers = []

        # batch fetch (max 50 at a time)
        for i in range(0, len(ref_ids), 50):
            batch = ref_ids[i:i+50]
            # convert to short ids (W12345 format)
            short_ids = [r.split("/")[-1] for r in batch]
            filter_str = "|".join(short_ids)

            # correct OpenAlex filter syntax: ids.openalex
            batch_data = self._request("works", {
                "filter": f"ids.openalex:{filter_str}",
                "per_page": 50
            })

            if batch_data:
                for work in batch_data.get("results", []):
                    paper = self._parse_work(work)
                    if paper:
                        papers.append(paper)

        return papers

    def get_citations(self, paper_id: str, limit: int = 30) -> List[Paper]:
        """get papers that cite this paper."""
        # cites filter requires OpenAlex work ID (W12345), not DOI
        if paper_id.startswith("10."):
            oa_id = self._doi_to_openalex_id(paper_id)
            if not oa_id:
                return []
        elif paper_id.startswith("W"):
            oa_id = paper_id
        elif paper_id.startswith("https://openalex.org/"):
            oa_id = paper_id.split("/")[-1]
        else:
            # try to look up assuming it's a DOI
            oa_id = self._doi_to_openalex_id(paper_id)
            if not oa_id:
                return []

        data = self._request("works", {
            "filter": f"cites:{oa_id}",
            "per_page": min(limit, 200),
            "sort": "cited_by_count:desc"
        })

        if not data:
            return []

        papers = []
        for work in data.get("results", []):
            paper = self._parse_work(work)
            if paper:
                papers.append(paper)

        return papers[:limit]

    def get_count_estimate(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> int:
        """get count estimate for query."""
        filters = []
        if year_min:
            filters.append(f"from_publication_date:{year_min}-01-01")
        if year_max:
            filters.append(f"to_publication_date:{year_max}-12-31")

        params = {
            "search": query,
            "per_page": 1
        }
        if filters:
            params["filter"] = ",".join(filters)

        data = self._request("works", params)
        if data:
            return data.get("meta", {}).get("count", 0)
        return -1

    # author methods

    def supports_authors(self) -> bool:
        return True

    def get_author(self, author_id: str) -> Optional[AuthorInfo]:
        """get author by OpenAlex author ID."""
        if author_id.startswith("A"):
            endpoint = f"authors/{author_id}"
        elif author_id.startswith("https://openalex.org/"):
            endpoint = f"authors/{author_id.split('/')[-1]}"
        else:
            endpoint = f"authors/{author_id}"

        data = self._request(endpoint)
        if not data:
            return None

        return self._parse_author(data)

    def get_author_by_orcid(self, orcid: str) -> Optional[AuthorInfo]:
        """get author by ORCID."""
        # normalize
        if orcid.startswith("https://orcid.org/"):
            orcid = orcid[18:]

        data = self._request("authors", {
            "filter": f"orcid:{orcid}"
        })

        if not data:
            return None

        results = data.get("results", [])
        if not results:
            return None

        return self._parse_author(results[0])

    def get_author_works(
        self,
        author_id: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 50
    ) -> List[Paper]:
        """get papers by author."""
        # normalize author id
        if author_id.startswith("A"):
            oa_author_id = author_id
        elif author_id.startswith("https://openalex.org/"):
            oa_author_id = author_id.split("/")[-1]
        else:
            oa_author_id = author_id

        filters = [f"author.id:{oa_author_id}"]
        if year_min:
            filters.append(f"from_publication_date:{year_min}-01-01")
        if year_max:
            filters.append(f"to_publication_date:{year_max}-12-31")

        data = self._request("works", {
            "filter": ",".join(filters),
            "per_page": min(limit, 200),
            "sort": "cited_by_count:desc"
        })

        if not data:
            return []

        papers = []
        for work in data.get("results", []):
            paper = self._parse_work(work)
            if paper:
                papers.append(paper)

        return papers[:limit]

    def resolve_author_id(
        self,
        name: str,
        affiliation: Optional[str] = None,
        coauthor_names: Optional[List[str]] = None
    ) -> Optional[AuthorInfo]:
        """try to resolve author by name + hints."""
        # search by name
        data = self._request("authors", {
            "search": name,
            "per_page": 5
        })

        if not data:
            return None

        results = data.get("results", [])
        if not results:
            return None

        # if only one result, return it
        if len(results) == 1:
            return self._parse_author(results[0])

        # try to match by affiliation
        if affiliation:
            for r in results:
                last_inst = r.get("last_known_institution", {})
                if last_inst:
                    inst_name = last_inst.get("display_name", "").lower()
                    if affiliation.lower() in inst_name:
                        return self._parse_author(r)

        # return highest cited as fallback
        return self._parse_author(results[0])

    # concept/field methods for trajectory

    def get_top_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """get top concepts for a query."""
        data = self._request("works", {
            "search": query,
            "per_page": 1,
            "group_by": "concepts.id"
        })

        if not data:
            return []

        groups = data.get("group_by", [])
        concepts = []
        for g in groups[:limit]:
            concepts.append({
                "id": g.get("key", ""),
                "name": g.get("key_display_name", ""),
                "count": g.get("count", 0)
            })

        return concepts

    # helpers

    def _doi_to_openalex_id(self, doi: str) -> Optional[str]:
        """convert DOI to OpenAlex ID."""
        paper = self.get_paper(doi)
        if paper and paper.openalex_id:
            return paper.openalex_id
        return None

    def _parse_work(self, work: Dict) -> Optional[Paper]:
        """parse OpenAlex work to Paper."""
        if not work:
            return None

        # extract openalex id
        oa_id = work.get("id", "")
        if oa_id.startswith("https://openalex.org/"):
            oa_id = oa_id.split("/")[-1]

        # authors
        authors = []
        author_ids = []
        for authorship in work.get("authorships", [])[:10]:
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)
            aid = author.get("id", "")
            if aid:
                author_ids.append(aid.split("/")[-1] if "/" in aid else aid)

        # concepts
        concepts = []
        for c in work.get("concepts", [])[:10]:
            concepts.append({
                "name": c.get("display_name", ""),
                "score": c.get("score", 0)
            })

        # check if review
        is_review = False
        doc_type = work.get("type", "")
        if doc_type == "review":
            is_review = True
        title = work.get("title", "") or ""
        if "review" in title.lower()[:50]:
            is_review = True

        # OA info
        oa_status = work.get("open_access", {})
        is_oa = oa_status.get("is_oa", False)
        oa_url = oa_status.get("oa_url")

        # venue
        venue = ""
        primary_location = work.get("primary_location", {})
        if primary_location:
            source = primary_location.get("source", {})
            if source:
                venue = source.get("display_name", "")

        return Paper(
            doi=work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
            openalex_id=oa_id,
            title=title,
            year=work.get("publication_year"),
            venue=venue,
            authors=authors,
            author_ids=author_ids,
            citation_count=work.get("cited_by_count"),
            reference_count=len(work.get("referenced_works", [])),
            is_review=is_review,
            is_oa=is_oa,
            oa_pdf_url=oa_url,
            concepts=concepts,
            abstract=work.get("abstract"),
            status=PaperStatus.CANDIDATE
        )

    def _parse_author(self, author: Dict) -> AuthorInfo:
        """parse OpenAlex author to AuthorInfo."""
        oa_id = author.get("id", "")
        if oa_id.startswith("https://openalex.org/"):
            oa_id = oa_id.split("/")[-1]

        # orcid
        orcid = None
        orcid_raw = author.get("orcid")
        if orcid_raw:
            if orcid_raw.startswith("https://orcid.org/"):
                orcid = orcid_raw[18:]
            else:
                orcid = orcid_raw

        # affiliations
        affiliations = []
        last_inst = author.get("last_known_institution", {})
        if last_inst:
            affiliations.append(last_inst.get("display_name", ""))

        return AuthorInfo(
            name=author.get("display_name", ""),
            orcid=orcid,
            openalex_id=oa_id,
            affiliations=affiliations,
            paper_count=author.get("works_count"),
            citation_count=author.get("cited_by_count")
        )

    def stats(self) -> Dict:
        """get provider statistics including resilience metrics."""
        return {
            "provider": self.name,
            **self._resilient.stats()
        }
