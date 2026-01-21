"""
base provider interface for paper/author data sources.
all providers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..core.models import Paper, Author


@dataclass
class AuthorInfo:
    """lightweight author info from provider."""
    name: str
    orcid: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    affiliations: List[str] = None
    paper_count: Optional[int] = None
    citation_count: Optional[int] = None

    def __post_init__(self):
        if self.affiliations is None:
            self.affiliations = []


class PaperProvider(ABC):
    """
    abstract base class for paper/author data providers.
    providers fetch data from external APIs (OpenAlex, S2, etc).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """provider name for logging."""
        pass

    # paper search methods

    @abstractmethod
    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[Paper]:
        """search for papers by query string."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        get paper by id.
        paper_id can be DOI, OpenAlex ID, S2 ID, etc.
        """
        pass

    @abstractmethod
    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """get papers cited by this paper (backward refs)."""
        pass

    @abstractmethod
    def get_citations(self, paper_id: str, limit: int = 30) -> List[Paper]:
        """get papers that cite this paper (forward cites)."""
        pass

    def get_count_estimate(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> int:
        """estimate total count for query (for triage)."""
        return -1  # not all providers support this

    # author methods (optional - default implementations)

    def get_author(self, author_id: str) -> Optional[AuthorInfo]:
        """get author info by id (ORCID, OpenAlex, S2)."""
        return None

    def get_author_works(
        self,
        author_id: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 50
    ) -> List[Paper]:
        """get papers by author."""
        return []

    def resolve_author_id(
        self,
        name: str,
        affiliation: Optional[str] = None,
        coauthor_names: Optional[List[str]] = None
    ) -> Optional[AuthorInfo]:
        """try to resolve author by name + hints."""
        return None

    # utility methods

    def supports_authors(self) -> bool:
        """does this provider support author queries?"""
        return False

    def supports_orcid(self) -> bool:
        """does this provider support ORCID lookups?"""
        return False


class ORCIDProvider:
    """
    ORCID API client for author identity resolution.
    separate from paper providers since ORCID is author-centric.
    """

    def __init__(self, rate_limit_delay: float = 1.0):
        self.base_url = "https://pub.orcid.org/v3.0"
        self.delay = rate_limit_delay
        self._last_request = 0

    def get_author_by_orcid(self, orcid: str) -> Optional[AuthorInfo]:
        """fetch author info from ORCID."""
        import httpx
        import time

        # normalize orcid
        orcid = orcid.strip()
        if orcid.startswith("https://orcid.org/"):
            orcid = orcid[18:]

        # rate limit
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            url = f"{self.base_url}/{orcid}/record"
            headers = {"Accept": "application/json"}
            resp = httpx.get(url, headers=headers, timeout=30)
            self._last_request = time.time()

            if resp.status_code != 200:
                return None

            data = resp.json()
            return self._parse_orcid_record(data, orcid)

        except Exception as e:
            print(f"[orcid] error fetching {orcid}: {e}")
            return None

    def get_works(self, orcid: str, limit: int = 50) -> List[Dict[str, Any]]:
        """fetch works list from ORCID (returns DOIs)."""
        import httpx
        import time

        orcid = orcid.strip()
        if orcid.startswith("https://orcid.org/"):
            orcid = orcid[18:]

        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            url = f"{self.base_url}/{orcid}/works"
            headers = {"Accept": "application/json"}
            resp = httpx.get(url, headers=headers, timeout=30)
            self._last_request = time.time()

            if resp.status_code != 200:
                return []

            data = resp.json()
            return self._parse_works(data, limit)

        except Exception as e:
            print(f"[orcid] error fetching works for {orcid}: {e}")
            return []

    def _parse_orcid_record(self, data: Dict, orcid: str) -> AuthorInfo:
        """parse ORCID API response."""
        person = data.get("person", {})
        name_data = person.get("name", {})

        given = name_data.get("given-names", {}).get("value", "")
        family = name_data.get("family-name", {}).get("value", "")
        name = f"{given} {family}".strip()

        # affiliations
        affiliations = []
        activities = data.get("activities-summary", {})
        employments = activities.get("employments", {}).get("affiliation-group", [])
        for emp in employments[:3]:
            summaries = emp.get("summaries", [])
            for s in summaries:
                org = s.get("employment-summary", {}).get("organization", {})
                org_name = org.get("name", "")
                if org_name and org_name not in affiliations:
                    affiliations.append(org_name)

        return AuthorInfo(
            name=name,
            orcid=orcid,
            affiliations=affiliations
        )

    def _parse_works(self, data: Dict, limit: int) -> List[Dict[str, Any]]:
        """parse works from ORCID response."""
        works = []
        groups = data.get("group", [])

        for group in groups[:limit]:
            work_summary = group.get("work-summary", [])
            if not work_summary:
                continue

            ws = work_summary[0]
            title_obj = ws.get("title", {}).get("title", {})
            title = title_obj.get("value", "") if title_obj else ""

            # get DOI
            doi = None
            external_ids = ws.get("external-ids", {}).get("external-id", [])
            for eid in external_ids:
                if eid.get("external-id-type") == "doi":
                    doi = eid.get("external-id-value")
                    break

            year = None
            pub_date = ws.get("publication-date", {})
            if pub_date and pub_date.get("year"):
                try:
                    year = int(pub_date["year"]["value"])
                except (ValueError, TypeError, KeyError):
                    pass

            works.append({
                "title": title,
                "doi": doi,
                "year": year
            })

        return works
