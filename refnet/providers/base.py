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
    h_index: Optional[int] = None  # from openalex summary_stats

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
        limit: int = 50,
        offset: int = 0
    ) -> List[Paper]:
        """get papers by author with pagination."""
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


@dataclass
class ORCIDEducation:
    """education record from ORCID."""
    institution: str
    degree: Optional[str] = None  # PhD, MS, BS, etc.
    department: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None


@dataclass
class ORCIDEmployment:
    """employment record from ORCID."""
    organization: str
    role: Optional[str] = None  # Professor, Postdoc, etc.
    department: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None  # None = current
    is_current: bool = False


@dataclass
class ORCIDWork:
    """work record from ORCID with type information."""
    title: str
    year: Optional[int] = None
    doi: Optional[str] = None
    work_type: str = "other"  # journal-article, conference-paper, etc.
    journal: Optional[str] = None
    is_conference: bool = False
    is_preprint: bool = False


@dataclass
class ORCIDFunding:
    """funding record from ORCID."""
    title: str
    funder: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grant_number: Optional[str] = None


@dataclass
class ORCIDProfile:
    """comprehensive ORCID profile."""
    orcid: str
    name: str

    # basic info
    biography: Optional[str] = None
    keywords: List[str] = None
    urls: List[Dict[str, str]] = None  # [{name, url}]

    # career
    educations: List[ORCIDEducation] = None
    employments: List[ORCIDEmployment] = None

    # works
    works: List[ORCIDWork] = None
    total_works: int = 0
    journal_articles: int = 0
    conference_papers: int = 0
    preprints: int = 0

    # funding
    fundings: List[ORCIDFunding] = None
    total_funding_count: int = 0

    # peer review
    peer_review_count: int = 0

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.urls is None:
            self.urls = []
        if self.educations is None:
            self.educations = []
        if self.employments is None:
            self.employments = []
        if self.works is None:
            self.works = []
        if self.fundings is None:
            self.fundings = []


class ORCIDProvider:
    """
    ORCID API client for comprehensive author identity resolution.

    extracts:
    - basic identity (name, bio, keywords)
    - education history (degrees, institutions)
    - employment history (positions, dates)
    - works with types (journal, conference, preprint)
    - funding information
    - peer review activity

    usage:
        provider = ORCIDProvider()
        profile = provider.get_full_profile("0000-0002-1234-5678")

        print(f"PhD from: {profile.educations[0].institution}")
        print(f"Conference papers: {profile.conference_papers}")
    """

    # ORCID work type mapping
    WORK_TYPE_MAP = {
        'journal-article': 'journal-article',
        'conference-paper': 'conference-paper',
        'conference-abstract': 'conference-paper',
        'conference-poster': 'conference-paper',
        'book': 'book',
        'book-chapter': 'book-chapter',
        'dissertation': 'dissertation',
        'preprint': 'preprint',
        'working-paper': 'preprint',
        'report': 'report',
        'patent': 'patent',
        'other': 'other',
    }

    def __init__(self, rate_limit_delay: float = 1.0):
        self.base_url = "https://pub.orcid.org/v3.0"
        self.delay = rate_limit_delay
        self._last_request = 0

    def _rate_limit(self):
        """apply rate limiting between requests."""
        import time
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def _normalize_orcid(self, orcid: str) -> str:
        """normalize ORCID to bare format."""
        orcid = orcid.strip()
        if orcid.startswith("https://orcid.org/"):
            return orcid[18:]
        if orcid.startswith("http://orcid.org/"):
            return orcid[17:]
        return orcid

    def _request(self, endpoint: str) -> Optional[Dict]:
        """make rate-limited request to ORCID API."""
        import httpx

        self._rate_limit()

        try:
            url = f"{self.base_url}/{endpoint}"
            headers = {"Accept": "application/json"}
            resp = httpx.get(url, headers=headers, timeout=30)

            if resp.status_code == 200:
                return resp.json()
            else:
                return None

        except Exception as e:
            print(f"[orcid] error fetching {endpoint}: {e}")
            return None

    def get_author_by_orcid(self, orcid: str) -> Optional[AuthorInfo]:
        """fetch basic author info from ORCID (backward compatible)."""
        orcid = self._normalize_orcid(orcid)
        data = self._request(f"{orcid}/record")

        if not data:
            return None

        return self._parse_basic_info(data, orcid)

    def get_full_profile(self, orcid: str) -> Optional[ORCIDProfile]:
        """fetch comprehensive ORCID profile."""
        orcid = self._normalize_orcid(orcid)
        data = self._request(f"{orcid}/record")

        if not data:
            return None

        return self._parse_full_profile(data, orcid)

    def get_works(self, orcid: str, limit: int = 100) -> List[ORCIDWork]:
        """fetch works with type information."""
        orcid = self._normalize_orcid(orcid)
        data = self._request(f"{orcid}/works")

        if not data:
            return []

        return self._parse_works_detailed(data, limit)

    def get_educations(self, orcid: str) -> List[ORCIDEducation]:
        """fetch education history."""
        orcid = self._normalize_orcid(orcid)
        data = self._request(f"{orcid}/educations")

        if not data:
            return []

        return self._parse_educations(data)

    def get_employments(self, orcid: str) -> List[ORCIDEmployment]:
        """fetch employment history."""
        orcid = self._normalize_orcid(orcid)
        data = self._request(f"{orcid}/employments")

        if not data:
            return []

        return self._parse_employments(data)

    def get_fundings(self, orcid: str) -> List[ORCIDFunding]:
        """fetch funding information."""
        orcid = self._normalize_orcid(orcid)
        data = self._request(f"{orcid}/fundings")

        if not data:
            return []

        return self._parse_fundings(data)

    # parsing methods

    def _parse_basic_info(self, data: Dict, orcid: str) -> AuthorInfo:
        """parse basic author info (backward compatible)."""
        person = data.get("person", {})
        name_data = person.get("name", {})

        given = name_data.get("given-names", {}).get("value", "") if name_data.get("given-names") else ""
        family = name_data.get("family-name", {}).get("value", "") if name_data.get("family-name") else ""
        name = f"{given} {family}".strip()

        # affiliations from employments
        affiliations = []
        activities = data.get("activities-summary", {})
        employments = activities.get("employments", {}).get("affiliation-group", [])
        for emp in employments[:5]:
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

    def _parse_full_profile(self, data: Dict, orcid: str) -> ORCIDProfile:
        """parse comprehensive ORCID profile."""
        person = data.get("person", {})
        activities = data.get("activities-summary", {})

        # name
        name_data = person.get("name", {})
        given = name_data.get("given-names", {}).get("value", "") if name_data.get("given-names") else ""
        family = name_data.get("family-name", {}).get("value", "") if name_data.get("family-name") else ""
        name = f"{given} {family}".strip()

        # biography
        bio_data = person.get("biography", {})
        biography = bio_data.get("content") if bio_data else None

        # keywords
        keywords = []
        kw_data = person.get("keywords", {}).get("keyword", [])
        for kw in kw_data:
            content = kw.get("content", "")
            if content:
                keywords.append(content)

        # urls
        urls = []
        url_data = person.get("researcher-urls", {}).get("researcher-url", [])
        for u in url_data:
            urls.append({
                "name": u.get("url-name", ""),
                "url": u.get("url", {}).get("value", "")
            })

        # educations
        educations = self._parse_educations_from_summary(
            activities.get("educations", {})
        )

        # employments
        employments = self._parse_employments_from_summary(
            activities.get("employments", {})
        )

        # works summary (counts)
        works_summary = activities.get("works", {})
        total_works = works_summary.get("group", [])

        # funding count
        funding_summary = activities.get("fundings", {})
        funding_groups = funding_summary.get("group", [])

        # peer review count
        peer_review_summary = activities.get("peer-reviews", {})
        peer_review_groups = peer_review_summary.get("group", [])
        peer_review_count = sum(
            len(g.get("peer-review-group", []))
            for g in peer_review_groups
        )

        return ORCIDProfile(
            orcid=orcid,
            name=name,
            biography=biography,
            keywords=keywords,
            urls=urls,
            educations=educations,
            employments=employments,
            total_works=len(total_works),
            total_funding_count=len(funding_groups),
            peer_review_count=peer_review_count
        )

    def _parse_educations_from_summary(self, edu_data: Dict) -> List[ORCIDEducation]:
        """parse educations from activities summary."""
        educations = []
        groups = edu_data.get("affiliation-group", [])

        for group in groups:
            summaries = group.get("summaries", [])
            for s in summaries:
                edu_summary = s.get("education-summary", {})
                if not edu_summary:
                    continue

                org = edu_summary.get("organization", {})

                # parse dates
                start_year = None
                end_year = None
                start_date = edu_summary.get("start-date", {})
                end_date = edu_summary.get("end-date", {})

                if start_date and start_date.get("year"):
                    try:
                        start_year = int(start_date["year"]["value"])
                    except (ValueError, TypeError, KeyError):
                        pass

                if end_date and end_date.get("year"):
                    try:
                        end_year = int(end_date["year"]["value"])
                    except (ValueError, TypeError, KeyError):
                        pass

                educations.append(ORCIDEducation(
                    institution=org.get("name", ""),
                    degree=edu_summary.get("role-title"),
                    department=edu_summary.get("department-name"),
                    start_year=start_year,
                    end_year=end_year
                ))

        # sort by end year (most recent first)
        educations.sort(key=lambda e: e.end_year or 9999, reverse=True)
        return educations

    def _parse_employments_from_summary(self, emp_data: Dict) -> List[ORCIDEmployment]:
        """parse employments from activities summary."""
        employments = []
        groups = emp_data.get("affiliation-group", [])

        for group in groups:
            summaries = group.get("summaries", [])
            for s in summaries:
                emp_summary = s.get("employment-summary", {})
                if not emp_summary:
                    continue

                org = emp_summary.get("organization", {})

                # parse dates
                start_year = None
                end_year = None
                start_date = emp_summary.get("start-date", {})
                end_date = emp_summary.get("end-date", {})

                if start_date and start_date.get("year"):
                    try:
                        start_year = int(start_date["year"]["value"])
                    except (ValueError, TypeError, KeyError):
                        pass

                if end_date and end_date.get("year"):
                    try:
                        end_year = int(end_date["year"]["value"])
                    except (ValueError, TypeError, KeyError):
                        pass

                is_current = (end_year is None)

                employments.append(ORCIDEmployment(
                    organization=org.get("name", ""),
                    role=emp_summary.get("role-title"),
                    department=emp_summary.get("department-name"),
                    start_year=start_year,
                    end_year=end_year,
                    is_current=is_current
                ))

        # sort by start year (most recent first), current jobs at top
        employments.sort(key=lambda e: (not e.is_current, -(e.start_year or 0)))
        return employments

    def _parse_educations(self, data: Dict) -> List[ORCIDEducation]:
        """parse educations endpoint response."""
        return self._parse_educations_from_summary(data)

    def _parse_employments(self, data: Dict) -> List[ORCIDEmployment]:
        """parse employments endpoint response."""
        return self._parse_employments_from_summary(data)

    def _parse_works_detailed(self, data: Dict, limit: int) -> List[ORCIDWork]:
        """parse works with type information."""
        works = []
        groups = data.get("group", [])

        for group in groups[:limit]:
            work_summary = group.get("work-summary", [])
            if not work_summary:
                continue

            ws = work_summary[0]

            # title
            title_obj = ws.get("title", {}).get("title", {})
            title = title_obj.get("value", "") if title_obj else ""

            # work type
            work_type_raw = ws.get("type", "other")
            work_type = self.WORK_TYPE_MAP.get(work_type_raw, "other")

            is_conference = work_type == "conference-paper"
            is_preprint = work_type == "preprint"

            # DOI and other identifiers
            doi = None
            external_ids = ws.get("external-ids", {}).get("external-id", [])
            for eid in external_ids:
                if eid.get("external-id-type") == "doi":
                    doi = eid.get("external-id-value")
                    break

            # year
            year = None
            pub_date = ws.get("publication-date", {})
            if pub_date and pub_date.get("year"):
                try:
                    year = int(pub_date["year"]["value"])
                except (ValueError, TypeError, KeyError):
                    pass

            # journal/venue
            journal = ws.get("journal-title", {}).get("value") if ws.get("journal-title") else None

            works.append(ORCIDWork(
                title=title,
                year=year,
                doi=doi,
                work_type=work_type,
                journal=journal,
                is_conference=is_conference,
                is_preprint=is_preprint
            ))

        return works

    def _parse_fundings(self, data: Dict) -> List[ORCIDFunding]:
        """parse funding information."""
        fundings = []
        groups = data.get("group", [])

        for group in groups:
            summaries = group.get("funding-summary", [])
            for fs in summaries:
                # title
                title_obj = fs.get("title", {}).get("title", {})
                title = title_obj.get("value", "") if title_obj else ""

                # funder
                org = fs.get("organization", {})
                funder = org.get("name", "")

                # amount
                amount_obj = fs.get("amount", {})
                amount = None
                currency = None
                if amount_obj:
                    try:
                        amount = float(amount_obj.get("value", 0))
                    except (ValueError, TypeError):
                        pass
                    currency = amount_obj.get("currency-code")

                # dates
                start_year = None
                end_year = None
                start_date = fs.get("start-date", {})
                end_date = fs.get("end-date", {})

                if start_date and start_date.get("year"):
                    try:
                        start_year = int(start_date["year"]["value"])
                    except (ValueError, TypeError, KeyError):
                        pass

                if end_date and end_date.get("year"):
                    try:
                        end_year = int(end_date["year"]["value"])
                    except (ValueError, TypeError, KeyError):
                        pass

                # grant number
                grant_number = None
                external_ids = fs.get("external-ids", {}).get("external-id", [])
                for eid in external_ids:
                    if eid.get("external-id-type") == "grant_number":
                        grant_number = eid.get("external-id-value")
                        break

                fundings.append(ORCIDFunding(
                    title=title,
                    funder=funder,
                    amount=amount,
                    currency=currency,
                    start_year=start_year,
                    end_year=end_year,
                    grant_number=grant_number
                ))

        return fundings
