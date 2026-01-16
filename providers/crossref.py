"""
crossref provider for DOI resolution and reference fetching.
docs: https://api.crossref.org/swagger-ui/index.html
free, polite pool with email (50 req/sec).
"""

import time
import requests
from typing import List, Optional, Dict, Any

from .base import PaperProvider, PaperStub


class CrossrefProvider(PaperProvider):
    """
    crossref api client.
    polite pool with email: 50 req/sec
    without: rate limited more aggressively.
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(self, email: str = "kiran@mcneese.edu"):
        self.email = email
        self.min_delay = 0.02  # 50 req/sec for polite pool
        self.last_request_time = 0
        self.headers = {
            'User-Agent': f'RefNetApp/1.0 (mailto:{email})'
        }

    def _rate_limit(self):
        """ensure we don't exceed rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, max_retries: int = 3) -> Dict[str, Any]:
        """make api request with retry logic."""
        if params is None:
            params = {}

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, headers=self.headers, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {}  # not found
                elif response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[crossref] rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"[crossref] server error {response.status_code}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[crossref] error: {response.status_code}")
                    return {}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[crossref] timeout, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        return {}

    def _parse_work(self, work: Dict[str, Any]) -> PaperStub:
        """parse crossref work to PaperStub."""
        # doi
        doi = work.get('DOI')

        # title (usually a list)
        titles = work.get('title', [])
        title = titles[0] if titles else ''

        # authors
        authors = []
        for author in work.get('author', [])[:5]:
            given = author.get('given', '')
            family = author.get('family', '')
            if family:
                name = f"{given} {family}".strip()
                authors.append(name)

        # year
        year = None
        if work.get('published-print'):
            parts = work['published-print'].get('date-parts', [[]])[0]
            if parts:
                year = parts[0]
        elif work.get('published-online'):
            parts = work['published-online'].get('date-parts', [[]])[0]
            if parts:
                year = parts[0]
        elif work.get('created'):
            parts = work['created'].get('date-parts', [[]])[0]
            if parts:
                year = parts[0]

        # venue
        venue = work.get('container-title', [''])[0] if work.get('container-title') else ''

        # citation count (crossref has "is-referenced-by-count")
        citation_count = work.get('is-referenced-by-count', 0)

        # reference count
        reference_count = work.get('references-count', 0)

        # type
        work_type = work.get('type', '')
        is_review = 'review' in work_type.lower() or 'review' in title.lower()

        # abstract
        abstract = work.get('abstract', '')
        # strip html tags from abstract
        if abstract:
            import re
            abstract = re.sub(r'<[^>]+>', '', abstract)

        return PaperStub(
            doi=doi,
            title=title,
            year=year,
            venue=venue,
            authors=authors,
            citation_count=citation_count,
            reference_count=reference_count,
            is_review=is_review,
            abstract=abstract
        )

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[PaperStub]:
        """search crossref for papers matching query."""
        params = {
            'query': query,
            'rows': min(limit, 1000),
            'sort': 'is-referenced-by-count',
            'order': 'desc'
        }

        # date filter
        filters = []
        if year_min:
            filters.append(f"from-pub-date:{year_min}")
        if year_max:
            filters.append(f"until-pub-date:{year_max}")

        if filters:
            params['filter'] = ','.join(filters)

        response = self._make_request('/works', params)
        items = response.get('message', {}).get('items', [])

        papers = []
        for item in items[:limit]:
            papers.append(self._parse_work(item))

        return papers

    def get_paper(self, paper_id: str) -> Optional[PaperStub]:
        """get paper by DOI."""
        # normalize doi
        doi = paper_id
        if doi.startswith('doi:'):
            doi = doi[4:]
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]

        response = self._make_request(f'/works/{doi}')
        work = response.get('message', {})

        if work and work.get('DOI'):
            return self._parse_work(work)
        return None

    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """get papers cited by this paper (from crossref references)."""
        # normalize doi
        doi = paper_id
        if doi.startswith('doi:'):
            doi = doi[4:]
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]

        response = self._make_request(f'/works/{doi}')
        work = response.get('message', {})

        if not work:
            return []

        references = work.get('reference', [])
        papers = []

        for ref in references[:limit]:
            # try to get DOI from reference
            ref_doi = ref.get('DOI')

            if ref_doi:
                # fetch full metadata
                paper = self.get_paper(ref_doi)
                if paper:
                    papers.append(paper)
            else:
                # create stub from reference data
                paper = PaperStub(
                    title=ref.get('article-title') or ref.get('unstructured', '')[:200],
                    year=int(ref.get('year')) if ref.get('year') else None,
                    venue=ref.get('journal-title'),
                    authors=[ref.get('author', '')] if ref.get('author') else []
                )
                if paper.title:
                    papers.append(paper)

        return papers

    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """
        crossref doesn't provide forward citations directly.
        use semantic scholar or openalex for this.
        """
        return []

    def get_count_estimate(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> int:
        """get estimated count of papers matching query."""
        params = {
            'query': query,
            'rows': 0  # just want the count
        }

        filters = []
        if year_min:
            filters.append(f"from-pub-date:{year_min}")
        if year_max:
            filters.append(f"until-pub-date:{year_max}")

        if filters:
            params['filter'] = ','.join(filters)

        response = self._make_request('/works', params)
        return response.get('message', {}).get('total-results', 0)

    def resolve_doi(self, doi: str) -> Optional[PaperStub]:
        """resolve doi to paper metadata (alias for get_paper)."""
        return self.get_paper(doi)


# simple test
if __name__ == "__main__":
    provider = CrossrefProvider()

    print("testing crossref search...")
    papers = provider.search_papers("ancestral protein reconstruction", year_min=2020, limit=5)
    for p in papers:
        print(f"  - {p.title[:60]}... ({p.year}) [cited: {p.citation_count}]")

    print("\ntesting crossref doi resolution...")
    paper = provider.get_paper("10.1038/s41586-021-03819-2")
    if paper:
        print(f"  - {paper.title}")
