"""
semantic scholar provider implementation.
docs: https://api.semanticscholar.org/api-docs/
uses API key from SEMANTIC_SCHOLAR_API_KEY env var for higher rate limits.
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any

from .base import PaperProvider, PaperStub


class SemanticScholarProvider(PaperProvider):
    """
    semantic scholar api client.
    with api key: 100 req/sec
    without: 100 req/5 min
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        # with key: 100 req/sec, without: ~0.3 req/sec
        self.min_delay = 0.01 if self.api_key else 3.0
        self.last_request_time = 0

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
        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, headers=headers, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # rate limited
                    wait_time = 2 ** (attempt + 1)
                    print(f"[s2] rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"[s2] server error {response.status_code}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[s2] error: {response.status_code} - {response.text[:200]}")
                    return {}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[s2] timeout, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        return {}

    def _parse_paper(self, paper: Dict[str, Any]) -> PaperStub:
        """convert s2 paper to PaperStub."""
        # extract identifiers
        external_ids = paper.get('externalIds', {}) or {}
        doi = external_ids.get('DOI')

        # s2 paper id
        s2_id = paper.get('paperId')

        # authors (first 5)
        authors = []
        for author in (paper.get('authors') or [])[:5]:
            name = author.get('name', '')
            if name:
                authors.append(name)

        # venue
        venue = paper.get('venue') or paper.get('journal', {}).get('name') if paper.get('journal') else None

        # check if review (heuristic)
        title = paper.get('title') or ''
        pub_types = paper.get('publicationTypes') or []
        is_review = 'Review' in pub_types or 'review' in title.lower()

        # open access
        is_oa = paper.get('isOpenAccess', False)
        oa_url = paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else None

        return PaperStub(
            doi=doi,
            s2_id=s2_id,
            title=title,
            year=paper.get('year'),
            venue=venue,
            authors=authors,
            citation_count=paper.get('citationCount'),
            reference_count=paper.get('referenceCount'),
            is_review=is_review,
            is_oa=is_oa,
            oa_pdf_url=oa_url,
            abstract=paper.get('abstract')
        )

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[PaperStub]:
        """search for papers matching query."""
        # s2 uses year range as "YYYY-YYYY" format
        year_filter = None
        if year_min and year_max:
            year_filter = f"{year_min}-{year_max}"
        elif year_min:
            year_filter = f"{year_min}-"
        elif year_max:
            year_filter = f"-{year_max}"

        params = {
            'query': query,
            'limit': min(limit, 100),  # s2 max is 100 per request
            'fields': 'paperId,externalIds,title,abstract,venue,year,authors,citationCount,referenceCount,isOpenAccess,openAccessPdf,publicationTypes,journal'
        }

        if year_filter:
            params['year'] = year_filter

        response = self._make_request('/paper/search', params)
        papers = []

        for paper in response.get('data', []):
            papers.append(self._parse_paper(paper))

        # handle pagination if needed
        total = response.get('total', 0)
        offset = response.get('offset', 0) + len(papers)

        while len(papers) < limit and offset < total:
            params['offset'] = offset
            response = self._make_request('/paper/search', params)
            for paper in response.get('data', []):
                papers.append(self._parse_paper(paper))
            offset += len(response.get('data', []))
            if not response.get('data'):
                break

        return papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[PaperStub]:
        """get paper by id (doi, s2_id, arxiv, etc)."""
        # s2 accepts various id formats
        if paper_id.startswith('doi:'):
            endpoint = f"/paper/DOI:{paper_id[4:]}"
        elif paper_id.startswith('s2:'):
            endpoint = f"/paper/{paper_id[3:]}"
        elif paper_id.startswith('10.'):  # raw doi
            endpoint = f"/paper/DOI:{paper_id}"
        elif paper_id.startswith('arxiv:'):
            endpoint = f"/paper/ARXIV:{paper_id[6:]}"
        else:
            endpoint = f"/paper/{paper_id}"

        params = {
            'fields': 'paperId,externalIds,title,abstract,venue,year,authors,citationCount,referenceCount,isOpenAccess,openAccessPdf,publicationTypes,journal'
        }

        response = self._make_request(endpoint, params)
        if response and 'paperId' in response:
            return self._parse_paper(response)
        return None

    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """get papers cited by this paper (backward citations)."""
        # normalize id
        if paper_id.startswith('doi:'):
            endpoint = f"/paper/DOI:{paper_id[4:]}/references"
        elif paper_id.startswith('s2:'):
            endpoint = f"/paper/{paper_id[3:]}/references"
        elif paper_id.startswith('10.'):
            endpoint = f"/paper/DOI:{paper_id}/references"
        else:
            endpoint = f"/paper/{paper_id}/references"

        params = {
            'fields': 'paperId,externalIds,title,abstract,venue,year,authors,citationCount,referenceCount,isOpenAccess,openAccessPdf',
            'limit': min(limit, 1000)
        }

        response = self._make_request(endpoint, params)
        papers = []

        for item in response.get('data', []):
            cited_paper = item.get('citedPaper', {})
            if cited_paper and cited_paper.get('paperId'):
                papers.append(self._parse_paper(cited_paper))

        return papers[:limit]

    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """get papers that cite this paper (forward citations)."""
        # normalize id
        if paper_id.startswith('doi:'):
            endpoint = f"/paper/DOI:{paper_id[4:]}/citations"
        elif paper_id.startswith('s2:'):
            endpoint = f"/paper/{paper_id[3:]}/citations"
        elif paper_id.startswith('10.'):
            endpoint = f"/paper/DOI:{paper_id}/citations"
        else:
            endpoint = f"/paper/{paper_id}/citations"

        params = {
            'fields': 'paperId,externalIds,title,abstract,venue,year,authors,citationCount,referenceCount,isOpenAccess,openAccessPdf',
            'limit': min(limit, 1000)
        }

        response = self._make_request(endpoint, params)
        papers = []

        for item in response.get('data', []):
            citing_paper = item.get('citingPaper', {})
            if citing_paper and citing_paper.get('paperId'):
                papers.append(self._parse_paper(citing_paper))

        return papers[:limit]

    def get_count_estimate(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> int:
        """get estimated count of papers matching query."""
        year_filter = None
        if year_min and year_max:
            year_filter = f"{year_min}-{year_max}"
        elif year_min:
            year_filter = f"{year_min}-"

        params = {
            'query': query,
            'limit': 1,
            'fields': 'paperId'
        }

        if year_filter:
            params['year'] = year_filter

        response = self._make_request('/paper/search', params)
        return response.get('total', 0)


# simple test
if __name__ == "__main__":
    provider = SemanticScholarProvider()

    print("testing s2 search...")
    papers = provider.search_papers("ancestral protein reconstruction", year_min=2020, limit=5)
    for p in papers:
        print(f"  - {p.title[:60]}... ({p.year}) [cited: {p.citation_count}]")

    print("\ntesting s2 count...")
    count = provider.get_count_estimate("ancestral protein reconstruction", year_min=2020)
    print(f"  count: {count}")
