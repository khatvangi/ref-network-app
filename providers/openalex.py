"""
openalex provider implementation.
docs: https://docs.openalex.org/
"""

import time
import requests
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin

from .base import PaperProvider, PaperStub


class OpenAlexProvider(PaperProvider):
    """
    openalex api client with rate limiting.
    uses polite pool (10 req/sec) when email is provided.
    """

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: str = "kiran@mcneese.edu"):
        self.email = email
        self.min_delay = 0.1  # 10 req/sec for polite pool
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

        # add email for polite pool
        params['mailto'] = self.email
        url = urljoin(self.BASE_URL, endpoint)

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    # rate limited, wait and retry
                    wait_time = 2 ** attempt
                    print(f"[openalex] rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"[openalex] server error {response.status_code}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[openalex] error: {response.status_code} - {response.text[:200]}")
                    return {}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[openalex] timeout, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        return {}

    def _parse_work(self, work: Dict[str, Any]) -> PaperStub:
        """convert openalex work to PaperStub."""
        # extract doi (remove https://doi.org/ prefix)
        doi = work.get('doi', '')
        if doi and doi.startswith('https://doi.org/'):
            doi = doi[16:]

        # extract openalex id (just the W... part)
        oaid = work.get('id', '')
        if oaid and '/' in oaid:
            oaid = oaid.split('/')[-1]

        # extract authors (first 5)
        authors = []
        for authorship in work.get('authorships', [])[:5]:
            author = authorship.get('author', {})
            name = author.get('display_name', '')
            if name:
                authors.append(name)

        # check if review (heuristic: type or title contains 'review')
        work_type = work.get('type', '')
        title = work.get('title', '') or ''
        is_review = work_type == 'review' or 'review' in title.lower()

        # open access info
        oa_info = work.get('open_access', {})
        is_oa = oa_info.get('is_oa', False)
        oa_pdf_url = oa_info.get('oa_url')

        # concepts/topics
        concepts = []
        for concept in work.get('concepts', [])[:10]:
            concepts.append({
                'name': concept.get('display_name', ''),
                'score': concept.get('score', 0)
            })

        # also include topics if available (newer openalex schema)
        for topic in work.get('topics', [])[:5]:
            concepts.append({
                'name': topic.get('display_name', ''),
                'score': topic.get('score', 0)
            })

        # venue/source
        source = work.get('primary_location', {}).get('source', {}) or {}
        venue = source.get('display_name', '')

        return PaperStub(
            doi=doi if doi else None,
            openalex_id=oaid if oaid else None,
            title=title,
            year=work.get('publication_year'),
            venue=venue,
            authors=authors,
            citation_count=work.get('cited_by_count', 0),
            reference_count=work.get('referenced_works_count', 0),
            is_review=is_review,
            is_oa=is_oa,
            oa_pdf_url=oa_pdf_url,
            abstract=work.get('abstract'),
            concepts=concepts
        )

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[PaperStub]:
        """search for papers matching query."""
        params = {
            'search': query,
            'per-page': min(limit, 200),
            'sort': 'cited_by_count:desc'  # prefer highly cited
        }

        # build filter string
        filters = []
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")
        elif year_max:
            filters.append(f"publication_year:<{year_max + 1}")

        if filters:
            params['filter'] = ','.join(filters)

        response = self._make_request('/works', params)
        results = response.get('results', [])

        papers = []
        for work in results[:limit]:
            papers.append(self._parse_work(work))

        return papers

    def get_paper(self, paper_id: str) -> Optional[PaperStub]:
        """get paper by id."""
        # handle different id formats
        if paper_id.startswith('doi:'):
            endpoint = f"/works/https://doi.org/{paper_id[4:]}"
        elif paper_id.startswith('oaid:'):
            endpoint = f"/works/{paper_id[5:]}"
        elif paper_id.startswith('10.'):  # raw doi
            endpoint = f"/works/https://doi.org/{paper_id}"
        elif paper_id.startswith('W'):  # raw openalex id
            endpoint = f"/works/{paper_id}"
        else:
            endpoint = f"/works/{paper_id}"

        response = self._make_request(endpoint)
        if response and 'id' in response:
            return self._parse_work(response)
        return None

    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """get papers cited by this paper (backward citations)."""
        # first get the paper to find its referenced_works
        paper = self.get_paper(paper_id)
        if not paper or not paper.openalex_id:
            return []

        # use filter to get referenced works
        params = {
            'filter': f"cited_by:{paper.openalex_id}",
            'per-page': min(limit, 200),
            'sort': 'cited_by_count:desc'
        }

        # actually, openalex uses a different approach
        # we need to get the work and then fetch its referenced_works
        endpoint = f"/works/{paper.openalex_id}"
        work = self._make_request(endpoint)

        referenced_ids = work.get('referenced_works', [])[:limit]
        if not referenced_ids:
            return []

        # batch fetch referenced works (up to 50 per request)
        papers = []
        for i in range(0, len(referenced_ids), 50):
            batch = referenced_ids[i:i+50]
            # extract just the id part
            batch_ids = [rid.split('/')[-1] for rid in batch]
            filter_str = '|'.join(batch_ids)

            params = {
                'filter': f"openalex_id:{filter_str}",
                'per-page': 50
            }
            response = self._make_request('/works', params)
            for w in response.get('results', []):
                papers.append(self._parse_work(w))

        return papers[:limit]

    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """get papers that cite this paper (forward citations)."""
        paper = self.get_paper(paper_id)
        if not paper or not paper.openalex_id:
            return []

        params = {
            'filter': f"cites:{paper.openalex_id}",
            'per-page': min(limit, 200),
            'sort': 'cited_by_count:desc'  # prefer highly cited citing papers
        }

        response = self._make_request('/works', params)
        papers = []
        for work in response.get('results', [])[:limit]:
            papers.append(self._parse_work(work))

        return papers

    def get_count_estimate(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> int:
        """get estimated count of papers matching query."""
        params = {
            'search': query,
            'per-page': 1  # we only need the count from meta
        }

        filters = []
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")
        elif year_max:
            filters.append(f"publication_year:<{year_max + 1}")

        if filters:
            params['filter'] = ','.join(filters)

        response = self._make_request('/works', params)
        return response.get('meta', {}).get('count', 0)

    def get_top_concepts(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """get top concepts for a query (for triage)."""
        params = {
            'search': query,
            'group_by': 'concepts.id'
        }

        filters = []
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")

        if filters:
            params['filter'] = ','.join(filters)

        response = self._make_request('/works', params)
        groups = response.get('group_by', [])[:limit]

        concepts = []
        for g in groups:
            concepts.append({
                'name': g.get('key_display_name', ''),
                'count': g.get('count', 0)
            })
        return concepts

    def get_top_venues(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """get top venues for a query (for triage)."""
        params = {
            'search': query,
            'group_by': 'primary_location.source.id'
        }

        filters = []
        if year_min and year_max:
            filters.append(f"publication_year:{year_min}-{year_max}")
        elif year_min:
            filters.append(f"publication_year:>{year_min - 1}")

        if filters:
            params['filter'] = ','.join(filters)

        response = self._make_request('/works', params)
        groups = response.get('group_by', [])[:limit]

        venues = []
        for g in groups:
            venues.append({
                'name': g.get('key_display_name', ''),
                'count': g.get('count', 0)
            })
        return venues


# simple test
if __name__ == "__main__":
    provider = OpenAlexProvider()

    # test search
    print("testing search...")
    papers = provider.search_papers("ancestral protein reconstruction", year_min=2020, limit=5)
    for p in papers:
        print(f"  - {p.title[:60]}... ({p.year}) [cited: {p.citation_count}]")

    # test count
    print("\ntesting count estimate...")
    count = provider.get_count_estimate("ancestral protein reconstruction", year_min=2020)
    print(f"  count: {count}")

    # test concepts
    print("\ntesting top concepts...")
    concepts = provider.get_top_concepts("ancestral protein reconstruction", year_min=2020)
    for c in concepts[:5]:
        print(f"  - {c['name']}: {c['count']}")
