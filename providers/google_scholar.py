"""
google scholar provider using scholarly package.
WARNING: google scholar has aggressive anti-bot measures.
uses very conservative rate limiting (1 req/5 sec) to avoid blocks.
optional: use SerpAPI for reliable access (requires paid key).
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any

from .base import PaperProvider, PaperStub

# try to import scholarly
try:
    from scholarly import scholarly, ProxyGenerator
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False


class GoogleScholarProvider(PaperProvider):
    """
    google scholar client using scholarly package.
    very slow due to rate limiting (1 req/5 sec to avoid blocks).
    optional serpapi support for faster, reliable access.
    """

    def __init__(
        self,
        use_proxy: bool = False,
        serpapi_key: Optional[str] = None,
        rate_limit_delay: float = 5.0  # seconds between requests
    ):
        self.serpapi_key = serpapi_key or os.environ.get('SERPAPI_KEY')
        self.use_serpapi = bool(self.serpapi_key)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        if not self.use_serpapi and SCHOLARLY_AVAILABLE and use_proxy:
            # set up proxy for scholarly to avoid blocks
            try:
                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)
                print("[gscholar] using free proxy")
            except Exception as e:
                print(f"[gscholar] proxy setup failed: {e}")

    def _rate_limit(self):
        """conservative rate limiting to avoid google blocks."""
        if self.use_serpapi:
            return  # serpapi has its own rate limiting

        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _parse_scholarly_result(self, result: Dict) -> PaperStub:
        """parse scholarly result to PaperStub."""
        # fill with scholarly
        bib = result.get('bib', {})

        title = bib.get('title', '')
        authors = bib.get('author', '').split(' and ') if bib.get('author') else []
        year = None
        if bib.get('pub_year'):
            try:
                year = int(bib.get('pub_year'))
            except ValueError:
                pass

        venue = bib.get('venue', '') or bib.get('journal', '')

        # citation count
        citation_count = result.get('num_citations', 0)

        # extract eprint url (often links to pdf)
        eprint_url = result.get('eprint_url')

        return PaperStub(
            title=title,
            year=year,
            venue=venue,
            authors=authors[:5],
            citation_count=citation_count,
            is_oa=bool(eprint_url),
            oa_pdf_url=eprint_url,
            abstract=bib.get('abstract')
        )

    def _serpapi_search(self, query: str, year_min: Optional[int], year_max: Optional[int], limit: int) -> List[PaperStub]:
        """search using serpapi (paid but reliable)."""
        papers = []
        start = 0

        while len(papers) < limit:
            params = {
                'engine': 'google_scholar',
                'q': query,
                'api_key': self.serpapi_key,
                'start': start,
                'num': min(20, limit - len(papers))  # max 20 per request
            }

            if year_min:
                params['as_ylo'] = year_min
            if year_max:
                params['as_yhi'] = year_max

            try:
                response = requests.get('https://serpapi.com/search', params=params, timeout=30)
                if response.status_code != 200:
                    print(f"[gscholar/serpapi] error: {response.status_code}")
                    break

                data = response.json()
                results = data.get('organic_results', [])

                if not results:
                    break

                for r in results:
                    paper = PaperStub(
                        title=r.get('title', ''),
                        year=r.get('publication_info', {}).get('year'),
                        venue=r.get('publication_info', {}).get('journal'),
                        authors=r.get('publication_info', {}).get('authors', [])[:5],
                        citation_count=r.get('inline_links', {}).get('cited_by', {}).get('total'),
                        is_oa=bool(r.get('resources')),
                        oa_pdf_url=r.get('resources', [{}])[0].get('link') if r.get('resources') else None,
                        abstract=r.get('snippet')
                    )
                    papers.append(paper)

                start += len(results)

            except Exception as e:
                print(f"[gscholar/serpapi] error: {e}")
                break

        return papers[:limit]

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[PaperStub]:
        """search google scholar for papers."""
        if self.use_serpapi:
            return self._serpapi_search(query, year_min, year_max, limit)

        if not SCHOLARLY_AVAILABLE:
            print("[gscholar] scholarly package not available")
            return []

        papers = []
        try:
            self._rate_limit()

            # construct query with year range
            search_query = query
            if year_min and year_max:
                search_query = f"{query} {year_min}..{year_max}"
            elif year_min:
                search_query = f"{query} after:{year_min}"

            search_results = scholarly.search_pubs(search_query)

            for i, result in enumerate(search_results):
                if i >= limit:
                    break

                self._rate_limit()

                paper = self._parse_scholarly_result(result)
                papers.append(paper)

                # progress indicator for slow searches
                if (i + 1) % 10 == 0:
                    print(f"[gscholar] fetched {i + 1}/{limit} papers...")

        except Exception as e:
            print(f"[gscholar] error: {e}")

        return papers

    def get_paper(self, paper_id: str) -> Optional[PaperStub]:
        """
        get paper by title (google scholar doesn't have stable ids).
        searches for the title and returns first match.
        """
        if not SCHOLARLY_AVAILABLE and not self.use_serpapi:
            return None

        papers = self.search_papers(f'"{paper_id}"', limit=1)
        return papers[0] if papers else None

    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """
        google scholar doesn't provide references directly.
        would need to parse the paper itself.
        """
        return []

    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """
        get papers that cite this paper.
        note: requires finding the paper first, then fetching citations.
        very slow due to rate limiting.
        """
        if not SCHOLARLY_AVAILABLE:
            return []

        try:
            self._rate_limit()

            # search for the paper
            search_results = scholarly.search_pubs(f'"{paper_id}"')
            result = next(search_results, None)

            if not result:
                return []

            self._rate_limit()

            # get citations
            citations = scholarly.citedby(result)
            papers = []

            for i, cite in enumerate(citations):
                if i >= limit:
                    break
                self._rate_limit()
                paper = self._parse_scholarly_result(cite)
                papers.append(paper)

            return papers

        except Exception as e:
            print(f"[gscholar] error getting citations: {e}")
            return []

    def get_count_estimate(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> int:
        """
        google scholar doesn't provide exact counts.
        returns -1 to indicate unknown.
        """
        return -1


# simple test
if __name__ == "__main__":
    provider = GoogleScholarProvider(rate_limit_delay=5.0)

    print("testing google scholar search (slow)...")
    papers = provider.search_papers("ancestral protein reconstruction", year_min=2020, limit=3)
    for p in papers:
        print(f"  - {p.title[:60]}... ({p.year}) [cited: {p.citation_count}]")
