"""
google scholar provider using scholarly package.
WARNING: google scholar has aggressive anti-bot measures.
uses conservative rate limiting to avoid blocks.

options:
1. scholarly (free but slow, risk of blocks)
2. SerpAPI (paid but reliable)
3. SearXNG (self-hosted, free)
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any

from .base import PaperProvider, AuthorInfo
from ..core.models import Paper

logger = logging.getLogger("refnet.providers.gscholar")

# try to import scholarly
try:
    from scholarly import scholarly, ProxyGenerator
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False
    logger.warning("scholarly package not available - pip install scholarly")


class GoogleScholarProvider(PaperProvider):
    """
    google scholar provider using scholarly package.
    very slow due to rate limiting (configurable delay to avoid blocks).
    optional serpapi support for faster, reliable access.
    """

    def __init__(
        self,
        use_proxy: bool = False,
        serpapi_key: Optional[str] = None,
        searxng_url: Optional[str] = None,
        rate_limit_delay: float = 5.0  # seconds between requests
    ):
        self.serpapi_key = serpapi_key or os.environ.get('SERPAPI_KEY')
        self.searxng_url = searxng_url or os.environ.get('SEARXNG_URL')
        self.use_serpapi = bool(self.serpapi_key)
        self.use_searxng = bool(self.searxng_url) and not self.use_serpapi
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self._call_count = 0

        if not self.use_serpapi and not self.use_searxng and SCHOLARLY_AVAILABLE and use_proxy:
            # set up proxy for scholarly to avoid blocks
            try:
                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)
                logger.info("using free proxy for scholarly")
            except Exception as e:
                logger.warning(f"proxy setup failed: {e}")

    @property
    def name(self) -> str:
        if self.use_serpapi:
            return "google_scholar_serpapi"
        elif self.use_searxng:
            return "google_scholar_searxng"
        return "google_scholar"

    def _rate_limit(self):
        """conservative rate limiting to avoid google blocks."""
        if self.use_serpapi or self.use_searxng:
            time.sleep(0.5)  # minimal delay for API
            return

        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _parse_to_paper(self, result: Dict, source: str = "scholarly") -> Paper:
        """convert search result to Paper object."""
        if source == "serpapi":
            pub_info = result.get('publication_info', {})
            title = result.get('title', '')
            year = None
            try:
                # serpapi sometimes has year in summary
                summary = pub_info.get('summary', '')
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', summary)
                if year_match:
                    year = int(year_match.group())
            except (ValueError, TypeError, AttributeError):
                pass

            authors = []
            if pub_info.get('authors'):
                authors = [a.get('name', '') for a in pub_info.get('authors', [])]

            citation_count = None
            cited_by = result.get('inline_links', {}).get('cited_by', {})
            if cited_by:
                citation_count = cited_by.get('total')

            # try to get pdf
            oa_pdf_url = None
            resources = result.get('resources', [])
            if resources:
                oa_pdf_url = resources[0].get('link')

            return Paper(
                id=f"gs:{title[:50]}",
                title=title,
                year=year,
                venue=pub_info.get('journal', ''),
                authors=authors[:10],
                citation_count=citation_count,
                abstract=result.get('snippet'),
                oa_pdf_url=oa_pdf_url
            )

        elif source == "searxng":
            title = result.get('title', '')
            content = result.get('content', '')

            # try to extract year from content
            year = None
            try:
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', content)
                if year_match:
                    year = int(year_match.group())
            except (ValueError, TypeError, AttributeError):
                pass

            return Paper(
                id=f"gs:{title[:50]}",
                title=title,
                year=year,
                abstract=content,
                url=result.get('url')
            )

        else:  # scholarly
            bib = result.get('bib', {})

            title = bib.get('title', '')
            authors = []
            author_data = bib.get('author')
            if author_data:
                # handle both list and string formats
                if isinstance(author_data, list):
                    authors = author_data
                else:
                    authors = author_data.split(' and ')

            year = None
            if bib.get('pub_year'):
                try:
                    year = int(bib.get('pub_year'))
                except (ValueError, TypeError):
                    pass

            venue = bib.get('venue', '') or bib.get('journal', '')
            citation_count = result.get('num_citations', 0)
            eprint_url = result.get('eprint_url')

            return Paper(
                id=f"gs:{title[:50]}",
                title=title,
                year=year,
                venue=venue,
                authors=authors[:10],
                citation_count=citation_count,
                abstract=bib.get('abstract'),
                oa_pdf_url=eprint_url
            )

    def _searxng_search(self, query: str, limit: int) -> List[Paper]:
        """search using self-hosted SearXNG instance."""
        import httpx

        papers = []
        try:
            params = {
                'q': f"{query} site:scholar.google.com",
                'format': 'json',
                'engines': 'google scholar',
                'pageno': 1
            }

            resp = httpx.get(
                f"{self.searxng_url}/search",
                params=params,
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                for result in data.get('results', [])[:limit]:
                    paper = self._parse_to_paper(result, source="searxng")
                    papers.append(paper)
                    self._call_count += 1

        except Exception as e:
            logger.error(f"searxng search failed: {e}")

        return papers

    def _serpapi_search(
        self,
        query: str,
        year_min: Optional[int],
        year_max: Optional[int],
        limit: int
    ) -> List[Paper]:
        """search using serpapi (paid but reliable)."""
        import httpx

        papers = []
        start = 0

        while len(papers) < limit:
            params = {
                'engine': 'google_scholar',
                'q': query,
                'api_key': self.serpapi_key,
                'start': start,
                'num': min(20, limit - len(papers))
            }

            if year_min:
                params['as_ylo'] = year_min
            if year_max:
                params['as_yhi'] = year_max

            try:
                self._rate_limit()
                resp = httpx.get(
                    'https://serpapi.com/search',
                    params=params,
                    timeout=30
                )
                self._call_count += 1

                if resp.status_code != 200:
                    logger.warning(f"serpapi error: {resp.status_code}")
                    break

                data = resp.json()
                results = data.get('organic_results', [])

                if not results:
                    break

                for r in results:
                    paper = self._parse_to_paper(r, source="serpapi")
                    papers.append(paper)

                start += len(results)

            except Exception as e:
                logger.error(f"serpapi search failed: {e}")
                break

        return papers[:limit]

    def _scholarly_search(
        self,
        query: str,
        year_min: Optional[int],
        year_max: Optional[int],
        limit: int
    ) -> List[Paper]:
        """search using scholarly package (free but slow)."""
        if not SCHOLARLY_AVAILABLE:
            logger.warning("scholarly package not available")
            return []

        papers = []
        try:
            # construct query with year range
            search_query = query
            if year_min and year_max:
                search_query = f"{query} {year_min}..{year_max}"
            elif year_min:
                search_query = f"{query} after:{year_min}"

            self._rate_limit()
            search_results = scholarly.search_pubs(search_query)
            self._call_count += 1

            for i, result in enumerate(search_results):
                if i >= limit:
                    break

                self._rate_limit()
                paper = self._parse_to_paper(result, source="scholarly")
                papers.append(paper)

                # progress indicator
                if (i + 1) % 10 == 0:
                    logger.info(f"fetched {i + 1}/{limit} papers...")

        except Exception as e:
            logger.error(f"scholarly search failed: {e}")

        return papers

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[Paper]:
        """search google scholar for papers."""
        if self.use_serpapi:
            return self._serpapi_search(query, year_min, year_max, limit)
        elif self.use_searxng:
            return self._searxng_search(query, limit)
        else:
            return self._scholarly_search(query, year_min, year_max, limit)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        get paper by title (google scholar doesn't have stable ids).
        searches for the title and returns first match.
        """
        # if it's a DOI, search for it
        if paper_id.startswith("10."):
            papers = self.search_papers(f'"{paper_id}"', limit=1)
        else:
            # assume title
            papers = self.search_papers(f'"{paper_id}"', limit=1)

        return papers[0] if papers else None

    def get_references(self, paper_id: str, limit: int = 50) -> List[Paper]:
        """
        google scholar doesn't provide references directly.
        would need to parse the paper PDF.
        """
        logger.debug("google scholar doesn't provide reference lists")
        return []

    def get_citations(self, paper_id: str, limit: int = 30) -> List[Paper]:
        """
        get papers that cite this paper.
        note: requires finding the paper first, then fetching citations.
        very slow due to rate limiting.
        """
        if not SCHOLARLY_AVAILABLE:
            logger.warning("scholarly not available for citation lookup")
            return []

        try:
            self._rate_limit()

            # search for the paper
            search_results = scholarly.search_pubs(f'"{paper_id}"')
            result = next(search_results, None)
            self._call_count += 1

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
                paper = self._parse_to_paper(cite, source="scholarly")
                papers.append(paper)
                self._call_count += 1

            return papers

        except Exception as e:
            logger.error(f"error getting citations: {e}")
            return []

    def get_count_estimate(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> int:
        """google scholar doesn't provide exact counts."""
        return -1

    def get_author(self, author_id: str) -> Optional[AuthorInfo]:
        """
        get author info from google scholar.
        author_id should be their google scholar profile id.
        """
        if not SCHOLARLY_AVAILABLE:
            return None

        try:
            self._rate_limit()
            author = scholarly.search_author_id(author_id)
            self._call_count += 1

            if author:
                author = scholarly.fill(author)
                return AuthorInfo(
                    name=author.get('name', ''),
                    affiliations=[author.get('affiliation', '')] if author.get('affiliation') else [],
                    citation_count=author.get('citedby'),
                    paper_count=len(author.get('publications', []))
                )
        except Exception as e:
            logger.error(f"error getting author: {e}")

        return None

    def resolve_author_id(
        self,
        name: str,
        affiliation: Optional[str] = None,
        coauthor_names: Optional[List[str]] = None
    ) -> Optional[AuthorInfo]:
        """search for author by name."""
        if not SCHOLARLY_AVAILABLE:
            return None

        try:
            self._rate_limit()
            query = name
            if affiliation:
                query = f"{name} {affiliation}"

            search_results = scholarly.search_author(query)
            result = next(search_results, None)
            self._call_count += 1

            if result:
                return AuthorInfo(
                    name=result.get('name', name),
                    affiliations=[result.get('affiliation', '')] if result.get('affiliation') else [],
                    citation_count=result.get('citedby')
                )

        except Exception as e:
            logger.error(f"error searching author: {e}")

        return None

    def get_author_works(
        self,
        author_id: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 50
    ) -> List[Paper]:
        """get papers by author from their google scholar profile."""
        if not SCHOLARLY_AVAILABLE:
            return []

        try:
            self._rate_limit()
            author = scholarly.search_author_id(author_id)
            self._call_count += 1

            if not author:
                return []

            author = scholarly.fill(author)
            publications = author.get('publications', [])

            papers = []
            for pub in publications[:limit]:
                self._rate_limit()
                try:
                    filled = scholarly.fill(pub)
                    paper = self._parse_to_paper(filled, source="scholarly")

                    # filter by year if specified
                    if year_min and paper.year and paper.year < year_min:
                        continue
                    if year_max and paper.year and paper.year > year_max:
                        continue

                    papers.append(paper)
                    self._call_count += 1
                except Exception as e:
                    logger.debug(f"failed to parse publication: {e}")
                    continue

            return papers

        except Exception as e:
            logger.error(f"error getting author works: {e}")
            return []

    def supports_authors(self) -> bool:
        return SCHOLARLY_AVAILABLE

    def stats(self) -> Dict[str, Any]:
        """return provider statistics."""
        return {
            'provider': self.name,
            'api_calls': self._call_count,
            'scholarly_available': SCHOLARLY_AVAILABLE,
            'serpapi_enabled': self.use_serpapi,
            'searxng_enabled': self.use_searxng
        }


# test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    provider = GoogleScholarProvider(rate_limit_delay=5.0)

    print("testing google scholar search (slow)...")
    papers = provider.search_papers("aminoacyl-tRNA synthetase evolution", year_min=2020, limit=3)
    for p in papers:
        print(f"  - {p.title[:60]}... ({p.year}) [cited: {p.citation_count}]")
