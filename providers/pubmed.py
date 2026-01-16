"""
pubmed provider using NCBI E-utilities.
docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
no api key required, but rate limited to 3 req/sec (10 with key).
"""

import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any

from .base import PaperProvider, PaperStub


class PubMedProvider(PaperProvider):
    """
    pubmed e-utilities api client.
    rate limit: 3 req/sec without key, 10 req/sec with key.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str = "kiran@mcneese.edu", api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key
        # 3 req/sec without key, 10 with key
        self.min_delay = 0.1 if api_key else 0.34
        self.last_request_time = 0

    def _rate_limit(self):
        """ensure we don't exceed rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict, max_retries: int = 3) -> requests.Response:
        """make api request with retry logic."""
        params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[pubmed] rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code >= 500:
                    wait_time = 2 ** attempt
                    print(f"[pubmed] server error {response.status_code}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[pubmed] error: {response.status_code}")
                    return None
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[pubmed] timeout, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        return None

    def _parse_article(self, article: ET.Element) -> PaperStub:
        """parse pubmed article xml to PaperStub."""
        medline = article.find('.//MedlineCitation')
        if medline is None:
            return None

        pmid = medline.findtext('.//PMID', '')

        # article info
        art = medline.find('.//Article')
        if art is None:
            return None

        title = art.findtext('.//ArticleTitle', '')

        # abstract
        abstract_parts = art.findall('.//Abstract/AbstractText')
        abstract = ' '.join(a.text or '' for a in abstract_parts if a.text)

        # authors (first 5)
        authors = []
        for author in art.findall('.//AuthorList/Author')[:5]:
            lastname = author.findtext('LastName', '')
            forename = author.findtext('ForeName', '')
            if lastname:
                name = f"{forename} {lastname}".strip()
                authors.append(name)

        # journal/venue
        journal = art.find('.//Journal')
        venue = journal.findtext('.//Title', '') if journal else ''

        # year
        year = None
        pub_date = art.find('.//Journal/JournalIssue/PubDate')
        if pub_date is not None:
            year_text = pub_date.findtext('Year')
            if year_text:
                try:
                    year = int(year_text)
                except ValueError:
                    pass

        # doi
        doi = None
        for id_elem in article.findall('.//PubmedData/ArticleIdList/ArticleId'):
            if id_elem.get('IdType') == 'doi':
                doi = id_elem.text
                break

        # publication type
        pub_types = [pt.text for pt in art.findall('.//PublicationTypeList/PublicationType') if pt.text]
        is_review = 'Review' in pub_types

        return PaperStub(
            doi=doi,
            openalex_id=None,  # would need crossref lookup
            s2_id=None,
            title=title,
            year=year,
            venue=venue,
            authors=authors,
            citation_count=None,  # pubmed doesn't provide this directly
            reference_count=None,
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
        """search pubmed for papers matching query."""
        # build query with date range
        search_query = query
        if year_min and year_max:
            search_query += f" AND {year_min}:{year_max}[dp]"
        elif year_min:
            search_query += f" AND {year_min}:3000[dp]"
        elif year_max:
            search_query += f" AND 1900:{year_max}[dp]"

        # step 1: esearch to get pmids
        search_params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': min(limit, 10000),
            'retmode': 'json',
            'sort': 'relevance'
        }

        search_response = self._make_request('esearch.fcgi', search_params)
        if not search_response:
            return []

        search_result = search_response.json()
        pmids = search_result.get('esearchresult', {}).get('idlist', [])

        if not pmids:
            return []

        # step 2: efetch to get article details
        papers = []
        # fetch in batches of 200
        for i in range(0, len(pmids), 200):
            batch = pmids[i:i+200]
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml'
            }

            fetch_response = self._make_request('efetch.fcgi', fetch_params)
            if not fetch_response:
                continue

            try:
                root = ET.fromstring(fetch_response.content)
                for article in root.findall('.//PubmedArticle'):
                    paper = self._parse_article(article)
                    if paper:
                        papers.append(paper)
            except ET.ParseError as e:
                print(f"[pubmed] xml parse error: {e}")
                continue

        return papers[:limit]

    def get_paper(self, paper_id: str) -> Optional[PaperStub]:
        """get paper by pmid or doi."""
        # determine if it's a pmid or doi
        if paper_id.startswith('pmid:'):
            pmid = paper_id[5:]
        elif paper_id.isdigit():
            pmid = paper_id
        elif paper_id.startswith('10.'):
            # search by doi
            search_params = {
                'db': 'pubmed',
                'term': f"{paper_id}[doi]",
                'retmax': 1,
                'retmode': 'json'
            }
            search_response = self._make_request('esearch.fcgi', search_params)
            if not search_response:
                return None
            result = search_response.json()
            pmids = result.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return None
            pmid = pmids[0]
        else:
            return None

        # fetch article
        fetch_params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }

        fetch_response = self._make_request('efetch.fcgi', fetch_params)
        if not fetch_response:
            return None

        try:
            root = ET.fromstring(fetch_response.content)
            article = root.find('.//PubmedArticle')
            if article:
                return self._parse_article(article)
        except ET.ParseError:
            pass

        return None

    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """
        get papers cited by this paper.
        note: pubmed doesn't directly provide references, would need pmc or external source.
        returns empty list for now - use crossref or s2 for this.
        """
        # pubmed doesn't provide reference lists directly
        # would need to use PMC or external services
        return []

    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """get papers that cite this paper using elink."""
        # normalize pmid
        if paper_id.startswith('pmid:'):
            pmid = paper_id[5:]
        elif paper_id.isdigit():
            pmid = paper_id
        else:
            # try to find pmid from doi
            paper = self.get_paper(paper_id)
            if not paper:
                return []
            # we don't have pmid stored, so can't proceed
            return []

        # use elink to find citing articles
        link_params = {
            'dbfrom': 'pubmed',
            'db': 'pubmed',
            'id': pmid,
            'linkname': 'pubmed_pubmed_citedin',
            'retmode': 'json'
        }

        link_response = self._make_request('elink.fcgi', link_params)
        if not link_response:
            return []

        try:
            result = link_response.json()
            linksets = result.get('linksets', [])
            if not linksets:
                return []

            links = linksets[0].get('linksetdbs', [])
            if not links:
                return []

            citing_pmids = [str(link['id']) for link in links[0].get('links', [])][:limit]

            if not citing_pmids:
                return []

            # fetch article details
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(citing_pmids),
                'retmode': 'xml'
            }

            fetch_response = self._make_request('efetch.fcgi', fetch_params)
            if not fetch_response:
                return []

            papers = []
            root = ET.fromstring(fetch_response.content)
            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)

            return papers

        except (KeyError, ET.ParseError) as e:
            print(f"[pubmed] error parsing citations: {e}")
            return []

    def get_count_estimate(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> int:
        """get estimated count of papers matching query."""
        search_query = query
        if year_min and year_max:
            search_query += f" AND {year_min}:{year_max}[dp]"
        elif year_min:
            search_query += f" AND {year_min}:3000[dp]"

        search_params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': 0,
            'retmode': 'json'
        }

        response = self._make_request('esearch.fcgi', search_params)
        if not response:
            return 0

        result = response.json()
        return int(result.get('esearchresult', {}).get('count', 0))


# simple test
if __name__ == "__main__":
    provider = PubMedProvider()

    print("testing pubmed search...")
    papers = provider.search_papers("ancestral protein reconstruction", year_min=2020, limit=5)
    for p in papers:
        print(f"  - {p.title[:60]}... ({p.year})")

    print("\ntesting pubmed count...")
    count = provider.get_count_estimate("ancestral protein reconstruction", year_min=2020)
    print(f"  count: {count}")
