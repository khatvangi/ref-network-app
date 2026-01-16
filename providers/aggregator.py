"""
multi-provider aggregator.
combines search results from multiple providers with deduplication.
"""

from typing import List, Optional, Dict, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import PaperProvider, PaperStub
from .openalex import OpenAlexProvider
from .semantic_scholar import SemanticScholarProvider
from .pubmed import PubMedProvider
from .crossref import CrossrefProvider
# google scholar is optional due to rate limits
try:
    from .google_scholar import GoogleScholarProvider, SCHOLARLY_AVAILABLE
except ImportError:
    GoogleScholarProvider = None
    SCHOLARLY_AVAILABLE = False


@dataclass
class AggregatorConfig:
    """configuration for aggregator."""
    use_openalex: bool = True
    use_s2: bool = True
    use_pubmed: bool = True
    use_crossref: bool = True
    use_gscholar: bool = False  # disabled by default due to rate limits
    gscholar_limit: int = 20  # max results from gscholar

    parallel: bool = True  # run providers in parallel
    dedupe_by_doi: bool = True
    dedupe_by_title: bool = True
    title_similarity_threshold: float = 0.8


class ProviderAggregator:
    """
    aggregates results from multiple paper providers.
    handles deduplication and merging of metadata.
    """

    def __init__(self, config: Optional[AggregatorConfig] = None):
        self.config = config or AggregatorConfig()
        self.providers: Dict[str, PaperProvider] = {}

        # initialize providers based on config
        if self.config.use_openalex:
            self.providers['openalex'] = OpenAlexProvider()

        if self.config.use_s2:
            self.providers['s2'] = SemanticScholarProvider()

        if self.config.use_pubmed:
            self.providers['pubmed'] = PubMedProvider()

        if self.config.use_crossref:
            self.providers['crossref'] = CrossrefProvider()

        if self.config.use_gscholar and GoogleScholarProvider:
            self.providers['gscholar'] = GoogleScholarProvider(rate_limit_delay=5.0)

    def _normalize_doi(self, doi: Optional[str]) -> Optional[str]:
        """normalize doi for comparison."""
        if not doi:
            return None
        doi = doi.lower().strip()
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]
        if doi.startswith('doi:'):
            doi = doi[4:]
        return doi

    def _title_hash(self, title: str) -> str:
        """create hash of title for fuzzy matching."""
        import re
        # lowercase, remove punctuation, remove common words
        title = title.lower()
        title = re.sub(r'[^\w\s]', '', title)
        stopwords = {'a', 'an', 'the', 'of', 'and', 'or', 'in', 'on', 'for', 'to', 'with'}
        words = [w for w in title.split() if w not in stopwords]
        return ' '.join(sorted(words[:10]))  # first 10 significant words, sorted

    def _merge_papers(self, papers: List[PaperStub]) -> List[PaperStub]:
        """merge papers with same DOI, keeping best metadata."""
        doi_map: Dict[str, PaperStub] = {}
        title_map: Dict[str, PaperStub] = {}
        result = []

        for paper in papers:
            doi = self._normalize_doi(paper.doi)

            # first try to match by DOI
            if doi and self.config.dedupe_by_doi:
                if doi in doi_map:
                    # merge: keep paper with more metadata
                    existing = doi_map[doi]
                    merged = self._merge_two(existing, paper)
                    doi_map[doi] = merged
                    continue
                else:
                    doi_map[doi] = paper
                    continue

            # then try to match by title
            if paper.title and self.config.dedupe_by_title:
                title_key = self._title_hash(paper.title)
                if title_key in title_map:
                    existing = title_map[title_key]
                    merged = self._merge_two(existing, paper)
                    title_map[title_key] = merged
                    continue
                else:
                    title_map[title_key] = paper
                    continue

            # no match found, add as new
            result.append(paper)

        # combine all unique papers
        result.extend(doi_map.values())
        result.extend(title_map.values())

        return result

    def _merge_two(self, p1: PaperStub, p2: PaperStub) -> PaperStub:
        """merge two papers, keeping best fields from each."""
        return PaperStub(
            doi=p1.doi or p2.doi,
            openalex_id=p1.openalex_id or p2.openalex_id,
            s2_id=p1.s2_id or p2.s2_id,
            title=p1.title if len(p1.title or '') >= len(p2.title or '') else p2.title,
            year=p1.year or p2.year,
            venue=p1.venue or p2.venue,
            authors=p1.authors if p1.authors else p2.authors,
            citation_count=max(p1.citation_count or 0, p2.citation_count or 0) or None,
            reference_count=max(p1.reference_count or 0, p2.reference_count or 0) or None,
            is_review=p1.is_review or p2.is_review,
            is_oa=p1.is_oa or p2.is_oa,
            oa_pdf_url=p1.oa_pdf_url or p2.oa_pdf_url,
            abstract=p1.abstract if p1.abstract else p2.abstract,
            concepts=p1.concepts if p1.concepts else p2.concepts
        )

    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[PaperStub]:
        """
        search all providers and merge results.
        """
        all_papers = []

        if self.config.parallel:
            # run providers in parallel
            with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
                futures = {}

                for name, provider in self.providers.items():
                    # adjust limit for gscholar
                    provider_limit = self.config.gscholar_limit if name == 'gscholar' else limit

                    future = executor.submit(
                        provider.search_papers,
                        query, year_min, year_max, provider_limit
                    )
                    futures[future] = name

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        papers = future.result()
                        print(f"[aggregator] {name}: {len(papers)} papers")
                        all_papers.extend(papers)
                    except Exception as e:
                        print(f"[aggregator] {name} error: {e}")
        else:
            # run sequentially
            for name, provider in self.providers.items():
                try:
                    provider_limit = self.config.gscholar_limit if name == 'gscholar' else limit
                    papers = provider.search_papers(query, year_min, year_max, provider_limit)
                    print(f"[aggregator] {name}: {len(papers)} papers")
                    all_papers.extend(papers)
                except Exception as e:
                    print(f"[aggregator] {name} error: {e}")

        # dedupe and merge
        merged = self._merge_papers(all_papers)
        print(f"[aggregator] total unique: {len(merged)} (from {len(all_papers)})")

        # sort by citation count
        merged.sort(key=lambda p: p.citation_count or 0, reverse=True)

        return merged[:limit]

    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """get references from all providers that support it."""
        all_refs = []

        # openalex and s2 are best for references
        if 'openalex' in self.providers:
            try:
                refs = self.providers['openalex'].get_references(paper_id, limit)
                all_refs.extend(refs)
            except Exception as e:
                print(f"[aggregator] openalex refs error: {e}")

        if 's2' in self.providers:
            try:
                refs = self.providers['s2'].get_references(paper_id, limit)
                all_refs.extend(refs)
            except Exception as e:
                print(f"[aggregator] s2 refs error: {e}")

        if 'crossref' in self.providers:
            try:
                refs = self.providers['crossref'].get_references(paper_id, limit)
                all_refs.extend(refs)
            except Exception as e:
                print(f"[aggregator] crossref refs error: {e}")

        return self._merge_papers(all_refs)[:limit]

    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """get citations from all providers that support it."""
        all_cites = []

        # openalex and s2 are best for forward citations
        if 'openalex' in self.providers:
            try:
                cites = self.providers['openalex'].get_citations(paper_id, limit)
                all_cites.extend(cites)
            except Exception as e:
                print(f"[aggregator] openalex cites error: {e}")

        if 's2' in self.providers:
            try:
                cites = self.providers['s2'].get_citations(paper_id, limit)
                all_cites.extend(cites)
            except Exception as e:
                print(f"[aggregator] s2 cites error: {e}")

        if 'pubmed' in self.providers:
            try:
                cites = self.providers['pubmed'].get_citations(paper_id, limit)
                all_cites.extend(cites)
            except Exception as e:
                print(f"[aggregator] pubmed cites error: {e}")

        return self._merge_papers(all_cites)[:limit]

    def get_count_estimates(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> Dict[str, int]:
        """get count estimates from all providers."""
        counts = {}

        for name, provider in self.providers.items():
            if name == 'gscholar':
                continue  # gscholar doesn't provide counts

            try:
                count = provider.get_count_estimate(query, year_min, year_max)
                counts[name] = count
            except Exception as e:
                print(f"[aggregator] {name} count error: {e}")
                counts[name] = -1

        return counts


# simple test
if __name__ == "__main__":
    config = AggregatorConfig(
        use_openalex=True,
        use_s2=True,
        use_pubmed=True,
        use_crossref=True,
        use_gscholar=False  # skip for speed
    )

    agg = ProviderAggregator(config)

    print("testing aggregated search...")
    papers = agg.search_papers("ancestral protein reconstruction", year_min=2020, limit=20)

    print(f"\nFound {len(papers)} unique papers:")
    for p in papers[:10]:
        sources = []
        if p.openalex_id:
            sources.append('OA')
        if p.s2_id:
            sources.append('S2')
        if p.doi:
            sources.append('DOI')
        print(f"  - {p.title[:50]}... ({p.year}) [{','.join(sources)}] cites={p.citation_count}")
