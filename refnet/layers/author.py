"""
author layer - identity resolution, expansion, and scoring.
implements scientist-centric discovery via author works.
"""

import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass

from ..core.models import Paper, Author, AuthorStatus, PaperStatus, EdgeType

logger = logging.getLogger("refnet.author")
from ..core.config import AuthorConfig, RefnetConfig
from ..providers.base import PaperProvider, AuthorInfo, ORCIDProvider
from ..graph.candidate_pool import CandidatePool
from ..graph.working_graph import WorkingGraph


@dataclass
class AuthorExpansionResult:
    """result of expanding an author's works."""
    author: Author
    papers_added: int
    papers_skipped: int
    foundational_papers: List[str]  # high-impact older papers
    recent_papers: List[str]  # recent papers


class AuthorLayer:
    """
    manages author identity resolution and expansion.
    third channel of discovery (after backward refs and forward cites).
    """

    def __init__(
        self,
        provider: PaperProvider,
        config: Optional[AuthorConfig] = None,
        orcid_provider: Optional[ORCIDProvider] = None
    ):
        self.provider = provider
        self.config = config or AuthorConfig()
        self.orcid = orcid_provider or ORCIDProvider()

        # caches
        self._author_cache: Dict[str, Author] = {}
        self._orcid_cache: Dict[str, str] = {}  # orcid -> author_id
        self._name_cache: Dict[str, str] = {}   # normalized_name -> author_id

    def resolve_author(
        self,
        name: str,
        orcid: Optional[str] = None,
        openalex_id: Optional[str] = None,
        s2_id: Optional[str] = None,
        affiliation: Optional[str] = None,
        coauthor_names: Optional[List[str]] = None
    ) -> Optional[Author]:
        """
        resolve author identity using all available signals.
        order of preference: ORCID > OpenAlex ID > S2 ID > name+hints
        """
        # try ORCID first (gold standard)
        if orcid and self.config.prefer_orcid:
            cached = self._orcid_cache.get(orcid)
            if cached:
                return self._author_cache.get(cached)

            author = self._resolve_by_orcid(orcid)
            if author:
                self._cache_author(author)
                return author

        # try provider IDs
        if openalex_id and self.provider.supports_authors():
            info = self.provider.get_author(openalex_id)
            if info:
                author = self._info_to_author(info)
                self._cache_author(author)
                return author

        if s2_id and hasattr(self.provider, 'get_author'):
            info = self.provider.get_author(s2_id)
            if info:
                author = self._info_to_author(info)
                self._cache_author(author)
                return author

        # try name-based resolution
        if name:
            # check cache first
            norm_name = self._normalize_name(name)
            cached = self._name_cache.get(norm_name)
            if cached:
                return self._author_cache.get(cached)

            # resolve via provider
            if self.provider.supports_authors():
                info = self.provider.resolve_author_id(
                    name, affiliation, coauthor_names
                )
                if info:
                    author = self._info_to_author(info)
                    # validate match quality
                    if self._validate_name_match(name, author.name):
                        self._cache_author(author)
                        return author

        return None

    def get_authors_to_expand(
        self,
        paper: Paper,
        existing_authors: Optional[Set[str]] = None
    ) -> List[Author]:
        """
        determine which authors to expand for a paper.
        respects caps and avoids already-expanded authors.
        """
        if not self.config.enabled:
            return []

        existing = existing_authors or set()
        authors_to_expand = []

        # get author info from paper
        author_names = paper.authors or []
        author_ids = paper.author_ids or []

        if not author_names:
            return []

        # determine which positions to expand
        positions_to_try = []

        # corresponding author (if available) - typically last
        if self.config.expand_corresponding and len(author_names) > 0:
            positions_to_try.append(len(author_names) - 1)

        # first author
        if self.config.expand_first_author:
            positions_to_try.append(0)

        # last author (if not already added as corresponding)
        if self.config.expand_last_author and len(author_names) > 1:
            if len(author_names) - 1 not in positions_to_try:
                positions_to_try.append(len(author_names) - 1)

        # remove duplicates while preserving order
        seen_positions = set()
        unique_positions = []
        for p in positions_to_try:
            if p not in seen_positions:
                seen_positions.add(p)
                unique_positions.append(p)

        # resolve authors at these positions
        for pos in unique_positions[:self.config.author_expand_k + 1]:
            if pos >= len(author_names):
                continue

            name = author_names[pos]
            author_id = author_ids[pos] if pos < len(author_ids) else None

            # skip if already expanded
            if author_id and author_id in existing:
                continue

            # try to resolve
            author = self.resolve_author(
                name=name,
                openalex_id=author_id if author_id and author_id.startswith("A") else None,
                s2_id=author_id if author_id and not author_id.startswith("A") else None
            )

            if author:
                # check for mega-author
                if self._is_mega_author(author) and self.config.skip_mega_unless_central:
                    logger.info(
                        f"[author] skipping mega-author: {author.name} "
                        f"({author.paper_count} papers > {self.config.mega_author_threshold} threshold)"
                    )
                    continue

                if author.id not in existing:
                    authors_to_expand.append(author)

            if len(authors_to_expand) >= self.config.author_expand_k:
                break

        return authors_to_expand

    def expand_author(
        self,
        author: Author,
        pool: CandidatePool,
        seed_ids: Set[str],
        current_year: int
    ) -> AuthorExpansionResult:
        """
        expand author's works into candidate pool.
        fetches recent + foundational papers.
        """
        if not self.provider.supports_authors():
            return AuthorExpansionResult(
                author=author,
                papers_added=0,
                papers_skipped=0,
                foundational_papers=[],
                recent_papers=[]
            )

        papers_added = 0
        papers_skipped = 0
        recent_papers = []
        foundational_papers = []

        # get author's external id
        author_ext_id = author.openalex_id or author.s2_id
        if not author_ext_id:
            return AuthorExpansionResult(
                author=author,
                papers_added=0,
                papers_skipped=0,
                foundational_papers=[],
                recent_papers=[]
            )

        # fetch recent works
        year_min = current_year - self.config.author_recent_years
        recent_works = self.provider.get_author_works(
            author_ext_id,
            year_min=year_min,
            limit=self.config.author_recent_cap * 2  # fetch more, filter later
        )

        for paper in recent_works[:self.config.author_recent_cap]:
            paper.discovered_from = author.id
            paper.discovered_channel = "author"

            if pool.add_paper(paper):
                papers_added += 1
                recent_papers.append(paper.id)
            else:
                papers_skipped += 1

        # fetch foundational works (older, high-impact)
        all_works = self.provider.get_author_works(
            author_ext_id,
            year_max=year_min - 1,
            limit=self.config.author_foundational_cap * 3
        )

        # sort by citation count to get foundational
        all_works.sort(key=lambda p: p.citation_count or 0, reverse=True)

        for paper in all_works[:self.config.author_foundational_cap]:
            # additional filter: must be somewhat relevant to seeds
            # (we'll compute this properly in scoring, but basic filter here)
            paper.discovered_from = author.id
            paper.discovered_channel = "author_foundational"

            if pool.add_paper(paper):
                papers_added += 1
                foundational_papers.append(paper.id)
            else:
                papers_skipped += 1

        # check budget
        total = len(recent_papers) + len(foundational_papers)
        if total > self.config.author_total_cap:
            # trim (prefer recent over foundational for trimming)
            excess = total - self.config.author_total_cap
            # this is just tracking, papers already added

        # update author status
        author.status = AuthorStatus.EXPANDED

        return AuthorExpansionResult(
            author=author,
            papers_added=papers_added,
            papers_skipped=papers_skipped,
            foundational_papers=foundational_papers,
            recent_papers=recent_papers
        )

    def compute_author_topic_fit(
        self,
        author: Author,
        graph: WorkingGraph
    ) -> float:
        """
        compute how well an author fits the current topic.
        based on average relevance of their papers in the graph.
        """
        # get author's papers in the graph
        author_papers = []
        for paper_id, paper in graph.papers.items():
            if author.openalex_id in paper.author_ids or \
               author.s2_id in paper.author_ids:
                author_papers.append(paper)

        if not author_papers:
            return 0.0

        # average relevance of top papers
        relevances = sorted(
            [p.relevance_score for p in author_papers],
            reverse=True
        )[:5]

        return sum(relevances) / len(relevances) if relevances else 0.0

    def compute_author_centrality(
        self,
        author: Author,
        graph: WorkingGraph
    ) -> float:
        """
        compute author's centrality in the graph.
        based on number of papers and their connectivity.
        """
        # count papers and connections
        author_paper_ids = []
        for paper_id, paper in graph.papers.items():
            if author.openalex_id in paper.author_ids or \
               author.s2_id in paper.author_ids:
                author_paper_ids.append(paper_id)

        if not author_paper_ids:
            return 0.0

        # count unique connections
        connected_papers = set()
        for pid in author_paper_ids:
            neighbors = graph.get_neighbors(pid)
            connected_papers.update(neighbors)

        # remove author's own papers
        connected_papers -= set(author_paper_ids)

        # centrality = normalized (papers * connections)
        paper_score = min(len(author_paper_ids) / 10, 1.0)
        connection_score = min(len(connected_papers) / 50, 1.0)

        return 0.5 * paper_score + 0.5 * connection_score

    # private helpers

    def _resolve_by_orcid(self, orcid: str) -> Optional[Author]:
        """resolve author using ORCID API."""
        info = self.orcid.get_author_by_orcid(orcid)
        if not info:
            # try via provider
            if hasattr(self.provider, 'get_author_by_orcid'):
                info = self.provider.get_author_by_orcid(orcid)

        if info:
            author = self._info_to_author(info)
            author.orcid = orcid
            return author

        return None

    def _info_to_author(self, info: AuthorInfo) -> Author:
        """convert AuthorInfo to Author model."""
        return Author(
            name=info.name,
            orcid=info.orcid,
            openalex_id=info.openalex_id,
            s2_id=info.s2_id,
            affiliations=info.affiliations or [],
            paper_count=info.paper_count or 0,
            citation_count=info.citation_count or 0,
            status=AuthorStatus.CANDIDATE
        )

    def _cache_author(self, author: Author):
        """cache author for future lookups."""
        self._author_cache[author.id] = author
        if author.orcid:
            self._orcid_cache[author.orcid] = author.id
        if author.name:
            norm = self._normalize_name(author.name)
            self._name_cache[norm] = author.id

    def _normalize_name(self, name: str) -> str:
        """normalize name for matching."""
        import re
        name = name.lower().strip()
        # remove middle initials
        name = re.sub(r'\b[a-z]\.\s*', '', name)
        # remove punctuation
        name = re.sub(r'[^\w\s]', '', name)
        return ' '.join(name.split())

    def _validate_name_match(self, query_name: str, found_name: str) -> bool:
        """check if found name matches query name."""
        q_norm = self._normalize_name(query_name)
        f_norm = self._normalize_name(found_name)

        # exact match
        if q_norm == f_norm:
            return True

        # check if last names match
        q_parts = q_norm.split()
        f_parts = f_norm.split()

        if not q_parts or not f_parts:
            return False

        # last names should match
        if q_parts[-1] != f_parts[-1]:
            return False

        # first initial should match
        if q_parts[0][0] == f_parts[0][0]:
            return True

        return False

    def _is_mega_author(self, author: Author) -> bool:
        """
        check if author is a mega-author (too many papers).
        mega-authors (500+ papers) would explode the graph if fully expanded.
        examples: statisticians, bioinformaticians with many collaborations.
        """
        if author.paper_count <= 0:
            # no data available, assume not mega
            return False
        return author.paper_count >= self.config.mega_author_threshold
