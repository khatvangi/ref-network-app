"""
corpus fetcher agent - get all papers by an author.

this is the heart of "investigate the author" - their corpus
reveals trajectory, methods, collaborators, and context.

input: author ID (OpenAlex or name to resolve)
output: author profile + all their papers
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .base import Agent, AgentResult, AgentStatus, AgentError
from ..core.models import Paper, Author
from ..providers.base import PaperProvider


@dataclass
class AuthorCorpus:
    """
    complete author corpus - profile + all papers.

    this is what we need to understand an author's research.
    """
    # author identity
    author_id: str
    name: str
    openalex_id: Optional[str] = None
    orcid: Optional[str] = None
    s2_id: Optional[str] = None

    # author metadata
    affiliations: List[str] = field(default_factory=list)
    h_index: Optional[int] = None
    total_citations: int = 0
    total_papers: int = 0

    # the corpus
    papers: List[Paper] = field(default_factory=list)

    # corpus statistics (computed)
    year_range: tuple = (None, None)
    top_venues: List[str] = field(default_factory=list)
    top_concepts: List[Dict[str, Any]] = field(default_factory=list)
    collaborators: List[str] = field(default_factory=list)

    def compute_stats(self):
        """compute corpus statistics from papers."""
        if not self.papers:
            return

        # year range
        years = [p.year for p in self.papers if p.year]
        if years:
            self.year_range = (min(years), max(years))

        # top venues
        venue_counts: Dict[str, int] = {}
        for p in self.papers:
            if p.venue:
                venue_counts[p.venue] = venue_counts.get(p.venue, 0) + 1
        self.top_venues = sorted(venue_counts.keys(), key=lambda v: -venue_counts[v])[:5]

        # top concepts (aggregate across papers)
        concept_weights: Dict[str, float] = {}
        for p in self.papers:
            for c in (p.concepts or [])[:5]:
                name = c.get('name', '')
                score = c.get('score', 0.5)
                concept_weights[name] = concept_weights.get(name, 0) + score

        sorted_concepts = sorted(concept_weights.items(), key=lambda x: -x[1])
        self.top_concepts = [
            {'name': name, 'weight': weight}
            for name, weight in sorted_concepts[:10]
        ]

        # collaborators (co-authors)
        coauthor_counts: Dict[str, int] = {}
        for p in self.papers:
            for author in (p.authors or []):
                if author != self.name:
                    coauthor_counts[author] = coauthor_counts.get(author, 0) + 1
        self.collaborators = sorted(coauthor_counts.keys(), key=lambda a: -coauthor_counts[a])[:20]


class CorpusFetcher(Agent):
    """
    fetch complete author corpus from API.

    handles:
    - author ID resolution (name → ID)
    - pagination (authors with 100s of papers)
    - rate limiting
    - partial failures (some papers fail to fetch)

    usage:
        fetcher = CorpusFetcher(provider)
        result = fetcher.run(author_id="A5009093641")  # OpenAlex ID
        # or
        result = fetcher.run(author_name="Charles W. Carter")

        if result.ok:
            corpus = result.data
            print(f"{corpus.name}: {len(corpus.papers)} papers")
    """

    def __init__(
        self,
        provider: PaperProvider,
        max_papers: int = 500,  # cap to avoid huge fetches
        batch_size: int = 50,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.provider = provider
        self.max_papers = max_papers
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "CorpusFetcher"

    def execute(
        self,
        author_id: Optional[str] = None,
        author_name: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> AgentResult[AuthorCorpus]:
        """
        fetch author's complete corpus.

        args:
            author_id: OpenAlex author ID (e.g., "A5009093641") or S2 ID
            author_name: author name to resolve (if no ID)
            min_year: only fetch papers from this year onwards
            max_year: only fetch papers up to this year

        returns:
            AgentResult with AuthorCorpus on success
        """
        result = AgentResult[AuthorCorpus](status=AgentStatus.SUCCESS)
        result.add_trace(f"starting corpus fetch: id={author_id}, name={author_name}")

        # need either ID or name
        if not author_id and not author_name:
            result.status = AgentStatus.FAILED
            result.add_error("MISSING_INPUT", "need author_id or author_name")
            return result

        # resolve name to ID if needed
        if not author_id and author_name:
            author_id = self._resolve_author_name(author_name, result)
            if not author_id:
                result.status = AgentStatus.FAILED
                result.add_error("AUTHOR_NOT_FOUND", f"could not resolve author: {author_name}")
                return result

        # fetch author profile
        author_info = self._fetch_author_profile(author_id, result)
        if not author_info:
            result.status = AgentStatus.FAILED
            result.add_error("PROFILE_FETCH_FAILED", f"could not fetch author profile: {author_id}")
            return result

        result.add_trace(f"found author: {author_info.name} ({getattr(author_info, 'paper_count', '?')} papers)")

        # create corpus object - use getattr for defensive access
        corpus = AuthorCorpus(
            author_id=author_id,
            name=author_info.name,
            openalex_id=getattr(author_info, 'openalex_id', None),
            orcid=getattr(author_info, 'orcid', None),
            s2_id=getattr(author_info, 's2_id', None),
            affiliations=getattr(author_info, 'affiliations', None) or [],
            h_index=getattr(author_info, 'h_index', None),
            total_citations=getattr(author_info, 'citation_count', 0) or 0,
            total_papers=getattr(author_info, 'paper_count', 0) or 0
        )

        # fetch papers
        papers = self._fetch_author_papers(
            author_id,
            min_year=min_year,
            max_year=max_year,
            result=result
        )

        corpus.papers = papers
        corpus.compute_stats()

        result.data = corpus
        result.add_trace(f"corpus complete: {len(papers)} papers fetched")

        # set status based on completeness
        if len(papers) == 0:
            result.status = AgentStatus.FAILED
            result.add_error("NO_PAPERS", "no papers found for author")
        elif len(result.errors) > 0:
            result.status = AgentStatus.PARTIAL
        else:
            result.status = AgentStatus.SUCCESS

        return result

    def _resolve_author_name(
        self,
        name: str,
        result: AgentResult
    ) -> Optional[str]:
        """resolve author name to ID."""
        result.add_trace(f"resolving author name: {name}")

        try:
            author_info = self.provider.resolve_author_id(name)
            result.api_calls += 1

            if author_info:
                author_id = author_info.openalex_id or author_info.s2_id
                result.add_trace(f"resolved to: {author_id}")
                return author_id
            else:
                result.add_trace("name resolution returned no results")
                return None

        except Exception as e:
            result.add_error(
                "NAME_RESOLUTION_ERROR",
                f"error resolving name: {e}",
                recoverable=True
            )
            return None

    def _fetch_author_profile(
        self,
        author_id: str,
        result: AgentResult
    ) -> Optional[Author]:
        """fetch author profile/metadata."""
        result.add_trace(f"fetching author profile: {author_id}")

        try:
            # try get_author if provider supports it
            if hasattr(self.provider, 'get_author'):
                author = self.provider.get_author(author_id)
                result.api_calls += 1
                if author:
                    return author

            # fallback: create minimal Author from ID
            result.add_warning("get_author not supported, using minimal profile")
            return Author(
                id=author_id,
                name=author_id,
                openalex_id=author_id if author_id.startswith('A') else None
            )

        except Exception as e:
            result.add_error(
                "PROFILE_FETCH_ERROR",
                f"error fetching profile: {e}",
                recoverable=True
            )
            return None

    def _fetch_author_papers(
        self,
        author_id: str,
        min_year: Optional[int],
        max_year: Optional[int],
        result: AgentResult
    ) -> List[Paper]:
        """fetch all papers by author with pagination."""
        result.add_trace(f"fetching papers for author: {author_id}")

        all_papers = []
        offset = 0
        consecutive_failures = 0
        max_consecutive_failures = 3

        while len(all_papers) < self.max_papers:
            try:
                # fetch batch
                batch = self.provider.get_author_works(
                    author_id,
                    limit=self.batch_size,
                    offset=offset
                )
                result.api_calls += 1

                if not batch:
                    result.add_trace(f"no more papers at offset {offset}")
                    break

                # filter by year if specified
                if min_year or max_year:
                    filtered = []
                    for p in batch:
                        if min_year and p.year and p.year < min_year:
                            continue
                        if max_year and p.year and p.year > max_year:
                            continue
                        filtered.append(p)
                    batch = filtered

                all_papers.extend(batch)
                result.add_trace(f"fetched {len(batch)} papers (total: {len(all_papers)})")

                # check if we got fewer than batch_size (end of results)
                if len(batch) < self.batch_size:
                    break

                offset += self.batch_size
                consecutive_failures = 0

                # rate limiting pause
                time.sleep(0.1)

            except Exception as e:
                consecutive_failures += 1
                result.add_error(
                    "PAPER_FETCH_ERROR",
                    f"error fetching papers at offset {offset}: {e}",
                    recoverable=True,
                    offset=offset
                )

                if consecutive_failures >= max_consecutive_failures:
                    result.add_trace(f"aborting after {consecutive_failures} consecutive failures")
                    break

                # try next batch
                offset += self.batch_size
                time.sleep(0.5)  # longer pause after error

        # sort by year (newest first)
        all_papers.sort(key=lambda p: p.year or 0, reverse=True)

        return all_papers
