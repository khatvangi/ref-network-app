"""
tiered search strategy - prioritizes high-impact sources.

instead of searching the entire ocean, we:
1. search tier 1 journals first (Nature, Science, etc.)
2. expand to tier 2 if needed (specialty journals)
3. fall back to broad search only if necessary

this reduces noise and improves quality of initial results.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from difflib import SequenceMatcher

from ..core.models import Paper
from ..providers.base import PaperProvider
from ..agents.field_resolver import FieldProfile, FieldResolution

logger = logging.getLogger("refnet.search.tiered")


@dataclass
class TieredSearchResult:
    """result of a tiered search operation."""
    papers: List[Paper] = field(default_factory=list)
    tier1_count: int = 0
    tier2_count: int = 0
    tier3_count: int = 0
    field_profile: Optional[FieldProfile] = None
    search_summary: str = ""


def venue_matches_journal(venue: str, journal: str, threshold: float = 0.8) -> bool:
    """check if a venue name matches a journal name."""
    if not venue or not journal:
        return False

    venue_lower = venue.lower().strip()
    journal_lower = journal.lower().strip()

    # exact match
    if venue_lower == journal_lower:
        return True

    # check if journal is a complete word in venue
    # (avoid "Science" matching "International Journal of Molecular Sciences")
    import re
    pattern = r'\b' + re.escape(journal_lower) + r'\b'
    if re.search(pattern, venue_lower):
        # for short journals (Science, Nature, Cell), require exact or very high match
        if len(journal_lower) <= 8:
            # short names need exact word match AND high similarity
            ratio = SequenceMatcher(None, venue_lower, journal_lower).ratio()
            return ratio >= 0.9
        return True

    # fuzzy match for longer journal names
    ratio = SequenceMatcher(None, venue_lower, journal_lower).ratio()
    return ratio >= threshold


def classify_paper_tier(
    paper: Paper,
    tier1_journals: List[str],
    tier2_journals: List[str]
) -> int:
    """classify a paper into tier 1, 2, or 3 based on venue."""
    if not paper.venue:
        return 3  # unknown venue = tier 3

    for j in tier1_journals:
        if venue_matches_journal(paper.venue, j):
            return 1

    for j in tier2_journals:
        if venue_matches_journal(paper.venue, j):
            return 2

    return 3


class TieredSearchStrategy:
    """
    search strategy that prioritizes high-impact sources.

    usage:
        from refnet.agents import FieldResolver
        from refnet.search import TieredSearchStrategy

        # resolve field first
        fr = FieldResolver()
        field_result = fr.execute(query="prebiotic chemistry")

        # create tiered search strategy
        strategy = TieredSearchStrategy(field_result.data.primary_field)

        # search with tier prioritization
        result = strategy.search(provider, query="RNA world", target_count=50)
        print(f"Found {result.tier1_count} tier 1, {result.tier2_count} tier 2")
    """

    def __init__(
        self,
        field_profile: FieldProfile,
        tier1_ratio: float = 0.5,
        tier2_ratio: float = 0.3,
        min_tier1_results: int = 5
    ):
        """
        initialize tiered search strategy.

        args:
            field_profile: field profile with journal tiers
            tier1_ratio: target ratio of tier 1 results
            tier2_ratio: target ratio of tier 2 results
            min_tier1_results: minimum tier 1 results before expanding
        """
        self.field = field_profile
        self.tier1_ratio = tier1_ratio
        self.tier2_ratio = tier2_ratio
        self.min_tier1_results = min_tier1_results

    def search(
        self,
        provider: PaperProvider,
        query: str,
        target_count: int = 50,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        oversample_factor: float = 3.0
    ) -> TieredSearchResult:
        """
        search for papers with tier prioritization.

        the strategy:
        1. fetch more papers than needed (oversampling)
        2. classify each by tier
        3. prioritize tier 1, then tier 2, then tier 3

        args:
            provider: paper provider to search with
            query: search query
            target_count: desired number of results
            year_min: minimum publication year
            year_max: maximum publication year
            oversample_factor: fetch this many times target_count

        returns:
            TieredSearchResult with prioritized papers
        """
        result = TieredSearchResult(field_profile=self.field)

        # fetch more papers than we need
        fetch_limit = int(target_count * oversample_factor)
        logger.info(f"[tiered] fetching {fetch_limit} papers for query: {query[:50]}...")

        all_papers = provider.search_papers(
            query=query,
            year_min=year_min,
            year_max=year_max,
            limit=fetch_limit
        )

        if not all_papers:
            result.search_summary = "no papers found"
            return result

        # classify papers by tier
        tier1_papers = []
        tier2_papers = []
        tier3_papers = []

        for paper in all_papers:
            tier = classify_paper_tier(
                paper,
                self.field.tier1_journals,
                self.field.tier2_journals
            )
            if tier == 1:
                tier1_papers.append(paper)
            elif tier == 2:
                tier2_papers.append(paper)
            else:
                tier3_papers.append(paper)

        logger.info(
            f"[tiered] classified: tier1={len(tier1_papers)}, "
            f"tier2={len(tier2_papers)}, tier3={len(tier3_papers)}"
        )

        # build result list with tier priority
        final_papers = []

        # take all tier 1
        final_papers.extend(tier1_papers)
        result.tier1_count = len(tier1_papers)

        # add tier 2 if we have room or need more
        remaining = target_count - len(final_papers)
        if remaining > 0:
            tier2_to_add = tier2_papers[:remaining]
            final_papers.extend(tier2_to_add)
            result.tier2_count = len(tier2_to_add)

        # add tier 3 only if we really need more
        remaining = target_count - len(final_papers)
        if remaining > 0 and len(final_papers) < target_count * 0.5:
            # only if we're below 50% of target
            tier3_to_add = tier3_papers[:remaining]
            final_papers.extend(tier3_to_add)
            result.tier3_count = len(tier3_to_add)

        result.papers = final_papers[:target_count]

        # summary
        total = len(result.papers)
        result.search_summary = (
            f"found {total} papers: "
            f"{result.tier1_count} tier1 ({100*result.tier1_count/total:.0f}%), "
            f"{result.tier2_count} tier2 ({100*result.tier2_count/total:.0f}%), "
            f"{result.tier3_count} tier3 ({100*result.tier3_count/total:.0f}%)"
        ) if total > 0 else "no papers found"

        logger.info(f"[tiered] {result.search_summary}")

        return result

    def prioritize_papers(
        self,
        papers: List[Paper],
        limit: int = 50,
        strategy: str = "tiered"
    ) -> TieredSearchResult:
        """
        prioritize an existing list of papers by tier.

        useful when you already have papers and want to re-rank them
        by journal importance.

        args:
            papers: list of papers to prioritize
            limit: maximum papers to return
            strategy: "tiered" (default), "ocean" (no filtering), or "tier1_only"

        returns:
            TieredSearchResult with prioritized papers
        """
        result = TieredSearchResult(field_profile=self.field)

        # ocean mode - no tier filtering, just return papers as-is
        if strategy == "ocean":
            result.papers = papers[:limit]
            result.tier3_count = len(result.papers)  # all counted as tier3 in ocean mode
            result.search_summary = f"ocean mode: returning {len(result.papers)} papers (no tier filtering)"
            return result

        tier1_papers = []
        tier2_papers = []
        tier3_papers = []

        for paper in papers:
            tier = classify_paper_tier(
                paper,
                self.field.tier1_journals,
                self.field.tier2_journals
            )
            if tier == 1:
                tier1_papers.append(paper)
            elif tier == 2:
                tier2_papers.append(paper)
            else:
                tier3_papers.append(paper)

        # tier1_only mode - strict filtering
        if strategy == "tier1_only":
            result.papers = tier1_papers[:limit]
            result.tier1_count = len(result.papers)
            result.search_summary = f"tier1 only: {result.tier1_count} papers"
            return result

        # default tiered mode - combine with priority
        final_papers = tier1_papers + tier2_papers + tier3_papers
        result.papers = final_papers[:limit]

        result.tier1_count = min(len(tier1_papers), limit)
        result.tier2_count = min(len(tier2_papers), max(0, limit - len(tier1_papers)))
        result.tier3_count = max(0, len(result.papers) - result.tier1_count - result.tier2_count)

        total = len(result.papers)
        result.search_summary = (
            f"prioritized {total} papers: "
            f"{result.tier1_count} tier1, {result.tier2_count} tier2, {result.tier3_count} tier3"
        ) if total > 0 else "no papers"

        return result

    def get_tier(self, paper: Paper) -> int:
        """get the tier classification for a single paper."""
        return classify_paper_tier(
            paper,
            self.field.tier1_journals,
            self.field.tier2_journals
        )

    def is_high_impact(self, paper: Paper) -> bool:
        """check if paper is from a tier 1 or tier 2 journal."""
        return self.get_tier(paper) <= 2
