"""
citation walker - unified service for fetching and classifying citations.
extracts the intro-hint logic that was duplicated across expansion modes.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..core.models import Paper, EdgeType
from ..core.config import ExpansionConfig
from ..providers.base import PaperProvider

logger = logging.getLogger("refnet.citation_walker")


@dataclass
class ClassifiedRef:
    """a reference with its classification (intro-hint or regular)."""
    paper: Paper
    is_intro_hint: bool
    edge_type: EdgeType
    edge_weight: float


@dataclass
class CitationResult:
    """result of fetching citations for a paper."""
    source_paper_id: str
    refs: List[ClassifiedRef] = field(default_factory=list)
    cites: List[Paper] = field(default_factory=list)
    refs_fetched: int = 0
    cites_fetched: int = 0
    api_calls: int = 0


class CitationWalker:
    """
    service for fetching and classifying paper citations.

    handles the INTRO_HINT_CITES heuristic:
    - first k refs (25% of total, clamped 10-40, max 20) are from introduction
    - intro refs get boosted edge weight (2.0) and special edge type
    - this helps identify problem/context papers vs methodology papers
    """

    def __init__(
        self,
        provider: PaperProvider,
        config: ExpansionConfig
    ):
        self.provider = provider
        self.config = config

    def compute_intro_hint_threshold(self, n_refs: int) -> int:
        """
        compute k: number of refs to classify as intro-hints.

        formula: k = clamp(round(n * intro_fraction), 10, 40)
                 then cap at max_intro_hint_per_node

        rationale: first ~25% of references in a paper typically come from
        the introduction where the problem/context is defined. these refs
        are more likely to be directly relevant to the paper's topic.
        """
        if n_refs == 0:
            return 0

        # base calculation: intro_fraction (default 0.25) of total refs
        k = int(n_refs * self.config.intro_fraction)

        # minimum of 10 if paper has enough refs
        if n_refs >= 10:
            k = max(k, 10)

        # hard cap at 40 (very long bibliographies are often reviews)
        k = min(k, 40)

        # finally cap at config limit (default 20)
        k = min(k, self.config.max_intro_hint_per_node)

        return k

    def fetch_refs_classified(
        self,
        paper_id: str,
        limit: int,
        source_paper_id: Optional[str] = None,
        discovered_from: Optional[str] = None,
        depth: int = 1
    ) -> Tuple[List[ClassifiedRef], int]:
        """
        fetch references and classify them as intro-hint or regular.

        returns:
            (classified_refs, api_calls)
        """
        refs = self.provider.get_references(paper_id, limit=limit)
        if not refs:
            return [], 1

        n_refs = len(refs)
        k = self.compute_intro_hint_threshold(n_refs)
        intro_weight = self.config.intro_hint_weight

        classified = []
        for i, ref in enumerate(refs):
            # set discovery metadata
            if discovered_from:
                ref.discovered_from = discovered_from
            ref.discovered_channel = "backward"
            ref.depth = depth

            # classify as intro-hint or regular
            is_intro = i < k
            classified.append(ClassifiedRef(
                paper=ref,
                is_intro_hint=is_intro,
                edge_type=EdgeType.INTRO_HINT_CITES if is_intro else EdgeType.CITES,
                edge_weight=intro_weight if is_intro else 1.0
            ))

        return classified, 1

    def fetch_refs_intro_only(
        self,
        paper_id: str,
        limit: int,
        discovered_from: Optional[str] = None,
        depth: int = 1
    ) -> Tuple[List[ClassifiedRef], int]:
        """
        fetch only intro-hint refs (for bucket expansion where we only want intro).

        this fetches all refs but only returns the first k (intro) ones.
        useful for dendrimer mode where we want focused expansion.
        """
        refs = self.provider.get_references(paper_id, limit=limit)
        if not refs:
            return [], 1

        n_refs = len(refs)
        k = self.compute_intro_hint_threshold(n_refs)
        intro_weight = self.config.intro_hint_weight

        classified = []
        for ref in refs[:k]:  # only intro refs
            if discovered_from:
                ref.discovered_from = discovered_from
            ref.discovered_channel = "backward"
            ref.depth = depth

            classified.append(ClassifiedRef(
                paper=ref,
                is_intro_hint=True,
                edge_type=EdgeType.INTRO_HINT_CITES,
                edge_weight=intro_weight
            ))

        return classified, 1

    def fetch_cites(
        self,
        paper_id: str,
        limit: int,
        discovered_from: Optional[str] = None,
        depth: int = 1
    ) -> Tuple[List[Paper], int]:
        """
        fetch forward citations (papers that cite this paper).

        returns:
            (cites, api_calls)
        """
        cites = self.provider.get_citations(paper_id, limit=limit)
        if not cites:
            return [], 1

        for cite in cites:
            if discovered_from:
                cite.discovered_from = discovered_from
            cite.discovered_channel = "forward"
            cite.depth = depth

        return cites, 1

    def walk_paper(
        self,
        paper: Paper,
        max_refs: int,
        max_cites: int,
        intro_only: bool = False
    ) -> CitationResult:
        """
        perform full citation walk for a paper.

        args:
            paper: the paper to expand
            max_refs: maximum refs to fetch
            max_cites: maximum cites to fetch
            intro_only: if True, only fetch intro-hint refs (for bucket mode)

        returns:
            CitationResult with classified refs and cites
        """
        paper_id = paper.doi or paper.openalex_id
        if not paper_id:
            logger.warning(f"[walker] no id for paper: {(paper.title or '?')[:30]}")
            return CitationResult(source_paper_id=paper.id)

        result = CitationResult(source_paper_id=paper.id)
        depth = (paper.depth or 0) + 1

        # fetch refs
        if max_refs > 0:
            try:
                if intro_only:
                    refs, calls = self.fetch_refs_intro_only(
                        paper_id, max_refs, paper.id, depth
                    )
                else:
                    refs, calls = self.fetch_refs_classified(
                        paper_id, max_refs, paper.id, paper.id, depth
                    )
                result.refs = refs
                result.refs_fetched = len(refs)
                result.api_calls += calls
            except Exception as e:
                logger.warning(f"[walker] refs failed for {(paper.title or '?')[:30]}: {e}")

        # fetch cites
        if max_cites > 0:
            try:
                cites, calls = self.fetch_cites(
                    paper_id, max_cites, paper.id, depth
                )
                result.cites = cites
                result.cites_fetched = len(cites)
                result.api_calls += calls
            except Exception as e:
                logger.warning(f"[walker] cites failed for {(paper.title or '?')[:30]}: {e}")

        return result
