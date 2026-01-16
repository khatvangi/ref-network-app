"""
hub detection - identify methodology papers and control explosion.
papers like AlphaFold, BLAST, etc. that are cited by everything.
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass

from ..core.models import Paper, PaperStatus
from ..core.config import ExpansionConfig


@dataclass
class HubAnalysis:
    """analysis result for a potential hub paper."""
    paper_id: str
    is_hub: bool
    is_mega_hub: bool
    hub_type: Optional[str]  # "methodology", "foundational", "review"
    citation_count: int
    relevance_score: float
    suppress_expansion: bool
    suppress_reason: Optional[str]


# known methodology paper patterns
METHODOLOGY_KEYWORDS = [
    "method", "software", "tool", "package", "algorithm",
    "database", "platform", "framework", "pipeline", "server",
    "web server", "online tool", "command-line"
]

METHODOLOGY_VENUES = [
    "nucleic acids research",
    "bioinformatics",
    "nature methods",
    "bmc bioinformatics",
    "plos computational biology",
    "genome biology",
    "journal of chemical information",
    "journal of computational chemistry"
]

# known mega-hubs (citation machines that add noise)
KNOWN_MEGA_HUBS = {
    "10.1038/nmeth.1923",  # ImageJ
    "10.1093/nar/gkr777",  # BLAST
    "10.1093/nar/gkv1189", # BLAST+
    "10.1038/s41586-021-03819-2",  # AlphaFold
    "10.1038/s41592-019-0686-2",   # UMAP
    "10.1093/bioinformatics/btp324",  # BWA
    "10.1038/nmeth.4285",  # RNA-seq analysis
    "10.1038/nature11247",  # ENCODE
}


class HubDetector:
    """
    detects hub/methodology papers and controls their expansion.
    prevents citation explosion from ubiquitous tools.
    """

    def __init__(self, config: Optional[ExpansionConfig] = None):
        self.config = config or ExpansionConfig()

    def analyze_paper(
        self,
        paper: Paper,
        seed_relevance: Optional[float] = None
    ) -> HubAnalysis:
        """
        analyze whether a paper is a hub and should have restricted expansion.
        """
        citation_count = paper.citation_count or 0
        relevance = seed_relevance if seed_relevance is not None else paper.relevance_score

        # check for known mega-hub
        if paper.doi and paper.doi.lower() in KNOWN_MEGA_HUBS:
            return HubAnalysis(
                paper_id=paper.id,
                is_hub=True,
                is_mega_hub=True,
                hub_type="methodology",
                citation_count=citation_count,
                relevance_score=relevance,
                suppress_expansion=True,
                suppress_reason="known methodology hub"
            )

        # check mega-hub threshold
        is_mega = citation_count >= self.config.mega_hub_threshold

        if is_mega:
            return HubAnalysis(
                paper_id=paper.id,
                is_hub=True,
                is_mega_hub=True,
                hub_type="mega_hub",
                citation_count=citation_count,
                relevance_score=relevance,
                suppress_expansion=True,
                suppress_reason=f"citations ({citation_count}) >= mega_hub_threshold"
            )

        # check hub threshold with low relevance
        is_hub = citation_count >= self.config.hub_citation_threshold
        is_low_relevance = relevance < self.config.hub_relevance_threshold

        if is_hub and is_low_relevance:
            hub_type = self._detect_hub_type(paper)
            return HubAnalysis(
                paper_id=paper.id,
                is_hub=True,
                is_mega_hub=False,
                hub_type=hub_type,
                citation_count=citation_count,
                relevance_score=relevance,
                suppress_expansion=True,
                suppress_reason=f"high citations ({citation_count}) + low relevance ({relevance:.2f})"
            )

        # not a hub
        return HubAnalysis(
            paper_id=paper.id,
            is_hub=is_hub,  # might be hub but relevant
            is_mega_hub=False,
            hub_type=self._detect_hub_type(paper) if is_hub else None,
            citation_count=citation_count,
            relevance_score=relevance,
            suppress_expansion=False,
            suppress_reason=None
        )

    def should_expand(
        self,
        paper: Paper,
        seed_relevance: Optional[float] = None
    ) -> bool:
        """
        quick check if paper should be fully expanded.
        """
        analysis = self.analyze_paper(paper, seed_relevance)
        return not analysis.suppress_expansion

    def get_expansion_limits(
        self,
        paper: Paper,
        seed_relevance: Optional[float] = None
    ) -> Dict[str, int]:
        """
        get adjusted expansion limits for a paper based on hub analysis.
        """
        analysis = self.analyze_paper(paper, seed_relevance)

        # default limits
        limits = {
            "max_refs": self.config.max_refs_per_node,
            "max_cites": self.config.max_cites_per_node,
            "max_author_works": self.config.max_author_works_per_node
        }

        if analysis.is_mega_hub:
            # don't expand at all
            return {"max_refs": 0, "max_cites": 0, "max_author_works": 0}

        if analysis.suppress_expansion:
            # reduced expansion
            return {
                "max_refs": 10,
                "max_cites": 5,
                "max_author_works": 0
            }

        if analysis.is_hub:
            # hub but relevant - moderate limits
            return {
                "max_refs": limits["max_refs"] // 2,
                "max_cites": limits["max_cites"] // 2,
                "max_author_works": limits["max_author_works"] // 2
            }

        return limits

    def compute_hub_penalty(
        self,
        paper: Paper,
        seed_relevance: Optional[float] = None
    ) -> float:
        """
        compute penalty factor for hub papers in scoring.
        returns value in [0, 1] where 1 = no penalty.
        """
        analysis = self.analyze_paper(paper, seed_relevance)

        if analysis.is_mega_hub:
            return 0.1  # severe penalty

        if analysis.suppress_expansion:
            return 0.3  # moderate penalty

        if analysis.is_hub:
            return 0.7  # mild penalty

        return 1.0  # no penalty

    def mark_methodology_papers(
        self,
        papers: List[Paper]
    ) -> int:
        """
        mark papers as methodology papers.
        returns count of papers marked.
        """
        count = 0
        for paper in papers:
            if self._is_methodology_paper(paper):
                paper.is_methodology = True
                count += 1
        return count

    # private helpers

    def _detect_hub_type(self, paper: Paper) -> Optional[str]:
        """detect the type of hub paper."""
        if self._is_methodology_paper(paper):
            return "methodology"

        if paper.is_review:
            return "review"

        # foundational if old and high citations
        if paper.year and paper.year < 2000:
            if paper.citation_count and paper.citation_count > 1000:
                return "foundational"

        return "high_citation"

    def _is_methodology_paper(self, paper: Paper) -> bool:
        """check if paper is a methodology/tool paper."""
        title = (paper.title or "").lower()
        venue = (paper.venue or "").lower()

        # check title for methodology keywords
        for kw in METHODOLOGY_KEYWORDS:
            if kw in title:
                return True

        # check venue
        for v in METHODOLOGY_VENUES:
            if v in venue:
                # also needs methodology keyword in title
                for kw in ["method", "tool", "software", "database", "server"]:
                    if kw in title:
                        return True

        # check for software naming patterns
        # (CamelCase tools like BLAST, ImageJ, etc.)
        import re
        if re.search(r'\b[A-Z]{2,}[a-z]*\b', paper.title or ""):
            # has acronym-like word, check context
            if any(kw in title for kw in ["software", "tool", "method", "algorithm"]):
                return True

        return False
