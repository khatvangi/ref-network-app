"""
citation walker agent - fetch and classify references and citations for a paper.

answers key questions:
- what papers does this paper cite?
- what papers cite this paper?
- which references are foundational vs methodological?
- which citations are most relevant?

input: Paper object
output: ClassifiedCitations with refs[], cites[]
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from collections import defaultdict

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper
from ..providers.openalex import OpenAlexProvider


@dataclass
class ClassifiedReference:
    """
    a reference with classification.
    """
    paper: Paper

    # classification
    ref_type: str = "regular"  # "foundational", "methodological", "regular", "self_cite"
    position_hint: str = "unknown"  # "early" (intro), "middle", "late" (methods/discussion)

    # relevance signals
    is_review: bool = False
    is_highly_cited: bool = False
    shared_authors: List[str] = field(default_factory=list)
    shared_concepts: List[str] = field(default_factory=list)

    # score (higher = more important)
    importance_score: float = 0.0


@dataclass
class ClassifiedCitation:
    """
    a citing paper with classification.
    """
    paper: Paper

    # classification
    cite_type: str = "regular"  # "extension", "application", "comparison", "self_cite"

    # relevance signals
    years_after: int = 0
    shared_authors: List[str] = field(default_factory=list)
    shared_concepts: List[str] = field(default_factory=list)

    # score (higher = more relevant to follow)
    relevance_score: float = 0.0


@dataclass
class ClassifiedCitations:
    """
    complete citation analysis for a paper.
    """
    # source paper
    source_paper_id: str
    source_title: str

    # references (papers this paper cites)
    references: List[ClassifiedReference] = field(default_factory=list)
    reference_count: int = 0

    # citations (papers that cite this paper)
    citations: List[ClassifiedCitation] = field(default_factory=list)
    citation_count: int = 0

    # summary stats
    foundational_refs: List[str] = field(default_factory=list)  # paper IDs
    methodological_refs: List[str] = field(default_factory=list)
    self_cites: int = 0

    # key papers to follow
    key_references: List[str] = field(default_factory=list)
    key_citations: List[str] = field(default_factory=list)

    # insights
    insights: List[str] = field(default_factory=list)

    def add_insight(self, insight: str):
        self.insights.append(insight)


class CitationWalker(Agent):
    """
    fetch and classify references and citations for a paper.

    classifications:
    - References:
      - foundational: highly cited, review papers, early in field
      - methodological: same author group, technical methods
      - regular: typical citation
      - self_cite: shared authors with source paper

    - Citations:
      - extension: builds directly on the work
      - application: applies methods to new domain
      - comparison: compares or evaluates
      - self_cite: shared authors

    usage:
        walker = CitationWalker(provider)
        result = walker.run(paper)

        if result.ok:
            citations = result.data
            print(f"Foundational refs: {citations.foundational_refs}")
    """

    # thresholds
    HIGH_CITATION_THRESHOLD = 100  # citations to be "highly cited"
    FOUNDATIONAL_YEAR_GAP = 10  # years before source = likely foundational

    def __init__(
        self,
        provider: OpenAlexProvider,
        max_references: int = 50,
        max_citations: int = 50,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.provider = provider
        self.max_references = max_references
        self.max_citations = max_citations

    @property
    def name(self) -> str:
        return "CitationWalker"

    def execute(
        self,
        paper: Paper,
        fetch_refs: bool = True,
        fetch_cites: bool = True
    ) -> AgentResult[ClassifiedCitations]:
        """
        fetch and classify citations for a paper.

        args:
            paper: Paper object to analyze
            fetch_refs: whether to fetch references (papers cited by this paper)
            fetch_cites: whether to fetch citations (papers that cite this paper)

        returns:
            AgentResult with ClassifiedCitations
        """
        result = AgentResult[ClassifiedCitations](status=AgentStatus.SUCCESS)
        result.add_trace(f"walking citations for: {paper.title[:60] if paper.title else paper.id}...")

        # get paper ID for API calls
        paper_id = paper.openalex_id or paper.doi or paper.id
        if not paper_id:
            result.status = AgentStatus.FAILED
            result.add_error("NO_ID", "paper has no usable ID for citation lookup")
            return result

        # create result object
        citations = ClassifiedCitations(
            source_paper_id=paper.id,
            source_title=paper.title or "Unknown"
        )

        # get source paper concepts for comparison
        source_concepts = self._get_concepts(paper)
        source_authors = set(a.lower() for a in (paper.authors or []))
        source_year = paper.year or 2020

        # fetch references
        if fetch_refs:
            refs = self._fetch_references(paper_id, result)
            classified_refs = self._classify_references(
                refs, source_concepts, source_authors, source_year, result
            )
            citations.references = classified_refs
            citations.reference_count = len(refs)
            result.api_calls += 1

        # fetch citations
        if fetch_cites:
            cites = self._fetch_citations(paper_id, result)
            classified_cites = self._classify_citations(
                cites, source_concepts, source_authors, source_year, result
            )
            citations.citations = classified_cites
            citations.citation_count = len(cites)
            result.api_calls += 1

        # compute summary stats
        self._compute_summary(citations, result)

        # generate insights
        self._generate_insights(citations, paper, result)

        result.data = citations
        result.add_trace(
            f"found {citations.reference_count} refs, {citations.citation_count} cites"
        )

        return result

    def _fetch_references(self, paper_id: str, result: AgentResult) -> List[Paper]:
        """fetch papers cited by this paper."""
        result.add_trace("fetching references...")

        refs = self.provider.get_references(paper_id, limit=self.max_references)
        result.add_trace(f"fetched {len(refs)} references")

        return refs

    def _fetch_citations(self, paper_id: str, result: AgentResult) -> List[Paper]:
        """fetch papers that cite this paper."""
        result.add_trace("fetching citations...")

        cites = self.provider.get_citations(paper_id, limit=self.max_citations)
        result.add_trace(f"fetched {len(cites)} citations")

        return cites

    def _classify_references(
        self,
        refs: List[Paper],
        source_concepts: Set[str],
        source_authors: Set[str],
        source_year: int,
        result: AgentResult
    ) -> List[ClassifiedReference]:
        """classify each reference."""
        result.add_trace("classifying references...")

        classified = []

        for i, ref in enumerate(refs):
            ref_authors = set(a.lower() for a in (ref.authors or []))
            ref_concepts = self._get_concepts(ref)
            ref_year = ref.year or source_year

            # determine shared elements
            shared_auth = list(source_authors & ref_authors)
            shared_conc = list(source_concepts & ref_concepts)

            # determine position hint (early refs = foundational)
            total_refs = len(refs)
            if i < total_refs * 0.2:
                position = "early"
            elif i > total_refs * 0.7:
                position = "late"
            else:
                position = "middle"

            # classify type
            ref_type = self._classify_ref_type(
                ref, shared_auth, ref_year, source_year, position
            )

            # compute importance score
            importance = self._compute_ref_importance(
                ref, shared_conc, ref_year, source_year, position
            )

            classified.append(ClassifiedReference(
                paper=ref,
                ref_type=ref_type,
                position_hint=position,
                is_review=ref.is_review,
                is_highly_cited=(ref.citation_count or 0) >= self.HIGH_CITATION_THRESHOLD,
                shared_authors=shared_auth,
                shared_concepts=shared_conc[:5],
                importance_score=importance
            ))

        # sort by importance
        classified.sort(key=lambda r: -r.importance_score)

        return classified

    def _classify_ref_type(
        self,
        ref: Paper,
        shared_authors: List[str],
        ref_year: int,
        source_year: int,
        position: str
    ) -> str:
        """classify reference type."""
        # self-cite
        if shared_authors:
            return "self_cite"

        # foundational: old, highly cited, early position
        year_gap = source_year - ref_year
        is_old = year_gap >= self.FOUNDATIONAL_YEAR_GAP
        is_highly_cited = (ref.citation_count or 0) >= self.HIGH_CITATION_THRESHOLD

        if (is_old or is_highly_cited) and position == "early":
            return "foundational"

        # methodological: review, methods paper, late position
        if ref.is_review or ref.is_methodology:
            return "methodological"

        if position == "late":
            return "methodological"

        return "regular"

    def _compute_ref_importance(
        self,
        ref: Paper,
        shared_concepts: List[str],
        ref_year: int,
        source_year: int,
        position: str
    ) -> float:
        """compute importance score for a reference."""
        import math

        score = 0.0

        # citation count (log scale)
        score += math.log(1 + (ref.citation_count or 0)) * 0.3

        # concept overlap
        score += len(shared_concepts) * 0.2

        # recency (newer = more relevant, but foundational old papers also important)
        year_gap = source_year - ref_year
        if year_gap <= 5:
            score += 0.3
        elif year_gap >= 10:
            # old papers are foundational
            score += 0.2

        # position bonus
        if position == "early":
            score += 0.2  # foundational
        elif position == "late":
            score += 0.1  # methodological

        return score

    def _classify_citations(
        self,
        cites: List[Paper],
        source_concepts: Set[str],
        source_authors: Set[str],
        source_year: int,
        result: AgentResult
    ) -> List[ClassifiedCitation]:
        """classify each citation."""
        result.add_trace("classifying citations...")

        classified = []

        for cite in cites:
            cite_authors = set(a.lower() for a in (cite.authors or []))
            cite_concepts = self._get_concepts(cite)
            cite_year = cite.year or source_year

            # determine shared elements
            shared_auth = list(source_authors & cite_authors)
            shared_conc = list(source_concepts & cite_concepts)

            years_after = cite_year - source_year

            # classify type
            cite_type = self._classify_cite_type(
                cite, shared_auth, shared_conc, years_after
            )

            # compute relevance score
            relevance = self._compute_cite_relevance(
                cite, shared_conc, years_after
            )

            classified.append(ClassifiedCitation(
                paper=cite,
                cite_type=cite_type,
                years_after=years_after,
                shared_authors=shared_auth,
                shared_concepts=shared_conc[:5],
                relevance_score=relevance
            ))

        # sort by relevance
        classified.sort(key=lambda c: -c.relevance_score)

        return classified

    def _classify_cite_type(
        self,
        cite: Paper,
        shared_authors: List[str],
        shared_concepts: List[str],
        years_after: int
    ) -> str:
        """classify citation type."""
        # self-cite
        if shared_authors:
            return "self_cite"

        # extension: high concept overlap, recent
        if len(shared_concepts) >= 3 and years_after <= 3:
            return "extension"

        # application: some overlap, different domain signals
        if 1 <= len(shared_concepts) <= 2:
            return "application"

        # comparison: review papers citing, or longer time gap
        if cite.is_review or years_after >= 5:
            return "comparison"

        return "regular"

    def _compute_cite_relevance(
        self,
        cite: Paper,
        shared_concepts: List[str],
        years_after: int
    ) -> float:
        """compute relevance score for a citation."""
        import math

        score = 0.0

        # citation count of citing paper
        score += math.log(1 + (cite.citation_count or 0)) * 0.3

        # concept overlap
        score += len(shared_concepts) * 0.3

        # recency (recent citations are more relevant)
        if years_after <= 2:
            score += 0.3
        elif years_after <= 5:
            score += 0.2
        else:
            score += 0.1

        return score

    def _get_concepts(self, paper: Paper) -> Set[str]:
        """extract concept names from paper."""
        if not paper.concepts:
            return set()
        return {c.get('name', '').lower() for c in paper.concepts if c.get('name')}

    def _compute_summary(self, citations: ClassifiedCitations, result: AgentResult):
        """compute summary statistics."""
        # foundational refs
        citations.foundational_refs = [
            r.paper.id for r in citations.references
            if r.ref_type == "foundational"
        ]

        # methodological refs
        citations.methodological_refs = [
            r.paper.id for r in citations.references
            if r.ref_type == "methodological"
        ]

        # self-cites
        citations.self_cites = sum(
            1 for r in citations.references if r.ref_type == "self_cite"
        ) + sum(
            1 for c in citations.citations if c.cite_type == "self_cite"
        )

        # key references (top 5 by importance, excluding self-cites)
        citations.key_references = [
            r.paper.id for r in citations.references[:10]
            if r.ref_type != "self_cite"
        ][:5]

        # key citations (top 5 by relevance, excluding self-cites)
        citations.key_citations = [
            c.paper.id for c in citations.citations[:10]
            if c.cite_type != "self_cite"
        ][:5]

    def _generate_insights(
        self,
        citations: ClassifiedCitations,
        paper: Paper,
        result: AgentResult
    ):
        """generate human-readable insights."""
        # reference breakdown
        ref_types = defaultdict(int)
        for r in citations.references:
            ref_types[r.ref_type] += 1

        if citations.references:
            citations.add_insight(
                f"References: {citations.reference_count} total - "
                f"{ref_types.get('foundational', 0)} foundational, "
                f"{ref_types.get('methodological', 0)} methodological, "
                f"{ref_types.get('self_cite', 0)} self-cites"
            )

        # citation breakdown
        cite_types = defaultdict(int)
        for c in citations.citations:
            cite_types[c.cite_type] += 1

        if citations.citations:
            citations.add_insight(
                f"Citations: {citations.citation_count} total - "
                f"{cite_types.get('extension', 0)} extensions, "
                f"{cite_types.get('application', 0)} applications, "
                f"{cite_types.get('self_cite', 0)} self-cites"
            )

        # highly cited references
        highly_cited = [r for r in citations.references if r.is_highly_cited]
        if highly_cited:
            titles = [r.paper.title[:40] + "..." for r in highly_cited[:3]]
            citations.add_insight(
                f"Highly cited references ({len(highly_cited)}): {', '.join(titles)}"
            )

        # recent extensions
        extensions = [
            c for c in citations.citations
            if c.cite_type == "extension" and c.years_after <= 3
        ]
        if extensions:
            citations.add_insight(
                f"Recent extensions: {len(extensions)} papers building directly on this work"
            )
