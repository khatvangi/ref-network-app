"""
relevance scorer agent - score paper relevance to a query/context.

used to:
- rank papers in search results
- filter candidate papers for expansion
- prioritize papers for reading lists

input: Paper + context (seed papers, query terms, target concepts)
output: RelevanceScore with score and explanation
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
from collections import Counter

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper


@dataclass
class RelevanceScore:
    """
    relevance score for a paper.
    """
    paper_id: str
    title: str

    # overall score (0-1)
    score: float = 0.0

    # component scores
    concept_score: float = 0.0      # topic/concept overlap
    author_score: float = 0.0       # author network overlap
    citation_score: float = 0.0     # citation importance
    recency_score: float = 0.0      # how recent
    quality_score: float = 0.0      # venue/citation quality

    # explanation
    explanation: str = ""
    matching_concepts: List[str] = field(default_factory=list)
    matching_authors: List[str] = field(default_factory=list)

    # flags
    is_highly_relevant: bool = False  # score > 0.8
    is_peripheral: bool = False       # score < 0.3


@dataclass
class ScoringContext:
    """
    context for relevance scoring.
    """
    # seed papers (the core of what user cares about)
    seed_papers: List[Paper] = field(default_factory=list)

    # target concepts (what topics to prioritize)
    target_concepts: List[str] = field(default_factory=list)

    # target authors (whose work to prioritize)
    target_authors: List[str] = field(default_factory=list)

    # query terms (if text search was used)
    query_terms: List[str] = field(default_factory=list)

    # year range (prefer papers in this range)
    min_year: Optional[int] = None
    max_year: Optional[int] = None


class RelevanceScorer(Agent):
    """
    score paper relevance to a given context.

    scoring components:
    1. Concept overlap: shared topics with seed papers/targets
    2. Author overlap: shared authors with seed papers/targets
    3. Citation importance: citation count, h-index of venue
    4. Recency: prefer recent papers (with decay)
    5. Quality signals: venue reputation, review status

    usage:
        scorer = RelevanceScorer()
        context = ScoringContext(
            seed_papers=[seed1, seed2],
            target_concepts=["aminoacyl tRNA synthetase"]
        )

        result = scorer.run(paper, context)
        if result.ok:
            print(f"Relevance: {result.data.score:.2f}")
    """

    # weights for score components
    CONCEPT_WEIGHT = 0.35
    AUTHOR_WEIGHT = 0.15
    CITATION_WEIGHT = 0.20
    RECENCY_WEIGHT = 0.15
    QUALITY_WEIGHT = 0.15

    # thresholds
    HIGH_RELEVANCE_THRESHOLD = 0.7
    LOW_RELEVANCE_THRESHOLD = 0.3
    CITATION_SCALE = 100  # citations for max score

    def __init__(
        self,
        concept_weight: float = 0.35,
        recency_decay: float = 0.1,  # decay per year
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.concept_weight = concept_weight
        self.recency_decay = recency_decay

    @property
    def name(self) -> str:
        return "RelevanceScorer"

    def execute(
        self,
        paper: Paper,
        context: ScoringContext
    ) -> AgentResult[RelevanceScore]:
        """
        score a paper's relevance to the given context.

        args:
            paper: Paper to score
            context: ScoringContext with seeds, targets, etc.

        returns:
            AgentResult with RelevanceScore
        """
        result = AgentResult[RelevanceScore](status=AgentStatus.SUCCESS)
        result.add_trace(f"scoring: {paper.title[:50] if paper.title else paper.id}...")

        # build context indices
        context_concepts = self._build_concept_index(context)
        context_authors = self._build_author_index(context)

        # compute component scores
        concept_score, matching_concepts = self._score_concepts(paper, context_concepts)
        author_score, matching_authors = self._score_authors(paper, context_authors)
        citation_score = self._score_citations(paper)
        recency_score = self._score_recency(paper, context)
        quality_score = self._score_quality(paper)

        # weighted combination
        overall = (
            concept_score * self.CONCEPT_WEIGHT +
            author_score * self.AUTHOR_WEIGHT +
            citation_score * self.CITATION_WEIGHT +
            recency_score * self.RECENCY_WEIGHT +
            quality_score * self.QUALITY_WEIGHT
        )

        # normalize to 0-1
        overall = min(1.0, max(0.0, overall))

        # build explanation
        explanation = self._build_explanation(
            paper, overall, concept_score, author_score,
            citation_score, recency_score, quality_score,
            matching_concepts, matching_authors
        )

        # create result
        score = RelevanceScore(
            paper_id=paper.id,
            title=paper.title or "Unknown",
            score=overall,
            concept_score=concept_score,
            author_score=author_score,
            citation_score=citation_score,
            recency_score=recency_score,
            quality_score=quality_score,
            explanation=explanation,
            matching_concepts=matching_concepts[:5],
            matching_authors=matching_authors[:5],
            is_highly_relevant=overall >= self.HIGH_RELEVANCE_THRESHOLD,
            is_peripheral=overall < self.LOW_RELEVANCE_THRESHOLD
        )

        result.data = score
        result.add_trace(f"score: {overall:.2f} (concept={concept_score:.2f}, auth={author_score:.2f})")

        return result

    def score_batch(
        self,
        papers: List[Paper],
        context: ScoringContext
    ) -> AgentResult[List[RelevanceScore]]:
        """
        score multiple papers efficiently.

        args:
            papers: list of Papers to score
            context: ScoringContext

        returns:
            AgentResult with list of RelevanceScores (sorted by score desc)
        """
        result = AgentResult[List[RelevanceScore]](status=AgentStatus.SUCCESS)
        result.add_trace(f"batch scoring {len(papers)} papers...")

        # build context indices once
        context_concepts = self._build_concept_index(context)
        context_authors = self._build_author_index(context)

        scores = []
        for paper in papers:
            # compute scores
            concept_score, matching_concepts = self._score_concepts(paper, context_concepts)
            author_score, matching_authors = self._score_authors(paper, context_authors)
            citation_score = self._score_citations(paper)
            recency_score = self._score_recency(paper, context)
            quality_score = self._score_quality(paper)

            overall = (
                concept_score * self.CONCEPT_WEIGHT +
                author_score * self.AUTHOR_WEIGHT +
                citation_score * self.CITATION_WEIGHT +
                recency_score * self.RECENCY_WEIGHT +
                quality_score * self.QUALITY_WEIGHT
            )
            overall = min(1.0, max(0.0, overall))

            explanation = self._build_explanation(
                paper, overall, concept_score, author_score,
                citation_score, recency_score, quality_score,
                matching_concepts, matching_authors
            )

            scores.append(RelevanceScore(
                paper_id=paper.id,
                title=paper.title or "Unknown",
                score=overall,
                concept_score=concept_score,
                author_score=author_score,
                citation_score=citation_score,
                recency_score=recency_score,
                quality_score=quality_score,
                explanation=explanation,
                matching_concepts=matching_concepts[:5],
                matching_authors=matching_authors[:5],
                is_highly_relevant=overall >= self.HIGH_RELEVANCE_THRESHOLD,
                is_peripheral=overall < self.LOW_RELEVANCE_THRESHOLD
            ))

        # sort by score descending
        scores.sort(key=lambda s: -s.score)

        result.data = scores
        result.add_trace(f"scored {len(scores)} papers, {sum(1 for s in scores if s.is_highly_relevant)} highly relevant")

        return result

    def _build_concept_index(self, context: ScoringContext) -> Dict[str, float]:
        """
        build concept -> weight index from context.
        """
        concepts: Dict[str, float] = Counter()

        # add target concepts with high weight
        for c in context.target_concepts:
            concepts[c.lower()] += 2.0

        # add concepts from seed papers
        for paper in context.seed_papers:
            if paper.concepts:
                for concept in paper.concepts:
                    name = concept.get('name', '').lower()
                    score = concept.get('score', 0.5)
                    if name:
                        concepts[name] += score

        # add query terms
        for term in context.query_terms:
            concepts[term.lower()] += 1.0

        return concepts

    def _build_author_index(self, context: ScoringContext) -> Set[str]:
        """
        build author name set from context.
        """
        authors = set()

        # add target authors
        for a in context.target_authors:
            authors.add(a.lower())

        # add authors from seed papers
        for paper in context.seed_papers:
            for a in (paper.authors or []):
                authors.add(a.lower())

        return authors

    def _score_concepts(
        self,
        paper: Paper,
        context_concepts: Dict[str, float]
    ) -> tuple[float, List[str]]:
        """
        score concept overlap.
        """
        if not paper.concepts or not context_concepts:
            return (0.0, [])

        matching = []
        total_weight = 0.0

        for concept in paper.concepts:
            name = concept.get('name', '').lower()
            paper_score = concept.get('score', 0.5)

            if name in context_concepts:
                context_weight = context_concepts[name]
                total_weight += paper_score * context_weight
                matching.append(name)

        # normalize by max possible
        max_possible = sum(context_concepts.values())
        if max_possible > 0:
            score = min(1.0, total_weight / (max_possible * 0.5))
        else:
            score = 0.0

        return (score, matching)

    def _score_authors(
        self,
        paper: Paper,
        context_authors: Set[str]
    ) -> tuple[float, List[str]]:
        """
        score author overlap.
        """
        if not paper.authors or not context_authors:
            return (0.0, [])

        matching = []
        for author in paper.authors:
            if author.lower() in context_authors:
                matching.append(author)

        # score based on fraction of paper authors that match
        if matching:
            score = min(1.0, len(matching) / 2)  # cap at 2 matching authors
        else:
            score = 0.0

        return (score, matching)

    def _score_citations(self, paper: Paper) -> float:
        """
        score based on citation count (log scale).
        """
        citations = paper.citation_count or 0

        if citations == 0:
            return 0.0

        # log scale with cap
        score = math.log(1 + citations) / math.log(1 + self.CITATION_SCALE)
        return min(1.0, score)

    def _score_recency(self, paper: Paper, context: ScoringContext) -> float:
        """
        score based on recency with decay.
        """
        if not paper.year:
            return 0.5  # unknown year gets neutral score

        current_year = 2025  # could be made dynamic

        # check if within target range
        if context.min_year and paper.year < context.min_year:
            return 0.2
        if context.max_year and paper.year > context.max_year:
            return 0.2

        # decay based on age
        age = current_year - paper.year
        score = math.exp(-self.recency_decay * age)

        return score

    def _score_quality(self, paper: Paper) -> float:
        """
        score based on quality signals.
        """
        score = 0.5  # baseline

        # open access bonus
        if paper.is_oa:
            score += 0.1

        # review paper (can be good or bad depending on use case)
        if paper.is_review:
            score += 0.1

        # has abstract
        if paper.abstract:
            score += 0.1

        # venue (would need venue reputation data)
        # for now, just check if venue is known
        if paper.venue:
            score += 0.1

        return min(1.0, score)

    def _build_explanation(
        self,
        paper: Paper,
        overall: float,
        concept_score: float,
        author_score: float,
        citation_score: float,
        recency_score: float,
        quality_score: float,
        matching_concepts: List[str],
        matching_authors: List[str]
    ) -> str:
        """
        build human-readable explanation of the score.
        """
        parts = []

        # overall assessment
        if overall >= self.HIGH_RELEVANCE_THRESHOLD:
            parts.append("Highly relevant.")
        elif overall >= 0.5:
            parts.append("Moderately relevant.")
        elif overall >= self.LOW_RELEVANCE_THRESHOLD:
            parts.append("Somewhat relevant.")
        else:
            parts.append("Low relevance.")

        # concept explanation
        if matching_concepts:
            parts.append(f"Matching topics: {', '.join(matching_concepts[:3])}.")
        elif concept_score < 0.2:
            parts.append("Few matching topics.")

        # author explanation
        if matching_authors:
            parts.append(f"Shared authors: {', '.join(matching_authors[:2])}.")

        # citation explanation
        if citation_score > 0.7:
            parts.append(f"Highly cited ({paper.citation_count}).")
        elif paper.citation_count and paper.citation_count > 10:
            parts.append(f"{paper.citation_count} citations.")

        # recency
        if paper.year:
            if paper.year >= 2023:
                parts.append("Recent publication.")
            elif paper.year <= 2010:
                parts.append("Older but potentially foundational.")

        return " ".join(parts)
