"""
verifier - orchestrates verification across all agents.
"""

from typing import List, Optional, Any
from datetime import datetime

from .core import (
    Check, VerificationResult, VerificationReport,
    Severity, error_check, warning_check, info_check
)
from .agent_checks import (
    verify_seed_resolver,
    verify_citation_walker,
    verify_author_resolver,
    verify_corpus_fetcher,
    verify_trajectory_analyzer,
    verify_collaborator_mapper,
    verify_topic_extractor,
    verify_gap_detector,
    verify_relevance_scorer,
    verify_field_resolver
)

from ..core.models import Paper
from ..agents.base import AgentResult
from ..agents.seed_resolver import ResolvedSeed
from ..agents.citation_walker import ClassifiedCitations
from ..agents.author_resolver import ResolvedAuthor
from ..agents.corpus_fetcher import AuthorCorpus
from ..agents.trajectory_analyzer import TrajectoryAnalysis
from ..agents.collaborator_mapper import CollaborationNetwork
from ..agents.topic_extractor import TopicAnalysis
from ..agents.gap_detector import GapAnalysis
from ..agents.relevance_scorer import RelevanceScore, ScoringContext
from ..agents.field_resolver import FieldResolution


class Verifier:
    """
    orchestrates verification across all agents.

    usage:
        verifier = Verifier()

        # verify individual agent results
        result = verifier.verify_seed_resolver(query, agent_result)

        # or verify pipeline results
        report = verifier.verify_pipeline(pipeline_results)
    """

    def __init__(self, strict: bool = False):
        """
        initialize verifier.

        args:
            strict: if True, warnings are treated as errors
        """
        self.strict = strict
        self._report = VerificationReport()

    def reset(self):
        """reset verification report."""
        self._report = VerificationReport()

    @property
    def report(self) -> VerificationReport:
        """get current verification report."""
        return self._report

    def _create_result(
        self,
        agent_name: str,
        input_summary: str,
        checks: List[Check]
    ) -> VerificationResult:
        """create verification result and add to report."""
        result = VerificationResult(
            agent_name=agent_name,
            input_summary=input_summary,
            checks=checks
        )
        self._report.add_result(result)
        return result

    # =========================================================================
    # Individual Agent Verification
    # =========================================================================

    def verify_seed_resolver(
        self,
        input_query: str,
        agent_result: AgentResult[ResolvedSeed]
    ) -> VerificationResult:
        """verify SeedResolver output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_seed_resolver(input_query, data)

        # add agent-level checks
        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "SeedResolver",
            f"query='{input_query[:50]}...'",
            checks
        )

    def verify_citation_walker(
        self,
        seed_paper: Paper,
        agent_result: AgentResult[ClassifiedCitations]
    ) -> VerificationResult:
        """verify CitationWalker output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_citation_walker(seed_paper, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "CitationWalker",
            f"paper='{seed_paper.title[:40]}...'",
            checks
        )

    def verify_author_resolver(
        self,
        query_name: str,
        context_paper: Optional[Paper],
        agent_result: AgentResult[ResolvedAuthor]
    ) -> VerificationResult:
        """verify AuthorResolver output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_author_resolver(query_name, context_paper, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "AuthorResolver",
            f"name='{query_name}'",
            checks
        )

    def verify_corpus_fetcher(
        self,
        author_id: str,
        expected_count: int,
        agent_result: AgentResult[AuthorCorpus]
    ) -> VerificationResult:
        """verify CorpusFetcher output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_corpus_fetcher(author_id, expected_count, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "CorpusFetcher",
            f"author={author_id}",
            checks
        )

    def verify_trajectory_analyzer(
        self,
        corpus: AuthorCorpus,
        agent_result: AgentResult[TrajectoryAnalysis]
    ) -> VerificationResult:
        """verify TrajectoryAnalyzer output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_trajectory_analyzer(corpus, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "TrajectoryAnalyzer",
            f"corpus={len(corpus.papers)} papers",
            checks
        )

    def verify_collaborator_mapper(
        self,
        author_name: str,
        corpus: AuthorCorpus,
        agent_result: AgentResult[CollaborationNetwork]
    ) -> VerificationResult:
        """verify CollaboratorMapper output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_collaborator_mapper(author_name, corpus, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "CollaboratorMapper",
            f"author='{author_name}'",
            checks
        )

    def verify_topic_extractor(
        self,
        papers: List[Paper],
        agent_result: AgentResult[TopicAnalysis]
    ) -> VerificationResult:
        """verify TopicExtractor output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_topic_extractor(papers, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "TopicExtractor",
            f"{len(papers)} papers",
            checks
        )

    def verify_gap_detector(
        self,
        papers: List[Paper],
        agent_result: AgentResult[GapAnalysis]
    ) -> VerificationResult:
        """verify GapDetector output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_gap_detector(papers, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "GapDetector",
            f"{len(papers)} papers",
            checks
        )

    def verify_relevance_scorer(
        self,
        paper: Paper,
        context: ScoringContext,
        agent_result: AgentResult[RelevanceScore]
    ) -> VerificationResult:
        """verify RelevanceScorer output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_relevance_scorer(paper, context, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        return self._create_result(
            "RelevanceScorer",
            f"paper='{paper.title[:40]}...'",
            checks
        )

    def verify_field_resolver(
        self,
        seed_paper: Optional[Paper],
        query: Optional[str],
        agent_result: AgentResult[FieldResolution]
    ) -> VerificationResult:
        """verify FieldResolver output."""
        data = agent_result.data if agent_result.ok else None
        checks = verify_field_resolver(seed_paper, query, data)

        if not agent_result.ok:
            checks.insert(0, error_check(
                name="agent_succeeded",
                passed=False,
                message=f"Agent failed: {agent_result.errors}",
                expected="success",
                actual="failed"
            ))

        input_summary = f"paper='{seed_paper.title[:30]}...'" if seed_paper else f"query='{query[:30]}...'"
        return self._create_result(
            "FieldResolver",
            input_summary,
            checks
        )

    # =========================================================================
    # Batch and Cross-Agent Verification
    # =========================================================================

    def add_cross_check(
        self,
        name: str,
        passed: bool,
        message: str,
        severity: Severity = Severity.WARNING
    ):
        """add a cross-agent check to the report."""
        # add to a special "CrossChecks" result
        if "CrossChecks" not in self._report.agent_results:
            self._report.agent_results["CrossChecks"] = VerificationResult(
                agent_name="CrossChecks",
                input_summary="cross-agent consistency checks"
            )

        self._report.agent_results["CrossChecks"].add_check(Check(
            name=name,
            passed=passed,
            severity=severity,
            message=message
        ))

    def verify_consistency(
        self,
        papers: List[Paper],
        authors: List[Any],  # AuthorProfile
        topics: Optional[TopicAnalysis],
        gaps: Optional[GapAnalysis]
    ):
        """run cross-agent consistency checks."""

        # check: paper count consistency
        unique_papers = len(set(p.id for p in papers))
        self.add_cross_check(
            name="papers_deduplicated",
            passed=unique_papers == len(papers),
            message=f"Papers should be deduplicated ({unique_papers} unique vs {len(papers)} total)"
        )

        # check: authors have papers in collection
        if authors:
            for author in authors[:3]:  # spot check first 3
                author_papers = [
                    p for p in papers
                    if p.authors and any(
                        author.name.lower() in a.lower()
                        for a in p.authors
                    )
                ]
                self.add_cross_check(
                    name=f"author_{author.name[:20]}_has_papers",
                    passed=len(author_papers) > 0,
                    message=f"Author '{author.name}' should have papers in collection"
                )

        # check: topics match papers
        if topics and topics.core_topics and papers:
            # verify top topic appears in papers
            top_topic = topics.core_topics[0].lower()
            papers_with_topic = sum(
                1 for p in papers
                if (p.title and top_topic in p.title.lower()) or
                   (p.concepts and any(top_topic in c.lower() for c in p.concepts))
            )
            self.add_cross_check(
                name="topics_match_papers",
                passed=papers_with_topic > 0,
                message=f"Top topic '{top_topic}' should appear in papers ({papers_with_topic} found)"
            )

        # check: gaps reference valid concepts
        if gaps and gaps.concept_gaps and topics and topics.core_topics:
            all_topics = set(t.lower() for t in topics.core_topics)
            if topics.emerging_topics:
                all_topics.update(t.lower() for t in topics.emerging_topics)

            for i, gap in enumerate(gaps.concept_gaps[:2]):
                a_in_topics = gap.concept_a.lower() in all_topics
                b_in_topics = gap.concept_b.lower() in all_topics

                self.add_cross_check(
                    name=f"gap_{i}_concepts_in_topics",
                    passed=a_in_topics or b_in_topics,
                    message=f"Gap concepts should relate to extracted topics",
                    severity=Severity.INFO
                )

    def finalize(self) -> VerificationReport:
        """finalize and return the verification report."""
        # add suggestions based on failures
        if self._report.failed_errors > 0:
            self._report.add_suggestion(
                "Critical errors found. Review agent inputs and data sources."
            )

        if self._report.failed_warnings > 3:
            self._report.add_suggestion(
                "Multiple warnings detected. Results may be incomplete or unreliable."
            )

        # check overall confidence
        if self._report.overall_confidence < 0.7:
            self._report.add_suggestion(
                f"Low confidence ({self._report.overall_confidence:.0%}). "
                "Consider verifying results manually."
            )

        return self._report
