"""
per-agent verification checks.

each function takes agent input + output and returns a list of checks.
"""

import random
from typing import List, Optional, Set
from datetime import datetime
from difflib import SequenceMatcher

from .core import Check, Severity, error_check, warning_check, info_check
from ..core.models import Paper
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


def name_similarity(name1: str, name2: str) -> float:
    """compute similarity between two names (0.0-1.0)."""
    if not name1 or not name2:
        return 0.0
    # normalize
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    return SequenceMatcher(None, n1, n2).ratio()


# =============================================================================
# SeedResolver Checks
# =============================================================================

def verify_seed_resolver(
    input_query: str,
    result: Optional[ResolvedSeed]
) -> List[Check]:
    """verify SeedResolver output."""
    checks = []
    current_year = datetime.now().year

    # check: result exists
    if result is None or result.paper is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="SeedResolver returned no result",
            expected="Paper object",
            actual=None
        ))
        return checks  # can't check further

    paper = result.paper

    # check: has required fields
    has_id = paper.id is not None and len(paper.id) > 0
    has_title = paper.title is not None and len(paper.title) > 0
    checks.append(error_check(
        name="has_required_fields",
        passed=has_id and has_title,
        message="Paper must have id and title",
        expected="id and title present",
        actual=f"id={'yes' if has_id else 'no'}, title={'yes' if has_title else 'no'}"
    ))

    # check: DOI matches (if DOI was input)
    if input_query.startswith("10."):
        doi_matches = paper.doi and input_query.lower() in paper.doi.lower()
        checks.append(error_check(
            name="doi_matches",
            passed=doi_matches,
            message="DOI in result should match input DOI",
            expected=input_query,
            actual=paper.doi
        ))

    # check: year plausible
    if paper.year:
        year_ok = 1800 <= paper.year <= current_year + 1
        checks.append(warning_check(
            name="year_plausible",
            passed=year_ok,
            message=f"Publication year should be between 1800 and {current_year+1}",
            expected=f"1800-{current_year+1}",
            actual=paper.year
        ))

    # check: has authors
    has_authors = paper.authors and len(paper.authors) > 0
    checks.append(warning_check(
        name="has_authors",
        passed=has_authors,
        message="Paper should have at least one author",
        expected=">=1 authors",
        actual=len(paper.authors) if paper.authors else 0
    ))

    # check: title not suspiciously short
    if paper.title:
        title_ok = len(paper.title) >= 10
        checks.append(info_check(
            name="title_length",
            passed=title_ok,
            message="Title should be at least 10 characters",
            expected=">=10 chars",
            actual=len(paper.title)
        ))

    return checks


# =============================================================================
# CitationWalker Checks
# =============================================================================

def verify_citation_walker(
    seed_paper: Paper,
    result: Optional[ClassifiedCitations]
) -> List[Check]:
    """verify CitationWalker output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="CitationWalker returned no result",
            expected="Citations object",
            actual=None
        ))
        return checks

    # check: has references (most papers have refs)
    has_refs = len(result.references) > 0
    checks.append(warning_check(
        name="has_references",
        passed=has_refs,
        message="Paper should have at least some references",
        expected=">=1 references",
        actual=len(result.references)
    ))

    # check: ref count reasonable (not impossibly high)
    ref_count_ok = len(result.references) < 500
    checks.append(warning_check(
        name="ref_count_reasonable",
        passed=ref_count_ok,
        message="Reference count should be < 500",
        expected="<500",
        actual=len(result.references)
    ))

    # check: references are older than seed (mostly)
    if seed_paper.year and result.references:
        older_refs = [
            r for r in result.references
            if r.paper.year and r.paper.year <= seed_paper.year
        ]
        refs_with_year = [r for r in result.references if r.paper.year]
        if refs_with_year:
            ratio = len(older_refs) / len(refs_with_year)
            checks.append(warning_check(
                name="refs_are_older",
                passed=ratio >= 0.7,
                message="Most references should predate the citing paper",
                expected=">=70% older",
                actual=f"{ratio*100:.0f}%"
            ))

    # check: citations are newer than seed (mostly)
    if seed_paper.year and result.citations:
        newer_cites = [
            c for c in result.citations
            if c.paper.year and c.paper.year >= seed_paper.year
        ]
        cites_with_year = [c for c in result.citations if c.paper.year]
        if cites_with_year:
            ratio = len(newer_cites) / len(cites_with_year)
            checks.append(warning_check(
                name="cites_are_newer",
                passed=ratio >= 0.7,
                message="Most citations should postdate the cited paper",
                expected=">=70% newer",
                actual=f"{ratio*100:.0f}%"
            ))

    # check: no self-reference
    ref_ids = {r.paper.id for r in result.references}
    self_ref = seed_paper.id in ref_ids
    checks.append(error_check(
        name="no_self_reference",
        passed=not self_ref,
        message="Paper should not appear in its own references",
        expected="seed not in refs",
        actual="seed in refs" if self_ref else "ok"
    ))

    return checks


# =============================================================================
# AuthorResolver Checks
# =============================================================================

def verify_author_resolver(
    query_name: str,
    context_paper: Optional[Paper],
    result: Optional[ResolvedAuthor]
) -> List[Check]:
    """verify AuthorResolver output."""
    checks = []

    if result is None or result.author_info is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="AuthorResolver returned no result",
            expected="AuthorInfo object",
            actual=None
        ))
        return checks

    author = result.author_info

    # check: has OpenAlex ID
    has_id = author.openalex_id is not None and len(author.openalex_id) > 0
    checks.append(error_check(
        name="has_openalex_id",
        passed=has_id,
        message="Author must have OpenAlex ID",
        expected="non-empty ID",
        actual=author.openalex_id
    ))

    # check: name similarity
    similarity = name_similarity(query_name, author.name)
    checks.append(warning_check(
        name="name_matches",
        passed=similarity >= 0.5,
        message=f"Name similarity to query: {similarity:.0%}",
        expected=query_name,
        actual=author.name
    ))

    # check: has papers
    has_papers = author.paper_count and author.paper_count > 0
    checks.append(warning_check(
        name="has_papers",
        passed=has_papers,
        message="Author should have at least 1 paper",
        expected=">=1 papers",
        actual=author.paper_count
    ))

    # check: context match (if context paper provided)
    if context_paper and context_paper.authors:
        matches_context = any(
            name_similarity(author.name, a) >= 0.7
            for a in context_paper.authors
        )
        checks.append(warning_check(
            name="context_match",
            passed=matches_context,
            message="Resolved author should appear on context paper",
            expected="author on paper",
            actual="found" if matches_context else "not found"
        ))

    # check: paper count plausible
    if author.paper_count:
        count_ok = 1 <= author.paper_count <= 5000
        checks.append(info_check(
            name="paper_count_plausible",
            passed=count_ok,
            message="Paper count should be between 1 and 5000",
            expected="1-5000",
            actual=author.paper_count
        ))

    return checks


# =============================================================================
# CorpusFetcher Checks
# =============================================================================

def verify_corpus_fetcher(
    author_id: str,
    expected_count: int,
    result: Optional[AuthorCorpus]
) -> List[Check]:
    """verify CorpusFetcher output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="CorpusFetcher returned no result",
            expected="AuthorCorpus object",
            actual=None
        ))
        return checks

    # check: has papers
    has_papers = len(result.papers) > 0
    checks.append(error_check(
        name="papers_not_empty",
        passed=has_papers,
        message="Corpus should have at least 1 paper",
        expected=">=1 papers",
        actual=len(result.papers)
    ))

    if not has_papers:
        return checks

    # check: no duplicate paper IDs
    ids = [p.id for p in result.papers]
    unique_ids = set(ids)
    no_dupes = len(ids) == len(unique_ids)
    checks.append(warning_check(
        name="no_duplicates",
        passed=no_dupes,
        message="Corpus should not contain duplicate papers",
        expected=f"{len(ids)} unique",
        actual=f"{len(unique_ids)} unique ({len(ids)-len(unique_ids)} dupes)"
    ))

    # check: count matches profile (within 50%)
    if expected_count > 0:
        ratio = len(result.papers) / expected_count
        count_ok = 0.3 <= ratio <= 1.5  # allow some variance
        checks.append(warning_check(
            name="count_matches_profile",
            passed=count_ok,
            message="Corpus size should approximate author's paper count",
            expected=f"~{expected_count} papers",
            actual=f"{len(result.papers)} papers ({ratio*100:.0f}%)"
        ))

    # check: years span reasonable range
    years = [p.year for p in result.papers if p.year]
    if years:
        year_span = max(years) - min(years)
        span_ok = year_span <= 70  # max 70 year career
        checks.append(info_check(
            name="years_span_reasonable",
            passed=span_ok,
            message="Year span should be <= 70 years",
            expected="<=70 years",
            actual=f"{year_span} years ({min(years)}-{max(years)})"
        ))

    # check: author appears in papers (sample check)
    # this is expensive, so we'd need author IDs in papers
    # for now, skip this check

    return checks


# =============================================================================
# TrajectoryAnalyzer Checks
# =============================================================================

def verify_trajectory_analyzer(
    corpus: AuthorCorpus,
    result: Optional[TrajectoryAnalysis]
) -> List[Check]:
    """verify TrajectoryAnalyzer output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="TrajectoryAnalyzer returned no result",
            expected="TrajectoryAnalysis object",
            actual=None
        ))
        return checks

    # check: has phases (if corpus has papers)
    if corpus.papers:
        has_phases = len(result.phases) > 0
        checks.append(warning_check(
            name="has_phases",
            passed=has_phases,
            message="Trajectory should have at least 1 phase",
            expected=">=1 phases",
            actual=len(result.phases)
        ))

    if not result.phases:
        return checks

    # check: phases cover corpus year range
    corpus_years = [p.year for p in corpus.papers if p.year]
    if corpus_years:
        corpus_min, corpus_max = min(corpus_years), max(corpus_years)
        phase_min = min(p.start_year for p in result.phases)
        phase_max = max(p.end_year for p in result.phases)

        covers_range = (
            phase_min <= corpus_min + 3 and
            phase_max >= corpus_max - 1
        )
        checks.append(warning_check(
            name="phases_cover_range",
            passed=covers_range,
            message="Phases should cover the corpus year range",
            expected=f"{corpus_min}-{corpus_max}",
            actual=f"{phase_min}-{phase_max}"
        ))

    # check: phases non-overlapping
    sorted_phases = sorted(result.phases, key=lambda p: p.start_year)
    overlaps = 0
    for i in range(len(sorted_phases) - 1):
        if sorted_phases[i].end_year > sorted_phases[i+1].start_year:
            overlaps += 1

    checks.append(error_check(
        name="phases_non_overlapping",
        passed=overlaps == 0,
        message="Career phases should not overlap",
        expected="no overlaps",
        actual=f"{overlaps} overlaps"
    ))

    # check: each phase has concepts
    phases_with_concepts = sum(
        1 for p in result.phases
        if p.dominant_concepts and len(p.dominant_concepts) > 0
    )
    checks.append(warning_check(
        name="phases_have_concepts",
        passed=phases_with_concepts == len(result.phases),
        message="Each phase should have dominant concepts",
        expected=f"{len(result.phases)} phases with concepts",
        actual=f"{phases_with_concepts} phases with concepts"
    ))

    return checks


# =============================================================================
# CollaboratorMapper Checks
# =============================================================================

def verify_collaborator_mapper(
    author_name: str,
    corpus: AuthorCorpus,
    result: Optional[CollaborationNetwork]
) -> List[Check]:
    """verify CollaboratorMapper output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="CollaboratorMapper returned no result",
            expected="CollaborationNetwork object",
            actual=None
        ))
        return checks

    # check: has collaborators (if multi-author papers exist)
    multi_author_papers = [
        p for p in corpus.papers
        if p.authors and len(p.authors) > 1
    ]
    if multi_author_papers:
        has_collabs = len(result.top_collaborators) > 0
        checks.append(warning_check(
            name="has_collaborators",
            passed=has_collabs,
            message="Should find collaborators in multi-author papers",
            expected=">=1 collaborators",
            actual=len(result.top_collaborators)
        ))

    # check: no self-collaboration
    self_collab = any(
        name_similarity(author_name, c) >= 0.85
        for c in result.top_collaborators
    )
    checks.append(error_check(
        name="no_self_collab",
        passed=not self_collab,
        message="Author should not appear as own collaborator",
        expected="author not in collaborators",
        actual="self found" if self_collab else "ok"
    ))

    # check: collab counts valid (not more than total papers)
    if result.collaborator_papers and corpus.papers:
        max_collab_count = max(result.collaborator_papers.values()) if result.collaborator_papers else 0
        counts_valid = max_collab_count <= len(corpus.papers)
        checks.append(warning_check(
            name="collab_counts_valid",
            passed=counts_valid,
            message="Collaboration counts should not exceed total papers",
            expected=f"<={len(corpus.papers)}",
            actual=max_collab_count
        ))

    # check: top collaborator verifiable (spot check)
    if result.top_collaborators:
        top_collab = result.top_collaborators[0]
        papers_with_collab = sum(
            1 for p in corpus.papers
            if p.authors and any(
                name_similarity(top_collab, a) >= 0.75
                for a in p.authors
            )
        )
        expected_count = result.collaborator_papers.get(top_collab, 0)

        # allow some variance due to name matching
        verifiable = papers_with_collab >= expected_count * 0.5
        checks.append(warning_check(
            name="top_collab_verifiable",
            passed=verifiable,
            message="Top collaborator paper count should be verifiable",
            expected=f"~{expected_count} papers",
            actual=f"{papers_with_collab} papers"
        ))

    return checks


# =============================================================================
# TopicExtractor Checks
# =============================================================================

def verify_topic_extractor(
    papers: List[Paper],
    result: Optional[TopicAnalysis]
) -> List[Check]:
    """verify TopicExtractor output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="TopicExtractor returned no result",
            expected="TopicAnalysis object",
            actual=None
        ))
        return checks

    # check: has topics
    has_topics = len(result.core_topics) > 0
    checks.append(warning_check(
        name="has_topics",
        passed=has_topics,
        message="Should extract at least 1 topic",
        expected=">=1 topics",
        actual=len(result.core_topics)
    ))

    if not papers:
        return checks

    # check: core topics are frequent (spot check top topic)
    if result.core_topics:
        top_topic = result.core_topics[0].lower()

        # count papers mentioning topic in title or concepts
        papers_with_topic = 0
        for p in papers:
            in_title = p.title and top_topic in p.title.lower()
            in_concepts = p.concepts and any(
                top_topic in c.lower() for c in p.concepts
            )
            if in_title or in_concepts:
                papers_with_topic += 1

        frequency = papers_with_topic / len(papers) if papers else 0
        checks.append(warning_check(
            name="core_topics_frequent",
            passed=frequency >= 0.03,  # at least 3%
            message=f"Core topic '{top_topic}' should appear in papers",
            expected=">=3% frequency",
            actual=f"{frequency*100:.1f}%"
        ))

    # check: no overlap between core and declining
    if result.core_topics and result.declining_topics:
        core_set = set(t.lower() for t in result.core_topics)
        declining_set = set(t.lower() for t in result.declining_topics)
        overlap = core_set & declining_set

        checks.append(warning_check(
            name="no_core_declining_overlap",
            passed=len(overlap) == 0,
            message="Topics shouldn't be both core and declining",
            expected="no overlap",
            actual=f"{len(overlap)} overlapping" if overlap else "ok"
        ))

    # check: emerging topics exist
    if result.emerging_topics:
        # emerging topics should be relatively recent
        # (would need topic-year data to verify properly)
        checks.append(info_check(
            name="has_emerging_topics",
            passed=True,
            message=f"Found {len(result.emerging_topics)} emerging topics",
            expected="emerging topics",
            actual=len(result.emerging_topics)
        ))

    return checks


# =============================================================================
# GapDetector Checks
# =============================================================================

def verify_gap_detector(
    papers: List[Paper],
    result: Optional[GapAnalysis]
) -> List[Check]:
    """verify GapDetector output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="GapDetector returned no result",
            expected="GapAnalysis object",
            actual=None
        ))
        return checks

    # build concept set from papers
    all_concepts: Set[str] = set()
    for p in papers:
        if p.concepts:
            all_concepts.update(c.lower() for c in p.concepts)

    # check: gap concepts exist in corpus (for top gaps)
    for i, gap in enumerate(result.concept_gaps[:3]):
        a_lower = gap.concept_a.lower()
        b_lower = gap.concept_b.lower()

        # fuzzy match
        a_exists = any(a_lower in c or c in a_lower for c in all_concepts)
        b_exists = any(b_lower in c or c in b_lower for c in all_concepts)

        checks.append(error_check(
            name=f"gap_{i}_concepts_exist",
            passed=a_exists and b_exists,
            message=f"Gap concepts '{gap.concept_a}' and '{gap.concept_b}' should exist",
            expected="both in corpus",
            actual=f"a:{a_exists}, b:{b_exists}"
        ))

    # check: gap scores bounded
    if result.concept_gaps:
        scores = [g.gap_score for g in result.concept_gaps]
        all_bounded = all(0.0 <= s <= 1.0 for s in scores)
        checks.append(error_check(
            name="gap_scores_bounded",
            passed=all_bounded,
            message="Gap scores should be between 0.0 and 1.0",
            expected="0.0-1.0",
            actual=f"min={min(scores):.2f}, max={max(scores):.2f}" if scores else "none"
        ))

    # check: gaps are actual gaps (low co-occurrence)
    if result.concept_gaps and papers:
        top_gap = result.concept_gaps[0]
        a_lower = top_gap.concept_a.lower()
        b_lower = top_gap.concept_b.lower()

        co_occur = 0
        for p in papers:
            if p.concepts:
                concepts_lower = [c.lower() for c in p.concepts]
                has_a = any(a_lower in c for c in concepts_lower)
                has_b = any(b_lower in c for c in concepts_lower)
                if has_a and has_b:
                    co_occur += 1

        co_occur_pct = co_occur / len(papers) if papers else 0
        checks.append(warning_check(
            name="gaps_are_actual_gaps",
            passed=co_occur_pct <= 0.1,  # <=10% co-occurrence
            message="Top gap should have low co-occurrence",
            expected="<=10% co-occurrence",
            actual=f"{co_occur_pct*100:.1f}%"
        ))

    return checks


# =============================================================================
# RelevanceScorer Checks
# =============================================================================

def verify_relevance_scorer(
    paper: Paper,
    context: ScoringContext,
    result: Optional[RelevanceScore]
) -> List[Check]:
    """verify RelevanceScorer output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="RelevanceScorer returned no result",
            expected="RelevanceScore object",
            actual=None
        ))
        return checks

    # check: scores bounded
    scores = [
        result.score, result.concept_score,
        result.author_score, result.citation_score, result.recency_score
    ]
    all_bounded = all(0.0 <= s <= 1.0 for s in scores)
    checks.append(error_check(
        name="scores_bounded",
        passed=all_bounded,
        message="All scores must be between 0.0 and 1.0",
        expected="0.0-1.0",
        actual=f"scores: {[f'{s:.2f}' for s in scores]}"
    ))

    # check: highly_relevant justified
    if result.is_highly_relevant:
        high_score = result.score >= 0.6
        checks.append(warning_check(
            name="highly_relevant_justified",
            passed=high_score,
            message="Highly relevant papers should have score >= 0.6",
            expected=">=0.6",
            actual=f"{result.score:.2f}"
        ))

    # check: paper_id matches
    id_matches = result.paper_id == paper.id
    checks.append(error_check(
        name="paper_id_matches",
        passed=id_matches,
        message="Score paper_id should match input paper",
        expected=paper.id,
        actual=result.paper_id
    ))

    # check: explanation present
    has_explanation = result.explanation and len(result.explanation) > 0
    checks.append(warning_check(
        name="explanation_present",
        passed=has_explanation,
        message="Score should have explanation",
        expected="non-empty explanation",
        actual="present" if has_explanation else "missing"
    ))

    return checks


# =============================================================================
# FieldResolver Checks
# =============================================================================

def verify_field_resolver(
    seed_paper: Optional[Paper],
    query: Optional[str],
    result: Optional[FieldResolution]
) -> List[Check]:
    """verify FieldResolver output."""
    checks = []

    if result is None:
        checks.append(error_check(
            name="result_exists",
            passed=False,
            message="FieldResolver returned no result",
            expected="FieldResolution object",
            actual=None
        ))
        return checks

    # check: has primary field
    has_field = result.primary_field is not None
    checks.append(error_check(
        name="has_primary_field",
        passed=has_field,
        message="FieldResolution must have a primary field",
        expected="FieldProfile object",
        actual="present" if has_field else "missing"
    ))

    if not has_field:
        return checks

    # check: field has name
    has_name = result.primary_field.name and len(result.primary_field.name) > 0
    checks.append(error_check(
        name="field_has_name",
        passed=has_name,
        message="Primary field must have a name",
        expected="non-empty name",
        actual=result.primary_field.name if has_name else "missing"
    ))

    # check: confidence bounded
    conf_bounded = 0.0 <= result.confidence <= 1.0
    checks.append(error_check(
        name="confidence_bounded",
        passed=conf_bounded,
        message="Confidence must be between 0.0 and 1.0",
        expected="0.0-1.0",
        actual=f"{result.confidence:.2f}"
    ))

    # check: has journal tiers
    has_tier1 = len(result.primary_field.tier1_journals) > 0
    has_tier2 = len(result.primary_field.tier2_journals) > 0
    checks.append(warning_check(
        name="has_journal_tiers",
        passed=has_tier1,
        message="Primary field should have tier 1 journals",
        expected=">=1 tier 1 journals",
        actual=len(result.primary_field.tier1_journals)
    ))

    # check: has key concepts
    has_concepts = len(result.primary_field.key_concepts) > 0
    checks.append(warning_check(
        name="has_key_concepts",
        passed=has_concepts,
        message="Primary field should have key concepts",
        expected=">=1 key concepts",
        actual=len(result.primary_field.key_concepts)
    ))

    # check: evidence present
    has_evidence = len(result.evidence) > 0
    checks.append(warning_check(
        name="has_evidence",
        passed=has_evidence,
        message="FieldResolution should have evidence",
        expected=">=1 evidence items",
        actual=len(result.evidence)
    ))

    # check: strategy valid
    valid_strategies = {"tier1_first", "balanced", "broad", "author_centric", "ocean"}
    strategy_valid = result.suggested_strategy in valid_strategies
    checks.append(error_check(
        name="strategy_valid",
        passed=strategy_valid,
        message="Strategy must be a valid option",
        expected=f"one of {valid_strategies}",
        actual=result.suggested_strategy
    ))

    # check: field relevance to input
    if seed_paper and seed_paper.title:
        # check if any key concept appears in paper title
        title_lower = seed_paper.title.lower()
        concept_in_title = any(
            c.lower() in title_lower
            for c in result.primary_field.key_concepts[:10]
        )
        checks.append(info_check(
            name="field_relevant_to_paper",
            passed=concept_in_title,
            message="Field key concepts should relate to paper title",
            expected="concept in title",
            actual="found" if concept_in_title else "not found"
        ))

    if query:
        # check if field relates to query
        query_lower = query.lower()
        field_name_lower = result.primary_field.name.lower()
        aliases_lower = [a.lower() for a in result.primary_field.aliases]

        name_match = (
            field_name_lower in query_lower or
            any(a in query_lower for a in aliases_lower) or
            any(c.lower() in query_lower for c in result.primary_field.key_concepts[:10])
        )
        checks.append(info_check(
            name="field_relevant_to_query",
            passed=name_match,
            message="Field should be relevant to query",
            expected="field or concepts in query",
            actual="found" if name_match else "not found"
        ))

    return checks
