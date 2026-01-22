"""
gap detector agent - find missing connections and unexplored areas.

answers key questions:
- what topics are rarely combined but could be?
- which author clusters don't overlap but work on related things?
- what methodological gaps exist (techniques not applied to problems)?
- where are the bridge opportunities?

input: collection of papers (from corpus, citations, or topic search)
output: GapAnalysis with gaps, bridges, and unexplored areas
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
import math

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper


# concepts too generic to be meaningful for gap detection
GENERIC_CONCEPTS = {
    'biology', 'chemistry', 'physics', 'medicine', 'computer science',
    'mathematics', 'engineering', 'science', 'technology',
    'research', 'study', 'analysis', 'method', 'approach', 'model',
    'experiment', 'data', 'results', 'conclusion', 'review',
}


@dataclass
class ConceptPair:
    """
    a pair of concepts that could be connected.

    gap_score indicates how disconnected they are:
    - high score = rarely appear together despite being related
    - low score = already well-connected
    """
    concept_a: str
    concept_b: str

    # co-occurrence stats
    papers_with_both: int = 0
    papers_with_a_only: int = 0
    papers_with_b_only: int = 0

    # similarity (from shared contexts)
    semantic_similarity: float = 0.0

    # gap metrics
    gap_score: float = 0.0  # high = underexplored connection

    # papers that bridge them
    bridge_papers: List[str] = field(default_factory=list)

    # potential relevance explanation
    potential: str = ""


@dataclass
class AuthorGap:
    """
    a gap between author clusters that don't collaborate
    but work on related topics.
    """
    cluster_a_names: List[str] = field(default_factory=list)
    cluster_b_names: List[str] = field(default_factory=list)

    # shared topics (why they should connect)
    shared_topics: List[str] = field(default_factory=list)

    # distinct topics (what each brings)
    topics_a_only: List[str] = field(default_factory=list)
    topics_b_only: List[str] = field(default_factory=list)

    # gap metrics
    gap_score: float = 0.0

    # potential bridge
    potential_bridge: str = ""


@dataclass
class MethodGap:
    """
    a methodological gap - a technique that could be
    applied to a domain but hasn't been.
    """
    method: str
    domain: str

    # evidence
    method_papers: int = 0
    domain_papers: int = 0
    combined_papers: int = 0

    # gap score
    gap_score: float = 0.0

    # potential
    potential: str = ""


@dataclass
class BridgePaper:
    """
    a paper that bridges multiple clusters or topics.
    """
    paper_id: str
    title: str
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)

    # bridging info
    clusters_bridged: List[str] = field(default_factory=list)
    bridge_score: float = 0.0

    # why it's a bridge
    reason: str = ""


@dataclass
class UnexploredArea:
    """
    an area with potential but few papers.
    """
    name: str
    description: str

    # evidence
    related_concepts: List[str] = field(default_factory=list)
    existing_papers: int = 0

    # metrics
    potential_score: float = 0.0

    # suggestions
    suggested_angles: List[str] = field(default_factory=list)


@dataclass
class GapAnalysis:
    """
    complete gap analysis for a paper set.
    """
    # source info
    total_papers: int = 0
    total_concepts: int = 0
    total_authors: int = 0

    # concept gaps (underexplored combinations)
    concept_gaps: List[ConceptPair] = field(default_factory=list)

    # author network gaps
    author_gaps: List[AuthorGap] = field(default_factory=list)

    # methodological gaps
    method_gaps: List[MethodGap] = field(default_factory=list)

    # bridge papers (papers that connect clusters)
    bridge_papers: List[BridgePaper] = field(default_factory=list)

    # unexplored areas
    unexplored_areas: List[UnexploredArea] = field(default_factory=list)

    # overall insights
    insights: List[str] = field(default_factory=list)

    def add_insight(self, insight: str):
        self.insights.append(insight)


class GapDetector(Agent):
    """
    detect gaps and bridge opportunities in a paper collection.

    analyzes:
    - concept co-occurrence (what's rarely combined?)
    - author networks (who should collaborate?)
    - methodological gaps (techniques not applied?)
    - unexplored areas (nascent topics?)

    usage:
        detector = GapDetector()
        result = detector.run(papers)

        if result.ok:
            analysis = result.data
            print(f"Found {len(analysis.concept_gaps)} concept gaps")
    """

    # thresholds
    MIN_CONCEPT_PAPERS = 5  # concept must appear in N+ papers
    MAX_COOCCUR_RATIO = 0.1  # below this = potential gap
    MIN_GAP_SCORE = 0.3  # minimum score to report

    # method/technique indicators
    METHOD_INDICATORS = {
        'machine learning', 'deep learning', 'neural network',
        'statistical analysis', 'bayesian', 'regression',
        'simulation', 'molecular dynamics', 'monte carlo',
        'clustering', 'classification', 'prediction',
        'spectroscopy', 'crystallography', 'sequencing',
        'imaging', 'microscopy', 'mass spectrometry',
        'network analysis', 'graph theory', 'optimization',
        'modeling', 'computational', 'algorithmic',
    }

    def __init__(
        self,
        min_concept_papers: int = 5,
        max_gaps_per_category: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.min_concept_papers = min_concept_papers
        self.max_gaps_per_category = max_gaps_per_category

    @property
    def name(self) -> str:
        return "GapDetector"

    def execute(
        self,
        papers: List[Paper],
        focus_concepts: Optional[List[str]] = None
    ) -> AgentResult[GapAnalysis]:
        """
        detect gaps in a paper collection.

        args:
            papers: list of Paper objects to analyze
            focus_concepts: optional list of concepts to focus gap detection on

        returns:
            AgentResult with GapAnalysis
        """
        result = AgentResult[GapAnalysis](status=AgentStatus.SUCCESS)
        result.add_trace(f"analyzing gaps in {len(papers)} papers")

        if len(papers) < 10:
            result.status = AgentStatus.FAILED
            result.add_error(
                "INSUFFICIENT_PAPERS",
                f"need at least 10 papers for gap detection, got {len(papers)}"
            )
            return result

        # create analysis object
        analysis = GapAnalysis(total_papers=len(papers))

        # step 1: build concept index
        concept_index, paper_concepts = self._build_concept_index(papers, result)
        analysis.total_concepts = len(concept_index)

        # step 2: build author index
        author_index = self._build_author_index(papers, result)
        analysis.total_authors = len(author_index)

        # step 3: find concept gaps
        concept_gaps = self._find_concept_gaps(
            concept_index, paper_concepts, focus_concepts, result
        )
        analysis.concept_gaps = concept_gaps[:self.max_gaps_per_category]

        # step 4: find method gaps
        method_gaps = self._find_method_gaps(concept_index, paper_concepts, result)
        analysis.method_gaps = method_gaps[:self.max_gaps_per_category]

        # step 5: find author gaps
        author_gaps = self._find_author_gaps(
            papers, author_index, concept_index, result
        )
        analysis.author_gaps = author_gaps[:self.max_gaps_per_category]

        # step 6: identify bridge papers
        bridge_papers = self._find_bridge_papers(papers, concept_gaps, result)
        analysis.bridge_papers = bridge_papers[:self.max_gaps_per_category]

        # step 7: identify unexplored areas
        unexplored = self._find_unexplored_areas(concept_index, paper_concepts, result)
        analysis.unexplored_areas = unexplored[:self.max_gaps_per_category]

        # step 8: generate insights
        self._generate_insights(analysis, result)

        result.data = analysis
        result.add_trace(
            f"found {len(analysis.concept_gaps)} concept gaps, "
            f"{len(analysis.method_gaps)} method gaps, "
            f"{len(analysis.bridge_papers)} bridge papers"
        )

        return result

    def _build_concept_index(
        self,
        papers: List[Paper],
        result: AgentResult
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        build concept -> paper_ids and paper_id -> concepts indices.

        returns:
            concept_index: concept -> set of paper_ids
            paper_concepts: paper_id -> set of concepts
        """
        result.add_trace("building concept index")

        concept_index: Dict[str, Set[str]] = defaultdict(set)
        paper_concepts: Dict[str, Set[str]] = defaultdict(set)

        for paper in papers:
            if not paper.concepts:
                continue

            for concept_data in paper.concepts:
                name = concept_data.get('name', '').lower()
                score = concept_data.get('score', 0.5)

                # filter generic concepts
                if name in GENERIC_CONCEPTS:
                    continue
                # filter low-relevance concepts
                if score < 0.3:
                    continue

                concept_index[name].add(paper.id)
                paper_concepts[paper.id].add(name)

        # filter to concepts with enough papers
        concept_index = {
            k: v for k, v in concept_index.items()
            if len(v) >= self.min_concept_papers
        }

        result.add_trace(f"indexed {len(concept_index)} concepts from {len(paper_concepts)} papers")
        return concept_index, paper_concepts

    def _build_author_index(
        self,
        papers: List[Paper],
        result: AgentResult
    ) -> Dict[str, Set[str]]:
        """
        build author -> paper_ids index.
        """
        result.add_trace("building author index")

        author_index: Dict[str, Set[str]] = defaultdict(set)

        for paper in papers:
            if not paper.authors:
                continue
            for author in paper.authors:
                author_index[author].add(paper.id)

        # filter to authors with multiple papers
        author_index = {
            k: v for k, v in author_index.items()
            if len(v) >= 2
        }

        result.add_trace(f"indexed {len(author_index)} authors")
        return author_index

    def _find_concept_gaps(
        self,
        concept_index: Dict[str, Set[str]],
        paper_concepts: Dict[str, Set[str]],
        focus_concepts: Optional[List[str]],
        result: AgentResult
    ) -> List[ConceptPair]:
        """
        find concept pairs that rarely co-occur but should.

        gap = concepts that appear in many papers separately
        but rarely together, suggesting an unexplored connection.
        """
        result.add_trace("finding concept gaps")

        concepts = list(concept_index.keys())

        # if focus concepts provided, filter
        if focus_concepts:
            focus_set = {c.lower() for c in focus_concepts}
            concepts = [c for c in concepts if c in focus_set or self._is_related(c, focus_set)]

        gaps = []

        # compare pairs
        for i, concept_a in enumerate(concepts):
            papers_a = concept_index[concept_a]

            for concept_b in concepts[i+1:]:
                papers_b = concept_index[concept_b]

                # compute co-occurrence
                papers_both = papers_a & papers_b
                papers_a_only = papers_a - papers_b
                papers_b_only = papers_b - papers_a

                # skip if already well-connected
                min_papers = min(len(papers_a), len(papers_b))
                if min_papers == 0:
                    continue

                cooccur_ratio = len(papers_both) / min_papers
                if cooccur_ratio > self.MAX_COOCCUR_RATIO:
                    continue

                # compute gap score
                # high when: many papers each, few together
                gap_score = self._compute_gap_score(
                    len(papers_a), len(papers_b), len(papers_both)
                )

                if gap_score < self.MIN_GAP_SCORE:
                    continue

                # compute semantic similarity from shared paper contexts
                similarity = self._compute_concept_similarity(
                    concept_a, concept_b, paper_concepts
                )

                gap = ConceptPair(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    papers_with_both=len(papers_both),
                    papers_with_a_only=len(papers_a_only),
                    papers_with_b_only=len(papers_b_only),
                    semantic_similarity=similarity,
                    gap_score=gap_score,
                    bridge_papers=list(papers_both)[:5],
                    potential=self._describe_gap_potential(concept_a, concept_b, gap_score)
                )

                gaps.append(gap)

        # sort by gap score
        gaps.sort(key=lambda g: -g.gap_score)

        result.add_trace(f"found {len(gaps)} concept gaps")
        return gaps

    def _is_related(self, concept: str, focus_set: Set[str]) -> bool:
        """check if concept is related to focus concepts (simple word overlap)."""
        concept_words = set(concept.lower().split())
        for focus in focus_set:
            focus_words = set(focus.lower().split())
            if concept_words & focus_words:
                return True
        return False

    def _compute_gap_score(
        self,
        papers_a: int,
        papers_b: int,
        papers_both: int
    ) -> float:
        """
        compute gap score for a concept pair.

        high score when:
        - both concepts have many papers
        - few papers have both
        """
        # geometric mean of individual counts (importance)
        importance = math.sqrt(papers_a * papers_b)

        # disconnect ratio (how separated they are)
        total_possible = papers_a + papers_b
        if total_possible == 0:
            return 0.0

        # jaccard distance
        union = papers_a + papers_b - papers_both
        if union == 0:
            return 0.0
        jaccard = papers_both / union
        disconnect = 1 - jaccard

        # combine: important + disconnected = high gap
        score = (math.log(1 + importance) / 5) * disconnect

        return min(1.0, score)

    def _compute_concept_similarity(
        self,
        concept_a: str,
        concept_b: str,
        paper_concepts: Dict[str, Set[str]]
    ) -> float:
        """
        compute semantic similarity between concepts
        based on shared neighboring concepts.
        """
        # get neighboring concepts for each
        neighbors_a: Dict[str, int] = defaultdict(int)
        neighbors_b: Dict[str, int] = defaultdict(int)

        for paper_id, concepts in paper_concepts.items():
            if concept_a in concepts:
                for c in concepts:
                    if c != concept_a:
                        neighbors_a[c] += 1
            if concept_b in concepts:
                for c in concepts:
                    if c != concept_b:
                        neighbors_b[c] += 1

        # jaccard similarity of neighbor sets
        set_a = set(neighbors_a.keys())
        set_b = set(neighbors_b.keys())

        intersection = set_a & set_b
        union = set_a | set_b

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _describe_gap_potential(
        self,
        concept_a: str,
        concept_b: str,
        gap_score: float
    ) -> str:
        """generate description of gap potential."""
        if gap_score > 0.7:
            return f"Strong potential: {concept_a} and {concept_b} are well-studied separately but rarely combined."
        elif gap_score > 0.5:
            return f"Moderate potential: connection between {concept_a} and {concept_b} appears underexplored."
        else:
            return f"Some potential: {concept_a} and {concept_b} could benefit from more cross-study."

    def _find_method_gaps(
        self,
        concept_index: Dict[str, Set[str]],
        paper_concepts: Dict[str, Set[str]],
        result: AgentResult
    ) -> List[MethodGap]:
        """
        find methodological gaps - techniques not applied to domains.
        """
        result.add_trace("finding method gaps")

        # identify method concepts
        methods = []
        domains = []

        for concept in concept_index:
            is_method = any(m in concept for m in self.METHOD_INDICATORS)
            if is_method:
                methods.append(concept)
            else:
                domains.append(concept)

        gaps = []

        for method in methods:
            method_papers = concept_index[method]

            for domain in domains:
                domain_papers = concept_index[domain]

                # combined
                combined = method_papers & domain_papers

                # gap when: method has papers, domain has papers, few combined
                if len(combined) < 2 and len(method_papers) >= 5 and len(domain_papers) >= 5:
                    # compute gap score
                    importance = math.sqrt(len(method_papers) * len(domain_papers))
                    disconnect = 1 - (len(combined) / min(len(method_papers), len(domain_papers)))
                    score = (math.log(1 + importance) / 5) * disconnect

                    if score >= self.MIN_GAP_SCORE:
                        gap = MethodGap(
                            method=method,
                            domain=domain,
                            method_papers=len(method_papers),
                            domain_papers=len(domain_papers),
                            combined_papers=len(combined),
                            gap_score=score,
                            potential=f"{method} techniques could be applied to {domain}"
                        )
                        gaps.append(gap)

        # sort by score
        gaps.sort(key=lambda g: -g.gap_score)

        result.add_trace(f"found {len(gaps)} method gaps")
        return gaps

    def _find_author_gaps(
        self,
        papers: List[Paper],
        author_index: Dict[str, Set[str]],
        concept_index: Dict[str, Set[str]],
        result: AgentResult
    ) -> List[AuthorGap]:
        """
        find author clusters that don't collaborate but work on related topics.
        """
        result.add_trace("finding author gaps")

        # build author -> concepts mapping
        author_concepts: Dict[str, Set[str]] = defaultdict(set)

        for paper in papers:
            if not paper.authors or not paper.concepts:
                continue

            paper_concepts_set = {
                c.get('name', '').lower()
                for c in paper.concepts[:5]
                if c.get('name', '').lower() not in GENERIC_CONCEPTS
            }

            for author in paper.authors:
                author_concepts[author].update(paper_concepts_set)

        # find co-author networks
        coauthor_graph: Dict[str, Set[str]] = defaultdict(set)

        for paper in papers:
            if not paper.authors:
                continue
            authors = paper.authors
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    coauthor_graph[a1].add(a2)
                    coauthor_graph[a2].add(a1)

        # find author pairs with similar topics but no collaboration
        gaps = []
        authors_list = list(author_index.keys())

        for i, author_a in enumerate(authors_list[:50]):  # limit computation
            concepts_a = author_concepts.get(author_a, set())
            coauthors_a = coauthor_graph.get(author_a, set())

            for author_b in authors_list[i+1:50]:
                # skip if they collaborate
                if author_b in coauthors_a:
                    continue

                concepts_b = author_concepts.get(author_b, set())

                # compute topic overlap
                shared = concepts_a & concepts_b
                if len(shared) < 2:
                    continue

                only_a = concepts_a - concepts_b
                only_b = concepts_b - concepts_a

                # gap score: high overlap, no collaboration
                overlap_ratio = len(shared) / max(len(concepts_a), len(concepts_b), 1)

                if overlap_ratio > 0.3:
                    gap = AuthorGap(
                        cluster_a_names=[author_a],
                        cluster_b_names=[author_b],
                        shared_topics=list(shared)[:5],
                        topics_a_only=list(only_a)[:3],
                        topics_b_only=list(only_b)[:3],
                        gap_score=overlap_ratio,
                        potential_bridge=f"Both work on {', '.join(list(shared)[:2])} but haven't collaborated"
                    )
                    gaps.append(gap)

        # sort by gap score
        gaps.sort(key=lambda g: -g.gap_score)

        result.add_trace(f"found {len(gaps)} author gaps")
        return gaps

    def _find_bridge_papers(
        self,
        papers: List[Paper],
        concept_gaps: List[ConceptPair],
        result: AgentResult
    ) -> List[BridgePaper]:
        """
        find papers that bridge multiple clusters/gaps.
        """
        result.add_trace("finding bridge papers")

        # get gap concepts
        gap_pairs = [(g.concept_a, g.concept_b) for g in concept_gaps[:20]]

        bridges = []

        for paper in papers:
            if not paper.concepts:
                continue

            paper_concepts = {
                c.get('name', '').lower()
                for c in paper.concepts
                if c.get('name', '').lower() not in GENERIC_CONCEPTS
            }

            # count how many gaps this paper bridges
            bridged_gaps = []
            for concept_a, concept_b in gap_pairs:
                if concept_a in paper_concepts and concept_b in paper_concepts:
                    bridged_gaps.append(f"{concept_a} + {concept_b}")

            if bridged_gaps:
                bridge = BridgePaper(
                    paper_id=paper.id,
                    title=paper.title or "Unknown",
                    year=paper.year,
                    authors=paper.authors[:3] if paper.authors else [],
                    clusters_bridged=bridged_gaps,
                    bridge_score=len(bridged_gaps),
                    reason=f"Connects {len(bridged_gaps)} concept gaps"
                )
                bridges.append(bridge)

        # sort by bridge score
        bridges.sort(key=lambda b: -b.bridge_score)

        result.add_trace(f"found {len(bridges)} bridge papers")
        return bridges

    def _find_unexplored_areas(
        self,
        concept_index: Dict[str, Set[str]],
        paper_concepts: Dict[str, Set[str]],
        result: AgentResult
    ) -> List[UnexploredArea]:
        """
        find nascent/emerging areas with few papers.
        """
        result.add_trace("finding unexplored areas")

        # find concept combinations that appear rarely (2-4 papers)
        # but whose individual concepts are well-established

        # count concept combinations
        combo_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for paper_id, concepts in paper_concepts.items():
            concepts_list = list(concepts)
            for i, c1 in enumerate(concepts_list):
                for c2 in concepts_list[i+1:]:
                    key = tuple(sorted([c1, c2]))
                    combo_counts[key] += 1

        # filter to nascent combinations
        unexplored = []

        for (c1, c2), count in combo_counts.items():
            # nascent: 2-4 papers
            if count < 2 or count > 4:
                continue

            # but concepts individually established
            papers_c1 = len(concept_index.get(c1, set()))
            papers_c2 = len(concept_index.get(c2, set()))

            if papers_c1 >= 10 and papers_c2 >= 10:
                # this is an emerging combination
                potential = math.sqrt(papers_c1 * papers_c2) / (count + 1)

                area = UnexploredArea(
                    name=f"{c1} × {c2}",
                    description=f"Intersection of {c1} ({papers_c1} papers) and {c2} ({papers_c2} papers) with only {count} papers",
                    related_concepts=[c1, c2],
                    existing_papers=count,
                    potential_score=potential,
                    suggested_angles=[
                        f"Apply {c1} approaches to {c2} problems",
                        f"Study {c2} phenomena using {c1} methods",
                        f"Review existing work connecting {c1} and {c2}"
                    ]
                )
                unexplored.append(area)

        # sort by potential
        unexplored.sort(key=lambda u: -u.potential_score)

        result.add_trace(f"found {len(unexplored)} unexplored areas")
        return unexplored

    def _generate_insights(
        self,
        analysis: GapAnalysis,
        result: AgentResult
    ):
        """generate human-readable insights."""
        # overview
        analysis.add_insight(
            f"Analyzed {analysis.total_papers} papers with {analysis.total_concepts} concepts "
            f"and {analysis.total_authors} authors."
        )

        # concept gaps
        if analysis.concept_gaps:
            top_gap = analysis.concept_gaps[0]
            analysis.add_insight(
                f"Top concept gap: '{top_gap.concept_a}' and '{top_gap.concept_b}' "
                f"appear separately ({top_gap.papers_with_a_only + top_gap.papers_with_both} vs "
                f"{top_gap.papers_with_b_only + top_gap.papers_with_both} papers) "
                f"but only together in {top_gap.papers_with_both} papers."
            )

        # method gaps
        if analysis.method_gaps:
            top_method = analysis.method_gaps[0]
            analysis.add_insight(
                f"Methodological opportunity: '{top_method.method}' ({top_method.method_papers} papers) "
                f"rarely applied to '{top_method.domain}' ({top_method.domain_papers} papers)."
            )

        # bridge papers
        if analysis.bridge_papers:
            analysis.add_insight(
                f"Found {len(analysis.bridge_papers)} bridge papers connecting multiple gaps. "
                f"Top bridge: '{analysis.bridge_papers[0].title[:60]}...'"
            )

        # unexplored areas
        if analysis.unexplored_areas:
            top_area = analysis.unexplored_areas[0]
            analysis.add_insight(
                f"Emerging area: {top_area.name} has only {top_area.existing_papers} papers "
                f"but high potential based on parent concept prevalence."
            )

        # author gaps
        if analysis.author_gaps:
            top_auth = analysis.author_gaps[0]
            if top_auth.cluster_a_names and top_auth.cluster_b_names:
                analysis.add_insight(
                    f"Potential collaboration: {top_auth.cluster_a_names[0]} and "
                    f"{top_auth.cluster_b_names[0]} work on overlapping topics "
                    f"({', '.join(top_auth.shared_topics[:2])}) but haven't collaborated."
                )
