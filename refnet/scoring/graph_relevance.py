"""
graph-native relevance scoring for SCIENTIST mode.
citations drive relevance, keywords are weak tie-breakers only.
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
import math

from ..core.models import Paper, EdgeType
from ..core.config import ScoringConfig
from ..graph.working_graph import WorkingGraph


@dataclass
class RelevanceBreakdown:
    """breakdown of relevance score components."""
    paper_id: str
    proximity_score: float
    multipath_score: float
    coupling_score: float
    cocitation_score: float
    portal_score: float
    keyword_score: float
    hub_penalty: float
    final_score: float


class GraphRelevanceScorer:
    """
    computes graph-native relevance scores.
    in SCIENTIST mode, citations drive relevance, not keywords.
    """

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

        # caches
        self._ref_sets: Dict[str, Set[str]] = {}  # paper_id -> set of refs
        self._citer_sets: Dict[str, Set[str]] = {}  # paper_id -> set of citers

    def compute_relevance(
        self,
        paper: Paper,
        graph: WorkingGraph,
        topic_keywords: Optional[List[str]] = None,
        hub_penalty: float = 1.0
    ) -> RelevanceBreakdown:
        """
        compute graph-native relevance score.
        """
        # proximity to seeds
        proximity = self._compute_proximity(paper, graph)

        # multipath support
        multipath = self._compute_multipath(paper, graph)

        # bibliographic coupling and co-citation
        coupling = self._compute_coupling(paper, graph)
        cocitation = self._compute_cocitation(paper, graph)
        best_coupling = max(coupling, cocitation)

        # portal support (reviewed by/cites reviews)
        portal = self._compute_portal_support(paper, graph)

        # keyword tie-breaker (weak)
        keyword = 0.0
        if topic_keywords:
            keyword = self._compute_keyword_match(paper, topic_keywords)

        # weighted combination
        final = (
            self.config.proximity_weight * proximity +
            self.config.multipath_weight * multipath +
            self.config.coupling_weight * best_coupling +
            self.config.portal_weight * portal +
            self.config.keyword_weight * keyword
        ) * hub_penalty

        return RelevanceBreakdown(
            paper_id=paper.id,
            proximity_score=proximity,
            multipath_score=multipath,
            coupling_score=coupling,
            cocitation_score=cocitation,
            portal_score=portal,
            keyword_score=keyword,
            hub_penalty=hub_penalty,
            final_score=min(final, 1.0)
        )

    def compute_materialization_score(
        self,
        paper: Paper,
        graph: WorkingGraph,
        bridge_score: float = 0.0
    ) -> float:
        """
        compute score for whether paper should enter working graph.
        similar to relevance but includes bridge potential.
        """
        proximity = self._compute_proximity(paper, graph)
        multipath = self._compute_multipath(paper, graph)
        coupling = max(
            self._compute_coupling(paper, graph),
            self._compute_cocitation(paper, graph)
        )
        portal = self._compute_portal_support(paper, graph)

        # recency/impact bonus
        recency = self._compute_recency_impact(paper)

        score = (
            self.config.mat_proximity_weight * proximity +
            self.config.mat_multipath_weight * multipath +
            self.config.mat_coupling_weight * coupling +
            self.config.mat_portal_weight * portal +
            self.config.mat_bridge_weight * bridge_score +
            self.config.mat_recency_weight * recency
        )

        return min(score, 1.0)

    def compute_bridge_score(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """
        compute bridge potential for a paper.
        high if paper connects different clusters or neighborhoods.
        """
        # check if paper connects different parts of graph
        neighbors = graph.get_neighbors(paper.id)

        if len(neighbors) < 2:
            return 0.0

        # check cluster distribution of neighbors
        cluster_counts: Dict[str, int] = {}
        for n in neighbors:
            cluster = graph.node_cluster_map.get(n)
            if cluster:
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        if len(cluster_counts) < 2:
            return 0.0

        # bridge score based on cluster diversity
        total = sum(cluster_counts.values())
        if total == 0:
            return 0.0

        # higher if connects multiple clusters evenly
        max_count = max(cluster_counts.values())
        diversity = 1.0 - (max_count / total)

        return diversity

    def batch_score(
        self,
        papers: List[Paper],
        graph: WorkingGraph,
        topic_keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        batch compute relevance scores.
        more efficient than individual calls.
        """
        scores = {}
        for paper in papers:
            breakdown = self.compute_relevance(paper, graph, topic_keywords)
            scores[paper.id] = breakdown.final_score
        return scores

    # private scoring methods

    def _compute_proximity(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """
        compute proximity score based on distance to seeds.
        ProximityScore = exp(-alpha * d)
        """
        if paper.id in graph.seed_ids:
            return 1.0

        dist = graph.min_distance_to_seeds(paper.id)

        if dist < 0:
            # not connected - use depth as fallback
            dist = paper.depth if paper.depth > 0 else 3

        return math.exp(-self.config.proximity_alpha * dist)

    def _compute_multipath(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """
        compute multipath support score.
        based on number of independent paths to seeds.
        """
        if paper.id in graph.seed_ids:
            return 1.0

        paths = graph.count_paths_to_seeds(paper.id, max_paths=6)

        if paths == 0:
            return 0.0

        # MultiPathScore = min(1.0, log(1+paths)/log(6))
        return min(1.0, math.log(1 + paths) / math.log(6))

    def _compute_coupling(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """
        compute bibliographic coupling score.
        based on overlap of references with seeds/graph.
        """
        # get paper's references (from edges)
        paper_refs = set()
        for edge in graph.get_edges_from(paper.id):
            if edge.edge_type in [EdgeType.CITES, EdgeType.INTRO_CITES, EdgeType.INTRO_HINT_CITES]:
                paper_refs.add(edge.target_id)

        if not paper_refs:
            return 0.0

        # get seed references
        seed_refs = set()
        for seed_id in graph.seed_ids:
            for edge in graph.get_edges_from(seed_id):
                if edge.edge_type in [EdgeType.CITES, EdgeType.INTRO_CITES, EdgeType.INTRO_HINT_CITES]:
                    seed_refs.add(edge.target_id)

        if not seed_refs:
            return 0.0

        # Jaccard similarity
        intersection = len(paper_refs & seed_refs)
        union = len(paper_refs | seed_refs)

        return intersection / union if union > 0 else 0.0

    def _compute_cocitation(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """
        compute co-citation score.
        based on overlap of papers that cite this paper vs cite seeds.
        """
        # get papers citing this paper
        paper_citers = set()
        for edge in graph.get_edges_to(paper.id):
            if edge.edge_type in [EdgeType.CITES, EdgeType.INTRO_CITES, EdgeType.INTRO_HINT_CITES]:
                paper_citers.add(edge.source_id)

        if not paper_citers:
            return 0.0

        # get seed citers
        seed_citers = set()
        for seed_id in graph.seed_ids:
            for edge in graph.get_edges_to(seed_id):
                if edge.edge_type in [EdgeType.CITES, EdgeType.INTRO_CITES, EdgeType.INTRO_HINT_CITES]:
                    seed_citers.add(edge.source_id)

        if not seed_citers:
            return 0.0

        # Jaccard
        intersection = len(paper_citers & seed_citers)
        union = len(paper_citers | seed_citers)

        return intersection / union if union > 0 else 0.0

    def _compute_portal_support(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """
        compute portal support score.
        boosted if paper is cited by reviews or cites reviews.
        """
        # check if paper is a review
        if paper.is_review:
            return 1.0

        # check connections to portals (reviews)
        portal_connections = 0

        for edge in graph.get_edges_to(paper.id):
            source = graph.get_paper(edge.source_id)
            if source and source.is_review:
                portal_connections += 1

        for edge in graph.get_edges_from(paper.id):
            target = graph.get_paper(edge.target_id)
            if target and target.is_review:
                portal_connections += 0.5  # citing reviews is less strong

        return min(portal_connections / 3, 1.0)

    def _compute_keyword_match(
        self,
        paper: Paper,
        keywords: List[str]
    ) -> float:
        """
        compute weak keyword match score.
        only used as tie-breaker (10% weight).
        """
        if not keywords:
            return 0.0

        text = f"{paper.title or ''} {paper.abstract or ''}"
        text = text.lower()

        matches = 0
        for kw in keywords:
            if kw.lower() in text:
                matches += 1

        return min(matches / len(keywords), 1.0)

    def _compute_recency_impact(self, paper: Paper) -> float:
        """
        compute recency and impact score for materialization.
        """
        from datetime import datetime
        current_year = datetime.now().year

        # recency bonus
        recency = 0.0
        if paper.year:
            years_old = current_year - paper.year
            recency = max(0, 1.0 - years_old / 10)  # decay over 10 years

        # impact bonus
        impact = 0.0
        if paper.citation_count:
            # log scale
            impact = min(math.log10(paper.citation_count + 1) / 4, 1.0)

        return 0.5 * recency + 0.5 * impact
