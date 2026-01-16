"""
gap analysis - identify bridges, missing links, unexplored clusters.
key output for understanding field structure.
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.models import Paper, Author, Cluster, GapCandidate, EdgeType
from ..core.config import GapAnalysisConfig
from ..graph.candidate_pool import CandidatePool
from ..graph.working_graph import WorkingGraph


@dataclass
class BridgeCandidate:
    """a potential bridge between clusters."""
    paper_id: str
    paper_title: str
    cluster_a_id: str
    cluster_b_id: str
    bridge_score: float
    evidence: Dict[str, Any]


@dataclass
class MissingLink:
    """a missing intermediate between two papers/clusters."""
    gap_type: str  # "citation_gap", "concept_gap", "temporal_gap"
    source_id: str
    target_id: str
    candidate_ids: List[str]  # papers that could fill the gap
    confidence: float
    explanation: str


@dataclass
class UnexploredCluster:
    """an unexplored region near the current graph."""
    cluster_id: str
    name: str
    representative_concepts: List[str]
    distance_to_graph: float
    candidate_paper_count: int
    top_candidates: List[str]


@dataclass
class GapAnalysisResult:
    """complete gap analysis result."""
    bridges: List[BridgeCandidate]
    missing_links: List[MissingLink]
    unexplored_clusters: List[UnexploredCluster]
    field_drift_map: Dict[int, List[str]]  # year -> emerging concepts
    summary: str


class GapAnalyzer:
    """
    analyzes gaps in the citation network.
    identifies bridges, missing links, and unexplored areas.
    """

    def __init__(self, config: Optional[GapAnalysisConfig] = None):
        self.config = config or GapAnalysisConfig()

    def analyze(
        self,
        graph: WorkingGraph,
        pool: CandidatePool
    ) -> GapAnalysisResult:
        """
        run complete gap analysis.
        """
        # detect clusters if not already done
        if not graph.clusters:
            self._detect_clusters(graph)

        # find bridges
        bridges = self._find_bridges(graph, pool)

        # find missing links
        missing = self._find_missing_links(graph, pool)

        # find unexplored clusters
        unexplored = self._find_unexplored(graph, pool)

        # compute field drift
        drift = self._compute_field_drift(graph)

        # generate summary
        summary = self._generate_summary(bridges, missing, unexplored, drift)

        return GapAnalysisResult(
            bridges=bridges,
            missing_links=missing,
            unexplored_clusters=unexplored,
            field_drift_map=drift,
            summary=summary
        )

    def _detect_clusters(self, graph: WorkingGraph):
        """
        detect clusters using concept-based grouping.
        lightweight alternative to full community detection.
        """
        # group papers by primary concept
        concept_groups: Dict[str, Set[str]] = defaultdict(set)

        for paper_id, paper in graph.papers.items():
            if paper.concepts:
                primary = paper.concepts[0].get('name', 'Unknown')
                concept_groups[primary].add(paper_id)
            else:
                concept_groups['Unknown'].add(paper_id)

        # create clusters from groups meeting size threshold
        cluster_id = 0
        for concept, paper_ids in concept_groups.items():
            if len(paper_ids) >= self.config.min_cluster_size:
                cluster = Cluster(
                    id=f"c_{cluster_id}",
                    name=concept,
                    paper_ids=paper_ids,
                    top_concepts=[{'name': concept, 'weight': 1.0}],
                    size=len(paper_ids)
                )
                graph.clusters[cluster.id] = cluster

                for pid in paper_ids:
                    graph.node_cluster_map[pid] = cluster.id

                cluster_id += 1

                if cluster_id >= self.config.max_clusters:
                    break

    def _find_bridges(
        self,
        graph: WorkingGraph,
        pool: CandidatePool
    ) -> List[BridgeCandidate]:
        """
        find papers that bridge between clusters.
        """
        bridges = []

        # analyze each paper in graph
        for paper_id, paper in graph.papers.items():
            # get neighbors and their clusters
            neighbors = graph.get_neighbors(paper_id)
            neighbor_clusters: Dict[str, int] = defaultdict(int)

            for n in neighbors:
                cluster = graph.node_cluster_map.get(n)
                if cluster:
                    neighbor_clusters[cluster] += 1

            # skip if only connected to one cluster
            if len(neighbor_clusters) < 2:
                continue

            # compute bridge score
            total = sum(neighbor_clusters.values())
            sorted_clusters = sorted(neighbor_clusters.items(), key=lambda x: x[1], reverse=True)

            # bridge if reasonably balanced connections
            c1, count1 = sorted_clusters[0]
            c2, count2 = sorted_clusters[1]

            balance = count2 / count1 if count1 > 0 else 0
            bridge_score = balance * (count1 + count2) / total

            if bridge_score >= self.config.min_bridge_score:
                bridges.append(BridgeCandidate(
                    paper_id=paper_id,
                    paper_title=paper.title,
                    cluster_a_id=c1,
                    cluster_b_id=c2,
                    bridge_score=bridge_score,
                    evidence={
                        "cluster_a_connections": count1,
                        "cluster_b_connections": count2,
                        "balance": balance
                    }
                ))

        # sort by bridge score
        bridges.sort(key=lambda b: b.bridge_score, reverse=True)
        return bridges[:self.config.max_bridges_to_show]

    def _find_missing_links(
        self,
        graph: WorkingGraph,
        pool: CandidatePool
    ) -> List[MissingLink]:
        """
        find missing intermediates between papers/clusters.
        """
        missing = []

        # analyze cluster pairs
        cluster_ids = list(graph.clusters.keys())

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                c1 = cluster_ids[i]
                c2 = cluster_ids[j]

                # check if clusters are connected
                connections = self._count_cluster_connections(graph, c1, c2)

                if connections == 0:
                    # completely disconnected - look for intermediates
                    candidates = self._find_intermediates(graph, pool, c1, c2)

                    if candidates:
                        c1_name = graph.clusters[c1].name
                        c2_name = graph.clusters[c2].name

                        missing.append(MissingLink(
                            gap_type="cluster_gap",
                            source_id=c1,
                            target_id=c2,
                            candidate_ids=candidates[:5],
                            confidence=0.7,
                            explanation=f"No direct links between '{c1_name}' and '{c2_name}' clusters"
                        ))

                elif connections < 3:
                    # weakly connected - might be missing links
                    candidates = self._find_intermediates(graph, pool, c1, c2)

                    if len(candidates) > connections:
                        c1_name = graph.clusters[c1].name
                        c2_name = graph.clusters[c2].name

                        missing.append(MissingLink(
                            gap_type="weak_link",
                            source_id=c1,
                            target_id=c2,
                            candidate_ids=candidates[:5],
                            confidence=0.5,
                            explanation=f"Weak connection ({connections} edges) between '{c1_name}' and '{c2_name}'"
                        ))

        return missing[:self.config.max_missing_to_show]

    def _find_unexplored(
        self,
        graph: WorkingGraph,
        pool: CandidatePool
    ) -> List[UnexploredCluster]:
        """
        find unexplored areas in the candidate pool.
        """
        unexplored = []

        # get concepts from candidate pool
        candidates = pool.get_top_candidates(limit=500, by="priority_score")

        # group by primary concept
        concept_counts: Dict[str, List[Paper]] = defaultdict(list)
        for paper in candidates:
            if paper.concepts:
                primary = paper.concepts[0].get('name', 'Unknown')
                concept_counts[primary].append(paper)

        # find concepts not well represented in graph
        graph_concepts = set()
        for paper in graph.papers.values():
            if paper.concepts:
                graph_concepts.add(paper.concepts[0].get('name', ''))

        for concept, papers in concept_counts.items():
            if concept not in graph_concepts and len(papers) >= 3:
                unexplored.append(UnexploredCluster(
                    cluster_id=f"unexplored_{concept[:20]}",
                    name=concept,
                    representative_concepts=[concept],
                    distance_to_graph=1.0,  # placeholder
                    candidate_paper_count=len(papers),
                    top_candidates=[p.id for p in papers[:5]]
                ))

        # sort by candidate count
        unexplored.sort(key=lambda u: u.candidate_paper_count, reverse=True)
        return unexplored[:5]

    def _compute_field_drift(
        self,
        graph: WorkingGraph
    ) -> Dict[int, List[str]]:
        """
        compute field drift over time.
        shows emerging concepts by year.
        """
        # group papers by year
        papers_by_year: Dict[int, List[Paper]] = defaultdict(list)
        for paper in graph.papers.values():
            if paper.year:
                papers_by_year[paper.year].append(paper)

        # compute concept prevalence by year
        concept_by_year: Dict[int, Dict[str, int]] = {}
        for year, papers in papers_by_year.items():
            concepts: Dict[str, int] = defaultdict(int)
            for paper in papers:
                for c in paper.concepts[:3]:
                    concepts[c.get('name', '')] += 1
            concept_by_year[year] = concepts

        # find emerging concepts (appearing in later years but not earlier)
        years = sorted(concept_by_year.keys())
        emerging: Dict[int, List[str]] = {}

        for i, year in enumerate(years):
            if i < 2:
                continue

            current = set(concept_by_year[year].keys())
            earlier = set()
            for j in range(max(0, i-3), i):
                earlier.update(concept_by_year[years[j]].keys())

            new_concepts = current - earlier
            if new_concepts:
                # sort by frequency in current year
                sorted_new = sorted(
                    new_concepts,
                    key=lambda c: concept_by_year[year].get(c, 0),
                    reverse=True
                )
                emerging[year] = sorted_new[:5]

        return emerging

    def _count_cluster_connections(
        self,
        graph: WorkingGraph,
        c1: str,
        c2: str
    ) -> int:
        """count edges between two clusters."""
        count = 0
        c1_papers = graph.clusters[c1].paper_ids if c1 in graph.clusters else set()
        c2_papers = graph.clusters[c2].paper_ids if c2 in graph.clusters else set()

        for edge in graph.edges.values():
            if (edge.source_id in c1_papers and edge.target_id in c2_papers) or \
               (edge.source_id in c2_papers and edge.target_id in c1_papers):
                count += 1

        return count

    def _find_intermediates(
        self,
        graph: WorkingGraph,
        pool: CandidatePool,
        c1: str,
        c2: str
    ) -> List[str]:
        """find papers that could connect two clusters."""
        c1_papers = graph.clusters[c1].paper_ids if c1 in graph.clusters else set()
        c2_papers = graph.clusters[c2].paper_ids if c2 in graph.clusters else set()

        # get concepts from each cluster
        c1_concepts = set()
        c2_concepts = set()

        for pid in c1_papers:
            paper = graph.get_paper(pid)
            if paper and paper.concepts:
                for c in paper.concepts[:3]:
                    c1_concepts.add(c.get('name', ''))

        for pid in c2_papers:
            paper = graph.get_paper(pid)
            if paper and paper.concepts:
                for c in paper.concepts[:3]:
                    c2_concepts.add(c.get('name', ''))

        # find candidates with concepts from both clusters
        candidates = pool.get_top_candidates(limit=200)
        intermediates = []

        for paper in candidates:
            if paper.id in graph.papers:
                continue  # already in graph

            paper_concepts = set()
            for c in paper.concepts[:5]:
                paper_concepts.add(c.get('name', ''))

            # check overlap with both clusters
            c1_overlap = len(paper_concepts & c1_concepts)
            c2_overlap = len(paper_concepts & c2_concepts)

            if c1_overlap > 0 and c2_overlap > 0:
                intermediates.append((paper.id, c1_overlap + c2_overlap))

        # sort by overlap
        intermediates.sort(key=lambda x: x[1], reverse=True)
        return [i[0] for i in intermediates[:10]]

    def _generate_summary(
        self,
        bridges: List[BridgeCandidate],
        missing: List[MissingLink],
        unexplored: List[UnexploredCluster],
        drift: Dict[int, List[str]]
    ) -> str:
        """generate human-readable summary."""
        lines = []

        if bridges:
            lines.append(f"Found {len(bridges)} bridge papers connecting different research areas.")
            lines.append(f"Top bridge: '{bridges[0].paper_title[:50]}...'")

        if missing:
            lines.append(f"Found {len(missing)} potential gaps between clusters.")

        if unexplored:
            lines.append(f"Found {len(unexplored)} unexplored areas in candidate pool.")
            top = unexplored[0]
            lines.append(f"Most prominent unexplored: '{top.name}' ({top.candidate_paper_count} papers)")

        if drift:
            latest_year = max(drift.keys())
            lines.append(f"Emerging concepts in {latest_year}: {', '.join(drift[latest_year][:3])}")

        return " ".join(lines)
