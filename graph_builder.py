"""
graph builder - builds citation network from seed papers.
implements the expansion loop from SPEC.md section 2.2.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
from datetime import datetime

from providers.base import PaperStub, PaperProvider
from providers.openalex import OpenAlexProvider
from scoring.relevance import compute_relevance_score


class EdgeType(Enum):
    CITES = "CITES"                    # backward: paper cites this
    CITED_BY = "CITED_BY"              # forward: this paper cited by
    INTRO_CITES = "INTRO_CITES"        # from intro section (pdf parsed)
    INTRO_HINT_CITES = "INTRO_HINT_CITES"  # heuristic: first k refs


@dataclass
class GraphNode:
    """node in the citation graph."""
    paper: PaperStub
    depth: int = 0
    relevance_score: float = 0.0
    priority_score: float = 0.0
    expanded: bool = False

    def __lt__(self, other):
        """for priority queue (max-heap, so negate)."""
        return self.priority_score > other.priority_score


@dataclass
class GraphEdge:
    """edge in the citation graph."""
    src_id: str   # canonical id of source paper
    dst_id: str   # canonical id of destination paper
    edge_type: EdgeType
    weight: float = 1.0


@dataclass
class GraphBuildConfig:
    """configuration for graph building."""
    years_back: int = 3
    max_depth: int = 2
    max_nodes: int = 200
    max_edges: int = 1000
    max_seeds: int = 10
    min_seed_citations: int = 30

    # per-node channel budgets (anti-explosion guardrails)
    max_refs_per_node: int = 50
    max_citedby_per_node: int = 30
    max_intro_hint_per_node: int = 20

    # relevance thresholds
    min_relevance: float = 0.15
    min_relevance_intro: float = 0.20

    # hub suppression
    hub_citation_threshold: int = 50000
    degree_cap: int = 80

    # intro hint heuristic
    intro_fraction: float = 0.25
    intro_hint_weight: float = 2.0

    # drift kill-switch
    drift_window: int = 30
    drift_threshold: float = 0.10


@dataclass
class GraphBuildResult:
    """result of graph building."""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    # stats
    expanded_count: int = 0
    total_candidates: int = 0
    drift_stopped: bool = False
    build_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary for JSON output."""
        return {
            'stats': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'expanded_count': self.expanded_count,
                'total_candidates': self.total_candidates,
                'drift_stopped': self.drift_stopped,
                'build_time_seconds': round(self.build_time_seconds, 2)
            },
            'nodes': [
                {
                    'id': node_id,
                    'title': node.paper.title,
                    'year': node.paper.year,
                    'doi': node.paper.doi,
                    'citation_count': node.paper.citation_count,
                    'is_review': node.paper.is_review,
                    'relevance_score': round(node.relevance_score, 3),
                    'depth': node.depth,
                    'expanded': node.expanded
                }
                for node_id, node in self.nodes.items()
            ],
            'edges': [
                {
                    'source': edge.src_id,
                    'target': edge.dst_id,
                    'type': edge.edge_type.value,
                    'weight': edge.weight
                }
                for edge in self.edges
            ],
            'edge_type_counts': self._edge_type_counts()
        }

    def _edge_type_counts(self) -> Dict[str, int]:
        """count edges by type."""
        counts = {}
        for edge in self.edges:
            t = edge.edge_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts


class GraphBuilder:
    """
    builds citation network via priority-based expansion.
    implements anti-explosion guardrails.
    """

    def __init__(
        self,
        provider: Optional[PaperProvider] = None,
        config: Optional[GraphBuildConfig] = None
    ):
        self.provider = provider or OpenAlexProvider()
        self.config = config or GraphBuildConfig()

    def build(self, topic: str, seeds: Optional[List[PaperStub]] = None) -> GraphBuildResult:
        """
        build citation graph for topic.

        args:
            topic: search query for topic
            seeds: optional pre-selected seed papers

        returns:
            GraphBuildResult with nodes and edges
        """
        start_time = datetime.now()
        result = GraphBuildResult()

        # track seen papers by canonical id
        seen_ids: Set[str] = set()

        # priority queue for expansion
        pq: List[GraphNode] = []

        # degree tracking for hub avoidance
        node_degree: Dict[str, int] = {}

        # drift tracking
        drift_window_relevance: List[bool] = []

        # 1. seed selection
        if seeds is None:
            seeds = self._select_seeds(topic)

        print(f"[build] starting with {len(seeds)} seeds")

        # add seeds to graph
        for paper in seeds:
            cid = paper.canonical_id()
            if cid in seen_ids:
                continue

            relevance = compute_relevance_score(paper, topic)
            priority = self._compute_priority(paper, relevance, depth=0)

            node = GraphNode(
                paper=paper,
                depth=0,
                relevance_score=relevance,
                priority_score=priority
            )

            result.nodes[cid] = node
            seen_ids.add(cid)
            node_degree[cid] = 0
            heapq.heappush(pq, node)

        # 2. expansion loop
        while pq and len(result.nodes) < self.config.max_nodes and len(result.edges) < self.config.max_edges:
            # check drift kill-switch
            if len(drift_window_relevance) >= self.config.drift_window:
                recent_good = sum(drift_window_relevance[-self.config.drift_window:])
                if recent_good / self.config.drift_window < self.config.drift_threshold:
                    print(f"[build] drift kill-switch triggered")
                    result.drift_stopped = True
                    break

            # pop highest priority node
            node = heapq.heappop(pq)
            cid = node.paper.canonical_id()

            # skip if already expanded or degree too high
            if node.expanded:
                continue
            if node_degree.get(cid, 0) >= self.config.degree_cap:
                print(f"[build] skipping hub: {node.paper.title[:40]}...")
                continue
            if node.depth >= self.config.max_depth:
                continue

            # expand this node
            node.expanded = True
            result.expanded_count += 1
            print(f"[build] expanding ({result.expanded_count}): {node.paper.title[:50]}...")

            # fetch backward citations (references)
            refs = self._fetch_references(node.paper)
            refs_added = 0

            for i, ref in enumerate(refs):
                if refs_added >= self.config.max_refs_per_node:
                    break

                ref_id = ref.canonical_id()

                # compute relevance
                relevance = compute_relevance_score(ref, topic)

                # apply intro hint heuristic
                n_refs = len(refs)
                k = min(int(n_refs * self.config.intro_fraction), self.config.max_intro_hint_per_node)
                k = max(k, 10) if n_refs >= 10 else n_refs

                is_intro_hint = i < k
                edge_type = EdgeType.INTRO_HINT_CITES if is_intro_hint else EdgeType.CITES
                edge_weight = self.config.intro_hint_weight if is_intro_hint else 1.0

                # relevance gate
                passes_gate = relevance >= self.config.min_relevance
                if is_intro_hint:
                    passes_gate = relevance >= self.config.min_relevance_intro

                # hub suppression
                if ref.citation_count and ref.citation_count >= self.config.hub_citation_threshold:
                    if relevance < 0.30:
                        edge_type = EdgeType.CITES  # don't boost
                        edge_weight = 0.5  # penalize
                        if relevance < 0.15:
                            continue  # skip entirely

                if not passes_gate:
                    drift_window_relevance.append(False)
                    continue

                drift_window_relevance.append(True)

                # add node if new
                if ref_id not in seen_ids:
                    priority = self._compute_priority(ref, relevance, node.depth + 1)
                    ref_node = GraphNode(
                        paper=ref,
                        depth=node.depth + 1,
                        relevance_score=relevance,
                        priority_score=priority
                    )
                    result.nodes[ref_id] = ref_node
                    seen_ids.add(ref_id)
                    node_degree[ref_id] = 0
                    heapq.heappush(pq, ref_node)

                # add edge
                result.edges.append(GraphEdge(
                    src_id=cid,
                    dst_id=ref_id,
                    edge_type=edge_type,
                    weight=edge_weight
                ))
                node_degree[cid] = node_degree.get(cid, 0) + 1
                node_degree[ref_id] = node_degree.get(ref_id, 0) + 1
                refs_added += 1

            # fetch forward citations (cited-by)
            citations = self._fetch_citations(node.paper)
            cites_added = 0

            for cite in citations:
                if cites_added >= self.config.max_citedby_per_node:
                    break

                cite_id = cite.canonical_id()
                relevance = compute_relevance_score(cite, topic)

                if relevance < self.config.min_relevance:
                    drift_window_relevance.append(False)
                    continue

                drift_window_relevance.append(True)

                # add node if new
                if cite_id not in seen_ids:
                    priority = self._compute_priority(cite, relevance, node.depth + 1)
                    cite_node = GraphNode(
                        paper=cite,
                        depth=node.depth + 1,
                        relevance_score=relevance,
                        priority_score=priority
                    )
                    result.nodes[cite_id] = cite_node
                    seen_ids.add(cite_id)
                    node_degree[cite_id] = 0
                    heapq.heappush(pq, cite_node)

                # add edge (direction: citing paper -> cited paper)
                result.edges.append(GraphEdge(
                    src_id=cite_id,
                    dst_id=cid,
                    edge_type=EdgeType.CITED_BY,
                    weight=1.0
                ))
                node_degree[cid] = node_degree.get(cid, 0) + 1
                node_degree[cite_id] = node_degree.get(cite_id, 0) + 1
                cites_added += 1

            result.total_candidates = len(pq)

        end_time = datetime.now()
        result.build_time_seconds = (end_time - start_time).total_seconds()

        print(f"[build] done: {len(result.nodes)} nodes, {len(result.edges)} edges")
        return result

    def _select_seeds(self, topic: str) -> List[PaperStub]:
        """select seed papers for topic."""
        from datetime import datetime
        current_year = datetime.now().year
        year_min = current_year - self.config.years_back

        # search for papers
        candidates = self.provider.search_papers(
            topic,
            year_min=year_min,
            limit=50
        )

        # prefer reviews and highly cited
        seeds = []
        reviews = [p for p in candidates if p.is_review]
        non_reviews = [p for p in candidates if not p.is_review]

        # add reviews first
        for paper in reviews[:3]:
            if paper.citation_count and paper.citation_count >= self.config.min_seed_citations:
                seeds.append(paper)

        # add highly cited non-reviews
        for paper in non_reviews:
            if len(seeds) >= self.config.max_seeds:
                break
            if paper.citation_count and paper.citation_count >= self.config.min_seed_citations:
                seeds.append(paper)

        # if not enough seeds, lower threshold for newer papers
        if len(seeds) < 5:
            for paper in candidates:
                if len(seeds) >= self.config.max_seeds:
                    break
                if paper.canonical_id() not in [s.canonical_id() for s in seeds]:
                    if paper.year and paper.year >= current_year - 1:
                        seeds.append(paper)  # accept newer papers with lower citation count

        return seeds

    def _fetch_references(self, paper: PaperStub) -> List[PaperStub]:
        """fetch papers cited by this paper."""
        paper_id = paper.openalex_id or paper.doi
        if not paper_id:
            return []

        if paper.openalex_id:
            paper_id = f"oaid:{paper.openalex_id}"
        else:
            paper_id = paper.doi

        return self.provider.get_references(paper_id, limit=self.config.max_refs_per_node)

    def _fetch_citations(self, paper: PaperStub) -> List[PaperStub]:
        """fetch papers that cite this paper."""
        paper_id = paper.openalex_id or paper.doi
        if not paper_id:
            return []

        if paper.openalex_id:
            paper_id = f"oaid:{paper.openalex_id}"
        else:
            paper_id = paper.doi

        return self.provider.get_citations(paper_id, limit=self.config.max_citedby_per_node)

    def _compute_priority(self, paper: PaperStub, relevance: float, depth: int) -> float:
        """
        compute priority score for expansion queue.
        priority = w1*relevance + w2*novelty + w3*recency + w4*citation_signal
        """
        from datetime import datetime
        current_year = datetime.now().year

        # relevance (0-1)
        w1 = 0.40
        relevance_component = relevance

        # novelty (penalize very high citation count = probably a hub)
        w2 = 0.15
        cite_count = paper.citation_count or 0
        if cite_count > 10000:
            novelty = 0.1
        elif cite_count > 1000:
            novelty = 0.3
        else:
            novelty = 0.8

        # recency (0-1)
        w3 = 0.25
        if paper.year:
            years_old = current_year - paper.year
            recency = max(0, 1 - years_old / 20)  # decay over 20 years
        else:
            recency = 0.5

        # citation signal (log-scaled)
        w4 = 0.20
        if cite_count > 0:
            import math
            citation_signal = min(1.0, math.log10(cite_count + 1) / 4)  # normalize: log10(10000)/4 = 1
        else:
            citation_signal = 0.0

        # depth penalty
        depth_penalty = 1.0 / (1 + depth * 0.3)

        priority = (
            w1 * relevance_component +
            w2 * novelty +
            w3 * recency +
            w4 * citation_signal
        ) * depth_penalty

        return priority


# simple test
if __name__ == "__main__":
    config = GraphBuildConfig(
        max_nodes=50,
        max_edges=200,
        max_depth=1
    )
    builder = GraphBuilder(config=config)

    print("building graph for 'ancestral protein reconstruction'...")
    result = builder.build("ancestral protein reconstruction")

    print(f"\nResult: {len(result.nodes)} nodes, {len(result.edges)} edges")
    print(f"Edge types: {result._edge_type_counts()}")
