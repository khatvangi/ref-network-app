"""
working graph - narrow, bounded graph for visualization.
materialized from candidate pool based on scores.
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import heapq

from ..core.models import Paper, Author, Edge, EdgeType, PaperStatus, Cluster
from ..core.config import WorkingGraphConfig, GraphSize, RefnetConfig


@dataclass
class GraphNode:
    """wrapper for nodes in working graph."""
    id: str
    node_type: str  # "paper" or "author"
    data: Any  # Paper or Author
    score: float = 0.0
    last_accessed: datetime = field(default_factory=datetime.now)
    is_seed: bool = False
    is_pinned: bool = False
    is_portal: bool = False  # reviews


@dataclass
class GraphEdge:
    """wrapper for edges in working graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    data: Optional[Edge] = None


class WorkingGraph:
    """
    narrow, bounded graph for UI visualization.
    maintains hard limits on nodes and edges.
    supports eviction of low-scoring nodes.
    """

    def __init__(self, config: Optional[WorkingGraphConfig] = None,
                 size: Optional[GraphSize] = None):
        self.config = config or WorkingGraphConfig()
        self._size = size or self.config.default_size

        # node storage
        self.nodes: Dict[str, GraphNode] = {}
        self.papers: Dict[str, Paper] = {}  # id -> Paper
        self.authors: Dict[str, Author] = {}  # id -> Author

        # edge storage
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency: Dict[str, Set[str]] = {}  # node_id -> {neighbor_ids}
        self.edge_index: Dict[Tuple[str, str], str] = {}  # (src, tgt) -> edge_id

        # seeds and pinned
        self.seed_ids: Set[str] = set()
        self.pinned_ids: Set[str] = set()
        self.portal_ids: Set[str] = set()  # reviews

        # clusters
        self.clusters: Dict[str, Cluster] = {}
        self.node_cluster_map: Dict[str, str] = {}  # node_id -> cluster_id

        # stats
        self.evictions = 0
        self.created_at = datetime.now()

    @property
    def max_nodes(self) -> int:
        if self._size == GraphSize.SMALL:
            return self.config.max_nodes_small
        if self._size == GraphSize.LARGE:
            return self.config.max_nodes_large
        return self.config.max_nodes_medium

    @property
    def max_edges(self) -> int:
        if self._size == GraphSize.SMALL:
            return self.config.max_edges_small
        if self._size == GraphSize.LARGE:
            return self.config.max_edges_large
        return self.config.max_edges_medium

    def add_seed(self, paper: Paper) -> bool:
        """add a seed paper (never evicted)."""
        paper.status = PaperStatus.SEED
        node = GraphNode(
            id=paper.id,
            node_type="paper",
            data=paper,
            score=1.0,  # max score for seeds
            is_seed=True
        )
        return self._add_node(node, paper)

    def add_paper(self, paper: Paper) -> bool:
        """add paper to working graph."""
        # check if already exists
        if paper.id in self.papers:
            return False

        # check capacity
        if len(self.nodes) >= self.max_nodes:
            if not self._evict_one():
                return False

        # check if it's a portal (review)
        is_portal = paper.is_review

        node = GraphNode(
            id=paper.id,
            node_type="paper",
            data=paper,
            score=paper.materialization_score,
            is_portal=is_portal
        )

        return self._add_node(node, paper)

    def add_author(self, author: Author) -> bool:
        """add author to working graph."""
        if author.id in self.authors:
            return False

        if len(self.nodes) >= self.max_nodes:
            if not self._evict_one():
                return False

        node = GraphNode(
            id=author.id,
            node_type="author",
            data=author,
            score=author.centrality
        )

        self.nodes[author.id] = node
        self.authors[author.id] = author
        self.adjacency[author.id] = set()
        return True

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType, weight: float = 1.0,
                 confidence: float = 1.0) -> bool:
        """add edge between nodes."""
        # both nodes must exist
        if source_id not in self.nodes or target_id not in self.nodes:
            return False

        # check if edge exists
        edge_key = (source_id, target_id)
        if edge_key in self.edge_index:
            return False

        # check capacity
        if len(self.edges) >= self.max_edges:
            # could evict edges, but for now just reject
            return False

        import uuid
        edge_id = str(uuid.uuid4())
        edge = GraphEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence
        )

        self.edges[edge_id] = edge
        self.edge_index[edge_key] = edge_id
        self.adjacency[source_id].add(target_id)
        self.adjacency[target_id].add(source_id)

        return True

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """check if edge exists between two nodes."""
        return (source_id, target_id) in self.edge_index

    def pin_paper(self, paper_id: str):
        """pin a paper (prevent eviction)."""
        if paper_id in self.nodes:
            self.nodes[paper_id].is_pinned = True
            self.pinned_ids.add(paper_id)

    def unpin_paper(self, paper_id: str):
        """unpin a paper."""
        if paper_id in self.nodes:
            self.nodes[paper_id].is_pinned = False
            self.pinned_ids.discard(paper_id)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """get paper by id."""
        return self.papers.get(paper_id)

    def get_author(self, author_id: str) -> Optional[Author]:
        """get author by id."""
        return self.authors.get(author_id)

    def get_neighbors(self, node_id: str) -> Set[str]:
        """get all neighbor ids."""
        return self.adjacency.get(node_id, set())

    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """get all edges from a node."""
        return [e for e in self.edges.values() if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> List[GraphEdge]:
        """get all edges to a node."""
        return [e for e in self.edges.values() if e.target_id == node_id]

    def get_papers_in_cluster(self, cluster_id: str) -> List[Paper]:
        """get all papers in a cluster."""
        if cluster_id not in self.clusters:
            return []
        return [self.papers[pid] for pid in self.clusters[cluster_id].paper_ids
                if pid in self.papers]

    def assign_cluster(self, node_id: str, cluster_id: str):
        """assign node to cluster."""
        self.node_cluster_map[node_id] = cluster_id
        if cluster_id in self.clusters:
            self.clusters[cluster_id].paper_ids.add(node_id)

    def compute_hop_distance(self, from_id: str, to_id: str) -> int:
        """compute shortest hop distance between nodes."""
        if from_id == to_id:
            return 0
        if from_id not in self.nodes or to_id not in self.nodes:
            return -1

        # bfs
        visited = {from_id}
        queue = [(from_id, 0)]

        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.adjacency.get(current, set()):
                if neighbor == to_id:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1  # not connected

    def min_distance_to_seeds(self, paper_id: str) -> int:
        """compute minimum distance to any seed."""
        if paper_id in self.seed_ids:
            return 0

        min_dist = float('inf')
        for seed_id in self.seed_ids:
            dist = self.compute_hop_distance(paper_id, seed_id)
            if dist >= 0 and dist < min_dist:
                min_dist = dist

        return min_dist if min_dist != float('inf') else -1

    def count_paths_to_seeds(self, paper_id: str, max_paths: int = 10) -> int:
        """approximate count of independent paths to seeds."""
        if paper_id in self.seed_ids:
            return max_paths

        paths = 0
        for seed_id in self.seed_ids:
            if self.compute_hop_distance(paper_id, seed_id) >= 0:
                paths += 1
                if paths >= max_paths:
                    break

        return paths

    def _add_node(self, node: GraphNode, paper: Paper) -> bool:
        """internal: add node to graph."""
        self.nodes[paper.id] = node
        self.papers[paper.id] = paper
        self.adjacency[paper.id] = set()

        if node.is_seed:
            self.seed_ids.add(paper.id)
        if node.is_portal:
            self.portal_ids.add(paper.id)

        return True

    def _evict_one(self) -> bool:
        """evict lowest-scoring non-protected node."""
        # find candidates for eviction
        candidates = []
        for node_id, node in self.nodes.items():
            if node.is_seed and self.config.never_evict_seeds:
                continue
            if node.is_pinned and self.config.never_evict_pinned:
                continue
            if node.is_portal and self.config.never_evict_portals:
                continue

            # score for eviction (lower = more likely to evict)
            # combine score with LRU
            age = (datetime.now() - node.last_accessed).total_seconds()
            eviction_score = (
                self.config.eviction_score_weight * node.score -
                self.config.eviction_lru_weight * (age / 3600)  # hours
            )
            heapq.heappush(candidates, (eviction_score, node_id))

        if not candidates:
            return False

        # evict lowest
        _, evict_id = heapq.heappop(candidates)
        self._remove_node(evict_id)
        self.evictions += 1
        return True

    def _remove_node(self, node_id: str):
        """remove node and all its edges."""
        if node_id not in self.nodes:
            return

        # remove edges
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            edge = self.edges.pop(edge_id)
            self.edge_index.pop((edge.source_id, edge.target_id), None)

        # update adjacency
        for neighbor in self.adjacency.get(node_id, set()):
            self.adjacency[neighbor].discard(node_id)

        # remove from collections
        self.nodes.pop(node_id, None)
        self.papers.pop(node_id, None)
        self.authors.pop(node_id, None)
        self.adjacency.pop(node_id, None)
        self.seed_ids.discard(node_id)
        self.pinned_ids.discard(node_id)
        self.portal_ids.discard(node_id)

    def stats(self) -> Dict[str, Any]:
        """get graph statistics."""
        edge_type_counts = {}
        for edge in self.edges.values():
            etype = edge.edge_type.value
            edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

        return {
            "nodes": len(self.nodes),
            "papers": len(self.papers),
            "authors": len(self.authors),
            "edges": len(self.edges),
            "seeds": len(self.seed_ids),
            "pinned": len(self.pinned_ids),
            "portals": len(self.portal_ids),
            "clusters": len(self.clusters),
            "evictions": self.evictions,
            "edge_types": edge_type_counts,
            "capacity_nodes": f"{len(self.nodes)}/{self.max_nodes}",
            "capacity_edges": f"{len(self.edges)}/{self.max_edges}"
        }

    def to_dict(self) -> Dict[str, Any]:
        """serialize for export."""
        return {
            "stats": self.stats(),
            "nodes": [self.nodes[nid].data.to_dict() for nid in self.nodes],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.value,
                    "weight": e.weight
                }
                for e in self.edges.values()
            ],
            "seeds": list(self.seed_ids),
            "pinned": list(self.pinned_ids)
        }
