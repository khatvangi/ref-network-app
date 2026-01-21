"""
core data models for refnet.
scientist-centric citation network builder.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from enum import Enum
from datetime import datetime
import uuid


class NodeType(Enum):
    """types of nodes in the graph."""
    PAPER = "paper"
    AUTHOR = "author"


class EdgeType(Enum):
    """types of edges in the graph."""
    # paper-paper edges
    CITES = "cites"                      # P -> R (P cites R)
    CITED_BY = "cited_by"                # derived convenience edge
    INTRO_CITES = "intro_cites"          # true intro extraction from PDF
    INTRO_HINT_CITES = "intro_hint_cites"  # heuristic intro refs

    # paper-author edges
    AUTHORED_BY = "authored_by"          # Paper -> Author
    AUTHORED = "authored"                # derived Author -> Paper

    # author-author edges
    COAUTHOR = "coauthor"                # Author <-> Author

    # trajectory edges
    TRAJECTORY_STEP = "trajectory_step"  # (A,Y) -> (A,Y+1)
    AUTHOR_BRIDGE = "author_bridge"      # cluster bridge via author drift


class PaperStatus(Enum):
    """status of a paper in the system."""
    CANDIDATE = "candidate"        # in candidate pool, not yet materialized
    MATERIALIZED = "materialized"  # in working graph
    EXPANDED = "expanded"          # fully expanded (refs + cites + authors fetched)
    REJECTED = "rejected"          # explicitly rejected (low score, duplicate)
    SEED = "seed"                  # user-provided seed (never evicted)
    PINNED = "pinned"              # user-pinned (never evicted)


class AuthorStatus(Enum):
    """status of an author in the system."""
    CANDIDATE = "candidate"
    MATERIALIZED = "materialized"
    EXPANDED = "expanded"          # works fetched
    TRAJECTORY_COMPUTED = "trajectory_computed"


@dataclass
class Paper:
    """
    full paper model with all metadata.
    used in working graph and for expanded papers.
    """
    # identity (at least one required)
    id: str = ""                          # internal uuid
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    pmid: Optional[str] = None

    # core metadata
    title: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    authors: List[str] = field(default_factory=list)  # author names
    author_ids: List[str] = field(default_factory=list)  # resolved author internal ids

    # extended metadata
    abstract: Optional[str] = None
    concepts: List[Dict[str, Any]] = field(default_factory=list)  # [{name, score}]
    fields: List[str] = field(default_factory=list)  # high-level fields

    # metrics
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None
    influential_citation_count: Optional[int] = None

    # flags
    is_review: bool = False
    is_oa: bool = False
    is_methodology: bool = False  # hub paper like AlphaFold

    # urls
    url: Optional[str] = None
    oa_pdf_url: Optional[str] = None

    # graph state
    status: PaperStatus = PaperStatus.CANDIDATE
    depth: int = 0                        # hops from seed
    discovered_from: Optional[str] = None  # source node id
    discovered_channel: Optional[str] = None  # "backward", "forward", "author", "portal"
    discovered_at: Optional[datetime] = None

    # scores (computed)
    relevance_score: float = 0.0          # GraphRelevance
    novelty_score: float = 0.0            # penalize duplicates + hubs
    priority_score: float = 0.0           # for expansion queue
    materialization_score: float = 0.0    # for working graph entry
    bridge_score: float = 0.0             # cross-cluster bridge potential

    # expansion state
    refs_fetched: bool = False
    cites_fetched: bool = False
    authors_fetched: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.discovered_at:
            self.discovered_at = datetime.now()

    def canonical_id(self) -> str:
        """return best available external id."""
        if self.doi:
            return f"doi:{self.doi}"
        if self.openalex_id:
            return f"oa:{self.openalex_id}"
        if self.s2_id:
            return f"s2:{self.s2_id}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        return f"internal:{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        """serialize for export."""
        return {
            "id": self.id,
            "canonical_id": self.canonical_id(),
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "s2_id": self.s2_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "authors": self.authors,
            "citation_count": self.citation_count,
            "is_review": self.is_review,
            "is_methodology": self.is_methodology,
            "status": self.status.value,
            "depth": self.depth,
            "relevance_score": self.relevance_score,
            "bridge_score": self.bridge_score,
            "concepts": self.concepts[:5] if self.concepts else []
        }


@dataclass
class Author:
    """
    author model with identity resolution.
    """
    # identity
    id: str = ""                          # internal uuid
    orcid: Optional[str] = None           # gold standard
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None

    # metadata
    name: str = ""
    display_name: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)
    paper_count: int = 0
    citation_count: int = 0

    # graph state
    status: AuthorStatus = AuthorStatus.CANDIDATE
    discovered_from: Optional[str] = None  # paper id that introduced this author

    # scores
    topic_fit: float = 0.0                # avg relevance of their papers
    centrality: float = 0.0               # importance in current graph
    priority: float = 0.0                 # for expansion

    # trajectory data
    year_profiles: Dict[int, 'AuthorYearProfile'] = field(default_factory=dict)
    trajectory_computed: bool = False
    drift_events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def canonical_id(self) -> str:
        """return best available external id."""
        if self.orcid:
            return f"orcid:{self.orcid}"
        if self.openalex_id:
            return f"oa_author:{self.openalex_id}"
        if self.s2_id:
            return f"s2_author:{self.s2_id}"
        return f"internal:{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canonical_id": self.canonical_id(),
            "orcid": self.orcid,
            "openalex_id": self.openalex_id,
            "name": self.name,
            "affiliations": self.affiliations[:3],
            "topic_fit": self.topic_fit,
            "centrality": self.centrality,
            "status": self.status.value,
            "trajectory_computed": self.trajectory_computed
        }


@dataclass
class AuthorYearProfile:
    """
    author's topic signature for a specific year.
    used for trajectory analysis.
    """
    author_id: str
    year: int

    # works in this year (bounded)
    work_ids: List[str] = field(default_factory=list)
    work_count: int = 0

    # topic signature
    top_concepts: List[Dict[str, float]] = field(default_factory=list)  # [{name, weight}]
    venues: List[str] = field(default_factory=list)

    # metrics
    total_citations: int = 0
    avg_relevance: float = 0.0

    # embedding centroid (optional, for similarity)
    embedding: Optional[List[float]] = None

    def concept_distribution(self) -> Dict[str, float]:
        """return normalized concept distribution."""
        if not self.top_concepts:
            return {}
        total = sum(c.get('weight', 0) for c in self.top_concepts)
        if total == 0:
            return {}
        return {c['name']: c.get('weight', 0) / total for c in self.top_concepts}


@dataclass
class Edge:
    """
    edge between nodes (paper-paper, paper-author, author-author).
    """
    id: str = ""
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.CITES

    # metadata
    weight: float = 1.0
    confidence: float = 1.0               # for heuristic edges

    # for trajectory edges
    drift_magnitude: Optional[float] = None
    drift_direction: Optional[str] = None  # "entering X, exiting Y"
    is_novelty_jump: bool = False

    # for author bridge edges
    bridge_years: Optional[List[int]] = None
    bridge_strength: Optional[float] = None
    exemplar_works: Optional[List[str]] = None

    created_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
            "confidence": self.confidence
        }


@dataclass
class Cluster:
    """
    cluster of related papers (concept-based or community-detected).
    """
    id: str = ""
    name: str = ""

    # membership
    paper_ids: Set[str] = field(default_factory=set)
    author_ids: Set[str] = field(default_factory=set)

    # signature
    top_concepts: List[Dict[str, float]] = field(default_factory=list)
    top_venues: List[str] = field(default_factory=list)
    year_range: Optional[tuple] = None

    # metrics
    size: int = 0
    avg_relevance: float = 0.0
    internal_density: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class GapCandidate:
    """
    a potential gap/bridge in the literature.
    """
    id: str = ""
    gap_type: str = ""  # "missing_bridge", "missing_intermediate", "unexplored_cluster"

    # what clusters/papers it connects
    cluster_a_id: Optional[str] = None
    cluster_b_id: Optional[str] = None

    # candidate papers that could fill the gap
    candidate_paper_ids: List[str] = field(default_factory=list)
    candidate_author_ids: List[str] = field(default_factory=list)

    # evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    description: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Bucket:
    """
    a bucket of papers for citation walking (dendrimer model).
    each paper creates its OWN child bucket, forming a tree structure.
    branches can be pruned independently based on relevance decay.
    """
    id: str = ""
    generation: int = 0  # 0 = seed bucket, 1 = first expansion, etc.
    papers: List[Paper] = field(default_factory=list)

    # dendrimer structure - track WHICH paper created this bucket
    source_paper_id: Optional[str] = None  # the paper that spawned this bucket
    source_paper_title: Optional[str] = None  # for debugging/visualization
    parent_bucket_id: Optional[str] = None  # parent bucket for lineage tracking
    child_bucket_ids: List[str] = field(default_factory=list)  # children for tree traversal

    avg_relevance: float = 0.0  # average relevance of papers in bucket
    is_alive: bool = True  # false if pruned due to relevance decay
    created_at: datetime = field(default_factory=datetime.now)

    # tracking
    total_refs_fetched: int = 0
    total_cites_fetched: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"bucket_{self.generation}_{str(uuid.uuid4())[:8]}"

    def compute_avg_relevance(self):
        """compute average relevance of papers in bucket."""
        if not self.papers:
            self.avg_relevance = 0.0
        else:
            self.avg_relevance = sum(
                p.relevance_score or 0.0 for p in self.papers
            ) / len(self.papers)

    def prune(self, reason: str = "relevance_decay"):
        """mark this bucket as pruned (dead branch)."""
        self.is_alive = False


@dataclass
class BucketExpansionState:
    """
    state for bucket-based citation walking (dendrimer model).
    tracks all buckets as a tree structure and global stopping conditions.
    """
    all_buckets: Dict[str, Bucket] = field(default_factory=dict)
    root_bucket_id: Optional[str] = None  # the initial seed bucket
    seen_paper_ids: Set[str] = field(default_factory=set)
    relevance_history: List[float] = field(default_factory=list)  # rolling window
    current_generation: int = 0
    total_papers_discovered: int = 0
    total_api_calls: int = 0
    topic: Optional[str] = None

    # dendrimer stats
    active_branches: int = 0  # buckets still alive
    pruned_branches: int = 0  # buckets killed due to relevance decay
    total_buckets_created: int = 0

    # adaptive depth
    max_generations: int = 10

    # stopping flags
    stopped_reason: Optional[str] = None
    is_exhausted: bool = False

    def get_alive_buckets_at_generation(self, gen: int) -> List[Bucket]:
        """get all alive (not pruned) buckets at a specific generation."""
        return [b for b in self.all_buckets.values()
                if b.generation == gen and b.is_alive]

    def count_alive_branches(self) -> int:
        """count total alive branches (buckets with is_alive=True)."""
        return sum(1 for b in self.all_buckets.values() if b.is_alive)
