"""
configuration for refnet.
all settings in one place, easily tunable.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from pathlib import Path


class ExpansionMode(Enum):
    """how the system expands the graph."""
    SCIENTIST = "scientist"  # citation-walk first, keywords weak
    KEYWORD = "keyword"      # legacy keyword-centric mode


class AggressivenessLevel(Enum):
    """how aggressively to explore."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GraphSize(Enum):
    """working graph size preset."""
    SMALL = "small"    # 500 nodes
    MEDIUM = "medium"  # 2000 nodes
    LARGE = "large"    # 5000 nodes


@dataclass
class ProviderConfig:
    """provider-specific settings."""
    # openalex
    openalex_email: str = "user@example.com"
    openalex_enabled: bool = True

    # semantic scholar
    s2_api_key: Optional[str] = None
    s2_enabled: bool = True

    # pubmed
    pubmed_api_key: Optional[str] = None
    pubmed_enabled: bool = True

    # crossref
    crossref_enabled: bool = True

    # google scholar
    gscholar_enabled: bool = False  # slow, disabled by default
    gscholar_delay: float = 5.0     # seconds between requests
    gscholar_limit: int = 20        # max results

    # orcid
    orcid_enabled: bool = True

    # rate limits (requests per second)
    openalex_rps: float = 10.0
    s2_rps: float = 100.0  # with key
    s2_rps_no_key: float = 0.3
    pubmed_rps: float = 3.0
    crossref_rps: float = 50.0
    orcid_rps: float = 1.0


@dataclass
class TriageConfig:
    """scope triage settings."""
    # risk score weights
    volume_weight: float = 0.45
    concept_entropy_weight: float = 0.25
    venue_dispersion_weight: float = 0.20
    specificity_weight: float = 0.10

    # band thresholds
    red_count_threshold: int = 20000
    red_score_threshold: float = 0.75
    yellow_count_min: int = 5000
    yellow_score_min: float = 0.55

    # preview limits
    seed_preview_count: int = 10
    top_concepts_count: int = 10
    top_venues_count: int = 10


@dataclass
class CandidatePoolConfig:
    """candidate pool (wide storage) settings."""
    max_size: int = 200000
    prune_threshold: float = 0.8  # prune when 80% full
    prune_keep_fraction: float = 0.6  # keep top 60% by score

    # sqlite settings
    db_path: str = "refnet_candidates.db"
    use_sqlite: bool = True

    # deduplication
    dedupe_by_doi: bool = True
    dedupe_by_title: bool = True
    title_similarity_threshold: float = 0.8


@dataclass
class WorkingGraphConfig:
    """working graph (narrow, visible) settings."""
    # size limits by preset
    max_nodes_small: int = 500
    max_nodes_medium: int = 2000
    max_nodes_large: int = 5000

    max_edges_small: int = 2000
    max_edges_medium: int = 8000
    max_edges_large: int = 20000

    # default
    default_size: GraphSize = GraphSize.MEDIUM

    # eviction
    never_evict_seeds: bool = True
    never_evict_pinned: bool = True
    never_evict_portals: bool = True  # reviews
    eviction_lru_weight: float = 0.3
    eviction_score_weight: float = 0.7


@dataclass
class ExpansionConfig:
    """graph expansion settings."""
    mode: ExpansionMode = ExpansionMode.SCIENTIST

    # depth limits
    max_depth: int = 3

    # per-node fetch limits
    max_refs_per_node: int = 50
    max_cites_per_node: int = 30
    max_author_works_per_node: int = 20

    # intro hint settings
    intro_fraction: float = 0.25
    intro_hint_weight: float = 2.0
    max_intro_hint_per_node: int = 20
    min_relevance_intro: float = 0.20

    # hub suppression (methodology papers)
    hub_citation_threshold: int = 5000
    hub_relevance_threshold: float = 0.30
    mega_hub_threshold: int = 50000  # never fully expand

    # frontier vs core expansion
    core_expand_fraction: float = 0.6
    frontier_expand_fraction: float = 0.4

    # api call budget
    max_api_calls_per_job: int = 2000

    # aggressiveness presets
    aggressiveness: AggressivenessLevel = AggressivenessLevel.MEDIUM

    # bucket expansion settings (citation walking)
    bucket_mode_enabled: bool = False  # enable bucket-based expansion
    base_max_generations: int = 10     # starting max generations (adaptive)
    min_bucket_relevance: float = 0.15 # minimum bucket avg relevance to continue branch
    drift_window: int = 30             # papers to consider for topic drift
    drift_kill_threshold: float = 0.10 # stop if <10% of recent papers are relevant
    min_relevance: float = 0.15        # minimum relevance to include paper in bucket

    def get_fetch_multiplier(self) -> float:
        """multiplier for fetch limits based on aggressiveness."""
        if self.aggressiveness == AggressivenessLevel.LOW:
            return 0.5
        if self.aggressiveness == AggressivenessLevel.HIGH:
            return 1.5
        return 1.0


@dataclass
class AuthorConfig:
    """author layer settings."""
    enabled: bool = True

    # which authors to expand per paper
    expand_corresponding: bool = True
    expand_first_author: bool = True
    expand_last_author: bool = True
    author_expand_k: int = 2  # max additional authors by centrality

    # per-author work limits
    author_recent_years: int = 5
    author_recent_cap: int = 10
    author_foundational_cap: int = 5
    author_total_cap: int = 15
    author_edges_budget_per_node: int = 20

    # identity resolution
    prefer_orcid: bool = True
    skip_ambiguous: bool = True
    min_coauthor_overlap: float = 0.3  # for fuzzy matching

    # mega-author suppression
    mega_author_threshold: int = 500  # papers
    skip_mega_unless_central: bool = True


@dataclass
class TrajectoryConfig:
    """author trajectory settings."""
    enabled: bool = True

    # which authors get trajectories
    max_trajectory_authors_auto: int = 30
    max_trajectory_authors_user: int = 200
    min_centrality_for_trajectory: float = 0.3

    # drift detection
    drift_jump_threshold: float = 0.55  # JSD threshold for novelty jump
    min_years_for_trajectory: int = 3
    year_window_size: int = 1  # compare adjacent years

    # profile settings
    max_works_per_year_profile: int = 20
    max_concepts_per_profile: int = 10


@dataclass
class ScoringConfig:
    """scoring weights for graph relevance and materialization."""

    # GraphRelevance weights (SCIENTIST mode)
    proximity_weight: float = 0.45
    multipath_weight: float = 0.20
    coupling_weight: float = 0.15
    portal_weight: float = 0.10
    keyword_weight: float = 0.10  # weak tie-breaker only

    # MaterializationScore weights
    mat_proximity_weight: float = 0.35
    mat_multipath_weight: float = 0.20
    mat_coupling_weight: float = 0.15
    mat_portal_weight: float = 0.10
    mat_bridge_weight: float = 0.10
    mat_recency_weight: float = 0.10

    # proximity decay
    proximity_alpha: float = 0.9  # exp(-alpha * distance)

    # hub penalty
    hubness_penalty_factor: float = 0.5


@dataclass
class GapAnalysisConfig:
    """gap analysis settings."""
    enabled: bool = True

    # cluster detection
    min_cluster_size: int = 5
    max_clusters: int = 20

    # bridge detection
    min_bridge_score: float = 0.3
    max_bridges_to_show: int = 10

    # missing intermediates
    max_missing_to_show: int = 20


@dataclass
class ExportConfig:
    """export settings."""
    output_dir: str = "output"
    default_format: str = "json"

    # graphml settings
    include_all_attributes: bool = True

    # csv settings
    nodes_filename: str = "nodes.csv"
    edges_filename: str = "edges.csv"

    # visualization
    generate_html_viewer: bool = True
    viewer_template: str = "cytoscape"  # or "flourish"


@dataclass
class RefnetConfig:
    """master configuration for refnet."""
    # sub-configs
    providers: ProviderConfig = field(default_factory=ProviderConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    candidate_pool: CandidatePoolConfig = field(default_factory=CandidatePoolConfig)
    working_graph: WorkingGraphConfig = field(default_factory=WorkingGraphConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    author: AuthorConfig = field(default_factory=AuthorConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    gap_analysis: GapAnalysisConfig = field(default_factory=GapAnalysisConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # global settings
    years_back: int = 5
    verbose: bool = True
    cache_dir: str = ".refnet_cache"

    @classmethod
    def default(cls) -> 'RefnetConfig':
        """return default configuration."""
        return cls()

    @classmethod
    def minimal(cls) -> 'RefnetConfig':
        """minimal config for quick testing."""
        config = cls()
        config.candidate_pool.max_size = 10000
        config.working_graph.default_size = GraphSize.SMALL
        config.expansion.max_depth = 2
        config.expansion.max_api_calls_per_job = 500
        config.author.enabled = False
        config.trajectory.enabled = False
        return config

    @classmethod
    def full(cls) -> 'RefnetConfig':
        """full config with all features."""
        config = cls()
        config.working_graph.default_size = GraphSize.LARGE
        config.expansion.max_depth = 4
        config.expansion.aggressiveness = AggressivenessLevel.HIGH
        return config

    def get_max_nodes(self) -> int:
        """get max nodes based on graph size preset."""
        size = self.working_graph.default_size
        if size == GraphSize.SMALL:
            return self.working_graph.max_nodes_small
        if size == GraphSize.LARGE:
            return self.working_graph.max_nodes_large
        return self.working_graph.max_nodes_medium

    def get_max_edges(self) -> int:
        """get max edges based on graph size preset."""
        size = self.working_graph.default_size
        if size == GraphSize.SMALL:
            return self.working_graph.max_edges_small
        if size == GraphSize.LARGE:
            return self.working_graph.max_edges_large
        return self.working_graph.max_edges_medium
