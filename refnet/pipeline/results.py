"""
pipeline results - comprehensive literature analysis output.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.models import Paper
from ..agents.corpus_fetcher import AuthorCorpus
from ..agents.trajectory_analyzer import TrajectoryAnalysis
from ..agents.collaborator_mapper import CollaborationNetwork
from ..agents.topic_extractor import TopicAnalysis, Topic
from ..agents.gap_detector import GapAnalysis, ConceptPair, BridgePaper
from ..agents.relevance_scorer import RelevanceScore
from ..agents.field_resolver import FieldResolution

# optional LLM extraction (may not be installed)
try:
    from ..llm.extractor import ExtractedInfo, PaperRelationship
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    ExtractedInfo = None
    PaperRelationship = None


@dataclass
class AuthorProfile:
    """
    complete profile of a key author.
    """
    name: str
    author_id: str

    # basic stats
    paper_count: int = 0
    citation_count: int = 0
    h_index: Optional[int] = None
    affiliations: List[str] = field(default_factory=list)

    # analysis results
    trajectory: Optional[TrajectoryAnalysis] = None
    collaboration_network: Optional[CollaborationNetwork] = None

    # relevance to query
    relevance_score: float = 0.0
    relevance_reason: str = ""

    # key papers by this author (in context)
    key_papers: List[str] = field(default_factory=list)  # paper IDs


@dataclass
class ReadingListItem:
    """
    a paper in the recommended reading list.
    """
    paper: Paper
    relevance_score: float = 0.0

    # why it's included
    category: str = "general"  # "seed", "foundational", "recent", "bridge", "key_author"
    reason: str = ""

    # reading priority
    priority: int = 0  # 1 = must read, 2 = should read, 3 = optional


@dataclass
class FieldLandscape:
    """
    overview of the research field.
    """
    # topic structure
    core_topics: List[str] = field(default_factory=list)
    emerging_topics: List[str] = field(default_factory=list)
    declining_topics: List[str] = field(default_factory=list)

    # topic details
    topic_analysis: Optional[TopicAnalysis] = None

    # temporal patterns
    year_range: tuple = (0, 0)
    peak_years: List[int] = field(default_factory=list)
    papers_per_year: Dict[int, int] = field(default_factory=dict)

    # field size
    total_papers: int = 0
    total_authors: int = 0


@dataclass
class ResearchGaps:
    """
    identified gaps and opportunities.
    """
    # concept gaps (underexplored combinations)
    concept_gaps: List[ConceptPair] = field(default_factory=list)

    # method gaps (techniques not applied)
    method_gaps: List[Dict[str, Any]] = field(default_factory=list)

    # bridge papers (connect different areas)
    bridge_papers: List[BridgePaper] = field(default_factory=list)

    # unexplored areas
    unexplored_areas: List[Dict[str, Any]] = field(default_factory=list)

    # full analysis
    gap_analysis: Optional[GapAnalysis] = None


@dataclass
class LiteratureAnalysis:
    """
    complete literature analysis result.

    this is the main output of the pipeline.
    """
    # metadata
    created_at: datetime = field(default_factory=datetime.now)
    seed_query: str = ""
    seed_type: str = ""  # "paper", "author", "topic"

    # the seed
    seed_paper: Optional[Paper] = None
    seed_author: Optional[AuthorProfile] = None

    # collected papers
    all_papers: List[Paper] = field(default_factory=list)
    paper_count: int = 0

    # key authors
    key_authors: List[AuthorProfile] = field(default_factory=list)

    # field landscape
    landscape: Optional[FieldLandscape] = None

    # gaps and opportunities
    gaps: Optional[ResearchGaps] = None

    # reading list (ranked)
    reading_list: List[ReadingListItem] = field(default_factory=list)

    # insights (human-readable takeaways)
    insights: List[str] = field(default_factory=list)

    # execution stats
    duration_seconds: float = 0.0
    api_calls: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # verification and field resolution
    resolved_field: Optional[FieldResolution] = None
    verification_summary: Optional[Dict[str, Any]] = None

    # LLM extraction results
    extracted_info: List[Any] = field(default_factory=list)  # List[ExtractedInfo]
    paper_relationships: List[Any] = field(default_factory=list)  # List[PaperRelationship]

    def add_insight(self, insight: str):
        """add a key insight."""
        self.insights.append(insight)

    def add_error(self, error: str):
        """add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """add a warning message."""
        self.warnings.append(warning)

    @property
    def success(self) -> bool:
        """true if analysis completed with usable results."""
        return self.paper_count > 0 and len(self.errors) == 0

    def summary(self) -> str:
        """generate a brief summary."""
        lines = [
            f"Literature Analysis: {self.seed_query}",
            f"  Papers: {self.paper_count}",
            f"  Key authors: {len(self.key_authors)}",
        ]

        if self.landscape:
            lines.append(f"  Core topics: {', '.join(self.landscape.core_topics[:3])}")

        if self.gaps and self.gaps.concept_gaps:
            lines.append(f"  Gaps identified: {len(self.gaps.concept_gaps)}")

        if self.reading_list:
            lines.append(f"  Reading list: {len(self.reading_list)} papers")

        if self.insights:
            lines.append(f"  Key insights: {len(self.insights)}")

        return "\n".join(lines)
