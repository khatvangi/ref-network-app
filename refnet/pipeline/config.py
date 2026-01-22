"""
pipeline configuration - controls how deep and wide to explore.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    """
    configuration for literature analysis pipeline.

    controls:
    - depth: how many hops from seed
    - width: how many items per hop
    - which analyses to run
    """

    # expansion limits
    max_references: int = 30       # refs per paper
    max_citations: int = 30        # cites per paper
    max_authors_to_follow: int = 5 # key authors to expand
    max_papers_per_author: int = 50  # papers per author corpus

    # depth control
    citation_depth: int = 1        # hops for citations (1 = direct only)
    author_depth: int = 1          # hops for author network

    # analysis toggles
    analyze_trajectories: bool = True
    analyze_collaborations: bool = True
    analyze_topics: bool = True
    analyze_gaps: bool = True
    score_relevance: bool = True

    # filtering
    min_citation_count: int = 0    # filter papers below this
    min_year: Optional[int] = None # filter papers before this
    max_year: Optional[int] = None # filter papers after this

    # focus
    focus_concepts: List[str] = field(default_factory=list)
    focus_authors: List[str] = field(default_factory=list)

    # output
    top_papers_count: int = 20     # how many papers in reading list
    top_authors_count: int = 10    # how many authors to highlight
    top_gaps_count: int = 5        # how many gaps to report


@dataclass
class QuickConfig(PipelineConfig):
    """fast exploration - minimal expansion."""
    max_references: int = 15
    max_citations: int = 15
    max_authors_to_follow: int = 3
    max_papers_per_author: int = 30
    analyze_gaps: bool = False


@dataclass
class DeepConfig(PipelineConfig):
    """thorough exploration - comprehensive analysis."""
    max_references: int = 50
    max_citations: int = 50
    max_authors_to_follow: int = 10
    max_papers_per_author: int = 100
    citation_depth: int = 2
    top_papers_count: int = 50


@dataclass
class AuthorFocusConfig(PipelineConfig):
    """author-focused exploration - deep author analysis."""
    max_references: int = 20
    max_citations: int = 20
    max_authors_to_follow: int = 3
    max_papers_per_author: int = 200
    analyze_trajectories: bool = True
    analyze_collaborations: bool = True
