"""
refnet - scientist-centric citation network builder.
"""

from .core.config import RefnetConfig
from .core.models import Paper, Author, Edge, EdgeType, PaperStatus
from .graph.candidate_pool import CandidatePool
from .graph.working_graph import WorkingGraph
from .graph.expansion import ExpansionEngine
from .providers.openalex import OpenAlexProvider
from .providers.semantic_scholar import SemanticScholarProvider
from .export.formats import GraphExporter
from .export.viewer import HTMLViewer

__version__ = "0.2.0"

__all__ = [
    "RefnetConfig",
    "Paper",
    "Author",
    "Edge",
    "EdgeType",
    "PaperStatus",
    "CandidatePool",
    "WorkingGraph",
    "ExpansionEngine",
    "OpenAlexProvider",
    "SemanticScholarProvider",
    "GraphExporter",
    "HTMLViewer"
]
