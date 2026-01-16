"""
base provider interface for paper data sources.
all providers (openalex, semantic scholar, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PaperStub:
    """minimal paper representation for graph nodes."""
    # identifiers (at least one should be present)
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None

    # core metadata
    title: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    authors: List[str] = field(default_factory=list)

    # metrics
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None

    # flags
    is_review: bool = False
    is_oa: bool = False
    oa_pdf_url: Optional[str] = None

    # abstract for relevance scoring
    abstract: Optional[str] = None

    # concepts/topics for scope analysis
    concepts: List[Dict[str, Any]] = field(default_factory=list)

    def canonical_id(self) -> str:
        """return best available identifier."""
        if self.doi:
            return f"doi:{self.doi}"
        if self.openalex_id:
            return f"oaid:{self.openalex_id}"
        if self.s2_id:
            return f"s2:{self.s2_id}"
        # fallback: hash of title+year
        return f"title:{hash((self.title, self.year))}"


class PaperProvider(ABC):
    """abstract base class for paper data providers."""

    @abstractmethod
    def search_papers(
        self,
        query: str,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        limit: int = 100
    ) -> List[PaperStub]:
        """search for papers matching query."""
        pass

    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[PaperStub]:
        """get paper by id (doi, openalex_id, s2_id)."""
        pass

    @abstractmethod
    def get_references(self, paper_id: str, limit: int = 50) -> List[PaperStub]:
        """get papers cited by this paper (backward citations)."""
        pass

    @abstractmethod
    def get_citations(self, paper_id: str, limit: int = 30) -> List[PaperStub]:
        """get papers that cite this paper (forward citations)."""
        pass

    @abstractmethod
    def get_count_estimate(self, query: str, year_min: Optional[int] = None, year_max: Optional[int] = None) -> int:
        """get estimated count of papers matching query (for triage)."""
        pass
