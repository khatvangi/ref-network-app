"""
candidate pool - wide storage for discovered papers and authors.
backed by sqlite for persistence and scale.
"""

from typing import List, Optional, Dict, Any, Set
from datetime import datetime

from ..core.models import Paper, Author, Edge, PaperStatus, AuthorStatus, EdgeType
from ..core.config import CandidatePoolConfig, RefnetConfig
from ..core.db import CandidateDB


class CandidatePool:
    """
    wide storage for all discovered papers/authors.
    can hold up to 200k items.
    provides deduplication and pruning.
    """

    def __init__(self, config: Optional[CandidatePoolConfig] = None,
                 db_path: Optional[str] = None):
        self.config = config or CandidatePoolConfig()
        self.db = CandidateDB(db_path or self.config.db_path)

        # in-memory caches for fast lookup
        self._doi_cache: Dict[str, str] = {}  # doi -> paper_id
        self._title_hash_cache: Dict[str, str] = {}  # title_hash -> paper_id

        # stats
        self.api_calls = 0
        self.duplicates_skipped = 0

    def add_paper(self, paper: Paper,
                  check_duplicate: bool = True) -> Optional[Paper]:
        """
        add paper to pool with deduplication.
        returns the paper if added, None if duplicate.
        """
        if check_duplicate:
            existing = self._find_duplicate(paper)
            if existing:
                self.duplicates_skipped += 1
                # merge metadata if useful
                return self._merge_if_better(existing, paper)

        # add to db
        if self.db.add_paper(paper):
            # update caches
            if paper.doi:
                self._doi_cache[self._normalize_doi(paper.doi)] = paper.id
            title_hash = self._title_hash(paper.title)
            if title_hash:
                self._title_hash_cache[title_hash] = paper.id
            return paper

        return None

    def add_papers_batch(self, papers: List[Paper]) -> int:
        """add multiple papers, returns count added."""
        added = 0
        for p in papers:
            if self.add_paper(p):
                added += 1
        return added

    def add_author(self, author: Author) -> Optional[Author]:
        """add author to pool with deduplication."""
        existing = self.find_author(
            orcid=author.orcid,
            openalex_id=author.openalex_id,
            s2_id=author.s2_id
        )
        if existing:
            return existing

        if self.db.add_author(author):
            return author
        return None

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """get paper by internal id."""
        return self.db.get_paper_by_id(paper_id)

    def find_paper(self, doi: Optional[str] = None,
                   openalex_id: Optional[str] = None,
                   s2_id: Optional[str] = None,
                   title: Optional[str] = None) -> Optional[Paper]:
        """find paper by any identifier."""
        # try exact id match first
        paper = self.db.find_paper(doi, openalex_id, s2_id)
        if paper:
            return paper

        # try title hash match
        if title and self.config.dedupe_by_title:
            title_hash = self._title_hash(title)
            if title_hash in self._title_hash_cache:
                return self.db.get_paper_by_id(self._title_hash_cache[title_hash])

        return None

    def find_author(self, orcid: Optional[str] = None,
                    openalex_id: Optional[str] = None,
                    s2_id: Optional[str] = None) -> Optional[Author]:
        """find author by any identifier."""
        return self.db.find_author(orcid, openalex_id, s2_id)

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType, weight: float = 1.0,
                 confidence: float = 1.0) -> bool:
        """add edge between nodes."""
        return self.db.add_edge(
            source_id, target_id, edge_type.value, weight, confidence
        )

    def get_top_candidates(self, limit: int = 100,
                           by: str = "priority_score") -> List[Paper]:
        """get top candidate papers for materialization."""
        return self.db.get_top_candidates(limit, by)

    def get_expansion_queue(self, limit: int = 10) -> List[Paper]:
        """get papers ready for expansion."""
        return self.db.get_papers_for_expansion(limit)

    def update_paper_status(self, paper_id: str, status: PaperStatus):
        """update paper status."""
        self.db.update_paper_status(paper_id, status)

    def update_paper_scores(self, paper_id: str, **scores):
        """update paper scores."""
        self.db.update_paper_scores(paper_id, **scores)

    def get_neighbors(self, paper_id: str) -> Dict[str, List[str]]:
        """get all neighbors of a paper by edge type."""
        neighbors: Dict[str, List[str]] = {}

        # outgoing edges
        for target, etype, weight, conf in self.db.get_edges_from(paper_id):
            if etype not in neighbors:
                neighbors[etype] = []
            neighbors[etype].append(target)

        # incoming edges
        for source, etype, weight, conf in self.db.get_edges_to(paper_id):
            reverse_type = f"reverse_{etype}"
            if reverse_type not in neighbors:
                neighbors[reverse_type] = []
            neighbors[reverse_type].append(source)

        return neighbors

    def get_edges_from_with_weight(self, paper_id: str) -> List[tuple]:
        """get all outgoing edges with full data (target, type, weight, confidence)."""
        return self.db.get_edges_from(paper_id)

    def get_edges_to_with_weight(self, paper_id: str) -> List[tuple]:
        """get all incoming edges with full data (source, type, weight, confidence)."""
        return self.db.get_edges_to(paper_id)

    def prune_if_needed(self):
        """prune low-scoring candidates if pool is too large."""
        total = self.db.count_papers()
        threshold = int(self.config.max_size * self.config.prune_threshold)

        if total > threshold:
            keep = int(self.config.max_size * self.config.prune_keep_fraction)
            pruned = self.db.prune_candidates(keep)
            print(f"[pool] pruned {pruned} candidates, kept {keep}")

    def stats(self) -> Dict[str, Any]:
        """get pool statistics."""
        db_stats = self.db.get_stats()
        return {
            **db_stats,
            "api_calls": self.api_calls,
            "duplicates_skipped": self.duplicates_skipped,
            "doi_cache_size": len(self._doi_cache),
            "title_cache_size": len(self._title_hash_cache)
        }

    def clear(self):
        """clear all data."""
        self.db.clear()
        self._doi_cache.clear()
        self._title_hash_cache.clear()
        self.api_calls = 0
        self.duplicates_skipped = 0

    # private helpers

    def _find_duplicate(self, paper: Paper) -> Optional[Paper]:
        """check if paper already exists."""
        # check by doi
        if paper.doi and self.config.dedupe_by_doi:
            doi = self._normalize_doi(paper.doi)
            if doi in self._doi_cache:
                return self.db.get_paper_by_id(self._doi_cache[doi])
            existing = self.db.get_paper_by_doi(doi)
            if existing:
                self._doi_cache[doi] = existing.id
                return existing

        # check by other ids
        if paper.openalex_id:
            existing = self.db.get_paper_by_openalex(paper.openalex_id)
            if existing:
                return existing

        # check by title hash
        if paper.title and self.config.dedupe_by_title:
            title_hash = self._title_hash(paper.title)
            if title_hash in self._title_hash_cache:
                return self.db.get_paper_by_id(self._title_hash_cache[title_hash])

        return None

    def _merge_if_better(self, existing: Paper, new: Paper) -> Paper:
        """merge new paper data into existing if better."""
        updated = False

        # fill in missing ids
        if not existing.doi and new.doi:
            existing.doi = new.doi
            updated = True
        if not existing.openalex_id and new.openalex_id:
            existing.openalex_id = new.openalex_id
            updated = True
        if not existing.s2_id and new.s2_id:
            existing.s2_id = new.s2_id
            updated = True

        # update citation count if higher
        if new.citation_count and (not existing.citation_count or
                                   new.citation_count > existing.citation_count):
            existing.citation_count = new.citation_count
            updated = True

        # update discovery metadata if new path is shorter (better discovery chain)
        # depth 0 = seed, so lower depth = closer to seed = more relevant path
        if new.depth is not None:
            if existing.depth is None or new.depth < existing.depth:
                existing.depth = new.depth
                existing.discovered_from = new.discovered_from
                existing.discovered_channel = new.discovered_channel
                updated = True

        # if updated, we could persist - for now just return existing
        return existing

    def _normalize_doi(self, doi: str) -> str:
        """normalize doi for comparison."""
        doi = doi.lower().strip()
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]
        if doi.startswith('doi:'):
            doi = doi[4:]
        return doi

    def _title_hash(self, title: str) -> str:
        """create hash of title for fuzzy matching."""
        if not title:
            return ""
        import re
        title = title.lower()
        title = re.sub(r'[^\w\s]', '', title)
        stopwords = {'a', 'an', 'the', 'of', 'and', 'or', 'in', 'on', 'for', 'to', 'with'}
        words = [w for w in title.split() if w not in stopwords]
        return ' '.join(sorted(words[:10]))
