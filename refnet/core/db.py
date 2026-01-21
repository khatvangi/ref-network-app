"""
sqlite database for candidate pool persistence.
stores lightweight candidate records for wide exploration.
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from .models import Paper, Author, PaperStatus, AuthorStatus


class CandidateDB:
    """
    sqlite-backed storage for candidate pool.
    stores minimal metadata for discovered papers/authors.
    """

    def __init__(self, db_path: str = "refnet_candidates.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                -- paper candidates
                CREATE TABLE IF NOT EXISTS paper_candidates (
                    id TEXT PRIMARY KEY,
                    doi TEXT,
                    openalex_id TEXT,
                    s2_id TEXT,
                    pmid TEXT,
                    title TEXT NOT NULL,
                    year INTEGER,
                    venue TEXT,
                    citation_count INTEGER,
                    reference_count INTEGER,
                    is_review INTEGER DEFAULT 0,
                    is_methodology INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'candidate',
                    depth INTEGER DEFAULT 0,
                    discovered_from TEXT,
                    discovered_channel TEXT,
                    discovered_at TEXT,
                    relevance_score REAL DEFAULT 0.0,
                    novelty_score REAL DEFAULT 0.0,
                    priority_score REAL DEFAULT 0.0,
                    materialization_score REAL DEFAULT 0.0,
                    bridge_score REAL DEFAULT 0.0,
                    concepts_json TEXT,
                    extra_json TEXT,
                    UNIQUE(doi),
                    UNIQUE(openalex_id),
                    UNIQUE(s2_id)
                );

                -- indexes for fast lookup
                CREATE INDEX IF NOT EXISTS idx_paper_doi ON paper_candidates(doi);
                CREATE INDEX IF NOT EXISTS idx_paper_openalex ON paper_candidates(openalex_id);
                CREATE INDEX IF NOT EXISTS idx_paper_s2 ON paper_candidates(s2_id);
                CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_candidates(status);
                CREATE INDEX IF NOT EXISTS idx_paper_priority ON paper_candidates(priority_score DESC);
                CREATE INDEX IF NOT EXISTS idx_paper_materialization ON paper_candidates(materialization_score DESC);

                -- author candidates
                CREATE TABLE IF NOT EXISTS author_candidates (
                    id TEXT PRIMARY KEY,
                    orcid TEXT,
                    openalex_id TEXT,
                    s2_id TEXT,
                    name TEXT NOT NULL,
                    affiliations_json TEXT,
                    status TEXT DEFAULT 'candidate',
                    discovered_from TEXT,
                    topic_fit REAL DEFAULT 0.0,
                    centrality REAL DEFAULT 0.0,
                    priority REAL DEFAULT 0.0,
                    extra_json TEXT,
                    UNIQUE(orcid),
                    UNIQUE(openalex_id),
                    UNIQUE(s2_id)
                );

                CREATE INDEX IF NOT EXISTS idx_author_orcid ON author_candidates(orcid);
                CREATE INDEX IF NOT EXISTS idx_author_openalex ON author_candidates(openalex_id);
                CREATE INDEX IF NOT EXISTS idx_author_priority ON author_candidates(priority DESC);

                -- edges (lightweight)
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    extra_json TEXT,
                    UNIQUE(source_id, target_id, edge_type)
                );

                CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_id);
                CREATE INDEX IF NOT EXISTS idx_edge_type ON edges(edge_type);

                -- metadata table
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

    @contextmanager
    def _conn(self):
        """context manager for db connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # paper operations

    def add_paper(self, paper: Paper) -> bool:
        """
        add paper to candidate pool.
        returns True if added, False if already exists.
        """
        with self._conn() as conn:
            try:
                conn.execute("""
                    INSERT INTO paper_candidates (
                        id, doi, openalex_id, s2_id, pmid,
                        title, year, venue, citation_count, reference_count,
                        is_review, is_methodology, status, depth,
                        discovered_from, discovered_channel, discovered_at,
                        relevance_score, novelty_score, priority_score,
                        materialization_score, bridge_score,
                        concepts_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.id, paper.doi, paper.openalex_id, paper.s2_id, paper.pmid,
                    paper.title, paper.year, paper.venue,
                    paper.citation_count, paper.reference_count,
                    1 if paper.is_review else 0,
                    1 if paper.is_methodology else 0,
                    paper.status.value, paper.depth,
                    paper.discovered_from, paper.discovered_channel,
                    paper.discovered_at.isoformat() if paper.discovered_at else None,
                    paper.relevance_score, paper.novelty_score, paper.priority_score,
                    paper.materialization_score, paper.bridge_score,
                    json.dumps(paper.concepts) if paper.concepts else None
                ))
                return True
            except sqlite3.IntegrityError:
                # already exists
                return False

    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """get paper by internal id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM paper_candidates WHERE id = ?",
                (paper_id,)
            ).fetchone()
            if row:
                return self._row_to_paper(row)
        return None

    def get_paper_by_doi(self, doi: str) -> Optional[Paper]:
        """get paper by doi."""
        doi = self._normalize_doi(doi)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM paper_candidates WHERE doi = ?",
                (doi,)
            ).fetchone()
            if row:
                return self._row_to_paper(row)
        return None

    def get_paper_by_openalex(self, oa_id: str) -> Optional[Paper]:
        """get paper by openalex id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM paper_candidates WHERE openalex_id = ?",
                (oa_id,)
            ).fetchone()
            if row:
                return self._row_to_paper(row)
        return None

    def find_paper(self, doi: Optional[str] = None,
                   openalex_id: Optional[str] = None,
                   s2_id: Optional[str] = None) -> Optional[Paper]:
        """find paper by any id."""
        if doi:
            p = self.get_paper_by_doi(doi)
            if p:
                return p
        if openalex_id:
            p = self.get_paper_by_openalex(openalex_id)
            if p:
                return p
        if s2_id:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM paper_candidates WHERE s2_id = ?",
                    (s2_id,)
                ).fetchone()
                if row:
                    return self._row_to_paper(row)
        return None

    def update_paper_scores(self, paper_id: str,
                            relevance: Optional[float] = None,
                            novelty: Optional[float] = None,
                            priority: Optional[float] = None,
                            materialization: Optional[float] = None,
                            bridge: Optional[float] = None):
        """update paper scores."""
        updates = []
        values = []
        if relevance is not None:
            updates.append("relevance_score = ?")
            values.append(relevance)
        if novelty is not None:
            updates.append("novelty_score = ?")
            values.append(novelty)
        if priority is not None:
            updates.append("priority_score = ?")
            values.append(priority)
        if materialization is not None:
            updates.append("materialization_score = ?")
            values.append(materialization)
        if bridge is not None:
            updates.append("bridge_score = ?")
            values.append(bridge)

        if not updates:
            return

        values.append(paper_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE paper_candidates SET {', '.join(updates)} WHERE id = ?",
                values
            )

    def update_paper_status(self, paper_id: str, status: PaperStatus):
        """update paper status."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE paper_candidates SET status = ? WHERE id = ?",
                (status.value, paper_id)
            )

    def update_paper_metadata(self, paper_id: str,
                              doi: Optional[str] = None,
                              openalex_id: Optional[str] = None,
                              s2_id: Optional[str] = None,
                              citation_count: Optional[int] = None,
                              depth: Optional[int] = None,
                              discovered_from: Optional[str] = None,
                              discovered_channel: Optional[str] = None):
        """
        update paper metadata fields that may improve during merge.
        only updates fields that are explicitly passed (not None).
        """
        updates = []
        values = []
        if doi is not None:
            updates.append("doi = ?")
            values.append(doi)
        if openalex_id is not None:
            updates.append("openalex_id = ?")
            values.append(openalex_id)
        if s2_id is not None:
            updates.append("s2_id = ?")
            values.append(s2_id)
        if citation_count is not None:
            updates.append("citation_count = ?")
            values.append(citation_count)
        if depth is not None:
            updates.append("depth = ?")
            values.append(depth)
        if discovered_from is not None:
            updates.append("discovered_from = ?")
            values.append(discovered_from)
        if discovered_channel is not None:
            updates.append("discovered_channel = ?")
            values.append(discovered_channel)

        if not updates:
            return

        values.append(paper_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE paper_candidates SET {', '.join(updates)} WHERE id = ?",
                values
            )

    def get_top_candidates(self, limit: int = 100,
                           by: str = "priority_score") -> List[Paper]:
        """get top candidate papers by score."""
        # whitelist mapping to prevent sql injection
        allowed_columns = {
            "priority_score": "priority_score",
            "materialization_score": "materialization_score",
            "relevance_score": "relevance_score",
            "bridge_score": "bridge_score",
        }
        order_col = allowed_columns.get(by, "priority_score")

        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT * FROM paper_candidates
                WHERE status = 'candidate'
                ORDER BY {order_col} DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_paper(r) for r in rows]

    def get_papers_for_expansion(self, limit: int = 10) -> List[Paper]:
        """get papers ready for expansion (materialized but not expanded)."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM paper_candidates
                WHERE status = 'materialized'
                ORDER BY priority_score DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_paper(r) for r in rows]

    def count_papers(self, status: Optional[PaperStatus] = None) -> int:
        """count papers, optionally by status."""
        with self._conn() as conn:
            if status:
                return conn.execute(
                    "SELECT COUNT(*) FROM paper_candidates WHERE status = ?",
                    (status.value,)
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM paper_candidates"
            ).fetchone()[0]

    def prune_candidates(self, keep_count: int):
        """prune lowest-scoring candidates to reduce pool size."""
        with self._conn() as conn:
            # count current candidates
            total = conn.execute(
                "SELECT COUNT(*) FROM paper_candidates WHERE status = 'candidate'"
            ).fetchone()[0]

            if total <= keep_count:
                return 0

            # delete lowest scoring
            delete_count = total - keep_count
            conn.execute("""
                DELETE FROM paper_candidates
                WHERE id IN (
                    SELECT id FROM paper_candidates
                    WHERE status = 'candidate'
                    ORDER BY priority_score ASC
                    LIMIT ?
                )
            """, (delete_count,))
            return delete_count

    # author operations

    def add_author(self, author: Author) -> bool:
        """add author to candidate pool."""
        with self._conn() as conn:
            try:
                conn.execute("""
                    INSERT INTO author_candidates (
                        id, orcid, openalex_id, s2_id, name,
                        affiliations_json, status, discovered_from,
                        topic_fit, centrality, priority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    author.id, author.orcid, author.openalex_id, author.s2_id,
                    author.name, json.dumps(author.affiliations),
                    author.status.value, author.discovered_from,
                    author.topic_fit, author.centrality, author.priority
                ))
                return True
            except sqlite3.IntegrityError:
                return False

    def get_author_by_id(self, author_id: str) -> Optional[Author]:
        """get author by internal id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM author_candidates WHERE id = ?",
                (author_id,)
            ).fetchone()
            if row:
                return self._row_to_author(row)
        return None

    def find_author(self, orcid: Optional[str] = None,
                    openalex_id: Optional[str] = None,
                    s2_id: Optional[str] = None) -> Optional[Author]:
        """find author by any id."""
        with self._conn() as conn:
            if orcid:
                row = conn.execute(
                    "SELECT * FROM author_candidates WHERE orcid = ?",
                    (orcid,)
                ).fetchone()
                if row:
                    return self._row_to_author(row)
            if openalex_id:
                row = conn.execute(
                    "SELECT * FROM author_candidates WHERE openalex_id = ?",
                    (openalex_id,)
                ).fetchone()
                if row:
                    return self._row_to_author(row)
            if s2_id:
                row = conn.execute(
                    "SELECT * FROM author_candidates WHERE s2_id = ?",
                    (s2_id,)
                ).fetchone()
                if row:
                    return self._row_to_author(row)
        return None

    def get_top_authors(self, limit: int = 50,
                        by: str = "centrality") -> List[Author]:
        """get top authors by score."""
        # whitelist mapping to prevent sql injection
        allowed_columns = {
            "centrality": "centrality",
            "topic_fit": "topic_fit",
            "priority": "priority",
        }
        order_col = allowed_columns.get(by, "centrality")

        with self._conn() as conn:
            rows = conn.execute(f"""
                SELECT * FROM author_candidates
                ORDER BY {order_col} DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_author(r) for r in rows]

    # edge operations

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: str, weight: float = 1.0,
                 confidence: float = 1.0) -> bool:
        """add edge between nodes."""
        import uuid
        with self._conn() as conn:
            try:
                conn.execute("""
                    INSERT INTO edges (id, source_id, target_id, edge_type, weight, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (str(uuid.uuid4()), source_id, target_id, edge_type, weight, confidence))
                return True
            except sqlite3.IntegrityError:
                return False

    def get_edges_from(self, source_id: str) -> List[Tuple[str, str, str, float]]:
        """get all edges from a source node."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT target_id, edge_type, weight, confidence FROM edges WHERE source_id = ?",
                (source_id,)
            ).fetchall()
            return [(r[0], r[1], r[2], r[3]) for r in rows]

    def get_edges_to(self, target_id: str) -> List[Tuple[str, str, str, float]]:
        """get all edges to a target node."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_id, edge_type, weight, confidence FROM edges WHERE target_id = ?",
                (target_id,)
            ).fetchall()
            return [(r[0], r[1], r[2], r[3]) for r in rows]

    def count_edges(self) -> int:
        """count total edges."""
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # helpers

    def _normalize_doi(self, doi: str) -> str:
        """normalize doi for storage."""
        doi = doi.lower().strip()
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]
        if doi.startswith('doi:'):
            doi = doi[4:]
        return doi

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        """convert db row to Paper object."""
        return Paper(
            id=row['id'],
            doi=row['doi'],
            openalex_id=row['openalex_id'],
            s2_id=row['s2_id'],
            pmid=row['pmid'],
            title=row['title'] or "",
            year=row['year'],
            venue=row['venue'],
            citation_count=row['citation_count'],
            reference_count=row['reference_count'],
            is_review=bool(row['is_review']),
            is_methodology=bool(row['is_methodology']),
            status=PaperStatus(row['status']),
            depth=row['depth'] or 0,
            discovered_from=row['discovered_from'],
            discovered_channel=row['discovered_channel'],
            discovered_at=datetime.fromisoformat(row['discovered_at']) if row['discovered_at'] else None,
            relevance_score=row['relevance_score'] or 0.0,
            novelty_score=row['novelty_score'] or 0.0,
            priority_score=row['priority_score'] or 0.0,
            materialization_score=row['materialization_score'] or 0.0,
            bridge_score=row['bridge_score'] or 0.0,
            concepts=json.loads(row['concepts_json']) if row['concepts_json'] else []
        )

    def _row_to_author(self, row: sqlite3.Row) -> Author:
        """convert db row to Author object."""
        return Author(
            id=row['id'],
            orcid=row['orcid'],
            openalex_id=row['openalex_id'],
            s2_id=row['s2_id'],
            name=row['name'] or "",
            affiliations=json.loads(row['affiliations_json']) if row['affiliations_json'] else [],
            status=AuthorStatus(row['status']),
            discovered_from=row['discovered_from'],
            topic_fit=row['topic_fit'] or 0.0,
            centrality=row['centrality'] or 0.0,
            priority=row['priority'] or 0.0
        )

    def clear(self):
        """clear all data."""
        with self._conn() as conn:
            conn.execute("DELETE FROM paper_candidates")
            conn.execute("DELETE FROM author_candidates")
            conn.execute("DELETE FROM edges")

    def get_stats(self) -> Dict[str, Any]:
        """get database statistics."""
        with self._conn() as conn:
            return {
                "papers_total": conn.execute("SELECT COUNT(*) FROM paper_candidates").fetchone()[0],
                "papers_candidate": conn.execute("SELECT COUNT(*) FROM paper_candidates WHERE status = 'candidate'").fetchone()[0],
                "papers_materialized": conn.execute("SELECT COUNT(*) FROM paper_candidates WHERE status = 'materialized'").fetchone()[0],
                "papers_expanded": conn.execute("SELECT COUNT(*) FROM paper_candidates WHERE status = 'expanded'").fetchone()[0],
                "authors_total": conn.execute("SELECT COUNT(*) FROM author_candidates").fetchone()[0],
                "edges_total": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            }
