"""
persistent garden - organic network growth across sessions.
one global database that accumulates knowledge over time.
"""

import os
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..core.models import Paper, Author, EdgeType, PaperStatus, AuthorStatus


logger = logging.getLogger("refnet.garden")


# default location
DEFAULT_GARDEN_PATH = Path.home() / ".refnet" / "garden.db"


class AuthorRole(Enum):
    """
    author roles based on trajectory + impact.

    LEADER: high centrality, many cite them, stable focus
    PLAYER: active contributor, moderate drift, connected
    FOLLOWER: cites leaders, low citations, copies trends
    DISRUPTOR: high drift, novelty jumps, bridges clusters (rule-breaker!)
    UNKNOWN: not enough data yet
    """
    LEADER = "leader"
    PLAYER = "player"
    FOLLOWER = "follower"
    DISRUPTOR = "disruptor"
    UNKNOWN = "unknown"


@dataclass
class AuthorProfile:
    """extended author profile with role classification."""
    id: str
    name: str
    orcid: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)

    # metrics
    paper_count: int = 0
    citation_count: int = 0
    centrality: float = 0.0

    # trajectory
    drift_magnitude_avg: float = 0.0
    novelty_jumps: int = 0
    clusters_bridged: int = 0
    trajectory_years: Tuple[int, int] = (0, 0)

    # role
    role: AuthorRole = AuthorRole.UNKNOWN
    role_confidence: float = 0.0

    # garden state
    is_seed: bool = False
    planted_at: Optional[str] = None
    last_expanded: Optional[str] = None


@dataclass
class GardenStats:
    """current state of the garden."""
    total_papers: int = 0
    total_authors: int = 0
    total_edges: int = 0
    seed_papers: int = 0
    seed_authors: int = 0
    islands: int = 0  # disconnected components

    # role distribution
    leaders: int = 0
    players: int = 0
    followers: int = 0
    disruptors: int = 0
    unknown: int = 0

    # growth history
    last_plant: Optional[str] = None
    last_grow: Optional[str] = None
    total_sessions: int = 0


class Garden:
    """
    persistent network garden.
    grows organically with each session.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_GARDEN_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """initialize garden database."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # papers table
        c.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                doi TEXT,
                openalex_id TEXT,
                s2_id TEXT,
                title TEXT,
                year INTEGER,
                venue TEXT,
                authors_json TEXT,
                author_ids_json TEXT,
                citation_count INTEGER,
                reference_count INTEGER,
                is_review INTEGER,
                is_seed INTEGER DEFAULT 0,
                concepts_json TEXT,
                relevance_score REAL,
                bridge_score REAL,
                status TEXT,
                planted_at TEXT,
                discovered_from TEXT,
                discovered_channel TEXT,
                UNIQUE(doi),
                UNIQUE(openalex_id),
                UNIQUE(s2_id)
            )
        """)

        # authors table with role
        c.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id TEXT PRIMARY KEY,
                name TEXT,
                orcid TEXT,
                openalex_id TEXT,
                s2_id TEXT,
                affiliations_json TEXT,
                paper_count INTEGER,
                citation_count INTEGER,
                centrality REAL DEFAULT 0,
                drift_magnitude_avg REAL DEFAULT 0,
                novelty_jumps INTEGER DEFAULT 0,
                clusters_bridged INTEGER DEFAULT 0,
                trajectory_start_year INTEGER,
                trajectory_end_year INTEGER,
                role TEXT DEFAULT 'unknown',
                role_confidence REAL DEFAULT 0,
                is_seed INTEGER DEFAULT 0,
                planted_at TEXT,
                last_expanded TEXT,
                UNIQUE(orcid),
                UNIQUE(openalex_id),
                UNIQUE(s2_id)
            )
        """)

        # edges table
        c.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                edge_type TEXT,
                weight REAL DEFAULT 1.0,
                created_at TEXT,
                PRIMARY KEY (source, target, edge_type)
            )
        """)

        # trajectory events
        c.execute("""
            CREATE TABLE IF NOT EXISTS trajectory_events (
                author_id TEXT,
                year INTEGER,
                drift_magnitude REAL,
                is_novelty_jump INTEGER,
                from_focus TEXT,
                to_focus TEXT,
                PRIMARY KEY (author_id, year)
            )
        """)

        # garden metadata
        c.execute("""
            CREATE TABLE IF NOT EXISTS garden_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_papers_openalex ON papers(openalex_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_authors_orcid ON authors(orcid)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_authors_openalex ON authors(openalex_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_authors_role ON authors(role)")

        conn.commit()
        conn.close()

        logger.info(f"[garden] initialized at {self.db_path}")

    def plant_paper(self, paper: Paper, is_seed: bool = True) -> bool:
        """
        plant a paper in the garden.
        returns True if new, False if already exists (but merges metadata).
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        now = datetime.now().isoformat()

        # check if exists
        existing = self._find_paper(c, paper)

        if existing:
            # merge metadata
            self._merge_paper(c, existing, paper)
            conn.commit()
            conn.close()
            logger.debug(f"[garden] merged paper: {paper.title[:40]}")
            return False

        # insert new
        c.execute("""
            INSERT INTO papers (
                id, doi, openalex_id, s2_id, title, year, venue,
                authors_json, author_ids_json, citation_count, reference_count,
                is_review, is_seed, concepts_json, relevance_score, bridge_score,
                status, planted_at, discovered_from, discovered_channel
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper.id,
            paper.doi,
            paper.openalex_id,
            paper.s2_id,
            paper.title,
            paper.year,
            paper.venue,
            json.dumps(paper.authors) if paper.authors else None,
            json.dumps(paper.author_ids) if paper.author_ids else None,
            paper.citation_count,
            paper.reference_count,
            1 if paper.is_review else 0,
            1 if is_seed else 0,
            json.dumps(paper.concepts) if paper.concepts else None,
            paper.relevance_score,
            paper.bridge_score,
            paper.status.value if paper.status else "candidate",
            now if is_seed else None,
            paper.discovered_from,
            paper.discovered_channel
        ))

        conn.commit()
        conn.close()

        logger.info(f"[garden] planted paper: {paper.title[:40]}")
        return True

    def plant_author(self, author: Author, is_seed: bool = True) -> bool:
        """
        plant an author in the garden.
        returns True if new, False if already exists (but merges metadata).
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        now = datetime.now().isoformat()

        # check if exists
        existing = self._find_author(c, author)

        if existing:
            # merge metadata
            self._merge_author(c, existing, author, is_seed)
            conn.commit()
            conn.close()
            logger.debug(f"[garden] merged author: {author.name}")
            return False

        # insert new
        c.execute("""
            INSERT INTO authors (
                id, name, orcid, openalex_id, s2_id, affiliations_json,
                paper_count, citation_count, is_seed, planted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            author.id,
            author.name,
            author.orcid,
            author.openalex_id,
            author.s2_id,
            json.dumps(author.affiliations) if author.affiliations else None,
            author.paper_count,
            author.citation_count,
            1 if is_seed else 0,
            now if is_seed else None
        ))

        conn.commit()
        conn.close()

        logger.info(f"[garden] planted author: {author.name}")
        return True

    def add_edge(self, source: str, target: str, edge_type: EdgeType) -> bool:
        """add edge to garden."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        try:
            c.execute("""
                INSERT OR IGNORE INTO edges (source, target, edge_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (source, target, edge_type.value, datetime.now().isoformat()))

            added = c.rowcount > 0
            conn.commit()
            return added
        except Exception as e:
            logger.warning(f"[garden] edge error: {e}")
            return False
        finally:
            conn.close()

    def add_trajectory_event(
        self,
        author_id: str,
        year: int,
        drift_magnitude: float,
        is_novelty_jump: bool,
        from_focus: str,
        to_focus: str
    ):
        """record trajectory event for author."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            INSERT OR REPLACE INTO trajectory_events
            (author_id, year, drift_magnitude, is_novelty_jump, from_focus, to_focus)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (author_id, year, drift_magnitude, 1 if is_novelty_jump else 0, from_focus, to_focus))

        conn.commit()
        conn.close()

    def compute_author_roles(self):
        """
        compute roles for all authors based on trajectory + impact.

        DISRUPTOR: high drift (>=0.4 avg) OR novelty jumps OR bridges clusters
        LEADER: high centrality (top 10%) AND high citations AND low drift
        PLAYER: moderate metrics, connected
        FOLLOWER: low citations, low drift, follows leaders
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # get all authors with their metrics
        c.execute("""
            SELECT id, name, citation_count, centrality,
                   drift_magnitude_avg, novelty_jumps, clusters_bridged
            FROM authors
        """)

        authors = c.fetchall()
        if not authors:
            conn.close()
            return

        # compute thresholds
        citations = [a[2] or 0 for a in authors]
        centralities = [a[3] or 0 for a in authors]

        citation_p90 = sorted(citations)[int(len(citations) * 0.9)] if citations else 0
        centrality_p90 = sorted(centralities)[int(len(centralities) * 0.9)] if centralities else 0

        # classify each author
        for author in authors:
            aid, name, cites, cent, drift, jumps, bridges = author
            cites = cites or 0
            cent = cent or 0
            drift = drift or 0
            jumps = jumps or 0
            bridges = bridges or 0

            # determine role
            if drift >= 0.4 or jumps >= 2 or bridges >= 2:
                # high trajectory change = disruptor
                role = AuthorRole.DISRUPTOR
                confidence = min(1.0, drift + jumps * 0.2 + bridges * 0.1)
            elif cent >= centrality_p90 and cites >= citation_p90 and drift < 0.3:
                # high impact, stable = leader
                role = AuthorRole.LEADER
                confidence = min(1.0, (cent / max(centrality_p90, 0.01)) * 0.5 + 0.5)
            elif cites < citation_p90 * 0.2 and drift < 0.2:
                # low impact, stable = follower
                role = AuthorRole.FOLLOWER
                confidence = 0.6
            elif cites > 0 or cent > 0:
                # active but not extreme = player
                role = AuthorRole.PLAYER
                confidence = 0.5
            else:
                role = AuthorRole.UNKNOWN
                confidence = 0.0

            # update
            c.execute("""
                UPDATE authors SET role = ?, role_confidence = ?
                WHERE id = ?
            """, (role.value, confidence, aid))

        conn.commit()
        conn.close()

        logger.info("[garden] computed author roles")

    def get_stats(self) -> GardenStats:
        """get current garden statistics."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        stats = GardenStats()

        # counts
        c.execute("SELECT COUNT(*) FROM papers")
        stats.total_papers = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM authors")
        stats.total_authors = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM edges")
        stats.total_edges = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM papers WHERE is_seed = 1")
        stats.seed_papers = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM authors WHERE is_seed = 1")
        stats.seed_authors = c.fetchone()[0]

        # role distribution
        c.execute("SELECT role, COUNT(*) FROM authors GROUP BY role")
        for role, count in c.fetchall():
            if role == "leader":
                stats.leaders = count
            elif role == "player":
                stats.players = count
            elif role == "follower":
                stats.followers = count
            elif role == "disruptor":
                stats.disruptors = count
            else:
                stats.unknown = count

        # metadata
        c.execute("SELECT value FROM garden_meta WHERE key = 'last_plant'")
        row = c.fetchone()
        stats.last_plant = row[0] if row else None

        c.execute("SELECT value FROM garden_meta WHERE key = 'last_grow'")
        row = c.fetchone()
        stats.last_grow = row[0] if row else None

        c.execute("SELECT value FROM garden_meta WHERE key = 'total_sessions'")
        row = c.fetchone()
        stats.total_sessions = int(row[0]) if row else 0

        conn.close()
        return stats

    def get_disruptors(self, limit: int = 20) -> List[AuthorProfile]:
        """get top disruptors (rule-breakers)."""
        return self._get_authors_by_role(AuthorRole.DISRUPTOR, limit)

    def get_leaders(self, limit: int = 20) -> List[AuthorProfile]:
        """get top leaders."""
        return self._get_authors_by_role(AuthorRole.LEADER, limit)

    def _get_authors_by_role(self, role: AuthorRole, limit: int) -> List[AuthorProfile]:
        """get authors by role."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT id, name, orcid, openalex_id, s2_id, affiliations_json,
                   paper_count, citation_count, centrality,
                   drift_magnitude_avg, novelty_jumps, clusters_bridged,
                   trajectory_start_year, trajectory_end_year,
                   role, role_confidence, is_seed, planted_at, last_expanded
            FROM authors
            WHERE role = ?
            ORDER BY role_confidence DESC, citation_count DESC
            LIMIT ?
        """, (role.value, limit))

        profiles = []
        for row in c.fetchall():
            profiles.append(AuthorProfile(
                id=row[0],
                name=row[1],
                orcid=row[2],
                openalex_id=row[3],
                s2_id=row[4],
                affiliations=json.loads(row[5]) if row[5] else [],
                paper_count=row[6] or 0,
                citation_count=row[7] or 0,
                centrality=row[8] or 0,
                drift_magnitude_avg=row[9] or 0,
                novelty_jumps=row[10] or 0,
                clusters_bridged=row[11] or 0,
                trajectory_years=(row[12] or 0, row[13] or 0),
                role=AuthorRole(row[14]) if row[14] else AuthorRole.UNKNOWN,
                role_confidence=row[15] or 0,
                is_seed=bool(row[16]),
                planted_at=row[17],
                last_expanded=row[18]
            ))

        conn.close()
        return profiles

    def get_frontier_papers(self, limit: int = 50) -> List[Paper]:
        """get papers at the frontier (not yet expanded)."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT id, doi, openalex_id, s2_id, title, year, venue,
                   authors_json, author_ids_json, citation_count, relevance_score
            FROM papers
            WHERE status != 'expanded'
            ORDER BY relevance_score DESC, citation_count DESC
            LIMIT ?
        """, (limit,))

        papers = []
        for row in c.fetchall():
            p = Paper(
                id=row[0],
                doi=row[1],
                openalex_id=row[2],
                s2_id=row[3],
                title=row[4],
                year=row[5],
                venue=row[6],
                authors=json.loads(row[7]) if row[7] else [],
                author_ids=json.loads(row[8]) if row[8] else [],
                citation_count=row[9],
                relevance_score=row[10] or 0
            )
            papers.append(p)

        conn.close()
        return papers

    def get_seed_authors(self) -> List[AuthorProfile]:
        """get all seed authors."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT id, name, orcid, openalex_id, s2_id, affiliations_json,
                   paper_count, citation_count, centrality,
                   drift_magnitude_avg, novelty_jumps, clusters_bridged,
                   role, role_confidence, is_seed, planted_at, last_expanded
            FROM authors
            WHERE is_seed = 1
        """)

        profiles = []
        for row in c.fetchall():
            profiles.append(AuthorProfile(
                id=row[0],
                name=row[1],
                orcid=row[2],
                openalex_id=row[3],
                s2_id=row[4],
                affiliations=json.loads(row[5]) if row[5] else [],
                paper_count=row[6] or 0,
                citation_count=row[7] or 0,
                centrality=row[8] or 0,
                drift_magnitude_avg=row[9] or 0,
                novelty_jumps=row[10] or 0,
                clusters_bridged=row[11] or 0,
                role=AuthorRole(row[12]) if row[12] else AuthorRole.UNKNOWN,
                role_confidence=row[13] or 0,
                is_seed=bool(row[14]),
                planted_at=row[15],
                last_expanded=row[16]
            ))

        conn.close()
        return profiles

    def update_meta(self, key: str, value: str):
        """update garden metadata."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO garden_meta (key, value) VALUES (?, ?)
        """, (key, value))
        conn.commit()
        conn.close()

    def increment_sessions(self):
        """increment session counter."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("SELECT value FROM garden_meta WHERE key = 'total_sessions'")
        row = c.fetchone()
        count = int(row[0]) if row else 0

        c.execute("""
            INSERT OR REPLACE INTO garden_meta (key, value) VALUES (?, ?)
        """, ("total_sessions", str(count + 1)))

        conn.commit()
        conn.close()

    def _find_paper(self, c, paper: Paper) -> Optional[str]:
        """find existing paper by any ID."""
        if paper.doi:
            c.execute("SELECT id FROM papers WHERE doi = ?", (paper.doi,))
            row = c.fetchone()
            if row:
                return row[0]

        if paper.openalex_id:
            c.execute("SELECT id FROM papers WHERE openalex_id = ?", (paper.openalex_id,))
            row = c.fetchone()
            if row:
                return row[0]

        if paper.s2_id:
            c.execute("SELECT id FROM papers WHERE s2_id = ?", (paper.s2_id,))
            row = c.fetchone()
            if row:
                return row[0]

        return None

    def _find_author(self, c, author: Author) -> Optional[str]:
        """find existing author by any ID."""
        if author.orcid:
            c.execute("SELECT id FROM authors WHERE orcid = ?", (author.orcid,))
            row = c.fetchone()
            if row:
                return row[0]

        if author.openalex_id:
            c.execute("SELECT id FROM authors WHERE openalex_id = ?", (author.openalex_id,))
            row = c.fetchone()
            if row:
                return row[0]

        if author.s2_id:
            c.execute("SELECT id FROM authors WHERE s2_id = ?", (author.s2_id,))
            row = c.fetchone()
            if row:
                return row[0]

        return None

    def _merge_paper(self, c, existing_id: str, paper: Paper):
        """merge new paper metadata into existing."""
        updates = []
        values = []

        if paper.doi:
            updates.append("doi = COALESCE(doi, ?)")
            values.append(paper.doi)
        if paper.openalex_id:
            updates.append("openalex_id = COALESCE(openalex_id, ?)")
            values.append(paper.openalex_id)
        if paper.s2_id:
            updates.append("s2_id = COALESCE(s2_id, ?)")
            values.append(paper.s2_id)
        if paper.citation_count:
            updates.append("citation_count = MAX(COALESCE(citation_count, 0), ?)")
            values.append(paper.citation_count)
        if paper.abstract:
            updates.append("abstract = COALESCE(abstract, ?)")
            values.append(paper.abstract)

        if updates:
            values.append(existing_id)
            c.execute(f"UPDATE papers SET {', '.join(updates)} WHERE id = ?", values)

    def _merge_author(self, c, existing_id: str, author: Author, is_seed: bool):
        """merge new author metadata into existing."""
        updates = []
        values = []

        if author.orcid:
            updates.append("orcid = COALESCE(orcid, ?)")
            values.append(author.orcid)
        if author.openalex_id:
            updates.append("openalex_id = COALESCE(openalex_id, ?)")
            values.append(author.openalex_id)
        if author.s2_id:
            updates.append("s2_id = COALESCE(s2_id, ?)")
            values.append(author.s2_id)
        if author.citation_count:
            updates.append("citation_count = MAX(COALESCE(citation_count, 0), ?)")
            values.append(author.citation_count)
        if is_seed:
            updates.append("is_seed = 1")
            updates.append("planted_at = COALESCE(planted_at, ?)")
            values.append(datetime.now().isoformat())

        if updates:
            values.append(existing_id)
            c.execute(f"UPDATE authors SET {', '.join(updates)} WHERE id = ?", values)

    def export_json(self, output_path: str) -> str:
        """export garden to JSON."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # papers
        c.execute("""
            SELECT id, doi, openalex_id, title, year, venue, authors_json,
                   citation_count, is_review, is_seed, relevance_score, bridge_score
            FROM papers
        """)
        papers = []
        for row in c.fetchall():
            papers.append({
                "id": row[0],
                "doi": row[1],
                "openalex_id": row[2],
                "title": row[3],
                "year": row[4],
                "venue": row[5],
                "authors": json.loads(row[6]) if row[6] else [],
                "citation_count": row[7],
                "is_review": bool(row[8]),
                "is_seed": bool(row[9]),
                "relevance_score": row[10],
                "bridge_score": row[11]
            })

        # authors with roles
        c.execute("""
            SELECT id, name, orcid, openalex_id, affiliations_json,
                   paper_count, citation_count, centrality,
                   drift_magnitude_avg, novelty_jumps, clusters_bridged,
                   role, role_confidence, is_seed
            FROM authors
        """)
        authors = []
        for row in c.fetchall():
            authors.append({
                "id": row[0],
                "name": row[1],
                "orcid": row[2],
                "openalex_id": row[3],
                "affiliations": json.loads(row[4]) if row[4] else [],
                "paper_count": row[5],
                "citation_count": row[6],
                "centrality": row[7],
                "drift_magnitude_avg": row[8],
                "novelty_jumps": row[9],
                "clusters_bridged": row[10],
                "role": row[11],
                "role_confidence": row[12],
                "is_seed": bool(row[13])
            })

        # edges
        c.execute("SELECT source, target, edge_type, weight FROM edges")
        edges = []
        for row in c.fetchall():
            edges.append({
                "source": row[0],
                "target": row[1],
                "type": row[2],
                "weight": row[3]
            })

        # trajectory events
        c.execute("""
            SELECT author_id, year, drift_magnitude, is_novelty_jump, from_focus, to_focus
            FROM trajectory_events
            ORDER BY author_id, year
        """)
        trajectories = []
        for row in c.fetchall():
            trajectories.append({
                "author_id": row[0],
                "year": row[1],
                "drift_magnitude": row[2],
                "is_novelty_jump": bool(row[3]),
                "from_focus": row[4],
                "to_focus": row[5]
            })

        conn.close()

        # build export
        stats = self.get_stats()
        data = {
            "meta": {
                "exported_at": datetime.now().isoformat(),
                "garden_path": str(self.db_path),
                "total_papers": stats.total_papers,
                "total_authors": stats.total_authors,
                "total_edges": stats.total_edges,
                "sessions": stats.total_sessions
            },
            "role_distribution": {
                "leaders": stats.leaders,
                "players": stats.players,
                "followers": stats.followers,
                "disruptors": stats.disruptors,
                "unknown": stats.unknown
            },
            "papers": papers,
            "authors": authors,
            "edges": edges,
            "trajectories": trajectories
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"[garden] exported to {output_path}")
        return output_path
