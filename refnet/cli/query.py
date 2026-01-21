#!/usr/bin/env python3
"""
refnet-query - fast CLI for exploring citation databases.
no graphics, just useful text output.

usage:
    refnet-query papers "aminoacyl tRNA"      # search by title/concepts
    refnet-query clusters                      # list topic clusters
    refnet-query cluster "Genetic code"        # papers in a cluster
    refnet-query connections <paper_id>        # what cites/is cited by
    refnet-query reading-list                  # curated reading list
    refnet-query stats                         # database statistics
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# default database location
DEFAULT_DB = Path.home() / ".refnet" / "candidates.db"


class QueryEngine:
    """fast query engine for citation databases."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"database not found: {db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def search_papers(
        self,
        query: str,
        limit: int = 20,
        min_citations: int = 0,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[Dict]:
        """search papers by title or concepts."""
        c = self.conn.cursor()

        # build query - search in title and concepts
        sql = """
            SELECT id, title, year, citation_count, venue, concepts_json, status
            FROM paper_candidates
            WHERE (
                title LIKE ? OR
                concepts_json LIKE ?
            )
        """
        params = [f"%{query}%", f"%{query}%"]

        if min_citations > 0:
            sql += " AND citation_count >= ?"
            params.append(min_citations)

        if year_from:
            sql += " AND year >= ?"
            params.append(year_from)

        if year_to:
            sql += " AND year <= ?"
            params.append(year_to)

        sql += " ORDER BY citation_count DESC LIMIT ?"
        params.append(limit)

        c.execute(sql, params)

        results = []
        for row in c.fetchall():
            concepts = []
            if row['concepts_json']:
                try:
                    concepts = [c['name'] for c in json.loads(row['concepts_json'])[:3]]
                except:
                    pass

            results.append({
                'id': row['id'],
                'title': row['title'],
                'year': row['year'],
                'citations': row['citation_count'],
                'venue': row['venue'],
                'concepts': concepts,
                'is_seed': row['status'] == 'seed'
            })

        return results

    def get_clusters(self, min_size: int = 10) -> List[Dict]:
        """get topic clusters by primary concept."""
        c = self.conn.cursor()

        c.execute("""
            SELECT concepts_json FROM paper_candidates
            WHERE concepts_json IS NOT NULL
        """)

        # count papers by primary concept
        concept_counts = defaultdict(int)
        concept_papers = defaultdict(list)

        for row in c.fetchall():
            try:
                concepts = json.loads(row['concepts_json'])
                if concepts:
                    primary = concepts[0]['name']
                    concept_counts[primary] += 1
            except:
                pass

        # filter by min size and sort
        clusters = [
            {'name': name, 'size': count}
            for name, count in concept_counts.items()
            if count >= min_size
        ]
        clusters.sort(key=lambda x: x['size'], reverse=True)

        return clusters

    def get_cluster_papers(
        self,
        cluster_name: str,
        limit: int = 20
    ) -> List[Dict]:
        """get papers in a specific cluster."""
        c = self.conn.cursor()

        # search for papers with this primary concept
        c.execute("""
            SELECT id, title, year, citation_count, venue, concepts_json, status
            FROM paper_candidates
            WHERE concepts_json LIKE ?
            ORDER BY citation_count DESC
            LIMIT ?
        """, [f'%"{cluster_name}"%', limit * 3])  # fetch more, filter below

        results = []
        for row in c.fetchall():
            try:
                concepts = json.loads(row['concepts_json'])
                if concepts and concepts[0]['name'].lower() == cluster_name.lower():
                    results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'year': row['year'],
                        'citations': row['citation_count'],
                        'venue': row['venue'],
                        'is_seed': row['status'] == 'seed'
                    })
                    if len(results) >= limit:
                        break
            except:
                pass

        return results

    def get_connections(self, paper_id: str) -> Dict:
        """get papers that cite/are cited by this paper."""
        c = self.conn.cursor()

        # get the paper itself
        c.execute("""
            SELECT id, title, year, citation_count FROM paper_candidates WHERE id = ?
        """, [paper_id])
        paper = c.fetchone()
        if not paper:
            return {'error': 'paper not found'}

        # get citations (papers this paper cites)
        c.execute("""
            SELECT p.id, p.title, p.year, p.citation_count, e.edge_type
            FROM edges e
            JOIN paper_candidates p ON p.id = e.target_id
            WHERE e.source_id = ?
            ORDER BY p.citation_count DESC
            LIMIT 20
        """, [paper_id])

        cites = [dict(row) for row in c.fetchall()]

        # get cited_by (papers that cite this paper)
        c.execute("""
            SELECT p.id, p.title, p.year, p.citation_count, e.edge_type
            FROM edges e
            JOIN paper_candidates p ON p.id = e.source_id
            WHERE e.target_id = ?
            ORDER BY p.citation_count DESC
            LIMIT 20
        """, [paper_id])

        cited_by = [dict(row) for row in c.fetchall()]

        return {
            'paper': dict(paper),
            'cites': cites,
            'cited_by': cited_by
        }

    def get_reading_list(self, limit: int = 50) -> List[Dict]:
        """generate curated reading list based on TOPICAL relevance (not citations).

        uses same logic as ReportGenerator: specific concepts weighted high,
        generic concepts (biology, chemistry) weighted low.
        """
        # generic concepts that match everything - low weight
        GENERIC = {
            'biology', 'chemistry', 'biochemistry', 'genetics', 'medicine',
            'cell biology', 'molecular biology', 'sequence (biology)', 'gene', 'protein',
            'computer science', 'artificial intelligence', 'machine learning',
            'deep learning', 'neural network', 'algorithm', 'database'
        }

        c = self.conn.cursor()

        # get seed papers first
        c.execute("""
            SELECT id, title, year, citation_count, venue, concepts_json
            FROM paper_candidates
            WHERE status = 'seed'
        """)
        seeds = [dict(row) for row in c.fetchall()]

        # get SPECIFIC seed concepts (excluding generic)
        seed_concepts = {}  # concept -> weight
        for seed in seeds:
            try:
                concepts = json.loads(seed['concepts_json'])
                for c_data in concepts[:7]:
                    name = c_data['name'].lower()
                    score = c_data.get('score', 0.5)
                    if score >= 0.5:  # high confidence only
                        if name in GENERIC:
                            seed_concepts[name] = max(seed_concepts.get(name, 0), score * 0.1)
                        else:
                            seed_concepts[name] = max(seed_concepts.get(name, 0), score)
            except:
                pass

        # get papers with concepts
        c.execute("""
            SELECT id, title, year, citation_count, venue, concepts_json, depth
            FROM paper_candidates
            WHERE status != 'seed'
            AND concepts_json IS NOT NULL
        """)

        # score papers by TOPICAL relevance
        import math
        candidates = []
        for row in c.fetchall():
            try:
                concepts = json.loads(row['concepts_json'])
            except:
                continue

            # compute weighted overlap
            topical_score = 0.0
            matching = []
            for c_data in concepts[:7]:
                name = c_data['name'].lower()
                paper_score = c_data.get('score', 0.5)
                if name in seed_concepts:
                    weight = seed_concepts[name]
                    contribution = paper_score * weight
                    topical_score += contribution
                    if name not in GENERIC:
                        matching.append(name)

            if topical_score > 0.1:  # minimum threshold
                depth = row['depth'] or 3
                # favor depth 1-2 (close to seeds)
                depth_factor = 1.0 / (1 + depth * 0.3)

                # small citation boost (log scale), but topical dominates
                cite_factor = math.log1p(row['citation_count'] or 0) * 0.1

                final_score = topical_score * depth_factor + cite_factor

                candidates.append({
                    'id': row['id'],
                    'title': row['title'],
                    'year': row['year'],
                    'citations': row['citation_count'],
                    'venue': row['venue'],
                    'topical': topical_score,
                    'score': final_score,
                    'reason': f"topical={topical_score:.2f}, depth={depth}, {', '.join(matching[:2])}"
                })

        # sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # combine seeds + top candidates
        reading_list = []
        for seed in seeds:
            reading_list.append({
                'id': seed['id'],
                'title': seed['title'],
                'year': seed['year'],
                'citations': seed['citation_count'],
                'venue': seed['venue'],
                'reason': '★ SEED PAPER'
            })

        reading_list.extend(candidates[:limit - len(seeds)])
        return reading_list

    def get_stats(self) -> Dict:
        """get database statistics."""
        c = self.conn.cursor()

        stats = {}

        # paper counts
        c.execute("SELECT COUNT(*) FROM paper_candidates")
        stats['total_papers'] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM paper_candidates WHERE status = 'seed'")
        stats['seeds'] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM paper_candidates WHERE status = 'materialized'")
        stats['materialized'] = c.fetchone()[0]

        # edge counts
        c.execute("SELECT COUNT(*) FROM edges")
        stats['total_edges'] = c.fetchone()[0]

        c.execute("SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type")
        stats['edges_by_type'] = {row[0]: row[1] for row in c.fetchall()}

        # year range
        c.execute("SELECT MIN(year), MAX(year) FROM paper_candidates WHERE year IS NOT NULL")
        row = c.fetchone()
        stats['year_range'] = f"{row[0]}-{row[1]}" if row[0] else "unknown"

        # discovery channels
        c.execute("""
            SELECT discovered_channel, COUNT(*)
            FROM paper_candidates
            GROUP BY discovered_channel
        """)
        stats['by_channel'] = {row[0] or 'seed': row[1] for row in c.fetchall()}

        return stats


def print_papers(papers: List[Dict], show_id: bool = False):
    """print paper list in readable format."""
    for i, p in enumerate(papers, 1):
        seed_mark = "★" if p.get('is_seed') else " "
        title = (p.get('title') or '?')[:60]
        year = p.get('year') or '?'
        cites = p.get('citations') or 0

        print(f"{seed_mark} {i:2d}. [{year}] {title}...")
        print(f"      {cites:,} citations", end="")
        if p.get('venue'):
            print(f" | {p['venue'][:30]}", end="")
        if p.get('reason'):
            print(f" | {p['reason']}", end="")
        print()
        if show_id:
            print(f"      ID: {p['id']}")
        if p.get('concepts'):
            print(f"      Concepts: {', '.join(p['concepts'][:3])}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Query citation database from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  refnet-query papers "aminoacyl tRNA"        # search papers
  refnet-query papers "evolution" --min-cites 100
  refnet-query clusters                        # list topic clusters
  refnet-query cluster "Genetic code"          # papers in cluster
  refnet-query connections <paper_id>          # see citations
  refnet-query reading-list                    # curated list
  refnet-query stats                           # database stats
        """
    )

    parser.add_argument(
        "--db", "-d",
        default=str(DEFAULT_DB),
        help=f"database path (default: {DEFAULT_DB})"
    )

    subparsers = parser.add_subparsers(dest="command", help="command")

    # papers command
    papers_p = subparsers.add_parser("papers", help="search papers")
    papers_p.add_argument("query", help="search query")
    papers_p.add_argument("--limit", "-n", type=int, default=20)
    papers_p.add_argument("--min-cites", type=int, default=0)
    papers_p.add_argument("--year-from", type=int)
    papers_p.add_argument("--year-to", type=int)
    papers_p.add_argument("--show-id", action="store_true")

    # clusters command
    clusters_p = subparsers.add_parser("clusters", help="list topic clusters")
    clusters_p.add_argument("--min-size", type=int, default=10)

    # cluster command
    cluster_p = subparsers.add_parser("cluster", help="papers in a cluster")
    cluster_p.add_argument("name", help="cluster name")
    cluster_p.add_argument("--limit", "-n", type=int, default=20)

    # connections command
    conn_p = subparsers.add_parser("connections", help="paper connections")
    conn_p.add_argument("paper_id", help="paper ID")

    # reading-list command
    read_p = subparsers.add_parser("reading-list", help="curated reading list")
    read_p.add_argument("--limit", "-n", type=int, default=50)

    # stats command
    subparsers.add_parser("stats", help="database statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        engine = QueryEngine(args.db)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Specify database with --db or set default at {DEFAULT_DB}")
        return 1

    try:
        if args.command == "papers":
            papers = engine.search_papers(
                args.query,
                limit=args.limit,
                min_citations=args.min_cites,
                year_from=args.year_from,
                year_to=args.year_to
            )
            print(f"\n=== Papers matching '{args.query}' ({len(papers)} results) ===\n")
            print_papers(papers, show_id=args.show_id)

        elif args.command == "clusters":
            clusters = engine.get_clusters(min_size=args.min_size)
            print(f"\n=== Topic Clusters ({len(clusters)} with {args.min_size}+ papers) ===\n")
            for c in clusters[:30]:
                print(f"  {c['name']:40s} {c['size']:5d} papers")

        elif args.command == "cluster":
            papers = engine.get_cluster_papers(args.name, limit=args.limit)
            print(f"\n=== Papers in '{args.name}' cluster ({len(papers)} shown) ===\n")
            print_papers(papers)

        elif args.command == "connections":
            result = engine.get_connections(args.paper_id)
            if 'error' in result:
                print(f"Error: {result['error']}")
                return 1

            p = result['paper']
            print(f"\n=== {p['title'][:60]}... ({p['year']}) ===")
            print(f"    {p['citation_count']:,} citations\n")

            print(f"--- Cites ({len(result['cites'])} papers) ---")
            for r in result['cites'][:10]:
                print(f"  → {(r['title'] or '?')[:50]}... ({r['year']})")

            print(f"\n--- Cited by ({len(result['cited_by'])} papers) ---")
            for r in result['cited_by'][:10]:
                print(f"  ← {(r['title'] or '?')[:50]}... ({r['year']})")

        elif args.command == "reading-list":
            papers = engine.get_reading_list(limit=args.limit)
            print(f"\n=== Curated Reading List ({len(papers)} papers) ===\n")
            print_papers(papers)

        elif args.command == "stats":
            stats = engine.get_stats()
            print("\n=== Database Statistics ===\n")
            print(f"Total papers:  {stats['total_papers']:,}")
            print(f"Seeds:         {stats['seeds']}")
            print(f"Materialized:  {stats['materialized']}")
            print(f"Total edges:   {stats['total_edges']:,}")
            print(f"Year range:    {stats['year_range']}")

            print("\nEdges by type:")
            for etype, count in sorted(stats['edges_by_type'].items(), key=lambda x: -x[1]):
                print(f"  {etype:20s} {count:,}")

            print("\nPapers by discovery channel:")
            for channel, count in sorted(stats['by_channel'].items(), key=lambda x: -x[1]):
                print(f"  {channel or 'seed':20s} {count:,}")

    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
