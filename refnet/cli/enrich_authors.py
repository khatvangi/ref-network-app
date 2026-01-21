#!/usr/bin/env python3
"""
enrich papers with author data from OpenAlex.
fetches author info for top topical papers and stores in authors_json.

usage:
    python enrich_authors.py --db candidates.db --limit 500 --email you@example.com
"""

import argparse
import json
import sqlite3
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

# add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from refnet.providers.openalex import OpenAlexProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enrich_authors")


# concepts that indicate methodology papers
METHODOLOGY_CONCEPTS = {
    'deep learning', 'machine learning', 'neural network', 'protein structure prediction',
    'sequence alignment', 'database', 'algorithm', 'software', 'web server',
    'bioinformatics', 'computational biology', 'statistics', 'data analysis',
    'computer science', 'artificial intelligence', 'language model', 'source code',
    'cluster analysis', 'inference'
}

GENERIC_CONCEPTS = {
    'biology', 'chemistry', 'biochemistry', 'genetics', 'medicine',
    'cell biology', 'molecular biology', 'sequence (biology)', 'gene', 'protein'
}


def get_seed_concepts(conn: sqlite3.Connection) -> Dict[str, float]:
    """get weighted concepts from seed papers."""
    c = conn.cursor()
    c.execute("SELECT concepts_json FROM paper_candidates WHERE status = 'seed'")

    concept_weights = defaultdict(float)
    for row in c.fetchall():
        if row[0]:
            try:
                concepts = json.loads(row[0])
                for concept in concepts[:7]:
                    name = concept.get('name', '').lower()
                    score = concept.get('score', 0.5)
                    if score >= 0.5 and name not in METHODOLOGY_CONCEPTS:
                        if name in GENERIC_CONCEPTS:
                            concept_weights[name] = max(concept_weights[name], score * 0.1)
                        else:
                            concept_weights[name] = max(concept_weights[name], score)
            except:
                pass

    return {k: v for k, v in concept_weights.items() if v >= 0.3}


def compute_topical_score(concepts_json: str, seed_concepts: Dict[str, float]) -> float:
    """compute topical relevance score."""
    if not concepts_json or not seed_concepts:
        return 0.0

    try:
        concepts = json.loads(concepts_json)
        score = 0.0
        for c in concepts[:7]:
            name = c.get('name', '').lower()
            paper_score = c.get('score', 0.5)
            if name in seed_concepts:
                score += paper_score * seed_concepts[name]
        return score
    except:
        return 0.0


def get_top_papers_needing_authors(
    conn: sqlite3.Connection,
    seed_concepts: Dict[str, float],
    limit: int = 500
) -> List[Dict]:
    """get top topical papers that don't have author data yet."""
    c = conn.cursor()

    # get papers without authors_json
    c.execute("""
        SELECT id, doi, openalex_id, title, concepts_json
        FROM paper_candidates
        WHERE (authors_json IS NULL OR authors_json = '')
        AND (doi IS NOT NULL OR openalex_id IS NOT NULL)
        AND concepts_json IS NOT NULL
    """)

    papers = []
    for row in c.fetchall():
        paper_id, doi, oa_id, title, concepts_json = row
        score = compute_topical_score(concepts_json, seed_concepts)
        if score > 0.1:  # minimum threshold
            papers.append({
                'id': paper_id,
                'doi': doi,
                'openalex_id': oa_id,
                'title': title,
                'topical_score': score
            })

    # sort by score and take top
    papers.sort(key=lambda x: x['topical_score'], reverse=True)
    return papers[:limit]


def fetch_author_data(provider: OpenAlexProvider, paper: Dict) -> Optional[List[Dict]]:
    """fetch author data from OpenAlex."""
    # try openalex_id first
    lookup_id = paper.get('openalex_id') or paper.get('doi')
    if not lookup_id:
        return None

    try:
        paper_data = provider.get_paper(lookup_id)
        if not paper_data:
            return None

        # extract author info
        authors = []
        if hasattr(paper_data, 'author_ids') and paper_data.author_ids:
            # we have author IDs, fetch full author info
            for i, author_id in enumerate(paper_data.author_ids[:10]):  # limit to 10 authors
                author_name = paper_data.authors[i] if i < len(paper_data.authors) else ""
                authors.append({
                    'id': author_id,
                    'name': author_name,
                    'position': i + 1
                })
        elif hasattr(paper_data, 'authors') and paper_data.authors:
            # only have names
            for i, name in enumerate(paper_data.authors[:10]):
                authors.append({
                    'name': name,
                    'position': i + 1
                })

        return authors if authors else None

    except Exception as e:
        logger.warning(f"failed to fetch {lookup_id}: {e}")
        return None


def enrich_papers(
    db_path: str,
    email: str,
    limit: int = 500,
    batch_size: int = 50
):
    """main enrichment function."""
    conn = sqlite3.connect(db_path)

    # get seed concepts
    logger.info("loading seed concepts...")
    seed_concepts = get_seed_concepts(conn)
    logger.info(f"found {len(seed_concepts)} seed concepts: {list(seed_concepts.keys())[:5]}...")

    # get papers needing enrichment
    logger.info(f"finding top {limit} papers needing author data...")
    papers = get_top_papers_needing_authors(conn, seed_concepts, limit)
    logger.info(f"found {len(papers)} papers to enrich")

    if not papers:
        logger.info("no papers need enrichment")
        conn.close()
        return

    # setup provider
    provider = OpenAlexProvider(email=email)

    # enrich in batches
    enriched = 0
    failed = 0

    for i, paper in enumerate(papers):
        if i > 0 and i % batch_size == 0:
            conn.commit()
            logger.info(f"progress: {i}/{len(papers)} ({enriched} enriched, {failed} failed)")

        authors = fetch_author_data(provider, paper)
        if authors:
            # update database
            c = conn.cursor()
            c.execute(
                "UPDATE paper_candidates SET authors_json = ? WHERE id = ?",
                (json.dumps(authors), paper['id'])
            )
            enriched += 1
        else:
            failed += 1

        # rate limiting (OpenAlex is generous but be polite)
        time.sleep(0.1)

    conn.commit()
    conn.close()

    logger.info(f"\n=== ENRICHMENT COMPLETE ===")
    logger.info(f"Total processed: {len(papers)}")
    logger.info(f"Enriched: {enriched}")
    logger.info(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Enrich papers with author data from OpenAlex")
    parser.add_argument("--db", "-d", required=True, help="database path")
    parser.add_argument("--email", "-e", default="kiran@mcneese.edu", help="email for OpenAlex API")
    parser.add_argument("--limit", "-n", type=int, default=500, help="max papers to enrich")
    parser.add_argument("--batch", "-b", type=int, default=50, help="commit batch size")

    args = parser.parse_args()

    enrich_papers(args.db, args.email, args.limit, args.batch)


if __name__ == "__main__":
    main()
