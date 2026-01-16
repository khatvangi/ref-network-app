"""
garden CLI commands - plant, grow, status, export.
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .persistent import Garden, GardenStats, AuthorRole, DEFAULT_GARDEN_PATH
from ..core.models import Paper, Author, PaperStatus
from ..core.resilience import setup_logging
from ..providers.openalex import OpenAlexProvider
from ..providers.composite import create_default_provider


logger = logging.getLogger("refnet.garden")


def cmd_plant(args):
    """plant seeds in the garden."""
    garden = Garden(args.garden)
    garden.increment_sessions()

    # setup provider
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if args.fallback:
        provider = create_default_provider(email=args.email, s2_api_key=s2_key)
    else:
        provider = OpenAlexProvider(email=args.email)

    planted = 0

    # plant DOIs
    if args.doi:
        for doi in args.doi:
            print(f"Planting DOI: {doi}")
            paper = provider.get_paper(doi)
            if paper:
                paper.status = PaperStatus.SEED
                if garden.plant_paper(paper, is_seed=True):
                    planted += 1
                    print(f"  ✓ {paper.title[:50]}")

                    # also plant primary authors
                    if paper.author_ids and paper.authors:
                        for i, aid in enumerate(paper.author_ids[:2]):
                            author = Author(
                                id=aid,
                                name=paper.authors[i] if i < len(paper.authors) else "",
                                openalex_id=aid if aid.startswith("A") else None
                            )
                            garden.plant_author(author, is_seed=False)
                else:
                    print(f"  (already in garden, merged)")
            else:
                print(f"  ✗ not found")

    # plant authors
    if args.author:
        for author_query in args.author:
            print(f"Planting author: {author_query}")

            # try to resolve author
            if author_query.startswith("A") and len(author_query) > 5:
                # OpenAlex author ID
                author_info = provider.get_author(author_query)
            else:
                # search by name
                author_info = provider.resolve_author_id(author_query)

            if author_info:
                author = Author(
                    id=author_info.openalex_id or author_info.s2_id or author_query,
                    name=author_info.name,
                    orcid=author_info.orcid,
                    openalex_id=author_info.openalex_id,
                    s2_id=author_info.s2_id,
                    affiliations=author_info.affiliations,
                    paper_count=author_info.paper_count,
                    citation_count=author_info.citation_count
                )

                if garden.plant_author(author, is_seed=True):
                    planted += 1
                    print(f"  ✓ {author.name} ({author.paper_count} papers)")

                    # fetch their top papers
                    print(f"  Fetching papers...")
                    works = provider.get_author_works(
                        author.openalex_id or author.s2_id,
                        limit=20
                    )
                    for paper in works:
                        garden.plant_paper(paper, is_seed=False)
                    print(f"  Added {len(works)} papers")
                else:
                    print(f"  (already in garden, merged)")
            else:
                print(f"  ✗ not found")

    garden.update_meta("last_plant", datetime.now().isoformat())

    # show stats
    stats = garden.get_stats()
    print(f"\n--- GARDEN STATUS ---")
    print(f"Papers: {stats.total_papers} ({stats.seed_papers} seeds)")
    print(f"Authors: {stats.total_authors} ({stats.seed_authors} seeds)")
    print(f"Edges: {stats.total_edges}")
    print(f"Sessions: {stats.total_sessions}")

    if planted > 0:
        print(f"\n✓ Planted {planted} new items")

    return 0


def cmd_grow(args):
    """grow the garden by expanding frontier nodes."""
    garden = Garden(args.garden)
    garden.increment_sessions()

    # setup provider
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if args.fallback:
        provider = create_default_provider(email=args.email, s2_api_key=s2_key)
    else:
        provider = OpenAlexProvider(email=args.email)

    print("Growing garden...")

    # get frontier papers
    frontier = garden.get_frontier_papers(limit=args.batch)
    print(f"Frontier papers: {len(frontier)}")

    papers_added = 0
    edges_added = 0

    for paper in frontier:
        paper_id = paper.doi or paper.openalex_id
        if not paper_id:
            continue

        print(f"  Expanding: {paper.title[:40]}...")

        # backward refs
        try:
            refs = provider.get_references(paper_id, limit=args.refs)
            for ref in refs:
                if garden.plant_paper(ref, is_seed=False):
                    papers_added += 1
                if garden.add_edge(paper.id, ref.id, EdgeType.CITES):
                    edges_added += 1
        except Exception as e:
            logger.warning(f"refs failed: {e}")

        # forward cites
        try:
            cites = provider.get_citations(paper_id, limit=args.cites)
            for cite in cites:
                if garden.plant_paper(cite, is_seed=False):
                    papers_added += 1
                if garden.add_edge(cite.id, paper.id, EdgeType.CITES):
                    edges_added += 1
        except Exception as e:
            logger.warning(f"cites failed: {e}")

        # mark as expanded (update status in db)
        conn = __import__('sqlite3').connect(str(garden.db_path))
        c = conn.cursor()
        c.execute("UPDATE papers SET status = 'expanded' WHERE id = ?", (paper.id,))
        conn.commit()
        conn.close()

    # grow authors if requested
    if args.authors:
        seed_authors = garden.get_seed_authors()
        print(f"\nExpanding {len(seed_authors)} seed authors...")

        for author in seed_authors:
            if author.last_expanded:
                continue  # already expanded

            author_id = author.openalex_id or author.s2_id
            if not author_id:
                continue

            print(f"  {author.name}...")

            try:
                works = provider.get_author_works(author_id, limit=30)
                for paper in works:
                    if garden.plant_paper(paper, is_seed=False):
                        papers_added += 1
                    # add authored edge
                    garden.add_edge(paper.id, author.id, EdgeType.AUTHORED)

                # mark expanded
                conn = __import__('sqlite3').connect(str(garden.db_path))
                c = conn.cursor()
                c.execute(
                    "UPDATE authors SET last_expanded = ? WHERE id = ?",
                    (datetime.now().isoformat(), author.id)
                )
                conn.commit()
                conn.close()

            except Exception as e:
                logger.warning(f"author expand failed: {e}")

    # compute trajectories (THIS IS THE KEY - populates drift data!)
    print("\nComputing author trajectories...")
    garden.compute_trajectories(min_papers_per_author=3)

    # compute cluster bridging
    print("Computing cluster bridging...")
    garden.compute_cluster_bridging()

    # compute author roles (now with real drift data!)
    print("Computing author roles...")
    garden.compute_author_roles()

    garden.update_meta("last_grow", datetime.now().isoformat())

    # show stats
    stats = garden.get_stats()
    print(f"\n--- GARDEN STATUS ---")
    print(f"Papers: {stats.total_papers}")
    print(f"Authors: {stats.total_authors}")
    print(f"Edges: {stats.total_edges}")
    print(f"\nRoles:")
    print(f"  Leaders: {stats.leaders}")
    print(f"  Players: {stats.players}")
    print(f"  Followers: {stats.followers}")
    print(f"  Disruptors: {stats.disruptors} ← rule-breakers!")
    print(f"\n✓ Added {papers_added} papers, {edges_added} edges")

    return 0


def cmd_status(args):
    """show garden status."""
    garden = Garden(args.garden)

    stats = garden.get_stats()

    print(f"\n{'='*50}")
    print(f"GARDEN: {garden.db_path}")
    print(f"{'='*50}")

    print(f"\n--- CONTENTS ---")
    print(f"Papers: {stats.total_papers} ({stats.seed_papers} seeds)")
    print(f"Authors: {stats.total_authors} ({stats.seed_authors} seeds)")
    print(f"Edges: {stats.total_edges}")
    print(f"Islands: {stats.islands}")

    print(f"\n--- AUTHOR ROLES ---")
    print(f"Leaders:    {stats.leaders:4d}  (high impact, stable focus)")
    print(f"Players:    {stats.players:4d}  (active contributors)")
    print(f"Followers:  {stats.followers:4d}  (follow trends)")
    print(f"Disruptors: {stats.disruptors:4d}  (rule-breakers, create new fields!)")
    print(f"Unknown:    {stats.unknown:4d}  (not enough data)")

    print(f"\n--- HISTORY ---")
    print(f"Sessions: {stats.total_sessions}")
    print(f"Last plant: {stats.last_plant or 'never'}")
    print(f"Last grow: {stats.last_grow or 'never'}")

    # show top disruptors
    disruptors = garden.get_disruptors(limit=5)
    if disruptors:
        print(f"\n--- TOP DISRUPTORS (Rule-Breakers) ---")
        for d in disruptors:
            jumps = f"{d.novelty_jumps} jumps" if d.novelty_jumps else ""
            bridges = f"{d.clusters_bridged} bridges" if d.clusters_bridged else ""
            extra = ", ".join(filter(None, [jumps, bridges]))
            print(f"  • {d.name}")
            print(f"    drift={d.drift_magnitude_avg:.2f} {extra}")

    # show top leaders
    leaders = garden.get_leaders(limit=5)
    if leaders:
        print(f"\n--- TOP LEADERS ---")
        for l in leaders:
            print(f"  • {l.name} ({l.citation_count} citations)")

    return 0


def cmd_export(args):
    """export garden to JSON."""
    garden = Garden(args.garden)

    output = args.output or str(Path.home() / ".refnet" / "garden_export.json")
    path = garden.export_json(output)
    print(f"Exported to: {path}")

    return 0


# need EdgeType import
from ..core.models import EdgeType


def main():
    """garden CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RefNet Garden - organic network growth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  plant   Add seeds (DOIs or authors) to the garden
  grow    Expand the network from frontier nodes
  status  Show garden statistics and top authors
  export  Export garden to JSON

examples:
  refnet-garden plant --doi 10.1073/pnas.1818339116
  refnet-garden plant --author "Charles Carter"
  refnet-garden grow --batch 20
  refnet-garden status
  refnet-garden export -o my_network.json
        """
    )

    parser.add_argument(
        "--garden", "-g",
        type=str,
        default=str(DEFAULT_GARDEN_PATH),
        help=f"garden database path (default: {DEFAULT_GARDEN_PATH})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="command")

    # plant command
    plant_parser = subparsers.add_parser("plant", help="plant seeds")
    plant_parser.add_argument("--doi", action="append", help="paper DOI")
    plant_parser.add_argument("--author", action="append", help="author name or ID")
    plant_parser.add_argument("--email", default="kiran@mcneese.edu", help="API email")
    plant_parser.add_argument("--fallback", action="store_true", help="use S2 fallback")

    # grow command
    grow_parser = subparsers.add_parser("grow", help="grow network")
    grow_parser.add_argument("--batch", type=int, default=10, help="papers to expand")
    grow_parser.add_argument("--refs", type=int, default=30, help="refs per paper")
    grow_parser.add_argument("--cites", type=int, default=20, help="cites per paper")
    grow_parser.add_argument("--authors", action="store_true", help="also expand seed authors")
    grow_parser.add_argument("--email", default="kiran@mcneese.edu", help="API email")
    grow_parser.add_argument("--fallback", action="store_true", help="use S2 fallback")

    # status command
    subparsers.add_parser("status", help="show status")

    # export command
    export_parser = subparsers.add_parser("export", help="export to JSON")
    export_parser.add_argument("--output", "-o", help="output path")

    args = parser.parse_args()

    # setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "plant":
        if not args.doi and not args.author:
            print("Error: need --doi or --author")
            return 1
        return cmd_plant(args)
    elif args.command == "grow":
        return cmd_grow(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "export":
        return cmd_export(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
