"""
refnet CLI - scientist-centric citation network builder.
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

from .core.config import RefnetConfig, GraphSize, AggressivenessLevel
from .core.models import Paper, PaperStatus
from .core.resilience import setup_logging
from .providers.openalex import OpenAlexProvider
from .providers.semantic_scholar import SemanticScholarProvider
from .providers.composite import CompositeProvider, create_default_provider
from .inputs.collection import load_collection, load_directory, enrich_papers_with_provider
from .graph.candidate_pool import CandidatePool
from .graph.working_graph import WorkingGraph
from .graph.expansion import ExpansionEngine
from .analysis.gap import GapAnalyzer
from .export.formats import GraphExporter
from .export.viewer import HTMLViewer

logger = logging.getLogger("refnet.cli")


def load_seeds_from_doi(doi: str, provider) -> Paper:
    """lookup paper by DOI."""
    paper = provider.get_paper(doi)
    if paper:
        paper.status = PaperStatus.SEED
    return paper


def load_seeds_from_title(title: str, provider) -> Paper:
    """lookup paper by title."""
    results = provider.search_papers(f'"{title}"', limit=5)
    if results:
        # find best match
        title_lower = title.lower()
        for r in results:
            if r.title and title_lower in r.title.lower():
                r.status = PaperStatus.SEED
                return r
        # return first if no exact match
        results[0].status = PaperStatus.SEED
        return results[0]
    return None


def load_seeds_from_bibtex(path: str, provider) -> list:
    """load seeds from BibTeX file."""
    try:
        import bibtexparser
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            bib = bibtexparser.load(f)

        seeds = []
        for entry in bib.entries[:20]:  # limit
            title = entry.get('title', '').strip('{}')
            doi = entry.get('doi', '').strip('{}')
            year = None
            try:
                year = int(entry.get('year', ''))
            except:
                pass

            if doi:
                paper = provider.get_paper(doi)
                if paper:
                    paper.status = PaperStatus.SEED
                    seeds.append(paper)
            elif title:
                paper = load_seeds_from_title(title, provider)
                if paper:
                    seeds.append(paper)

        return seeds
    except Exception as e:
        print(f"[cli] bibtex error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Scientist-centric citation network builder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
input modes (scientist-centric, seeds preferred):
  --doi DOI           start from paper DOI (preferred)
  --title TEXT        start from paper title
  --bib FILE          import from BibTeX file
  TOPIC               topic string (bootstrap only - less preferred)

examples:
  refnet --doi 10.1038/s41586-021-03819-2
  refnet --title "Highly accurate protein structure prediction"
  refnet --bib my_library.bib
  refnet "ancestral protein reconstruction" --max-nodes 100
        """
    )

    # input modes
    input_group = parser.add_argument_group('input modes')
    input_group.add_argument(
        "topic",
        nargs='?',
        help="topic for seed search (bootstrap only)"
    )
    input_group.add_argument(
        "--doi",
        type=str,
        action='append',
        metavar="DOI",
        help="paper DOI (can specify multiple)"
    )
    input_group.add_argument(
        "--title",
        type=str,
        metavar="TEXT",
        help="paper title to lookup"
    )
    input_group.add_argument(
        "--bib",
        type=str,
        metavar="FILE",
        help="BibTeX file"
    )
    input_group.add_argument(
        "--collection", "-c",
        type=str,
        metavar="PATH",
        help="JSON/CSV collection file or directory"
    )
    input_group.add_argument(
        "--enrich",
        action="store_true",
        help="enrich collection papers with provider lookups"
    )
    input_group.add_argument(
        "--seed-limit",
        type=int,
        default=50,
        help="max seeds to load from collection (default: 50)"
    )

    # expansion settings
    parser.add_argument(
        "--max-nodes", "-n",
        type=int,
        default=200,
        help="max nodes in working graph (default: 200)"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=3,
        help="max expansion depth (default: 3)"
    )
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=5,
        choices=[3, 5, 10, 20, 50],
        help="years back for seed search (default: 5)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="graph size preset (default: medium)"
    )
    parser.add_argument(
        "--aggressiveness",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="exploration aggressiveness (default: medium)"
    )

    # feature toggles
    parser.add_argument(
        "--no-authors",
        action="store_true",
        help="disable author expansion"
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="disable trajectory analysis"
    )
    parser.add_argument(
        "--no-gap",
        action="store_true",
        help="disable gap analysis"
    )

    # output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="output directory (default: output)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="all",
        choices=["json", "graphml", "csv", "html", "all"],
        help="export format (default: all)"
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="don't generate HTML viewer"
    )

    # misc
    parser.add_argument(
        "--email",
        type=str,
        default="kiran@mcneese.edu",
        help="email for API polite pools"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="minimal output"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="verbose/debug output"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="path for candidate pool database"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="enable fallback to Semantic Scholar if OpenAlex fails"
    )

    args = parser.parse_args()

    # setup logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    setup_logging(level=log_level)

    # validate input
    if not any([args.topic, args.doi, args.title, args.bib, args.collection]):
        parser.error("at least one input required: --doi, --title, --bib, --collection, or TOPIC")

    # configure
    config = RefnetConfig()
    config.providers.openalex_email = args.email
    config.expansion.max_depth = args.max_depth
    config.author.enabled = not args.no_authors
    config.trajectory.enabled = not args.no_trajectory
    config.gap_analysis.enabled = not args.no_gap

    # size preset
    if args.size == "small":
        config.working_graph.default_size = GraphSize.SMALL
    elif args.size == "large":
        config.working_graph.default_size = GraphSize.LARGE
    else:
        config.working_graph.default_size = GraphSize.MEDIUM

    # override max nodes if specified
    if args.max_nodes:
        config.working_graph.max_nodes_small = min(args.max_nodes, 500)
        config.working_graph.max_nodes_medium = min(args.max_nodes, 2000)
        config.working_graph.max_nodes_large = min(args.max_nodes, 5000)

    # aggressiveness
    if args.aggressiveness == "low":
        config.expansion.aggressiveness = AggressivenessLevel.LOW
    elif args.aggressiveness == "high":
        config.expansion.aggressiveness = AggressivenessLevel.HIGH

    # initialize provider(s)
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    if args.fallback:
        # use composite provider with fallback
        provider = create_default_provider(email=args.email, s2_api_key=s2_key)
        logger.info("[cli] using composite provider (OpenAlex + S2 fallback)")
    else:
        provider = OpenAlexProvider(email=args.email)
        if s2_key:
            logger.info("[cli] semantic scholar API key found (use --fallback to enable)")

    # load seeds
    seeds = []

    if args.collection:
        print(f"\n{'='*60}")
        print(f"INPUT: Collection - {args.collection}")
        print(f"{'='*60}\n")

        coll_path = Path(args.collection)
        if coll_path.is_dir():
            papers = load_directory(str(coll_path), limit_per_file=args.seed_limit)
        else:
            papers = load_collection(str(coll_path), limit=args.seed_limit)

        print(f"Loaded {len(papers)} papers from collection")

        # enrich if requested
        if args.enrich:
            print(f"Enriching papers with provider lookups...")
            papers = enrich_papers_with_provider(papers, provider, batch_size=10)

        # filter to papers with DOIs for proper expansion
        papers_with_doi = [p for p in papers if p.doi]
        print(f"Papers with DOIs: {len(papers_with_doi)}")

        # take top papers as seeds
        for p in papers_with_doi[:args.seed_limit]:
            p.status = PaperStatus.SEED
            seeds.append(p)

        print(f"Using {len(seeds)} papers as seeds")

    elif args.doi:
        print(f"\n{'='*60}")
        print(f"INPUT: DOI lookup")
        print(f"{'='*60}\n")

        for doi in args.doi:
            paper = load_seeds_from_doi(doi, provider)
            if paper:
                print(f"Found: {paper.title}")
                print(f"  Year: {paper.year}, Citations: {paper.citation_count}")
                seeds.append(paper)
            else:
                print(f"Not found: {doi}")

    elif args.title:
        print(f"\n{'='*60}")
        print(f"INPUT: Title lookup")
        print(f"{'='*60}\n")

        paper = load_seeds_from_title(args.title, provider)
        if paper:
            print(f"Found: {paper.title}")
            print(f"  Year: {paper.year}, DOI: {paper.doi}")
            seeds.append(paper)
        else:
            print(f"Not found: {args.title}")
            return 1

    elif args.bib:
        print(f"\n{'='*60}")
        print(f"INPUT: BibTeX file - {args.bib}")
        print(f"{'='*60}\n")

        seeds = load_seeds_from_bibtex(args.bib, provider)
        print(f"Loaded {len(seeds)} seeds from file")

    elif args.topic:
        print(f"\n{'='*60}")
        print(f"INPUT: Topic bootstrap - {args.topic}")
        print(f"{'='*60}\n")

        # topic search for bootstrap seeds
        current_year = datetime.now().year
        year_min = current_year - args.years

        count = provider.get_count_estimate(args.topic, year_min=year_min)
        print(f"Count estimate: {count:,}")

        if count > 20000:
            print(f"\n!!! TOPIC TOO BROAD ({count:,} papers) !!!")
            print("Please provide anchor papers with --doi or --title")
            print("Or narrow your topic search terms")
            return 1

        results = provider.search_papers(args.topic, year_min=year_min, limit=20)
        print(f"Found {len(results)} seed papers")

        for r in results[:10]:
            r.status = PaperStatus.SEED
            seeds.append(r)

    if not seeds:
        print("No seeds found. Please provide valid input.")
        return 1

    # create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # initialize storage
    db_path = args.db_path or str(output_dir / "candidates.db")
    config.candidate_pool.db_path = db_path
    pool = CandidatePool(config.candidate_pool, db_path)
    graph = WorkingGraph(config.working_graph)

    # run expansion
    print(f"\n--- BUILDING GRAPH ---")
    print(f"Seeds: {len(seeds)}")
    print(f"Max nodes: {config.get_max_nodes()}")
    print(f"Max depth: {config.expansion.max_depth}")
    print(f"Author expansion: {'enabled' if config.author.enabled else 'disabled'}")

    engine = ExpansionEngine(provider, config)
    job = engine.build(seeds, topic=args.topic, pool=pool, graph=graph)

    # export results
    print(f"\n--- EXPORTING RESULTS ---")
    exporter = GraphExporter(str(output_dir))

    gap_result = job.gap_analysis if hasattr(job, 'gap_analysis') else None

    if args.format in ["json", "all"]:
        path = exporter.export_json(graph, "graph.json", gap_result)
        print(f"JSON: {path}")

    if args.format in ["graphml", "all"]:
        path = exporter.export_graphml(graph, "graph.graphml")
        print(f"GraphML: {path}")

    if args.format in ["csv", "all"]:
        nodes_path, edges_path = exporter.export_csv(graph)
        print(f"CSV: {nodes_path}, {edges_path}")

    if args.format in ["html", "all"] and not args.no_viewer:
        viewer = HTMLViewer(str(output_dir))
        title = args.topic or seeds[0].title[:50] if seeds else "Citation Network"
        path = viewer.generate(graph, "viewer.html", title, gap_result)
        print(f"HTML Viewer: {path}")

    # print summary
    stats = graph.stats()
    print(f"\n--- SUMMARY ---")
    print(f"Nodes: {stats['nodes']} (papers: {stats['papers']}, authors: {stats['authors']})")
    print(f"Edges: {stats['edges']}")
    print(f"Edge types: {stats['edge_types']}")
    print(f"Seeds: {stats['seeds']}")
    print(f"Clusters: {stats['clusters']}")
    print(f"Duration: {job.stats.duration_seconds:.1f}s")
    print(f"API calls: {job.stats.api_calls}")

    # show errors if any
    if job.stats.errors > 0:
        print(f"\n--- ERRORS ---")
        print(f"Total errors: {job.stats.errors}")
        print(f"Papers failed: {job.stats.papers_failed}")
        if job.error:
            print(f"Job error: {job.error}")

    # show provider stats if available
    if hasattr(provider, 'stats'):
        pstats = provider.stats()
        if pstats.get('failed_calls', 0) > 0 or pstats.get('fallback_count', 0) > 0:
            print(f"\n--- PROVIDER STATS ---")
            print(f"Provider: {pstats.get('provider', 'unknown')}")
            if 'fallback_count' in pstats:
                print(f"Fallbacks used: {pstats['fallback_count']}")
            if pstats.get('circuit_state') != 'closed':
                print(f"Circuit state: {pstats['circuit_state']}")

    if gap_result:
        print(f"\n--- GAP ANALYSIS ---")
        print(gap_result.summary)

    print(f"\nOutput directory: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
