#!/usr/bin/env python3
"""
REFNET - Universal Citation Network Builder
============================================

One command for any input. Auto-detects format, runs all layers.

SUPPORTED INPUTS:
  • PDF file        - paper.pdf (extracts DOIs from references)
  • DOI             - 10.1038/s41586-021-03819-2
  • BibTeX/RIS      - library.bib, references.ris (Zotero/EndNote export)
  • JSON/CSV        - collection.json, papers.csv
  • Topic string    - "aminoacyl-tRNA synthetase evolution"
  • Directory       - /path/to/papers/ (processes all files)

LAYERS (all automatic):
  1. Citation expansion - references (backward) + citations (forward)
  2. Author expansion   - first/last author works (3rd discovery axis)
  3. Trajectory         - JSD drift detection, novelty jumps
  4. Clustering         - Louvain community detection
  5. Gap analysis       - unexplored areas in candidate pool

GUARDRAILS:
  • Hub suppression     - methodology papers (>5k cites) limited
  • Relevance filter    - topic drift prevented
  • Mega-author skip    - prolific authors don't explode graph

USAGE:
    python refnet.py paper.pdf
    python refnet.py --doi 10.1038/s41586-021-03819-2
    python refnet.py library.bib --max-nodes 300
    python refnet.py "CRISPR evolution" --with-gscholar
    python refnet.py --collection papers.json -o my_network/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from providers import (
    OpenAlexProvider,
    SemanticScholarProvider,
    PubMedProvider,
    CrossrefProvider,
    ProviderAggregator,
    AggregatorConfig,
    PaperStub
)
from topic_triage import TopicTriageEngine, ScopeBand
from graph_builder import GraphBuilder, GraphBuildConfig


def load_seeds_from_pdf(pdf_path: str) -> tuple:
    """load seed papers from pdf references."""
    from inputs.pdf_parser import extract_references_from_pdf
    main_paper, refs = extract_references_from_pdf(pdf_path)
    return main_paper, refs


def load_seeds_from_bibtex(bib_path: str) -> list:
    """load seed papers from bibtex/ris file."""
    from inputs.bibtex_parser import parse_file
    return parse_file(bib_path)


def lookup_seed_by_title(title: str) -> PaperStub:
    """lookup a single paper by title."""
    from inputs.title_lookup import lookup_paper_by_title
    return lookup_paper_by_title(title)


def lookup_seed_by_doi(doi: str) -> PaperStub:
    """lookup a single paper by doi."""
    from inputs.title_lookup import lookup_paper_by_doi
    return lookup_paper_by_doi(doi)


def main():
    parser = argparse.ArgumentParser(
        description="Build citation networks from various sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
input modes:
  TOPIC           search by topic string (default)
  --pdf FILE      extract references from PDF
  --bib FILE      import from BibTeX/RIS file
  --title TEXT    lookup paper by title
  --doi DOI       lookup paper by DOI

examples:
  %(prog)s "ancestral protein reconstruction"
  %(prog)s "CRISPR gene editing" --years 5 --max-nodes 100
  %(prog)s --pdf paper.pdf --max-nodes 200
  %(prog)s --bib zotero_export.bib
  %(prog)s --title "Highly accurate protein structure prediction"
  %(prog)s --doi 10.1038/s41586-021-03819-2
  %(prog)s "chemistry" --triage-only
  %(prog)s "protein folding" --use-gscholar --output graph.json
        """
    )

    # input modes
    input_group = parser.add_argument_group('input modes')
    input_group.add_argument(
        "topic",
        nargs='?',
        help="research topic to search for"
    )
    input_group.add_argument(
        "--pdf",
        type=str,
        metavar="FILE",
        help="extract references from PDF file"
    )
    input_group.add_argument(
        "--bib",
        type=str,
        metavar="FILE",
        help="import from BibTeX or RIS file"
    )
    input_group.add_argument(
        "--title",
        type=str,
        metavar="TEXT",
        help="lookup paper by title and expand"
    )
    input_group.add_argument(
        "--doi",
        type=str,
        metavar="DOI",
        help="lookup paper by DOI and expand"
    )

    # time range
    parser.add_argument(
        "--years", "-y",
        type=int,
        default=3,
        choices=[3, 5, 10, 20, 50],
        help="look back N years (default: 3)"
    )

    # graph limits
    parser.add_argument(
        "--max-nodes", "-n",
        type=int,
        default=200,
        help="maximum nodes in graph (default: 200)"
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=2,
        help="maximum expansion depth (default: 2)"
    )

    # provider options
    provider_group = parser.add_argument_group('providers')
    provider_group.add_argument(
        "--use-gscholar",
        action="store_true",
        help="include google scholar (slow, 1 req/5 sec)"
    )
    provider_group.add_argument(
        "--no-pubmed",
        action="store_true",
        help="skip pubmed"
    )
    provider_group.add_argument(
        "--no-s2",
        action="store_true",
        help="skip semantic scholar"
    )
    provider_group.add_argument(
        "--openalex-only",
        action="store_true",
        help="only use openalex (fastest)"
    )

    # output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="output JSON file"
    )
    parser.add_argument(
        "--triage-only",
        action="store_true",
        help="only run triage, don't build graph"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force build even if triage is RED"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="minimal output"
    )

    # misc
    parser.add_argument(
        "--email",
        type=str,
        default="kiran@mcneese.edu",
        help="email for api polite pools"
    )

    args = parser.parse_args()

    # validate input - at least one input mode required
    if not any([args.topic, args.pdf, args.bib, args.title, args.doi]):
        parser.error("at least one input mode required: TOPIC, --pdf, --bib, --title, or --doi")

    # determine input mode and load seeds
    seeds = []
    topic = None
    main_paper = None

    if args.pdf:
        print(f"\n{'='*60}")
        print(f"INPUT: PDF file - {args.pdf}")
        print(f"{'='*60}\n")

        main_paper, refs = load_seeds_from_pdf(args.pdf)
        seeds = refs
        if main_paper:
            print(f"Main paper: {main_paper.title}")
            seeds.insert(0, main_paper)
        print(f"Extracted {len(seeds)} references")

        # use main paper title as topic for relevance scoring
        if main_paper and main_paper.title:
            topic = main_paper.title[:100]

    elif args.bib:
        print(f"\n{'='*60}")
        print(f"INPUT: BibTeX/RIS file - {args.bib}")
        print(f"{'='*60}\n")

        seeds = load_seeds_from_bibtex(args.bib)
        print(f"Loaded {len(seeds)} papers from file")

        # use first paper title as topic
        if seeds and seeds[0].title:
            topic = seeds[0].title[:100]

    elif args.title:
        print(f"\n{'='*60}")
        print(f"INPUT: Paper title lookup")
        print(f"{'='*60}\n")

        paper = lookup_seed_by_title(args.title)
        if paper:
            print(f"Found: {paper.title}")
            print(f"  Year: {paper.year}, DOI: {paper.doi}")
            seeds = [paper]
            topic = paper.title
        else:
            print(f"Paper not found: {args.title}")
            return 1

    elif args.doi:
        print(f"\n{'='*60}")
        print(f"INPUT: DOI lookup - {args.doi}")
        print(f"{'='*60}\n")

        paper = lookup_seed_by_doi(args.doi)
        if paper:
            print(f"Found: {paper.title}")
            print(f"  Year: {paper.year}")
            seeds = [paper]
            topic = paper.title
        else:
            print(f"Paper not found for DOI: {args.doi}")
            return 1

    else:
        # topic search mode
        topic = args.topic
        print(f"\n{'='*60}")
        print(f"INPUT: Topic search - {topic}")
        print(f"{'='*60}\n")

    # configure providers
    if args.openalex_only:
        agg_config = AggregatorConfig(
            use_openalex=True,
            use_s2=False,
            use_pubmed=False,
            use_crossref=False,
            use_gscholar=False
        )
    else:
        agg_config = AggregatorConfig(
            use_openalex=True,
            use_s2=not args.no_s2,
            use_pubmed=not args.no_pubmed,
            use_crossref=True,
            use_gscholar=args.use_gscholar,
            gscholar_limit=20
        )

    aggregator = ProviderAggregator(agg_config)

    # if we have seeds but no topic search, skip triage and go straight to build
    if seeds and not args.topic:
        print(f"\n--- SEED-BASED BUILD ---")
        print(f"Seeds: {len(seeds)}")
        print(f"Max nodes: {args.max_nodes}, Max depth: {args.max_depth}")

        config = GraphBuildConfig(
            years_back=args.years,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth
        )

        # use openalex as primary provider for graph building
        builder = GraphBuilder(OpenAlexProvider(email=args.email), config)
        result = builder.build(topic or "research", seeds=seeds)

        # output
        output_data = {
            'input_mode': 'seeds',
            'seed_count': len(seeds),
            'generated_at': datetime.now().isoformat(),
            'graph': result.to_dict()
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nGraph saved to: {args.output}")
        else:
            print(f"\n--- GRAPH SUMMARY ---")
            print(f"Nodes: {len(result.nodes)}")
            print(f"Edges: {len(result.edges)}")
            print(f"Edge types: {result._edge_type_counts()}")
            print(f"Build time: {result.build_time_seconds:.1f}s")

        return 0

    # topic search mode - run triage first
    current_year = datetime.now().year
    year_min = current_year - args.years

    print(f"--- MULTI-PROVIDER SEARCH ---")
    print(f"Providers: {', '.join(aggregator.providers.keys())}")
    print(f"Years: {year_min}-{current_year}")

    # get count estimates from all providers
    counts = aggregator.get_count_estimates(topic, year_min=year_min)
    print(f"\nCount estimates:")
    for name, count in counts.items():
        if count >= 0:
            print(f"  {name}: {count:,}")

    # run triage using openalex (most reliable for this)
    triage_engine = TopicTriageEngine(OpenAlexProvider(email=args.email))
    triage = triage_engine.triage(topic, years_back=args.years)

    # display triage results
    if not args.quiet:
        print(f"\n--- TRIAGE RESULTS ---")
        print(f"OpenAlex count: {triage.count_estimate:,}")
        print(f"Risk score: {triage.scope_risk_score:.3f}")
        print(f"Band: {triage.scope_band.value}")

        if triage.top_concepts:
            print(f"\nTop concepts:")
            for c in triage.top_concepts[:5]:
                print(f"  - {c['name']}: {c['count']}")

    if args.triage_only:
        if args.output:
            output_data = {
                'topic': topic,
                'counts_by_provider': counts,
                'triage': triage.to_dict()
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nTriage saved to: {args.output}")
        return 0

    # check if topic is too broad
    if triage.scope_band == ScopeBand.RED and not args.force:
        print(f"\n!!! TOPIC TOO BROAD !!!")
        print(f"Count {triage.count_estimate:,} > 20000 or risk {triage.scope_risk_score:.3f} > 0.75")
        print(f"\nSuggestions to narrow your topic:")

        if triage.recommended_facets.get('subdomain_options'):
            print(f"  Add subdomain: {', '.join(triage.recommended_facets['subdomain_options'][:3])}")

        print(f"\nUse --force to build anyway (not recommended)")
        return 1

    if triage.scope_band == ScopeBand.YELLOW and not args.quiet:
        print(f"\n*** YELLOW WARNING ***")
        print(f"Topic may be somewhat broad. Proceeding...")

    # search all providers for seeds
    print(f"\n--- AGGREGATED SEARCH ---")
    agg_seeds = aggregator.search_papers(topic, year_min=year_min, limit=50)

    # combine with triage seeds, prefer papers with more metadata
    all_seeds = agg_seeds + triage.seed_preview
    # dedupe
    seen = set()
    unique_seeds = []
    for s in all_seeds:
        key = s.doi or s.title[:50].lower()
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    # sort by citation count
    unique_seeds.sort(key=lambda p: p.citation_count or 0, reverse=True)
    seeds = unique_seeds[:20]  # top 20 seeds

    print(f"Combined seeds: {len(seeds)} unique papers")

    if not args.quiet:
        print(f"\nTop seeds:")
        for p in seeds[:5]:
            print(f"  - {p.title[:50]}... ({p.year}) [cites: {p.citation_count}]")

    # build graph
    print(f"\n--- BUILDING GRAPH ---")
    print(f"Max nodes: {args.max_nodes}, Max depth: {args.max_depth}")

    config = GraphBuildConfig(
        years_back=args.years,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth
    )

    builder = GraphBuilder(OpenAlexProvider(email=args.email), config)
    result = builder.build(topic, seeds=seeds)

    # output
    output_data = {
        'topic': topic,
        'input_mode': 'topic_search',
        'providers': list(aggregator.providers.keys()),
        'generated_at': datetime.now().isoformat(),
        'triage': triage.to_dict(),
        'graph': result.to_dict()
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nGraph saved to: {args.output}")
    else:
        print(f"\n--- GRAPH SUMMARY ---")
        print(f"Nodes: {len(result.nodes)}")
        print(f"Edges: {len(result.edges)}")
        print(f"Edge types: {result._edge_type_counts()}")
        print(f"Build time: {result.build_time_seconds:.1f}s")

    return 0


def run_modern_pipeline(args, seeds):
    """
    run the modern pipeline with all layers.
    uses refnet package (layers, trajectory, gap analysis).

    this delegates to the proper modern architecture which uses:
    - ExpansionEngine.build() method (not expand)
    - CandidatePool with (config, db_path)
    - WorkingGraph with (config)
    """
    from pathlib import Path
    from refnet.providers.composite import create_default_provider
    from refnet.inputs.collection import enrich_papers_with_provider
    from refnet.graph.candidate_pool import CandidatePool
    from refnet.graph.working_graph import WorkingGraph
    from refnet.graph.expansion import ExpansionEngine
    from refnet.export.formats import GraphExporter
    from refnet.export.viewer import HTMLViewer
    from refnet.core.config import RefnetConfig
    from refnet.core.models import Paper, PaperStatus

    # convert seeds to modern Paper format if needed
    modern_seeds = []
    for s in seeds:
        if hasattr(s, 'doi') and s.doi:
            p = Paper(
                doi=s.doi,
                title=getattr(s, 'title', '') or '',
                year=getattr(s, 'year', None),
                authors=getattr(s, 'authors', [])[:5] if hasattr(s, 'authors') else [],
                status=PaperStatus.SEED
            )
            modern_seeds.append(p)

    if not modern_seeds:
        print("ERROR: No valid seeds with DOIs")
        return 1

    # create output directory
    output_dir = Path(args.output) if hasattr(args, 'output') and args.output else Path('refnet_output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # create provider with optional google scholar
    email = getattr(args, 'email', 'kiran@mcneese.edu')
    provider = create_default_provider(
        email=email,
        use_google_scholar=getattr(args, 'use_gscholar', False)
    )

    # enrich seeds
    print(f"\nEnriching {len(modern_seeds)} seeds...")
    modern_seeds = enrich_papers_with_provider(modern_seeds, provider)
    seeds_with_doi = [s for s in modern_seeds if s.doi]
    print(f"Seeds with DOIs: {len(seeds_with_doi)}")

    if not seeds_with_doi:
        print("ERROR: No seeds with valid DOIs after enrichment")
        return 1

    # config - use proper RefnetConfig initialization
    config = RefnetConfig()
    config.providers.openalex_email = email

    # set max nodes using proper config fields
    max_nodes = getattr(args, 'max_nodes', 200)
    config.working_graph.max_nodes_small = min(max_nodes, 500)
    config.working_graph.max_nodes_medium = min(max_nodes, 2000)
    config.working_graph.max_nodes_large = min(max_nodes, 5000)

    # set expansion depth
    config.expansion.max_depth = getattr(args, 'max_depth', 2)

    # toggle layers
    config.author.enabled = not getattr(args, 'no_authors', False)
    config.trajectory.enabled = not getattr(args, 'no_trajectory', False)
    config.gap_analysis.enabled = True

    # create storage components
    db_path = str(output_dir / 'candidates.db')
    config.candidate_pool.db_path = db_path
    pool = CandidatePool(config.candidate_pool, db_path)
    graph = WorkingGraph(config.working_graph)

    # run expansion engine (the correct way)
    bucket_mode = getattr(args, 'bucket_mode', False)
    mode_name = "bucket expansion" if bucket_mode else "depth-based"

    print(f"\n--- BUILDING GRAPH ({mode_name}) ---")
    print(f"Seeds: {len(seeds_with_doi)}, Max nodes: {max_nodes}")
    if bucket_mode:
        print(f"Bucket mode: adaptive depth, relevance decay stopping")
        print(f"  base_max_generations: {config.expansion.base_max_generations}")
        print(f"  min_bucket_relevance: {config.expansion.min_bucket_relevance}")
        print(f"  drift_kill_threshold: {config.expansion.drift_kill_threshold}")
    else:
        print(f"Depth: {config.expansion.max_depth}")
    print(f"Author expansion: {'ON' if config.author.enabled else 'OFF'}")
    print(f"Trajectory analysis: {'ON' if config.trajectory.enabled else 'OFF'}")

    # ExpansionEngine takes (provider, config) - not pool/graph
    engine = ExpansionEngine(provider, config)

    # build() or build_with_buckets() based on mode
    topic = getattr(args, 'topic', None) or (args.inputs[0] if hasattr(args, 'inputs') and args.inputs else None)

    if bucket_mode:
        # use bucket-based expansion (citation walking until exhaustion)
        job = engine.build_with_buckets(seeds_with_doi, topic=topic, pool=pool, graph=graph)
    else:
        # use traditional depth-based expansion
        job = engine.build(seeds_with_doi, topic=topic, pool=pool, graph=graph)

    # export results
    print(f"\n--- EXPORTING ---")
    exporter = GraphExporter(str(output_dir))

    gap_result = job.gap_analysis if hasattr(job, 'gap_analysis') else None

    exporter.export_json(graph, "graph.json", gap_result)
    exporter.export_graphml(graph, "graph.graphml")
    exporter.export_csv(graph)

    viewer = HTMLViewer(str(output_dir))
    title = topic or (seeds_with_doi[0].title[:50] if seeds_with_doi else "Citation Network")
    viewer.generate(graph, "viewer.html", title, gap_result)

    # summary
    stats = graph.stats()

    pool_stats = pool.stats()

    print(f"\n--- RESULTS ---")
    print(f"Working graph: {stats['papers']} papers, {stats['edges']} edges, {stats['clusters']} clusters")
    print(f"Candidate pool: {pool_stats.get('papers', 0)} papers discovered")
    print(f"Duration: {job.stats.duration_seconds:.1f}s, API calls: {job.stats.api_calls}")

    if gap_result:
        print(f"\n--- GAP ANALYSIS ---")
        print(gap_result.summary)

    print(f"\nOutput: {output_dir.absolute()}/")
    print(f"View: firefox {output_dir}/viewer.html")

    return 0


if __name__ == "__main__":
    # check for --modern flag to use new pipeline
    if '--modern' in sys.argv or '--all-layers' in sys.argv:
        # parse minimal args and run modern pipeline
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('inputs', nargs='*')
        parser.add_argument('--pdf', type=str)
        parser.add_argument('--bib', type=str)
        parser.add_argument('--doi', type=str, action='append', help='DOI (can specify multiple)')
        parser.add_argument('--title', type=str)
        parser.add_argument('--collection', '-c', type=str)
        parser.add_argument('--output', '-o', type=str, default='refnet_output')
        parser.add_argument('--max-nodes', '-n', type=int, default=300)
        parser.add_argument('--max-depth', '-d', type=int, default=2)
        parser.add_argument('--email', type=str, default='user@example.com')
        parser.add_argument('--use-gscholar', action='store_true')
        parser.add_argument('--no-authors', action='store_true')
        parser.add_argument('--no-trajectory', action='store_true')
        parser.add_argument('--seed-limit', type=int, default=30, help='max seeds from collection')
        parser.add_argument('--modern', action='store_true')
        parser.add_argument('--all-layers', action='store_true')
        parser.add_argument('--bucket-mode', action='store_true',
                            help='use bucket expansion (citation walking until exhaustion)')
        args = parser.parse_args()

        # collect seeds based on input
        seeds = []

        if args.collection:
            from refnet.inputs.collection import load_collection
            seeds = load_collection(args.collection, limit=args.seed_limit)
        elif args.pdf:
            main_paper, refs = load_seeds_from_pdf(args.pdf)
            seeds = refs
            if main_paper:
                seeds.insert(0, main_paper)
        elif args.bib:
            seeds = load_seeds_from_bibtex(args.bib)
        elif args.doi:
            # support multiple DOIs
            for doi in args.doi:
                paper = lookup_seed_by_doi(doi)
                if paper:
                    seeds.append(paper)
        elif args.title:
            paper = lookup_seed_by_title(args.title)
            if paper:
                seeds = [paper]
        elif args.inputs:
            # topic search - use Google Scholar if requested, otherwise OpenAlex
            topic = ' '.join(args.inputs)
            if getattr(args, 'use_gscholar', False):
                print(f"[topic] searching Google Scholar for: {topic}")
                from refnet.providers.google_scholar import GoogleScholarProvider
                from refnet.providers.openalex import OpenAlexProvider as ModernOpenAlex
                gs_provider = GoogleScholarProvider(rate_limit_delay=3.0)
                gs_results = gs_provider.search_papers(topic, limit=30)
                print(f"[topic] found {len(gs_results)} papers from Google Scholar")

                # enrich with DOIs via OpenAlex title lookup
                print(f"[topic] looking up DOIs via OpenAlex...")
                oa_provider = ModernOpenAlex(email=args.email)
                enriched = []
                for gs_paper in gs_results:
                    if gs_paper.doi:
                        enriched.append(gs_paper)
                    elif gs_paper.title:
                        # search by title
                        oa_results = oa_provider.search_papers(f'"{gs_paper.title}"', limit=3)
                        for oa in oa_results:
                            if oa.doi and oa.title and gs_paper.title.lower()[:30] in oa.title.lower():
                                print(f"  + {oa.title[:50]}... -> {oa.doi}")
                                enriched.append(oa)
                                break
                results = enriched
                print(f"[topic] {len(results)} papers with DOIs")
            else:
                from providers import OpenAlexProvider
                provider = OpenAlexProvider(email=args.email)
                results = provider.search_papers(topic, limit=30)
            seeds = results

        if seeds:
            sys.exit(run_modern_pipeline(args, seeds))
        else:
            print("ERROR: No seeds found")
            sys.exit(1)
    else:
        sys.exit(main())
