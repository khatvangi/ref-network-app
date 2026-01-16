#!/usr/bin/env python3
"""
refnet - citation network builder CLI.
builds citation networks from various input sources:
  - topic string
  - pdf file (extracts references)
  - bibtex/ris file (zotero export)
  - paper title or doi

searches multiple databases: openalex, semantic scholar, pubmed, crossref, google scholar

usage:
    python refnet.py "ancestral protein reconstruction"
    python refnet.py --pdf paper.pdf
    python refnet.py --bib references.bib
    python refnet.py --title "Highly accurate protein structure prediction"
    python refnet.py --doi 10.1038/s41586-021-03819-2
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


if __name__ == "__main__":
    sys.exit(main())
