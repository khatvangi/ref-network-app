"""
pipeline CLI - literature analysis from command line.

usage:
    python -m refnet.pipeline.cli paper <doi>
    python -m refnet.pipeline.cli author <name>

examples:
    python -m refnet.pipeline.cli paper 10.1073/pnas.2116840119
    python -m refnet.pipeline.cli author "Charles W. Carter" --affiliation UNC
    python -m refnet.pipeline.cli paper 10.1038/s41586-021-03819-2 --config deep --output json
"""

import argparse
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..providers.openalex import OpenAlexProvider
from ..providers.base import ORCIDProvider
from .orchestrator import Pipeline
from .config import PipelineConfig, QuickConfig, DeepConfig, AuthorFocusConfig
from .results import LiteratureAnalysis


def setup_logging(verbose: bool = False, quiet: bool = False):
    """configure logging."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def format_summary(result: LiteratureAnalysis) -> str:
    """format result as human-readable summary."""
    lines = []

    # header
    lines.append("=" * 60)
    lines.append(f"LITERATURE ANALYSIS: {result.seed_query}")
    lines.append("=" * 60)
    lines.append("")

    # basic stats
    lines.append(f"Papers analyzed: {result.paper_count}")
    lines.append(f"Key authors: {len(result.key_authors)}")
    lines.append(f"Duration: {result.duration_seconds:.1f}s")
    lines.append(f"API calls: {result.api_calls}")
    lines.append("")

    # seed info
    if result.seed_paper:
        lines.append("--- SEED PAPER ---")
        lines.append(f"Title: {result.seed_paper.title}")
        lines.append(f"Year: {result.seed_paper.year}")
        lines.append(f"Citations: {result.seed_paper.citation_count}")
        lines.append("")

    if result.seed_author:
        lines.append("--- SEED AUTHOR ---")
        lines.append(f"Name: {result.seed_author.name}")
        lines.append(f"Papers: {result.seed_author.paper_count}")
        lines.append(f"Citations: {result.seed_author.citation_count}")
        if result.seed_author.affiliations:
            lines.append(f"Affiliations: {', '.join(result.seed_author.affiliations[:2])}")
        lines.append("")

    # key authors
    if result.key_authors:
        lines.append("--- KEY AUTHORS ---")
        for i, author in enumerate(result.key_authors[:5], 1):
            lines.append(f"  {i}. {author.name}")
            lines.append(f"     Papers: {author.paper_count}, Citations: {author.citation_count}")
            if author.trajectory and author.trajectory.phases:
                phase_topics = [p.dominant_concepts[0] if p.dominant_concepts else '?'
                               for p in author.trajectory.phases[:3]]
                lines.append(f"     Research: {' → '.join(phase_topics)}")
        lines.append("")

    # landscape
    if result.landscape:
        lines.append("--- FIELD LANDSCAPE ---")
        if result.landscape.core_topics:
            lines.append(f"Core topics: {', '.join(result.landscape.core_topics[:5])}")
        if result.landscape.emerging_topics:
            lines.append(f"Emerging: {', '.join(result.landscape.emerging_topics[:3])}")
        if result.landscape.declining_topics:
            lines.append(f"Declining: {', '.join(result.landscape.declining_topics[:3])}")
        if result.landscape.year_range[0]:
            lines.append(f"Year range: {result.landscape.year_range[0]}-{result.landscape.year_range[1]}")
        lines.append("")

    # gaps
    if result.gaps and result.gaps.concept_gaps:
        lines.append("--- RESEARCH OPPORTUNITIES ---")
        for i, gap in enumerate(result.gaps.concept_gaps[:3], 1):
            lines.append(f"  {i}. {gap.concept_a} × {gap.concept_b}")
            lines.append(f"     Gap score: {gap.gap_score:.2f}")
        lines.append("")

    # reading list
    if result.reading_list:
        lines.append("--- READING LIST ---")
        for i, item in enumerate(result.reading_list[:10], 1):
            priority_marker = "★" if item.priority == 1 else "○" if item.priority == 2 else "·"
            lines.append(f"  {priority_marker} [{item.category}] {item.paper.title[:60]}...")
            lines.append(f"    Year: {item.paper.year}, Score: {item.relevance_score:.2f}")
        if len(result.reading_list) > 10:
            lines.append(f"  ... and {len(result.reading_list) - 10} more papers")
        lines.append("")

    # insights
    if result.insights:
        lines.append("--- KEY INSIGHTS ---")
        for i, insight in enumerate(result.insights, 1):
            lines.append(f"  {i}. {insight}")
        lines.append("")

    # errors/warnings
    if result.errors:
        lines.append("--- ERRORS ---")
        for error in result.errors:
            lines.append(f"  ! {error}")
        lines.append("")

    if result.warnings:
        lines.append("--- WARNINGS ---")
        for warning in result.warnings:
            lines.append(f"  * {warning}")
        lines.append("")

    return "\n".join(lines)


def format_markdown(result: LiteratureAnalysis) -> str:
    """format result as markdown."""
    lines = []

    # header
    lines.append(f"# Literature Analysis: {result.seed_query}")
    lines.append("")
    lines.append(f"*Generated: {result.created_at.strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # summary stats
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Papers analyzed**: {result.paper_count}")
    lines.append(f"- **Key authors**: {len(result.key_authors)}")
    lines.append(f"- **Duration**: {result.duration_seconds:.1f}s")
    lines.append("")

    # seed
    if result.seed_paper:
        lines.append("## Seed Paper")
        lines.append("")
        lines.append(f"**{result.seed_paper.title}**")
        lines.append(f"- Year: {result.seed_paper.year}")
        lines.append(f"- Citations: {result.seed_paper.citation_count}")
        if result.seed_paper.doi:
            lines.append(f"- DOI: [{result.seed_paper.doi}](https://doi.org/{result.seed_paper.doi})")
        lines.append("")

    if result.seed_author:
        lines.append("## Seed Author")
        lines.append("")
        lines.append(f"**{result.seed_author.name}**")
        lines.append(f"- Papers: {result.seed_author.paper_count}")
        lines.append(f"- Citations: {result.seed_author.citation_count}")
        lines.append("")

    # key authors
    if result.key_authors:
        lines.append("## Key Authors")
        lines.append("")
        for author in result.key_authors[:5]:
            lines.append(f"### {author.name}")
            lines.append(f"- Papers: {author.paper_count}, Citations: {author.citation_count}")
            if author.affiliations:
                lines.append(f"- Affiliations: {', '.join(author.affiliations[:2])}")
            if author.trajectory and author.trajectory.phases:
                lines.append("- Research trajectory:")
                for phase in author.trajectory.phases[:3]:
                    topics = ', '.join(phase.dominant_concepts[:3]) if phase.dominant_concepts else 'N/A'
                    lines.append(f"  - {phase.start_year}-{phase.end_year}: {topics}")
            lines.append("")

    # landscape
    if result.landscape:
        lines.append("## Field Landscape")
        lines.append("")
        if result.landscape.core_topics:
            lines.append(f"**Core topics**: {', '.join(result.landscape.core_topics[:5])}")
        if result.landscape.emerging_topics:
            lines.append(f"**Emerging**: {', '.join(result.landscape.emerging_topics[:3])}")
        if result.landscape.year_range[0]:
            lines.append(f"**Year range**: {result.landscape.year_range[0]}-{result.landscape.year_range[1]}")
        lines.append("")

    # gaps
    if result.gaps and result.gaps.concept_gaps:
        lines.append("## Research Opportunities")
        lines.append("")
        lines.append("| Concept A | Concept B | Gap Score |")
        lines.append("|-----------|-----------|-----------|")
        for gap in result.gaps.concept_gaps[:5]:
            lines.append(f"| {gap.concept_a} | {gap.concept_b} | {gap.gap_score:.2f} |")
        lines.append("")

    # reading list
    if result.reading_list:
        lines.append("## Reading List")
        lines.append("")
        lines.append("| Priority | Category | Title | Year | Score |")
        lines.append("|----------|----------|-------|------|-------|")
        for item in result.reading_list[:15]:
            prio = "★★★" if item.priority == 1 else "★★" if item.priority == 2 else "★"
            title = item.paper.title[:50] + "..." if len(item.paper.title) > 50 else item.paper.title
            lines.append(f"| {prio} | {item.category} | {title} | {item.paper.year} | {item.relevance_score:.2f} |")
        lines.append("")

    # insights
    if result.insights:
        lines.append("## Key Insights")
        lines.append("")
        for insight in result.insights:
            lines.append(f"- {insight}")
        lines.append("")

    return "\n".join(lines)


def format_json(result: LiteratureAnalysis) -> str:
    """format result as JSON."""
    data = {
        "metadata": {
            "created_at": result.created_at.isoformat(),
            "seed_query": result.seed_query,
            "seed_type": result.seed_type,
            "duration_seconds": result.duration_seconds,
            "api_calls": result.api_calls,
            "success": result.success
        },
        "summary": {
            "paper_count": result.paper_count,
            "key_authors_count": len(result.key_authors),
            "reading_list_count": len(result.reading_list)
        },
        "key_authors": [
            {
                "name": a.name,
                "author_id": a.author_id,
                "paper_count": a.paper_count,
                "citation_count": a.citation_count,
                "affiliations": a.affiliations,
                "relevance_score": a.relevance_score
            }
            for a in result.key_authors
        ],
        "reading_list": [
            {
                "title": item.paper.title,
                "year": item.paper.year,
                "doi": item.paper.doi,
                "category": item.category,
                "priority": item.priority,
                "relevance_score": item.relevance_score,
                "reason": item.reason
            }
            for item in result.reading_list
        ],
        "insights": result.insights,
        "errors": result.errors,
        "warnings": result.warnings
    }

    # add landscape
    if result.landscape:
        data["landscape"] = {
            "core_topics": result.landscape.core_topics,
            "emerging_topics": result.landscape.emerging_topics,
            "declining_topics": result.landscape.declining_topics,
            "year_range": list(result.landscape.year_range),
            "total_papers": result.landscape.total_papers,
            "total_authors": result.landscape.total_authors
        }

    # add gaps
    if result.gaps:
        data["gaps"] = {
            "concept_gaps": [
                {
                    "concept_a": g.concept_a,
                    "concept_b": g.concept_b,
                    "gap_score": g.gap_score,
                    "papers_a": g.papers_a,
                    "papers_b": g.papers_b
                }
                for g in (result.gaps.concept_gaps or [])
            ],
            "method_gaps": result.gaps.method_gaps or [],
            "unexplored_areas": result.gaps.unexplored_areas or []
        }

    return json.dumps(data, indent=2)


def get_config(name: str) -> PipelineConfig:
    """get config by name."""
    configs = {
        "quick": QuickConfig(),
        "default": PipelineConfig(),
        "deep": DeepConfig(),
        "author": AuthorFocusConfig()
    }
    return configs.get(name, PipelineConfig())


def cmd_paper(args):
    """analyze starting from a paper."""
    setup_logging(args.verbose, args.quiet)

    # setup
    provider = OpenAlexProvider()
    config = get_config(args.config)

    # apply overrides
    if args.focus:
        config.focus_concepts = args.focus
    if args.min_year:
        config.min_year = args.min_year

    pipeline = Pipeline(provider, config=config)

    if not args.quiet:
        print(f"\nAnalyzing paper: {args.query}")
        print(f"Config: {args.config}")
        print("-" * 40)

    # run
    result = pipeline.analyze_paper(args.query, config=config)

    # output
    output_result(result, args)

    return 0 if result.success else 1


def cmd_author(args):
    """analyze starting from an author."""
    setup_logging(args.verbose, args.quiet)

    # setup
    provider = OpenAlexProvider()
    orcid = ORCIDProvider()

    # use author-focused config if specified, otherwise use requested config
    if args.config == "default":
        config = AuthorFocusConfig()
    else:
        config = get_config(args.config)

    # apply overrides
    if args.focus:
        config.focus_concepts = args.focus
    if args.min_year:
        config.min_year = args.min_year

    pipeline = Pipeline(provider, orcid_provider=orcid, config=config)

    if not args.quiet:
        print(f"\nAnalyzing author: {args.name}")
        if args.affiliation:
            print(f"Affiliation hint: {args.affiliation}")
        print(f"Config: {args.config}")
        print("-" * 40)

    # run
    result = pipeline.analyze_author(
        name=args.name,
        affiliation_hint=args.affiliation,
        config=config
    )

    # output
    output_result(result, args)

    return 0 if result.success else 1


def output_result(result: LiteratureAnalysis, args):
    """output result in requested format."""
    # format
    if args.output == "json":
        output = format_json(result)
    elif args.output == "markdown":
        output = format_markdown(result)
    else:
        output = format_summary(result)

    # destination
    if args.file:
        path = Path(args.file)
        path.write_text(output)
        if not args.quiet:
            print(f"\nOutput written to: {path}")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Literature analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s paper 10.1073/pnas.2116840119
  %(prog)s author "Charles W. Carter" --affiliation UNC
  %(prog)s paper 10.1038/s41586-021-03819-2 --config deep --output json
  %(prog)s author "Jack Szostak" --output markdown -f report.md
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="analysis mode")

    # paper subcommand
    paper_parser = subparsers.add_parser("paper", help="analyze from paper DOI/title")
    paper_parser.add_argument(
        "query",
        help="paper DOI or title"
    )
    paper_parser.set_defaults(func=cmd_paper)

    # author subcommand
    author_parser = subparsers.add_parser("author", help="analyze from author name")
    author_parser.add_argument(
        "name",
        help="author name"
    )
    author_parser.add_argument(
        "--affiliation", "-a",
        type=str,
        help="affiliation hint for disambiguation"
    )
    author_parser.set_defaults(func=cmd_author)

    # common arguments (add to both)
    for subparser in [paper_parser, author_parser]:
        subparser.add_argument(
            "--config", "-c",
            type=str,
            default="default",
            choices=["quick", "default", "deep", "author"],
            help="config preset (default: default)"
        )
        subparser.add_argument(
            "--output", "-o",
            type=str,
            default="summary",
            choices=["summary", "json", "markdown"],
            help="output format (default: summary)"
        )
        subparser.add_argument(
            "--file", "-f",
            type=str,
            help="output to file"
        )
        subparser.add_argument(
            "--focus",
            type=str,
            nargs='+',
            help="focus concepts to highlight"
        )
        subparser.add_argument(
            "--min-year",
            type=int,
            help="filter papers before this year"
        )
        subparser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="verbose output"
        )
        subparser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="minimal output"
        )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
