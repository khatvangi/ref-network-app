#!/usr/bin/env python3
"""
test agents - verify CorpusFetcher and TrajectoryAnalyzer work with real data.

tests with Charles W. Carter Jr. (aaRS researcher from seed papers).
"""

import sys
sys.path.insert(0, '/storage/kiran-stuff/ref-network-app')

from refnet.agents import CorpusFetcher, TrajectoryAnalyzer
from refnet.agents.base import AgentStatus
from refnet.providers.openalex import OpenAlexProvider


def test_corpus_fetcher():
    """test CorpusFetcher with Charles Carter."""
    print("=" * 60)
    print("TEST: CorpusFetcher")
    print("=" * 60)

    # charles carter's openalex id (from aaRS research)
    # let's first resolve by name to get correct ID
    provider = OpenAlexProvider(email="test@example.com")

    # resolve author by name first
    print("\n1. Resolving author by name: 'Charles W. Carter'")
    author_info = provider.resolve_author_id("Charles W. Carter Jr")

    if author_info:
        print(f"   Found: {author_info.name}")
        print(f"   OpenAlex ID: {author_info.openalex_id}")
        print(f"   Paper count: {author_info.paper_count}")
        print(f"   Citations: {author_info.citation_count}")
        author_id = author_info.openalex_id
    else:
        print("   FAILED: Could not resolve author")
        return None

    # now test CorpusFetcher
    print(f"\n2. Fetching corpus for {author_id}")
    fetcher = CorpusFetcher(provider, max_papers=200)  # larger sample for trajectory
    result = fetcher.run(author_id=author_id)

    print(f"\n3. Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Duration: {result.duration_ms:.1f}ms")
    print(f"   API calls: {result.api_calls}")

    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for e in result.errors[:3]:
            print(f"     - [{e.code}] {e.message}")

    if result.ok and result.data:
        corpus = result.data
        print(f"\n4. Corpus Stats:")
        print(f"   Author: {corpus.name}")
        print(f"   Papers fetched: {len(corpus.papers)}")
        print(f"   Year range: {corpus.year_range}")
        print(f"   Top venues: {corpus.top_venues[:3]}")
        print(f"   Top concepts: {[c['name'] for c in corpus.top_concepts[:5]]}")
        print(f"   Top collaborators: {corpus.collaborators[:5]}")

        # show sample papers
        print(f"\n5. Sample papers (5 most cited):")
        sorted_papers = sorted(corpus.papers, key=lambda p: p.citation_count or 0, reverse=True)
        for p in sorted_papers[:5]:
            print(f"   [{p.year}] {p.title[:60]}... ({p.citation_count} cites)")

        return corpus
    else:
        print("   FAILED: No data returned")
        return None


def test_trajectory_analyzer(corpus):
    """test TrajectoryAnalyzer with corpus from previous test."""
    print("\n" + "=" * 60)
    print("TEST: TrajectoryAnalyzer (v2 - tuned)")
    print("=" * 60)

    if not corpus:
        print("   SKIPPED: No corpus data")
        return

    print(f"\n1. Analyzing trajectory for {corpus.name} ({len(corpus.papers)} papers)")

    # optionally test with ORCID provider
    from refnet.providers.base import ORCIDProvider
    orcid_provider = ORCIDProvider()

    analyzer = TrajectoryAnalyzer(orcid_provider=orcid_provider)
    # TrajectoryAnalyzer expects the full AuthorCorpus object
    result = analyzer.run(corpus=corpus)

    print(f"\n2. Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Duration: {result.duration_ms:.1f}ms")

    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for e in result.errors[:3]:
            print(f"     - [{e.code}] {e.message}")

    if result.ok and result.data:
        traj = result.data
        print(f"\n3. Trajectory Analysis:")
        print(f"   Type: {traj.trajectory_type}")
        print(f"   Stability: {traj.stability_score:.2f}")
        print(f"   Phases: {len(traj.phases)}")
        print(f"   Drift events: {len(traj.drift_events)}")
        print(f"   Core concepts: {traj.core_concepts[:5]}")

        if traj.phases:
            print(f"\n4. Research Phases:")
            for phase in traj.phases:
                concepts = ", ".join(phase.dominant_concepts[:3])
                print(f"   Phase {phase.phase_id}: {phase.start_year}-{phase.end_year}")
                print(f"      Focus: {concepts}")
                print(f"      Papers: {phase.paper_count}")

        if traj.drift_events:
            print(f"\n5. Drift Events:")
            for drift in traj.drift_events:
                print(f"   {drift.year}: magnitude={drift.drift_magnitude:.3f}")
                if drift.concepts_entering:
                    print(f"      +{drift.concepts_entering[:2]}")
                if drift.concepts_exiting:
                    print(f"      -{drift.concepts_exiting[:2]}")

        if traj.emerging_concepts:
            print(f"\n6. Emerging Concepts: {traj.emerging_concepts[:5]}")

        if traj.insights:
            print(f"\n7. Insights:")
            for insight in traj.insights[:3]:
                print(f"   - {insight}")
    else:
        print("   FAILED: No data returned")


if __name__ == "__main__":
    corpus = test_corpus_fetcher()
    test_trajectory_analyzer(corpus)
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
