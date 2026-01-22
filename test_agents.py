#!/usr/bin/env python3
"""
test agents - verify all agents work with real data.

tests with Charles W. Carter Jr. (aaRS researcher from seed papers).
"""

import sys
sys.path.insert(0, '/storage/kiran-stuff/ref-network-app')

from refnet.agents import (
    CorpusFetcher, TrajectoryAnalyzer,
    CollaboratorMapper, TopicExtractor, GapDetector,
    AgentStatus
)
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
            for insight in traj.insights:
                print(f"   - {insight}")

        # show ORCID data if available
        if traj.education_history:
            print(f"\n8. Education (from ORCID):")
            for edu in traj.education_history[:3]:
                degree = edu.degree or "Degree"
                years = f"{edu.start_year or '?'}-{edu.end_year or 'present'}"
                print(f"   - {degree} @ {edu.institution} ({years})")

        if traj.employment_history:
            print(f"\n9. Employment (from ORCID):")
            for emp in traj.employment_history[:3]:
                role = emp.role or "Position"
                status = "(current)" if emp.is_current else ""
                years = f"{emp.start_year or '?'}-{emp.end_year or 'present'}"
                print(f"   - {role} @ {emp.organization} {years} {status}")

        if traj.work_types:
            print(f"\n10. Work Types:")
            wt = traj.work_types
            print(f"    Journal articles: {wt.journal_articles}")
            print(f"    Conference papers: {wt.conference_papers}")
            print(f"    Preprints: {wt.preprints}")
    else:
        print("   FAILED: No data returned")


def test_collaborator_mapper(corpus):
    """test CollaboratorMapper with corpus from previous test."""
    print("\n" + "=" * 60)
    print("TEST: CollaboratorMapper")
    print("=" * 60)

    if not corpus:
        print("   SKIPPED: No corpus data")
        return

    print(f"\n1. Mapping collaborations for {corpus.name} ({len(corpus.papers)} papers)")

    mapper = CollaboratorMapper()
    result = mapper.run(corpus=corpus)

    print(f"\n2. Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Duration: {result.duration_ms:.1f}ms")

    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for e in result.errors[:3]:
            print(f"     - [{e.code}] {e.message}")

    if result.ok and result.data:
        network = result.data
        print(f"\n3. Network Stats:")
        print(f"   Total collaborators: {network.total_collaborators}")
        print(f"   Collaborative papers: {network.total_collaborative_papers}")
        print(f"   Solo papers: {network.solo_papers}")
        print(f"   Avg authors/paper: {network.avg_authors_per_paper:.1f}")
        print(f"   Collaboration style: {network.collaboration_style}")

        if network.top_collaborators:
            print(f"\n4. Top Collaborators:")
            for name in network.top_collaborators[:5]:
                collab = next((c for c in network.collaborators if c.name == name), None)
                if collab:
                    print(f"   - {name}: {collab.paper_count} papers, {collab.collaboration_years} years")

        if network.long_term_collaborators:
            print(f"\n5. Long-term Collaborators (3+ years):")
            for name in network.long_term_collaborators[:5]:
                print(f"   - {name}")

        if network.clusters:
            print(f"\n6. Collaboration Clusters:")
            for cluster in network.clusters[:3]:
                print(f"   - {cluster.name}: {len(cluster.collaborator_names)} people")

        if network.insights:
            print(f"\n7. Insights:")
            for insight in network.insights[:3]:
                print(f"   - {insight}")
    else:
        print("   FAILED: No data returned")


def test_topic_extractor(corpus):
    """test TopicExtractor with corpus from previous test."""
    print("\n" + "=" * 60)
    print("TEST: TopicExtractor")
    print("=" * 60)

    if not corpus:
        print("   SKIPPED: No corpus data")
        return

    print(f"\n1. Extracting topics from {len(corpus.papers)} papers")

    extractor = TopicExtractor()
    result = extractor.run(papers=corpus.papers)

    print(f"\n2. Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Duration: {result.duration_ms:.1f}ms")

    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for e in result.errors[:3]:
            print(f"     - [{e.code}] {e.message}")

    if result.ok and result.data:
        analysis = result.data
        print(f"\n3. Topic Analysis:")
        print(f"   Total topics: {len(analysis.topics)}")
        print(f"   Core topics: {analysis.core_topics[:5]}")
        print(f"   Emerging topics: {analysis.emerging_topics[:5]}")
        print(f"   Declining topics: {analysis.declining_topics[:5]}")

        if analysis.topics:
            print(f"\n4. Top Topics by Weight:")
            for topic in analysis.topics[:10]:
                trend_indicator = {
                    'emerging': '↑',
                    'declining': '↓',
                    'new': '★',
                    'stable': '─'
                }.get(topic.trend, '?')
                print(f"   - {topic.name}: {topic.paper_count} papers, {topic.trend} {trend_indicator}")

        if analysis.topic_clusters:
            print(f"\n5. Topic Clusters:")
            for cluster in analysis.topic_clusters[:5]:
                topics = ', '.join(cluster.topics[:3])
                print(f"   - {cluster.name}: {topics}")

        if analysis.insights:
            print(f"\n6. Insights:")
            for insight in analysis.insights[:3]:
                print(f"   - {insight}")
    else:
        print("   FAILED: No data returned")


def test_gap_detector(corpus):
    """test GapDetector with corpus from previous test."""
    print("\n" + "=" * 60)
    print("TEST: GapDetector")
    print("=" * 60)

    if not corpus:
        print("   SKIPPED: No corpus data")
        return

    print(f"\n1. Detecting gaps in {len(corpus.papers)} papers")

    detector = GapDetector()
    result = detector.run(papers=corpus.papers)

    print(f"\n2. Result:")
    print(f"   Status: {result.status.value}")
    print(f"   Duration: {result.duration_ms:.1f}ms")

    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for e in result.errors[:3]:
            print(f"     - [{e.code}] {e.message}")

    if result.ok and result.data:
        analysis = result.data
        print(f"\n3. Gap Analysis:")
        print(f"   Total concepts: {analysis.total_concepts}")
        print(f"   Total authors: {analysis.total_authors}")
        print(f"   Concept gaps: {len(analysis.concept_gaps)}")
        print(f"   Method gaps: {len(analysis.method_gaps)}")
        print(f"   Bridge papers: {len(analysis.bridge_papers)}")
        print(f"   Unexplored areas: {len(analysis.unexplored_areas)}")

        if analysis.concept_gaps:
            print(f"\n4. Top Concept Gaps:")
            for gap in analysis.concept_gaps[:5]:
                print(f"   - {gap.concept_a} × {gap.concept_b}")
                print(f"     ({gap.papers_with_a_only}+{gap.papers_with_both}) vs ({gap.papers_with_b_only}+{gap.papers_with_both}), gap={gap.gap_score:.2f}")

        if analysis.method_gaps:
            print(f"\n5. Method Gaps:")
            for gap in analysis.method_gaps[:3]:
                print(f"   - {gap.method} → {gap.domain}")
                print(f"     {gap.method_papers} method papers, {gap.domain_papers} domain papers, {gap.combined_papers} combined")

        if analysis.bridge_papers:
            print(f"\n6. Bridge Papers:")
            for bridge in analysis.bridge_papers[:3]:
                print(f"   - [{bridge.year}] {bridge.title[:50]}...")
                print(f"     Bridges: {', '.join(bridge.clusters_bridged[:2])}")

        if analysis.unexplored_areas:
            print(f"\n7. Unexplored Areas:")
            for area in analysis.unexplored_areas[:3]:
                print(f"   - {area.name}: {area.existing_papers} papers")
                print(f"     {area.description[:60]}...")

        if analysis.insights:
            print(f"\n8. Insights:")
            for insight in analysis.insights[:3]:
                print(f"   - {insight}")
    else:
        print("   FAILED: No data returned")


if __name__ == "__main__":
    corpus = test_corpus_fetcher()
    test_trajectory_analyzer(corpus)
    test_collaborator_mapper(corpus)
    test_topic_extractor(corpus)
    test_gap_detector(corpus)
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
