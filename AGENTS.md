# RefNet Agent System

## Overview

RefNet uses a modular agent architecture where each agent performs a single well-defined task. Agents are composable, traceable, and fault-tolerant.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                              │
├─────────────────────────────────────────────────────────────────┤
│  SeedResolver          AuthorResolver                           │
│  (DOI/title → Paper)   (name → Author)                          │
└──────────┬─────────────────────┬────────────────────────────────┘
           │                     │
           ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GRAPH EXPANSION                             │
├─────────────────────────────────────────────────────────────────┤
│  CitationWalker        CorpusFetcher                            │
│  (Paper → refs/cites)  (Author → papers)                        │
└──────────┬─────────────────────┬────────────────────────────────┘
           │                     │
           ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ANALYSIS                                  │
├─────────────────────────────────────────────────────────────────┤
│  TrajectoryAnalyzer    CollaboratorMapper    TopicExtractor     │
│  (research drift)      (co-author network)   (theme extraction) │
│                                                                  │
│  GapDetector           RelevanceScorer                          │
│  (missing links)       (paper ranking)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Principles

1. **Single Responsibility** - One agent, one task
2. **Composable** - Agents can call other agents
3. **Traceable** - Every decision logged with reason
4. **Fallible** - Handle failures gracefully, never crash pipeline
5. **Testable** - Each agent can be tested in isolation

## Base Classes

All agents inherit from `Agent` and return `AgentResult`:

```python
from refnet.agents import Agent, AgentResult, AgentStatus

class AgentResult:
    status: AgentStatus  # SUCCESS, PARTIAL, FAILED, SKIPPED
    data: T              # the result data
    errors: List[AgentError]
    warnings: List[str]
    duration_ms: float
    api_calls: int
    trace: List[str]     # debug trace

    @property
    def ok(self) -> bool:  # True if usable data
```

---

## Entry Point Agents

### SeedResolver

Resolves paper identifiers to full Paper objects. The entry point for the pipeline.

**Supported inputs:**
- DOI: `10.1038/s41586-020-2649-2`
- DOI URL: `https://doi.org/10.1038/...`
- OpenAlex ID: `W2741809807`
- PMID: `32908161` or `PMID:32908161`
- arXiv: `2103.00020` or `https://arxiv.org/abs/2103.00020`
- Title: `"Attention Is All You Need"` (fuzzy search)

**Usage:**
```python
from refnet.agents import SeedResolver
from refnet.providers.openalex import OpenAlexProvider

provider = OpenAlexProvider(email="you@example.com")
resolver = SeedResolver(provider)

# resolve by DOI
result = resolver.run(query="10.1038/s41586-020-2649-2")

# resolve by title with hints
result = resolver.run(
    query="Attention Is All You Need",
    hint_year=2017,
    hint_author="Vaswani"
)

if result.ok:
    paper = result.data.paper
    print(f"Title: {paper.title}")
    print(f"Confidence: {result.data.confidence:.0%}")
```

**Output:** `ResolvedSeed`
- `paper`: Full Paper object
- `input_type`: Detected type (doi, title, pmid, etc.)
- `confidence`: Match confidence (1.0 for exact, lower for title search)
- `alternatives`: Other potential matches (for title search)

---

### AuthorResolver

Resolves author names to full profiles with disambiguation.

**Disambiguation strategies:**
1. Name similarity (handles initials, variations)
2. Affiliation hints
3. Coauthor network overlap
4. Paper context

**Usage:**
```python
from refnet.agents import AuthorResolver

resolver = AuthorResolver(provider)

result = resolver.run(
    name="C. W. Carter",
    affiliation_hint="UNC Chapel Hill",
    coauthor_hints=["Violetta Weinreb"]
)

if result.ok:
    author = result.data.author_info
    print(f"Resolved: {author.name}")
    print(f"OpenAlex ID: {author.openalex_id}")
    print(f"Confidence: {result.data.confidence:.0%}")

    # check alternatives for ambiguous cases
    for alt in result.data.alternatives:
        print(f"  Alternative: {alt.author_info.name} ({alt.confidence:.0%})")
```

**Output:** `ResolvedAuthor`
- `author_info`: AuthorInfo with name, IDs, affiliations, paper count
- `confidence`: Resolution confidence
- `alternatives`: Other candidate matches
- `hints_used`: Which hints were applied

---

## Graph Expansion Agents

### CitationWalker

Fetches and classifies references (what paper cites) and citations (what cites paper).

**Reference classifications:**
- `foundational`: Highly cited, old, cited early in paper
- `methodological`: Methods papers, cited late in paper
- `self_cite`: Shared authors with source paper
- `regular`: Standard citation

**Citation classifications:**
- `extension`: Builds directly on the work (high concept overlap, recent)
- `application`: Applies methods to new domain
- `comparison`: Reviews or evaluates the work
- `self_cite`: Shared authors

**Usage:**
```python
from refnet.agents import CitationWalker

walker = CitationWalker(provider, max_references=50, max_citations=50)
result = walker.run(paper=seed_paper)

if result.ok:
    cites = result.data

    # references this paper cites
    print(f"References: {cites.reference_count}")
    print(f"  Foundational: {len(cites.foundational_refs)}")
    print(f"  Methodological: {len(cites.methodological_refs)}")

    for ref in cites.references[:5]:
        print(f"  [{ref.ref_type}] {ref.paper.title}")
        print(f"    Importance: {ref.importance_score:.2f}")

    # papers that cite this paper
    print(f"Citations: {cites.citation_count}")
    for cite in cites.citations[:5]:
        print(f"  [{cite.cite_type}] {cite.paper.title}")
        print(f"    Years after: {cite.years_after}")
```

**Output:** `ClassifiedCitations`
- `references`: List of ClassifiedReference
- `citations`: List of ClassifiedCitation
- `foundational_refs`, `methodological_refs`: Paper IDs by type
- `key_references`, `key_citations`: Top papers to follow
- `insights`: Human-readable observations

---

### CorpusFetcher

Fetches all papers by an author (their corpus).

**Usage:**
```python
from refnet.agents import CorpusFetcher

fetcher = CorpusFetcher(provider, max_papers=200)
result = fetcher.run(author_id="A5009093641")

if result.ok:
    corpus = result.data
    print(f"Author: {corpus.name}")
    print(f"Papers: {len(corpus.papers)}")
    print(f"Year range: {corpus.year_range}")
    print(f"Top venues: {corpus.top_venues[:3]}")
    print(f"Top concepts: {corpus.top_concepts[:5]}")
    print(f"Top collaborators: {corpus.collaborators[:5]}")
```

**Output:** `AuthorCorpus`
- `papers`: List of Paper objects
- `year_range`: (min_year, max_year)
- `top_venues`, `top_concepts`, `collaborators`: Aggregated stats

---

## Analysis Agents

### TrajectoryAnalyzer

Analyzes an author's research trajectory over time.

**Features:**
- Phase detection (distinct research periods)
- Drift events (topic changes)
- Core vs emerging concepts
- ORCID integration (education, employment, work types)

**Usage:**
```python
from refnet.agents import TrajectoryAnalyzer
from refnet.providers.base import ORCIDProvider

analyzer = TrajectoryAnalyzer(orcid_provider=ORCIDProvider())
result = analyzer.run(corpus=corpus)

if result.ok:
    traj = result.data

    print(f"Trajectory type: {traj.trajectory_type}")  # focused, shifter, explorer
    print(f"Stability: {traj.stability_score:.2f}")
    print(f"Core concepts: {traj.core_concepts[:5]}")

    # research phases
    for phase in traj.phases:
        print(f"Phase {phase.phase_id}: {phase.start_year}-{phase.end_year}")
        print(f"  Focus: {phase.dominant_concepts[:3]}")
        print(f"  Papers: {phase.paper_count}")

    # drift events (major topic changes)
    for drift in traj.drift_events:
        print(f"Drift in {drift.year}: magnitude={drift.drift_magnitude:.2f}")
        print(f"  Entering: {drift.concepts_entering[:2]}")
        print(f"  Exiting: {drift.concepts_exiting[:2]}")

    # ORCID data if available
    if traj.education_history:
        for edu in traj.education_history:
            print(f"  {edu.degree} @ {edu.institution}")
```

**Output:** `AuthorTrajectory`
- `trajectory_type`: focused, shifter, explorer
- `stability_score`: 0-1 (higher = more stable)
- `phases`: List of ResearchPhase
- `drift_events`: List of DriftEvent
- `core_concepts`, `emerging_concepts`: Topic lists
- `education_history`, `employment_history`: From ORCID
- `work_types`: Journal articles, conference papers, preprints

---

### CollaboratorMapper

Maps an author's collaboration network.

**Usage:**
```python
from refnet.agents import CollaboratorMapper

mapper = CollaboratorMapper()
result = mapper.run(corpus=corpus)

if result.ok:
    network = result.data

    print(f"Collaboration style: {network.collaboration_style}")
    # solo, small_team, stable_group, large_network

    print(f"Total collaborators: {network.total_collaborators}")
    print(f"Long-term (3+ years): {network.long_term_collaborators[:5]}")
    print(f"Recent: {network.recent_collaborators[:5]}")

    # top collaborators
    for collab in network.collaborators[:5]:
        print(f"  {collab.name}: {collab.paper_count} papers")
        print(f"    Years: {collab.first_year}-{collab.last_year}")
        print(f"    Topics: {collab.shared_concepts[:3]}")

    # clusters by topic
    for cluster in network.clusters:
        print(f"  Cluster '{cluster.name}': {cluster.collaborator_names}")
```

**Output:** `CollaborationNetwork`
- `collaborators`: List of Collaborator with metrics
- `clusters`: CollaboratorCluster grouped by topic
- `collaboration_style`: Classification
- `top_collaborators`, `long_term_collaborators`, `recent_collaborators`

---

### TopicExtractor

Extracts and analyzes topics from a paper collection.

**Sources:**
- OpenAlex concepts (primary)
- Title terms
- Abstract terms (if available)

**Usage:**
```python
from refnet.agents import TopicExtractor

extractor = TopicExtractor()
result = extractor.run(papers=corpus.papers)

if result.ok:
    analysis = result.data

    print(f"Total topics: {len(analysis.topics)}")
    print(f"Core topics: {analysis.core_topics[:5]}")
    print(f"Emerging: {analysis.emerging_topics[:5]}")
    print(f"Declining: {analysis.declining_topics[:5]}")

    # topic details
    for topic in analysis.topics[:10]:
        trend_icon = {"emerging": "↑", "declining": "↓", "stable": "─", "new": "★"}
        print(f"  {topic.name}: {topic.paper_count} papers")
        print(f"    Trend: {topic.trend} {trend_icon.get(topic.trend, '')}")

    # topic clusters
    for cluster in analysis.topic_clusters:
        print(f"  Cluster '{cluster.name}': {cluster.topics[:3]}")
```

**Output:** `TopicAnalysis`
- `topics`: List of Topic with trends
- `core_topics`, `emerging_topics`, `declining_topics`
- `topic_clusters`: Related topic groups
- `topic_cooccurrence`: Which topics appear together

---

### GapDetector

Finds missing connections and unexplored areas in a paper collection.

**Gap types:**
- **Concept gaps**: Topics rarely combined but potentially related
- **Method gaps**: Techniques not applied to domains
- **Author gaps**: Researchers on similar topics who don't collaborate
- **Unexplored areas**: Nascent topic combinations

**Usage:**
```python
from refnet.agents import GapDetector

detector = GapDetector()
result = detector.run(papers=all_papers)

if result.ok:
    analysis = result.data

    # concept gaps
    for gap in analysis.concept_gaps[:5]:
        print(f"Gap: {gap.concept_a} × {gap.concept_b}")
        print(f"  Papers with A: {gap.papers_with_a_only}")
        print(f"  Papers with B: {gap.papers_with_b_only}")
        print(f"  Papers with both: {gap.papers_with_both}")
        print(f"  Gap score: {gap.gap_score:.2f}")

    # method gaps
    for gap in analysis.method_gaps[:3]:
        print(f"Method gap: {gap.method} → {gap.domain}")
        print(f"  {gap.potential}")

    # bridge papers
    for bridge in analysis.bridge_papers[:3]:
        print(f"Bridge: {bridge.title}")
        print(f"  Connects: {bridge.clusters_bridged}")

    # unexplored areas
    for area in analysis.unexplored_areas[:3]:
        print(f"Unexplored: {area.name}")
        print(f"  {area.description}")
```

**Output:** `GapAnalysis`
- `concept_gaps`: ConceptPair with gap scores
- `method_gaps`: MethodGap (technique → domain)
- `author_gaps`: AuthorGap (non-collaborating similar researchers)
- `bridge_papers`: Papers connecting multiple gaps
- `unexplored_areas`: Nascent combinations

---

### RelevanceScorer

Scores paper relevance to a query context.

**Score components:**
- Concept overlap (35%)
- Author overlap (15%)
- Citation importance (20%)
- Recency (15%)
- Quality signals (15%)

**Usage:**
```python
from refnet.agents import RelevanceScorer
from refnet.agents.relevance_scorer import ScoringContext

scorer = RelevanceScorer()

context = ScoringContext(
    seed_papers=[seed1, seed2],
    target_concepts=["aminoacyl tRNA synthetase", "genetic code"],
    target_authors=["Charles W. Carter"],
    min_year=2015
)

# single paper
result = scorer.execute(paper, context)
if result.ok:
    score = result.data
    print(f"Score: {score.score:.2f}")
    print(f"Explanation: {score.explanation}")

# batch scoring
result = scorer.score_batch(papers, context)
if result.ok:
    for score in result.data:  # sorted by relevance
        if score.is_highly_relevant:
            print(f"★ {score.title}: {score.score:.2f}")
```

**Output:** `RelevanceScore`
- `score`: Overall relevance (0-1)
- `concept_score`, `author_score`, `citation_score`, `recency_score`, `quality_score`
- `matching_concepts`, `matching_authors`
- `explanation`: Human-readable explanation
- `is_highly_relevant`: score >= 0.7
- `is_peripheral`: score < 0.3

---

## Complete Pipeline Example

```python
from refnet.agents import (
    SeedResolver, CitationWalker, AuthorResolver, CorpusFetcher,
    TrajectoryAnalyzer, CollaboratorMapper, TopicExtractor,
    GapDetector, RelevanceScorer
)
from refnet.agents.relevance_scorer import ScoringContext
from refnet.providers.openalex import OpenAlexProvider

provider = OpenAlexProvider(email="you@example.com")

# 1. resolve seed paper
resolver = SeedResolver(provider)
seed_result = resolver.run(query="10.1038/s41586-020-2649-2")
seed_paper = seed_result.data.paper

# 2. walk citations
walker = CitationWalker(provider)
cite_result = walker.run(paper=seed_paper)
refs = [r.paper for r in cite_result.data.references]
cites = [c.paper for c in cite_result.data.citations]

# 3. resolve key authors
author_resolver = AuthorResolver(provider)
key_authors = []
for author_name in seed_paper.authors[:3]:
    result = author_resolver.run(name=author_name)
    if result.ok:
        key_authors.append(result.data.author_info)

# 4. fetch author corpora
fetcher = CorpusFetcher(provider)
all_papers = list(refs) + list(cites)
for author in key_authors:
    result = fetcher.run(author_id=author.openalex_id)
    if result.ok:
        all_papers.extend(result.data.papers)

# 5. analyze trajectories
analyzer = TrajectoryAnalyzer()
for author in key_authors:
    corpus_result = fetcher.run(author_id=author.openalex_id)
    if corpus_result.ok:
        traj_result = analyzer.run(corpus=corpus_result.data)
        # process trajectory...

# 6. extract topics
extractor = TopicExtractor()
topic_result = extractor.run(papers=all_papers)
topics = topic_result.data

# 7. detect gaps
detector = GapDetector()
gap_result = detector.run(papers=all_papers)
gaps = gap_result.data

# 8. score and rank papers
scorer = RelevanceScorer()
context = ScoringContext(
    seed_papers=[seed_paper],
    target_concepts=topics.core_topics[:5]
)
score_result = scorer.score_batch(all_papers, context)
ranked_papers = score_result.data

# 9. output results
print("Key papers to read:")
for score in ranked_papers[:20]:
    if score.is_highly_relevant:
        print(f"  ★ {score.title}")

print("\nResearch gaps to explore:")
for gap in gaps.concept_gaps[:5]:
    print(f"  {gap.concept_a} × {gap.concept_b}")
```

---

## Error Handling

All agents return `AgentResult` which captures errors without crashing:

```python
result = agent.run(...)

if result.ok:
    # use result.data
    pass
elif result.status == AgentStatus.PARTIAL:
    # some data available, check warnings
    for warning in result.warnings:
        print(f"Warning: {warning}")
    # still use result.data with caution
elif result.status == AgentStatus.FAILED:
    # handle failure
    for error in result.errors:
        print(f"Error [{error.code}]: {error.message}")
        if error.recoverable:
            # can retry or skip
            pass
```

---

## Performance Notes

- **Rate limiting**: OpenAlexProvider handles rate limiting (10 req/sec)
- **Batch operations**: Use `score_batch()` for multiple papers
- **Caching**: Provider caches are per-session
- **API calls**: Check `result.api_calls` for usage tracking

---

## Testing

Run the test suite:

```bash
python test_agents.py
```

This tests all agents with real data (Charles W. Carter, aaRS researcher).
