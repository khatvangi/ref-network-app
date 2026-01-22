# RefNet Verification & Domain Intelligence

## Overview

Two complementary systems to make RefNet robust:

1. **Verification System** - Ensures agents do what they claim
2. **Domain Intelligence** - Guides search with field knowledge

---

## Part 1: Verification Architecture

### Why Verification?

Agents can fail silently:
- Return plausible-looking but wrong data
- Miss important results
- Make incorrect associations
- Return stale or incomplete data

We can't anticipate all failure modes, so we need defense in depth.

### Four Verification Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    VERIFICATION LAYERS                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: SELF-CHECKS (in each agent, immediate)            │
│  ├── Schema validation (required fields present)            │
│  ├── Type checks (strings are strings, ints are ints)       │
│  └── Basic sanity (non-empty, non-null where required)      │
│                                                              │
│  Layer 2: SEMANTIC VERIFICATION (Verifier agent)            │
│  ├── Cross-reference checks (do relationships hold?)        │
│  ├── Consistency checks (do numbers add up?)                │
│  └── Plausibility checks (is this reasonable?)              │
│                                                              │
│  Layer 3: EXTERNAL VALIDATION (optional, expensive)         │
│  ├── Cross-source verification (check against other APIs)   │
│  └── Sample spot-checks (randomly verify subset)            │
│                                                              │
│  Layer 4: ANOMALY DETECTION (learn over time)               │
│  ├── Statistical bounds (is this within normal range?)      │
│  └── Pattern detection (have we seen this failure before?)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Core Data Structures

```python
@dataclass
class Check:
    """single verification check."""
    name: str           # "doi_matches", "author_in_papers"
    passed: bool
    expected: Any       # what we expected
    actual: Any         # what we got
    severity: str       # "error", "warning", "info"
    message: str        # human explanation

@dataclass
class VerificationResult:
    """result of verifying an agent's output."""
    agent_name: str
    input_summary: str        # what was the input
    passed: bool              # all critical checks passed
    confidence: float         # 0.0 - 1.0 overall confidence
    checks: List[Check]       # all checks run
    errors: List[Check]       # severity=error that failed
    warnings: List[Check]     # severity=warning that failed
    suggestions: List[str]    # how to fix issues
```

---

## Part 2: Per-Agent Verification Checks

### SeedResolver

Resolves DOI/title to full paper.

| Check | Severity | Description |
|-------|----------|-------------|
| `doi_matches` | error | If DOI input, result DOI must match |
| `title_similarity` | warning | If title input, result title should be similar |
| `has_required_fields` | error | Paper must have id, title |
| `year_plausible` | warning | Year between 1900 and current+1 |
| `has_authors` | warning | Paper should have at least 1 author |

```python
def verify_seed_resolver(input_query: str, result: Paper) -> List[Check]:
    checks = []

    # doi_matches
    if input_query.startswith("10."):
        checks.append(Check(
            name="doi_matches",
            passed=result.doi and input_query.lower() in result.doi.lower(),
            expected=input_query,
            actual=result.doi,
            severity="error",
            message="DOI in result should match input DOI"
        ))

    # year_plausible
    current_year = datetime.now().year
    checks.append(Check(
        name="year_plausible",
        passed=result.year is None or (1900 <= result.year <= current_year + 1),
        expected=f"1900-{current_year+1}",
        actual=result.year,
        severity="warning",
        message="Publication year should be plausible"
    ))

    # ... more checks
    return checks
```

### CitationWalker

Fetches references and citations for a paper.

| Check | Severity | Description |
|-------|----------|-------------|
| `has_references` | warning | Should find at least some references |
| `has_citations` | info | May have citations (unless very new) |
| `ref_count_reasonable` | warning | Refs should be < 500 (sanity) |
| `refs_are_older` | warning | Most refs should predate the paper |
| `cites_are_newer` | warning | Most citations should postdate the paper |
| `no_self_reference` | error | Paper shouldn't cite itself in refs |

```python
def verify_citation_walker(seed_paper: Paper, result: Citations) -> List[Check]:
    checks = []

    # refs_are_older
    if seed_paper.year and result.references:
        older_refs = [r for r in result.references
                      if r.paper.year and r.paper.year <= seed_paper.year]
        ratio = len(older_refs) / len(result.references)
        checks.append(Check(
            name="refs_are_older",
            passed=ratio >= 0.8,
            expected=">=80% refs older than seed",
            actual=f"{ratio*100:.0f}%",
            severity="warning",
            message="References should mostly predate the citing paper"
        ))

    # no_self_reference
    ref_ids = {r.paper.id for r in result.references}
    checks.append(Check(
        name="no_self_reference",
        passed=seed_paper.id not in ref_ids,
        expected="seed not in refs",
        actual="seed in refs" if seed_paper.id in ref_ids else "ok",
        severity="error",
        message="Paper should not appear in its own references"
    ))

    return checks
```

### AuthorResolver

Resolves author name to profile with IDs.

| Check | Severity | Description |
|-------|----------|-------------|
| `name_matches` | warning | Result name should be similar to query |
| `has_openalex_id` | error | Must have OpenAlex ID |
| `has_papers` | warning | Author should have at least 1 paper |
| `paper_count_positive` | error | Paper count should be > 0 |
| `context_match` | warning | If context paper given, author should be on it |

```python
def verify_author_resolver(
    query_name: str,
    context_paper: Optional[Paper],
    result: AuthorInfo
) -> List[Check]:
    checks = []

    # name_matches (fuzzy)
    name_similarity = compute_name_similarity(query_name, result.name)
    checks.append(Check(
        name="name_matches",
        passed=name_similarity >= 0.7,
        expected=query_name,
        actual=result.name,
        severity="warning",
        message=f"Name similarity: {name_similarity:.0%}"
    ))

    # context_match
    if context_paper and context_paper.authors:
        author_on_paper = any(
            compute_name_similarity(result.name, a) >= 0.8
            for a in context_paper.authors
        )
        checks.append(Check(
            name="context_match",
            passed=author_on_paper,
            expected="author on context paper",
            actual="found" if author_on_paper else "not found",
            severity="warning",
            message="Resolved author should appear on context paper"
        ))

    return checks
```

### CorpusFetcher

Gets all papers by an author.

| Check | Severity | Description |
|-------|----------|-------------|
| `papers_not_empty` | error | Should return at least 1 paper |
| `author_in_papers` | error | Author ID should appear in each paper |
| `count_matches_profile` | warning | Paper count ~= author's reported count |
| `years_span_reasonable` | warning | Years should span reasonable range |
| `no_duplicates` | warning | No duplicate paper IDs |

```python
def verify_corpus_fetcher(
    author_id: str,
    expected_count: int,
    result: AuthorCorpus
) -> List[Check]:
    checks = []

    # author_in_papers (sample check - verify 10 random papers)
    sample = random.sample(result.papers, min(10, len(result.papers)))
    papers_with_author = 0
    for paper in sample:
        # would need to check paper's author IDs
        # this is expensive, so we sample
        pass

    # count_matches_profile
    ratio = len(result.papers) / expected_count if expected_count > 0 else 0
    checks.append(Check(
        name="count_matches_profile",
        passed=0.5 <= ratio <= 1.5,
        expected=f"~{expected_count} papers",
        actual=f"{len(result.papers)} papers",
        severity="warning",
        message="Corpus size should approximate author's paper count"
    ))

    # no_duplicates
    ids = [p.id for p in result.papers]
    unique_ids = set(ids)
    checks.append(Check(
        name="no_duplicates",
        passed=len(ids) == len(unique_ids),
        expected=f"{len(ids)} unique",
        actual=f"{len(unique_ids)} unique ({len(ids)-len(unique_ids)} dupes)",
        severity="warning",
        message="Corpus should not contain duplicate papers"
    ))

    return checks
```

### TrajectoryAnalyzer

Computes author's research drift over time.

| Check | Severity | Description |
|-------|----------|-------------|
| `phases_cover_range` | warning | Phases should cover corpus year range |
| `phases_non_overlapping` | error | Phase years shouldn't overlap |
| `phases_have_papers` | warning | Each phase should have ≥1 paper |
| `concepts_exist` | warning | Dominant concepts should appear in phase papers |
| `drift_events_valid` | warning | Drift events should be between phases |

```python
def verify_trajectory_analyzer(
    corpus: AuthorCorpus,
    result: TrajectoryAnalysis
) -> List[Check]:
    checks = []

    # phases_cover_range
    if corpus.papers:
        corpus_years = [p.year for p in corpus.papers if p.year]
        if corpus_years:
            corpus_min, corpus_max = min(corpus_years), max(corpus_years)
            phase_min = min(p.start_year for p in result.phases) if result.phases else 0
            phase_max = max(p.end_year for p in result.phases) if result.phases else 0
            checks.append(Check(
                name="phases_cover_range",
                passed=phase_min <= corpus_min + 2 and phase_max >= corpus_max - 1,
                expected=f"{corpus_min}-{corpus_max}",
                actual=f"{phase_min}-{phase_max}",
                severity="warning",
                message="Phases should cover the corpus year range"
            ))

    # phases_non_overlapping
    if len(result.phases) >= 2:
        overlaps = []
        sorted_phases = sorted(result.phases, key=lambda p: p.start_year)
        for i in range(len(sorted_phases) - 1):
            if sorted_phases[i].end_year > sorted_phases[i+1].start_year:
                overlaps.append((sorted_phases[i], sorted_phases[i+1]))
        checks.append(Check(
            name="phases_non_overlapping",
            passed=len(overlaps) == 0,
            expected="no overlaps",
            actual=f"{len(overlaps)} overlaps",
            severity="error",
            message="Career phases should not overlap"
        ))

    return checks
```

### CollaboratorMapper

Finds co-author network.

| Check | Severity | Description |
|-------|----------|-------------|
| `has_collaborators` | warning | Should find at least 1 collaborator |
| `no_self_collab` | error | Author shouldn't collaborate with self |
| `collab_counts_valid` | warning | Collab paper counts ≤ total papers |
| `top_collab_verifiable` | warning | Top collaborator appears in actual papers |

```python
def verify_collaborator_mapper(
    author_name: str,
    corpus: AuthorCorpus,
    result: CollaborationNetwork
) -> List[Check]:
    checks = []

    # no_self_collab
    self_collab = any(
        compute_name_similarity(author_name, c) >= 0.9
        for c in result.top_collaborators
    )
    checks.append(Check(
        name="no_self_collab",
        passed=not self_collab,
        expected="author not in collaborators",
        actual="self found" if self_collab else "ok",
        severity="error",
        message="Author should not appear as own collaborator"
    ))

    # top_collab_verifiable
    if result.top_collaborators:
        top_collab = result.top_collaborators[0]
        papers_with_collab = sum(
            1 for p in corpus.papers
            if p.authors and any(
                compute_name_similarity(top_collab, a) >= 0.8
                for a in p.authors
            )
        )
        expected_count = result.collaborator_papers.get(top_collab, 0)
        checks.append(Check(
            name="top_collab_verifiable",
            passed=papers_with_collab >= expected_count * 0.8,
            expected=f"{expected_count} papers",
            actual=f"{papers_with_collab} papers",
            severity="warning",
            message="Top collaborator paper count should be verifiable"
        ))

    return checks
```

### TopicExtractor

Extracts themes from paper collection.

| Check | Severity | Description |
|-------|----------|-------------|
| `has_topics` | warning | Should extract at least 1 topic |
| `core_topics_frequent` | warning | Core topics should appear in ≥10% of papers |
| `emerging_are_recent` | warning | Emerging topics should be weighted toward recent years |
| `declining_are_older` | warning | Declining topics should be weighted toward older years |
| `no_duplicate_topics` | warning | Same topic shouldn't be core AND declining |

```python
def verify_topic_extractor(
    papers: List[Paper],
    result: TopicAnalysis
) -> List[Check]:
    checks = []

    # core_topics_frequent
    if result.core_topics and papers:
        # verify top core topic appears frequently
        top_topic = result.core_topics[0].lower()
        papers_with_topic = sum(
            1 for p in papers
            if p.title and top_topic in p.title.lower()
            or p.abstract and top_topic in p.abstract.lower()
        )
        frequency = papers_with_topic / len(papers)
        checks.append(Check(
            name="core_topics_frequent",
            passed=frequency >= 0.05,  # at least 5%
            expected=">=5% frequency",
            actual=f"{frequency*100:.1f}%",
            severity="warning",
            message=f"Core topic '{top_topic}' should appear frequently"
        ))

    # no_duplicate_topics
    core_set = set(t.lower() for t in result.core_topics)
    declining_set = set(t.lower() for t in result.declining_topics)
    overlap = core_set & declining_set
    checks.append(Check(
        name="no_duplicate_topics",
        passed=len(overlap) == 0,
        expected="no overlap",
        actual=f"{len(overlap)} overlapping: {overlap}" if overlap else "ok",
        severity="warning",
        message="Topics shouldn't be both core and declining"
    ))

    return checks
```

### GapDetector

Finds missing connections and unexplored areas.

| Check | Severity | Description |
|-------|----------|-------------|
| `gap_concepts_exist` | error | Gap concepts must exist in corpus |
| `gaps_are_actual_gaps` | warning | Gap pairs should have low co-occurrence |
| `bridge_papers_valid` | warning | Bridge papers should mention both concepts |
| `gap_scores_bounded` | warning | Gap scores should be 0.0-1.0 |

```python
def verify_gap_detector(
    papers: List[Paper],
    result: GapAnalysis
) -> List[Check]:
    checks = []

    # gap_concepts_exist
    all_concepts = set()
    for p in papers:
        if p.concepts:
            all_concepts.update(c.lower() for c in p.concepts)

    for gap in result.concept_gaps[:3]:  # check top 3
        a_exists = gap.concept_a.lower() in all_concepts
        b_exists = gap.concept_b.lower() in all_concepts
        checks.append(Check(
            name=f"gap_concepts_exist_{gap.concept_a}_{gap.concept_b}",
            passed=a_exists and b_exists,
            expected="both concepts in corpus",
            actual=f"a:{a_exists}, b:{b_exists}",
            severity="error",
            message=f"Gap concepts must exist in corpus"
        ))

    # gaps_are_actual_gaps (verify low co-occurrence)
    if result.concept_gaps:
        top_gap = result.concept_gaps[0]
        co_occur = sum(
            1 for p in papers
            if p.concepts and
            top_gap.concept_a.lower() in [c.lower() for c in p.concepts] and
            top_gap.concept_b.lower() in [c.lower() for c in p.concepts]
        )
        checks.append(Check(
            name="gaps_are_actual_gaps",
            passed=co_occur <= len(papers) * 0.05,  # <=5% co-occurrence
            expected="<=5% co-occurrence",
            actual=f"{co_occur}/{len(papers)} ({co_occur/len(papers)*100:.1f}%)",
            severity="warning",
            message="Top gap should have low co-occurrence"
        ))

    return checks
```

### RelevanceScorer

Scores paper relevance to context.

| Check | Severity | Description |
|-------|----------|-------------|
| `scores_bounded` | error | All scores should be 0.0-1.0 |
| `component_scores_sum` | warning | Component scores should justify total |
| `highly_relevant_justified` | warning | If highly_relevant=True, score should be high |
| `explanation_present` | warning | Should have explanation for score |

```python
def verify_relevance_scorer(
    paper: Paper,
    context: ScoringContext,
    result: RelevanceScore
) -> List[Check]:
    checks = []

    # scores_bounded
    scores = [
        result.score, result.concept_score,
        result.author_score, result.citation_score, result.recency_score
    ]
    all_bounded = all(0.0 <= s <= 1.0 for s in scores)
    checks.append(Check(
        name="scores_bounded",
        passed=all_bounded,
        expected="all scores 0.0-1.0",
        actual=f"scores: {scores}",
        severity="error",
        message="All scores must be between 0.0 and 1.0"
    ))

    # highly_relevant_justified
    if result.is_highly_relevant:
        checks.append(Check(
            name="highly_relevant_justified",
            passed=result.score >= 0.7,
            expected="score >= 0.7",
            actual=f"score = {result.score}",
            severity="warning",
            message="Highly relevant papers should have high scores"
        ))

    return checks
```

---

## Part 3: FieldResolver & Domain Intelligence

### The Problem

OpenAlex has millions of papers. Searching everything:
- Returns too much noise
- Misses field-specific context
- Can't distinguish important from obscure journals
- No understanding of field structure

### The Solution: Domain-Aware Search

```
Seed Paper: "Palladium-catalyzed C-H activation..."
     │
     ▼
FieldResolver identifies: Organic Chemistry / Catalysis
     │
     ▼
Domain knowledge activates:
  ├── Tier 1 Journals: JACS, Angew. Chem., Nature Chem.
  ├── Tier 2 Journals: JOC, Org. Lett., Chem. Sci.
  ├── Known Leaders: Hartwig, Buchwald, Yu, Sanford
  └── Key Concepts: cross-coupling, C-H activation, ligand
     │
     ▼
CHECKPOINT: "I think this is Organic Chemistry / Catalysis.
             Key journals: JACS, Angew. Chem.
             Am I on the right track?"
     │
     ├── User: "Yes" → Continue with domain filter
     └── User: "No, it's biochemistry" → Re-resolve
```

### FieldProfile Data Structure

```python
@dataclass
class JournalTier:
    """journals grouped by importance."""
    tier: int                    # 1, 2, or 3
    journals: List[str]          # journal names or ISSNs
    description: str             # "top-tier", "specialty", "general"

@dataclass
class FieldProfile:
    """domain knowledge for a research field."""
    name: str                    # "Organic Chemistry"
    aliases: List[str]           # ["orgo", "synthetic chemistry"]
    parent_field: str            # "Chemistry"

    # journal tiers
    tier1_journals: List[str]    # Nature, Science, JACS, Angew. Chem.
    tier2_journals: List[str]    # JOC, Org. Lett., specialty journals
    tier3_sources: List[str]     # everything else

    # bootstrap knowledge
    known_leaders: List[str]     # famous researchers in field
    key_concepts: List[str]      # domain terminology

    # OpenAlex mapping
    openalex_concepts: List[str] # OpenAlex concept IDs

    # search hints
    exclude_terms: List[str]     # terms that indicate wrong field
    require_terms: List[str]     # terms that should appear

@dataclass
class FieldResolution:
    """result of field identification."""
    primary_field: FieldProfile
    confidence: float            # 0.0 - 1.0
    secondary_fields: List[FieldProfile]  # related fields
    evidence: List[str]          # why we think this
    suggested_strategy: str      # "tier1_first", "author_centric", etc.
```

### FieldResolver Agent

```python
class FieldResolver(Agent):
    """
    identifies research field from seed and provides domain knowledge.

    strategies:
    1. Use OpenAlex concepts on seed paper
    2. Match journal to known field journals
    3. Match concepts in title/abstract to field terminology
    4. Ask user if uncertain
    """

    def run(
        self,
        seed_paper: Optional[Paper] = None,
        query: Optional[str] = None
    ) -> AgentResult[FieldResolution]:
        # 1. extract signals from seed
        signals = self._extract_signals(seed_paper, query)

        # 2. match against known fields
        candidates = self._match_fields(signals)

        # 3. rank by confidence
        ranked = self._rank_candidates(candidates)

        # 4. build resolution
        if ranked and ranked[0].confidence >= 0.7:
            return AgentResult.success(ranked[0])
        else:
            # low confidence - might need user input
            return AgentResult.partial(
                ranked[0] if ranked else None,
                warnings=["Low confidence field identification"]
            )
```

### Domain Knowledge Store

```python
# could be JSON/YAML file or database
FIELD_PROFILES = {
    "organic_chemistry": FieldProfile(
        name="Organic Chemistry",
        aliases=["synthetic chemistry", "orgo"],
        parent_field="Chemistry",
        tier1_journals=[
            "Journal of the American Chemical Society",
            "Angewandte Chemie",
            "Nature Chemistry",
            "Science",
            "Nature"
        ],
        tier2_journals=[
            "Journal of Organic Chemistry",
            "Organic Letters",
            "Chemical Science",
            "ACS Catalysis",
            "Chemistry - A European Journal"
        ],
        known_leaders=[
            "John F. Hartwig",
            "Stephen L. Buchwald",
            "Jin-Quan Yu",
            "Melanie S. Sanford"
        ],
        key_concepts=[
            "catalysis", "synthesis", "cross-coupling",
            "C-H activation", "asymmetric", "ligand"
        ],
        openalex_concepts=["C178790620"],  # Organic chemistry concept ID
        exclude_terms=["biological", "in vivo", "clinical"],
        require_terms=[]
    ),

    "biochemistry": FieldProfile(
        name="Biochemistry",
        aliases=["biological chemistry"],
        parent_field="Biology",
        tier1_journals=[
            "Nature", "Science", "Cell",
            "Nature Chemical Biology",
            "Journal of Biological Chemistry"
        ],
        # ... etc
    ),

    # ... more fields
}
```

### Tiered Search Strategy

```python
class TieredSearchStrategy:
    """
    search strategy that prioritizes high-impact sources.
    """

    def __init__(self, field_profile: FieldProfile):
        self.field = field_profile

    def search(
        self,
        provider: OpenAlexProvider,
        query_params: dict,
        target_count: int = 50
    ) -> List[Paper]:
        results = []

        # Tier 1: High-impact journals first
        tier1_results = provider.search_papers(
            **query_params,
            journals=self.field.tier1_journals,
            limit=target_count
        )
        results.extend(tier1_results)

        # if we have enough, stop
        if len(results) >= target_count:
            return results[:target_count]

        # Tier 2: Specialty journals
        remaining = target_count - len(results)
        tier2_results = provider.search_papers(
            **query_params,
            journals=self.field.tier2_journals,
            limit=remaining
        )
        results.extend(tier2_results)

        # Tier 3: Everything else (if still need more)
        if len(results) < target_count * 0.5:
            # only go to tier 3 if we have very few results
            remaining = target_count - len(results)
            tier3_results = provider.search_papers(
                **query_params,
                limit=remaining
            )
            results.extend(tier3_results)

        return results
```

### User Checkpoint

```python
@dataclass
class FieldCheckpoint:
    """checkpoint for user verification."""
    identified_field: str
    confidence: float
    evidence: List[str]
    suggested_journals: List[str]
    suggested_authors: List[str]
    question: str  # "Is this the right field?"

def create_checkpoint(resolution: FieldResolution) -> FieldCheckpoint:
    return FieldCheckpoint(
        identified_field=resolution.primary_field.name,
        confidence=resolution.confidence,
        evidence=resolution.evidence,
        suggested_journals=resolution.primary_field.tier1_journals[:5],
        suggested_authors=resolution.primary_field.known_leaders[:5],
        question=f"I identified this as {resolution.primary_field.name}. "
                 f"Key journals: {', '.join(resolution.primary_field.tier1_journals[:3])}. "
                 f"Is this correct?"
    )
```

---

## Part 4: Integration

### Pipeline with Verification & Domain Intelligence

```
User Input: DOI or Query
      │
      ▼
┌─────────────────────────────────────────┐
│ SeedResolver                             │
│   └── Self-checks (Layer 1)             │
│   └── Verifier checks (Layer 2)         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ FieldResolver                            │
│   └── Identify field from seed          │
│   └── Load domain knowledge             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ CHECKPOINT: User Verification            │
│   "Is this Organic Chemistry?"          │
│   └── Yes → Continue                    │
│   └── No → Re-resolve with feedback     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ TieredSearch (Domain-Aware)              │
│   └── Tier 1 journals first             │
│   └── Tier 2 if needed                  │
│   └── Tier 3 only if sparse             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Agent Pipeline (with verification)       │
│   CitationWalker → verify               │
│   AuthorResolver → verify               │
│   CorpusFetcher → verify                │
│   TrajectoryAnalyzer → verify           │
│   TopicExtractor → verify               │
│   GapDetector → verify                  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Final Verification                       │
│   └── Cross-check all results           │
│   └── Flag inconsistencies              │
│   └── Compute overall confidence        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         LiteratureAnalysis
         (with confidence scores)
```

### Verification Report

```python
@dataclass
class VerificationReport:
    """overall verification report for pipeline run."""
    overall_passed: bool
    overall_confidence: float

    # per-agent results
    agent_results: Dict[str, VerificationResult]

    # summary
    total_checks: int
    passed_checks: int
    failed_errors: int
    failed_warnings: int

    # issues
    critical_issues: List[str]
    warnings: List[str]
    suggestions: List[str]

    def summary(self) -> str:
        return (
            f"Verification: {'PASSED' if self.overall_passed else 'FAILED'}\n"
            f"Confidence: {self.overall_confidence:.0%}\n"
            f"Checks: {self.passed_checks}/{self.total_checks} passed\n"
            f"Errors: {self.failed_errors}, Warnings: {self.failed_warnings}"
        )
```

---

## Implementation Order

1. **Core verification framework**
   - Check, VerificationResult dataclasses
   - Verifier base class

2. **Per-agent verification** (one at a time)
   - SeedResolver checks
   - CitationWalker checks
   - AuthorResolver checks
   - CorpusFetcher checks
   - TrajectoryAnalyzer checks
   - CollaboratorMapper checks
   - TopicExtractor checks
   - GapDetector checks
   - RelevanceScorer checks

3. **FieldResolver agent**
   - FieldProfile data structure
   - Field matching logic
   - Domain knowledge store (start with 5-10 fields)

4. **Tiered search strategy**
   - Integrate with provider
   - Journal filtering

5. **User checkpoint**
   - CLI integration
   - Pipeline integration

6. **Integration**
   - Wire verification into pipeline
   - Wire field resolution into pipeline
   - Verification report generation
