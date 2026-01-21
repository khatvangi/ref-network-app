# Bucket Expansion Model - Design Document

## Overview

The bucket expansion model implements "citation walking" where papers discovered through citations become seeds for further expansion. This creates recursive bucket-based discovery until the topic is exhausted.

## Core Concept

```
INITIAL SEED
     │
     ├─── INTRO REFS (first 25% of references)
     │         │
     │         └──► These papers cite foundational work
     │               that defines the problem space
     │
     └─── CITATIONS (papers citing the seed)
               │
               └──► These papers build on the seed,
                    represent the field's evolution

     Both directions form BUCKET₀
```

## Bidirectional Expansion

Unlike traditional BFS/DFS by depth, bucket expansion treats discovery bidirectionally:

```
                         SEED PAPER
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       BACKWARD (refs)                  FORWARD (cites)
       "What does seed                  "Who uses seed's
        build upon?"                     findings?"
              │                               │
              ▼                               ▼
       intro_refs[:k]                   all_citations
       (k = 25% clamped)                (with relevance gate)
              │                               │
              └───────────┬───────────────────┘
                          ▼
                    BUCKET₀ = {papers}
                          │
                    ┌─────┴─────┐
                    ▼           ▼
               For each paper in bucket:
                    │
         ┌─────────┴─────────┐
         ▼                   ▼
    intro_refs          citations
         │                   │
         └─────────┬─────────┘
                   ▼
             BUCKET₁ = {new papers}
                   │
                  ...
                   ▼
             BUCKETₙ (until exhaustion)
```

## Key Differences from Current Implementation

| Aspect | Current (Depth-based) | Bucket-based |
|--------|----------------------|--------------|
| Stopping | Fixed max_depth=3 | Relevance exhaustion |
| Expansion unit | Single paper | Bucket of papers |
| Discovery scope | Limited by depth | Naturally bounded by topic |
| Branches | Treated equally | Intro refs get priority (weight 2.0) |
| Direction | Sequential depth levels | Bidirectional per paper |

## Data Structures

### Bucket
```python
@dataclass
class Bucket:
    generation: int                    # 0 = seed bucket, 1 = first expansion, etc.
    papers: List[Paper]                # papers in this bucket
    source_bucket_id: Optional[str]    # parent bucket (for lineage tracking)
    avg_relevance: float               # average relevance of papers in bucket
    created_at: datetime
```

### BucketExpansionState
```python
@dataclass
class BucketExpansionState:
    all_buckets: Dict[str, Bucket]     # bucket_id -> Bucket
    seen_paper_ids: Set[str]           # global deduplication
    relevance_history: List[float]     # rolling window for drift detection
    current_generation: int            # current bucket generation
    total_papers_discovered: int
    total_api_calls: int
```

## Stopping Conditions (Guardrails)

### 1. Relevance Decay (PRIMARY)
```python
def check_relevance_decay(bucket: Bucket, config: Config) -> bool:
    """
    Stop expanding a bucket if average relevance drops too low.
    This is branch-local, not global.
    """
    if bucket.avg_relevance < config.min_bucket_relevance:
        return True  # stop this branch
    return False
```

- `min_bucket_relevance` default: 0.15
- Per-bucket decision (allows some branches to die while others continue)

### 2. Topic Drift Kill-Switch (GLOBAL)
```python
def check_topic_drift(state: BucketExpansionState, config: Config) -> bool:
    """
    Emergency stop if overall discovery quality drops.
    Rolling window: last N papers discovered.
    """
    if len(state.relevance_history) < config.drift_window:
        return False

    recent = state.relevance_history[-config.drift_window:]
    relevant_ratio = sum(1 for r in recent if r >= config.min_relevance) / len(recent)

    if relevant_ratio < config.drift_kill_threshold:
        return True  # stop ALL expansion
    return False
```

- `drift_window` default: 30
- `drift_kill_threshold` default: 0.10 (if <10% relevant, stop)

### 3. Adaptive Depth
```python
def compute_adaptive_max_generation(bucket_0: Bucket, config: Config) -> int:
    """
    Let initial bucket inform depth expectation.
    Large initial bucket = expect more generations.
    Small initial bucket = may exhaust quickly.
    """
    base_depth = config.base_max_generations  # default: 10
    bucket_size = len(bucket_0.papers)

    if bucket_size < 10:
        return min(base_depth, 5)
    elif bucket_size < 50:
        return base_depth
    else:
        return base_depth + int(bucket_size / 50)  # +1 per 50 papers
```

- Start with generous depth, let data decide
- Cut down if relevance decays

### 4. API Budget
```python
def check_api_budget(state: BucketExpansionState, config: Config) -> bool:
    if state.total_api_calls >= config.max_api_calls:
        return True  # hard stop
    return False
```

- `max_api_calls` default: 2000
- Hard cap, non-negotiable

### 5. Natural Exhaustion
```python
def check_natural_exhaustion(new_bucket: Bucket) -> bool:
    """No new papers found = topic exhausted."""
    return len(new_bucket.papers) == 0
```

## Algorithm

```python
def expand_buckets_until_exhausted(
    initial_seeds: List[Paper],
    provider: PaperProvider,
    config: Config
) -> BucketExpansionState:

    state = BucketExpansionState()

    # Create BUCKET₀ from initial seeds
    bucket_0 = create_initial_bucket(initial_seeds)
    state.all_buckets[bucket_0.id] = bucket_0
    state.current_generation = 0

    # Compute adaptive max generations
    max_generations = compute_adaptive_max_generation(bucket_0, config)

    current_buckets = [bucket_0]

    while current_buckets:
        # Check global stopping conditions
        if check_api_budget(state, config):
            break
        if check_topic_drift(state, config):
            break
        if state.current_generation >= max_generations:
            break

        next_generation_buckets = []

        for bucket in current_buckets:
            # Check bucket-local relevance decay
            if check_relevance_decay(bucket, config):
                continue  # kill this branch, others may continue

            # Expand each paper in bucket
            new_bucket = expand_bucket(bucket, state, provider, config)

            if not check_natural_exhaustion(new_bucket):
                next_generation_buckets.append(new_bucket)
                state.all_buckets[new_bucket.id] = new_bucket

        current_buckets = next_generation_buckets
        state.current_generation += 1

    return state


def expand_bucket(
    bucket: Bucket,
    state: BucketExpansionState,
    provider: PaperProvider,
    config: Config
) -> Bucket:
    """Expand all papers in bucket → new bucket."""

    new_papers = []

    for paper in bucket.papers:
        # Get intro refs (first 25%)
        refs = provider.get_references(paper.id)
        state.total_api_calls += 1

        k = compute_intro_k(len(refs))  # 25% clamped 10-40, max 20
        for ref in refs[:k]:
            if ref.id not in state.seen_paper_ids:
                ref.relevance = compute_relevance(ref, state.topic)
                state.relevance_history.append(ref.relevance)

                if ref.relevance >= config.min_relevance:
                    new_papers.append(ref)
                    state.seen_paper_ids.add(ref.id)
                    state.total_papers_discovered += 1

        # Get citations
        cites = provider.get_citations(paper.id)
        state.total_api_calls += 1

        for cite in cites:
            if cite.id not in state.seen_paper_ids:
                cite.relevance = compute_relevance(cite, state.topic)
                state.relevance_history.append(cite.relevance)

                if cite.relevance >= config.min_relevance:
                    new_papers.append(cite)
                    state.seen_paper_ids.add(cite.id)
                    state.total_papers_discovered += 1

    # Create new bucket
    new_bucket = Bucket(
        generation=bucket.generation + 1,
        papers=new_papers,
        source_bucket_id=bucket.id,
        avg_relevance=sum(p.relevance for p in new_papers) / len(new_papers) if new_papers else 0.0
    )

    return new_bucket
```

## Configuration Parameters

```python
@dataclass
class BucketExpansionConfig:
    # Relevance thresholds
    min_relevance: float = 0.15           # minimum to include paper
    min_bucket_relevance: float = 0.15    # minimum bucket avg to continue branch

    # Topic drift
    drift_window: int = 30                # papers to consider for drift
    drift_kill_threshold: float = 0.10    # stop if <10% relevant

    # Adaptive depth
    base_max_generations: int = 10        # starting max generations

    # API budget
    max_api_calls: int = 2000             # hard cap

    # Intro hint
    intro_fraction: float = 0.25          # first 25% of refs
    max_intro_per_paper: int = 20         # cap per paper
```

## Integration with Existing Code

This design will be implemented as a new method `build_with_buckets()` in `ExpansionEngine`, alongside the existing `build()` method:

```python
class ExpansionEngine:
    def build(self, seeds, ...):
        """Existing depth-based expansion."""
        ...

    def build_with_buckets(self, seeds, ...):
        """New bucket-based expansion until exhaustion."""
        ...
```

The new method will:
1. Reuse existing `CandidatePool` and `WorkingGraph`
2. Reuse existing `_expand_paper()` logic for fetching refs/cites
3. Add bucket tracking and relevance-based stopping
4. Maintain backward compatibility

## Test Case: aaRS Literature

Using the aaRS (aminoacyl-tRNA synthetase) literature as test case:
- Seed: Key aaRS paper from existing collection
- Expected behavior: Should discover related papers about tRNA synthetases, protein evolution, editing domains
- Stopping: Should stop when expanding into unrelated areas (e.g., clinical medicine)

## Verification Criteria

1. **Bucket formation**: Initial seed creates bucket with intro refs + citations
2. **Recursive expansion**: Buckets create child buckets
3. **Relevance decay**: Low-relevance branches stop expanding
4. **Topic drift**: Global stop when overall quality drops
5. **Deduplication**: Same paper never expanded twice
6. **API budget**: Hard stop at max_api_calls

---

*Design document created 2026-01-16*
*Status: DESIGN COMPLETE - Ready for implementation*
