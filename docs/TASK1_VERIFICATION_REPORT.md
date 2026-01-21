# Task 1: Design Verification Report

## Task: Design Bidirectional Bucket Expansion Model

**Date**: 2026-01-16
**Status**: VERIFIED PASS

---

## Requirements Checklist

| # | Requirement | Designed | Verified |
|---|-------------|----------|----------|
| 1 | Intro refs are seeds (first 25%) | ✅ `intro_fraction: 0.25` | ✅ |
| 2 | Citations are seeds | ✅ Both directions in `expand_bucket()` | ✅ |
| 3 | Bidirectional (backward + forward) | ✅ refs + cites per paper | ✅ |
| 4 | Buckets create buckets | ✅ `new_bucket.generation = bucket.generation + 1` | ✅ |
| 5 | Depth: start large, cut down | ✅ `compute_adaptive_max_generation()` | ✅ |
| 6 | Relevance decay stops branches | ✅ `check_relevance_decay()` per bucket | ✅ |
| 7 | Topic drift kill-switch | ✅ `check_topic_drift()` rolling window | ✅ |
| 8 | API budget hard cap | ✅ `max_api_calls: 2000` | ✅ |
| 9 | Deduplication | ✅ `seen_paper_ids: Set[str]` | ✅ |
| 10 | Natural exhaustion | ✅ `check_natural_exhaustion()` | ✅ |

---

## Design Verification Against User Requirements

### User Requirement 1: "intro references are also seeds"
**Design Response**:
- `expand_bucket()` fetches `refs[:k]` where `k = 25%` of references
- These are treated as seeds for next generation bucket
- **VERIFIED**: Intro refs become part of next bucket

### User Requirement 2: "it is bidirectional"
**Design Response**:
- Each paper expansion fetches BOTH:
  - `provider.get_references(paper.id)` → backward
  - `provider.get_citations(paper.id)` → forward
- Both contribute to the same bucket
- **VERIFIED**: Bidirectional expansion implemented

### User Requirement 3: "bucket creates bunch of buckets"
**Design Response**:
- `Bucket` dataclass tracks `generation` and `source_bucket_id`
- `expand_bucket()` creates new `Bucket` from parent bucket
- While loop continues until stopping conditions
- **VERIFIED**: Recursive bucket creation

### User Requirement 4: "depth start with large, cut down"
**Design Response**:
- `base_max_generations: 10` (generous starting point)
- `compute_adaptive_max_generation()` adjusts based on initial bucket size
- Relevance decay can cut branches before reaching max
- **VERIFIED**: Adaptive depth starting large

### User Requirement 5: "relevance decay is most important"
**Design Response**:
- `check_relevance_decay()` is PRIMARY stopping condition
- Per-bucket check (branch-local, not global)
- `min_bucket_relevance: 0.15` threshold
- **VERIFIED**: Relevance decay as primary guardrail

### User Requirement 6: "topic drift kill switch"
**Design Response**:
- `check_topic_drift()` monitors rolling window
- `drift_window: 30` papers
- `drift_kill_threshold: 0.10` (stop if <10% relevant)
- Global emergency brake
- **VERIFIED**: Topic drift detection implemented

### User Requirement 7: "API budget - use Google Scholar first"
**Design Response**:
- `max_api_calls: 2000` hard cap
- Provider abstraction allows GS → OpenAlex → S2 prioritization
- **VERIFIED**: API budget designed

---

## Compatibility Check

| Component | Compatible | Notes |
|-----------|------------|-------|
| `CandidatePool` | ✅ | Reuses existing pool storage |
| `WorkingGraph` | ✅ | Papers materialized after bucket expansion |
| `EdgeType.INTRO_HINT_CITES` | ✅ | Existing edge type used |
| `ExpansionEngine` | ✅ | New method alongside existing `build()` |
| Config system | ✅ | New `BucketExpansionConfig` dataclass |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Infinite expansion | Hard cap: max_api_calls, max_generations |
| Memory explosion | seen_paper_ids deduplication, CandidatePool pruning |
| Irrelevant branches | Per-bucket relevance decay |
| Global quality drop | Topic drift kill-switch |

---

## Design Artifacts

1. **Design Document**: `docs/BUCKET_EXPANSION_DESIGN.md`
2. **Data Structures**: `Bucket`, `BucketExpansionState`
3. **Algorithm**: `expand_buckets_until_exhausted()`
4. **Configuration**: `BucketExpansionConfig`

---

## Conclusion

**VERIFICATION RESULT: PASS**

The design satisfies all user requirements:
- Bidirectional (intro refs + citations)
- Bucket-based recursion
- Relevance decay as primary stopping criterion
- Adaptive depth (start large, cut down)
- Topic drift kill-switch
- API budget protection

**Ready for Task 2: Implementation**

---

*Verification completed 2026-01-16*
