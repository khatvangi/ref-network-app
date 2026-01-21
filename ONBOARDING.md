# RefNet Onboarding Guide

**For developers and maintainers** - understand the architecture and logic.

---

## What is RefNet?

RefNet answers: **"Given a paper I care about, what else should I read?"**

It's a citation network builder that:
1. Starts from seed papers you provide
2. Expands through citation relationships (who cites whom)
3. Analyzes author patterns and career trajectories
4. Detects research clusters and gaps
5. Produces a curated reading list with field structure

---

## The Core Insight: Why Dendrimer?

Traditional citation crawlers use BFS (breadth-first) or DFS (depth-first):
- **BFS**: Explore all papers at depth 1, then depth 2, etc. → Explodes quickly, loses focus
- **DFS**: Follow one chain deep, then backtrack → Misses parallel developments

**RefNet uses a dendrimer model** (named after branched polymer molecules):

```
Traditional BFS:                    RefNet Dendrimer:

Level 0:    Seed                    Seed
            /|\                       │
Level 1:  A B C D E F              Bucket_0 (A, B, C, D, E, F)
         /|\ etc...                   │
Level 2: explodes                   Each paper creates OWN bucket:
         (thousands)                  A → Bucket_A (A's refs + cites)
                                      B → Bucket_B (B's refs + cites)
                                      C → [PRUNED - low relevance]
                                      ...

                                    Bucket_A expands:
                                      A1 → Bucket_A1
                                      A2 → [PRUNED]

                                    Result: Focused growth, weak branches die
```

**Key properties:**
1. **Independent branches**: Each paper's expansion is tracked separately
2. **Relevance-based pruning**: Branches die when avg_relevance < 0.15
3. **Natural exhaustion**: Stops when no more relevant papers to fetch
4. **Bidirectional**: Follows both refs (backward) and citations (forward)

---

## Data Flow: From Seed to Output

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT                                                                │
│   - DOI(s)          python3 refnet.py --doi 10.1234/example         │
│   - PDF             python3 refnet.py --pdf paper.pdf               │
│   - BibTeX          python3 refnet.py --bib library.bib             │
│   - Collection      python3 refnet.py -c my_collection              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Citation Expansion (Dendrimer)                              │
│                                                                      │
│   For each seed paper:                                               │
│     1. Fetch paper metadata from APIs (S2, OpenAlex, PubMed)         │
│     2. Get references (papers it cites) - backward expansion         │
│     3. Get citations (papers citing it) - forward expansion          │
│     4. Mark first 25% of refs as "intro hints" (weight 2.0)          │
│     5. Create bucket for this paper's children                       │
│     6. Calculate bucket avg_relevance                                │
│     7. If avg_relevance < 0.15 → PRUNE (stop this branch)            │
│     8. Else → add children to expansion queue                        │
│                                                                      │
│   Repeat until: all branches pruned OR API budget (2000) exhausted   │
│                 OR max generations (10) OR natural exhaustion        │
│                                                                      │
│   Result: CandidatePool with 10,000-20,000 papers                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 2: Author Expansion                                            │
│                                                                      │
│   1. Identify "key authors" (appear in ≥3 papers in the pool)        │
│   2. For each key author:                                            │
│      - Fetch their publication list from APIs                        │
│      - Score each paper for relevance to the network                 │
│      - Add relevant papers to pool                                   │
│                                                                      │
│   Why: Authors who appear often are field experts.                   │
│        Their other work is likely relevant but may not be cited.     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3: Trajectory Analysis                                         │
│                                                                      │
│   For top authors (by paper count):                                  │
│     1. Get their publications ordered by year                        │
│     2. Extract concepts/topics from each paper                       │
│     3. Compute Jensen-Shannon Divergence between time periods        │
│     4. Detect "pivots" where JSD exceeds threshold                   │
│                                                                      │
│   Why: Shows how authors (and the field) evolved over time.          │
│        Pivots indicate when new ideas entered the field.             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 4: Clustering                                                  │
│                                                                      │
│   1. Build citation graph from pool                                  │
│   2. Run Louvain community detection                                 │
│   3. Assign each paper to a cluster                                  │
│   4. Label clusters by common terms in titles                        │
│                                                                      │
│   Why: Reveals field structure. Sub-communities become visible.      │
│        e.g., "structural biology" vs "evolution" vs "disease"        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 5: Gap Analysis                                                │
│                                                                      │
│   Finds three types of gaps:                                         │
│                                                                      │
│   BRIDGES: Papers with edges to multiple clusters                    │
│     → Interdisciplinary work connecting sub-fields                   │
│                                                                      │
│   MISSING LINKS: Expected connections that don't exist               │
│     - Citation gaps: Highly-cited papers you haven't covered         │
│     - Concept gaps: Topics mentioned but not explored                │
│     - Temporal gaps: Time periods with few papers                    │
│                                                                      │
│   UNEXPLORED CLUSTERS: Nearby areas not in your network              │
│     → Papers that cite your papers but aren't in pool                │
│                                                                      │
│   Why: Answers "what am I missing?"                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MATERIALIZATION                                                      │
│                                                                      │
│   CandidatePool has 10,000-20,000 papers (too many to visualize)     │
│                                                                      │
│   Select top N papers by relevance score:                            │
│     - Citation count                                                 │
│     - Intro hint weight (2x for first 25% of refs)                   │
│     - Connection to seed papers                                      │
│     - Recency                                                        │
│                                                                      │
│   Mark selected papers as "materialized"                             │
│   Copy their edges to WorkingGraph                                   │
│                                                                      │
│   Result: WorkingGraph with 150-300 papers                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                               │
│                                                                      │
│   candidates.db     SQLite database with ALL discovered papers       │
│                     - paper_candidates table (10,000-20,000 rows)    │
│                     - edges table (20,000-30,000 rows)               │
│                     - status: 'candidate' or 'materialized'          │
│                                                                      │
│   graph.json        Working graph (top N papers) + gap analysis      │
│   graph.graphml     For Gephi/Cytoscape                              │
│   nodes.csv         Paper metadata for Excel/R                       │
│   edges.csv         Citation relationships                           │
│   viewer.html       Interactive D3 visualization                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Two-Tier Architecture: Why?

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CANDIDATE POOL (SQLite)                          │
│                                                                      │
│   Purpose: Store EVERYTHING discovered                               │
│   Size: 10,000 - 20,000+ papers                                      │
│   Storage: candidates.db (persists to disk)                          │
│   Status: 'candidate' (discovered) or 'materialized' (in graph)      │
│                                                                      │
│   Why keep all?                                                      │
│     - Query later: "find all papers about X"                         │
│     - Re-materialize with different criteria                         │
│     - Analysis on full network (centrality, communities)             │
│     - Resume interrupted runs                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Top N by relevance
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     WORKING GRAPH (in-memory)                        │
│                                                                      │
│   Purpose: Visualization and gap analysis                            │
│   Size: 150 - 300 papers (configurable via --max-nodes)              │
│   Storage: graph.json, viewer.html                                   │
│                                                                      │
│   Why limit?                                                         │
│     - Can't visualize 20,000 nodes effectively                       │
│     - Gap analysis needs manageable graph                            │
│     - Human reading list should be focused                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Code Architecture

```
refnet/
├── core/                      # Foundational data structures
│   ├── models.py              # Paper, Author, Bucket, Edge dataclasses
│   ├── config.py              # All configuration (expansion, author, gap)
│   └── db.py                  # CandidateDB - SQLite persistence layer
│
├── graph/                     # Core expansion logic
│   ├── expansion.py           # ExpansionEngine - MAIN ORCHESTRATOR
│   │                          #   - run_expansion() entry point
│   │                          #   - _expand_bucket() dendrimer logic
│   │                          #   - _process_paper() fetch refs/cites
│   │                          #   - Calls all 5 layers in sequence
│   │
│   ├── candidate_pool.py      # CandidatePool - wide paper storage
│   │                          #   - add_paper() with deduplication
│   │                          #   - add_edge() with ID alignment
│   │                          #   - get_papers_by_status()
│   │
│   └── working_graph.py       # WorkingGraph - narrow visualization graph
│                              #   - materialize_from_pool()
│                              #   - sync edges from pool
│
├── layers/                    # Analysis layers 2-3
│   ├── author.py              # AuthorLayer
│   │                          #   - identify_key_authors()
│   │                          #   - fetch_author_papers()
│   │
│   └── trajectory.py          # TrajectoryLayer
│                              #   - compute_jsd_drift()
│                              #   - detect_pivots()
│
├── analysis/                  # Analysis layers 4-5
│   ├── gap.py                 # GapAnalyzer
│   │                          #   - find_bridges()
│   │                          #   - find_missing_links()
│   │                          #   - find_unexplored_clusters()
│   │
│   └── hub.py                 # HubDetector
│                              #   - suppress overly-cited generic papers
│                              #   - (e.g., BLAST, Bradford assay)
│
├── providers/                 # External API integrations
│   ├── composite.py           # CompositeProvider - tries multiple APIs
│   ├── semantic_scholar.py    # Semantic Scholar API
│   ├── openalex.py            # OpenAlex API
│   └── pubmed.py              # PubMed/NCBI API
│
└── export/                    # Output generation
    ├── viewer.py              # HTML visualization with D3
    └── exporter.py            # JSON, GraphML, CSV export
```

---

## Key Classes and Their Responsibilities

### ExpansionEngine (`graph/expansion.py`)
**The main orchestrator.** Controls the entire pipeline.

```python
class ExpansionEngine:
    def run_expansion(self, seeds: List[Paper]) -> WorkingGraph:
        # Layer 1: Dendrimer expansion
        self._expand_dendrimer(seeds)

        # Layer 2: Author expansion
        self._run_author_layer()

        # Layer 3: Trajectory analysis
        self._run_trajectory_layer()

        # Materialize top papers to working graph
        self._materialize_working_graph()

        # Layer 4: Clustering
        self._run_clustering()

        # Layer 5: Gap analysis
        self._run_gap_analysis()

        return self.working_graph
```

### CandidatePool (`graph/candidate_pool.py`)
**Wide storage for all discovered papers.**

```python
class CandidatePool:
    def add_paper(self, paper: Paper) -> Paper:
        # Check for duplicate by DOI
        # If exists, return EXISTING paper (critical for ID alignment!)
        # If new, insert and return

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType):
        # Store citation relationship
        # MUST use IDs returned by add_paper() for alignment
```

### Bucket (`core/models.py`)
**Tracks one branch of the dendrimer.**

```python
@dataclass
class Bucket:
    parent_paper_id: str          # Paper that spawned this bucket
    papers: List[Paper]           # Papers in this bucket
    generation: int               # Depth from seed
    avg_relevance: float          # Average relevance score

    def should_prune(self) -> bool:
        return self.avg_relevance < MIN_BUCKET_RELEVANCE  # 0.15
```

---

## Critical Bug Fix: ID Alignment

**Problem (fixed):** When adding papers, duplicates were detected but the code used the NEW paper's ID instead of the EXISTING paper's ID for edges.

```python
# WRONG (old code):
pool.add_paper(ref)  # Returns existing paper if duplicate
pool.add_edge(paper.id, ref.id, EdgeType.CITES)  # BUG: uses new ref.id

# CORRECT (fixed code):
added_ref = pool.add_paper(ref)  # Returns existing paper if duplicate
if added_ref:
    pool.add_edge(paper.id, added_ref.id, EdgeType.CITES)  # Uses returned ID
```

**Symptom:** Low edge count in working graph (< 1 edge per paper).
**Root cause:** Edges stored with wrong IDs, couldn't sync to graph.

---

## Configuration (`core/config.py`)

```python
# Expansion settings
max_api_calls_per_job = 2000      # API budget limit
base_max_generations = 10          # Max dendrimer depth
min_bucket_relevance = 0.15        # Prune threshold
intro_fraction = 0.25              # First 25% of refs = intro hints
intro_hint_weight = 2.0            # Weight multiplier for intro refs
max_refs_per_node = 100            # Max refs to fetch per paper
max_cites_per_node = 50            # Max citations to fetch per paper

# Author layer
author_enabled = True
max_authors_per_paper = 5
min_papers_for_key_author = 3      # Need 3+ papers to be "key author"

# Trajectory layer
trajectory_enabled = True
max_trajectory_authors = 10

# Gap analysis
gap_enabled = True
```

---

## API Providers and Fallback

RefNet uses multiple APIs with fallback:

```
┌─────────────────────────────────────────────────────────────────────┐
│ CompositeProvider                                                    │
│                                                                      │
│   Try in order:                                                      │
│   1. Semantic Scholar (best metadata, but rate-limited)              │
│   2. OpenAlex (good coverage, fast)                                  │
│   3. PubMed (biomedical focus)                                       │
│                                                                      │
│   If one fails → try next                                            │
│   If all fail → log warning, continue with other papers              │
└─────────────────────────────────────────────────────────────────────┘
```

**"rate limited" and "all providers failed" messages are normal.** The system continues with papers that DO have data.

---

## Stopping Conditions

The dendrimer stops when ANY of these occur:

| Condition | Threshold | Scope |
|-----------|-----------|-------|
| Relevance decay | avg_relevance < 0.15 | Per-branch (prunes one branch) |
| API budget | 2000 calls | Global (stops everything) |
| Max generations | 10 levels deep | Global |
| Natural exhaustion | No more papers to fetch | Global |

---

## Session Continuity

When starting a new session on this project:

1. **Read STATUS_REPORT.md** for current state and recent changes
2. **Check running processes**: `ps aux | grep refnet.py`
3. **Check output directories**: `ls -la /tmp/aars_*` or wherever output is
4. **Check database counts**:
   ```bash
   sqlite3 output/candidates.db "SELECT COUNT(*) FROM paper_candidates; SELECT COUNT(*) FROM edges;"
   ```

---

## Common Development Tasks

### Add a new API provider
1. Create `refnet/providers/newapi.py`
2. Implement `get_paper()`, `get_references()`, `get_citations()`
3. Add to `CompositeProvider` in `providers/composite.py`

### Modify relevance scoring
1. Edit `_calculate_relevance()` in `graph/expansion.py`
2. Adjust `intro_hint_weight` in `core/config.py`

### Add a new analysis layer
1. Create `refnet/layers/newlayer.py` or `refnet/analysis/newlayer.py`
2. Call it from `ExpansionEngine.run_expansion()`

### Change visualization
1. Edit `refnet/export/viewer.py`
2. Modify the D3.js template

---

## Testing

```bash
# Quick test (50 nodes, no author layer)
python3 refnet.py --all-layers --bucket-mode --doi 10.1093/bioinformatics/btae061 -o /tmp/test --max-nodes 50 --no-authors

# Check results
sqlite3 /tmp/test/candidates.db "SELECT COUNT(*) FROM paper_candidates"
firefox /tmp/test/viewer.html
```

---

## Summary: The Mental Model

1. **Seeds define the network** - choose domain-specific papers, not methods papers
2. **Dendrimer expands bidirectionally** - refs (backward) + citations (forward)
3. **Branches die independently** - weak branches prune, strong branches grow
4. **Pool stores everything** - 10,000-20,000 papers in SQLite
5. **Graph shows the best** - top 150-300 papers for visualization
6. **Gap analysis finds what's missing** - bridges, missing links, unexplored areas

**The result:** Not just a list of papers, but the STRUCTURE of a research field.
