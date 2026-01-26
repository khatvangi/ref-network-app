# RefNet Status Report
**Date**: 2026-01-17
**Session**: Dendrimer Implementation & ID Fix

---

## Current Test Running

```bash
# aaRS literature test (running with nohup)
PID: 2240495
Command: python3 refnet.py --all-layers --bucket-mode --doi 10.1093/bioinformatics/btae061 -o /tmp/aars_fixed --max-nodes 150
Log: /tmp/aars_fixed.log
```

**Progress** (as of last check):
- Papers: ~14,200
- Edges: ~24,800
- Status: Layer 1 (dendrimer expansion) running
- Will auto-stop via branch pruning or natural exhaustion

---

## Critical Bug Fix: ID Alignment

### Problem
Papers in the candidate pool had different UUIDs than the same papers in the working graph. Edges stored in pool couldn't be synced to graph because IDs didn't match.

**Example of the bug**:
```
AlphaFold2 paper in pool:  ID = c3a5aa08-7da5-4dc0-ac3e-926c1751934a
AlphaFold2 paper in graph: ID = eef7262a-e847-44ef-b9f5-ed6517f15d6b
Edge in pool: c3a5aa08... -> some_target
Edge sync fails because graph has eef7262a..., not c3a5aa08...
```

### Root Cause
When `pool.add_paper(ref)` found a duplicate, it returned the EXISTING paper with its original ID. But the code ignored the return value and used `ref.id` (the NEW paper's ID) for creating edges.

### Fix Applied
**File**: `refnet/graph/expansion.py`

Changed all occurrences of:
```python
pool.add_paper(ref)
pool.add_edge(paper.id, ref.id, EdgeType.CITES)  # WRONG: uses new ID
bucket_papers.append(ref)
```

To:
```python
added_ref = pool.add_paper(ref)
if added_ref:
    pool.add_edge(paper.id, added_ref.id, EdgeType.CITES)  # CORRECT: uses returned ID
    bucket_papers.append(added_ref)
```

**Lines fixed**: 307-322, 336-350, 846-859, 878-891, 952-965, 984-997, 1160-1173, 1191-1204

### Verification
```python
# Quick test confirmed fix works:
Paper1 added with ID: 61f54ea0...
Paper2 (duplicate) returns: 61f54ea0... (original ID, not new)
Edge correctly points to: 61f54ea0...
```

---

## Architecture: Dendrimer Model

### Concept
Like a dendrimer polymer in chemistry - exponential branching where each branch can die independently.

```
Seed Paper
    └── Bucket_0 (refs + cites of seed)
        ├── Paper_A → Bucket_A (refs + cites of A)
        │   ├── Paper_A1 → Bucket_A1
        │   └── Paper_A2 → Bucket_A2 [PRUNED - low relevance]
        ├── Paper_B → Bucket_B (refs + cites of B)
        └── Paper_C → Bucket_C [PRUNED - low relevance]
```

### Key Properties
1. **Per-branch pruning**: Each bucket tracks its own `avg_relevance`
2. **Independent death**: Branch dies when `avg_relevance < min_bucket_relevance` (0.15)
3. **Natural exhaustion**: Stops when no more papers to expand
4. **Bidirectional expansion**: Intro refs (first 25%, weight=2.0) + citations (weight=1.0)

### Stopping Conditions
| Condition | Threshold | Scope |
|-----------|-----------|-------|
| Relevance decay | avg_relevance < 0.15 | Per-branch |
| Topic drift | overall quality drop | Global |
| API budget | 2000 calls | Global |
| Max generations | 10 (adaptive) | Global |
| Natural exhaustion | no more papers | Global |

---

## 5-Layer System

| Layer | Name | What it does |
|-------|------|--------------|
| 1 | Citation Expansion | Dendrimer: refs (backward) + cites (forward) |
| 2 | Author Expansion | Key authors' other works |
| 3 | Trajectory Analysis | JSD drift detection for author careers |
| 4 | Clustering | Community detection in graph |
| 5 | Gap Analysis | Bridges, missing links, unexplored clusters |

**Execution order**: Layer 1 runs until exhaustion → then 2-5 run sequentially

---

## Key Files

### Core
| File | Purpose |
|------|---------|
| `refnet/core/models.py` | Paper, Author, Bucket, BucketExpansionState dataclasses |
| `refnet/core/config.py` | All configuration (expansion, author, trajectory, gap) |
| `refnet/core/db.py` | CandidateDB - SQLite persistence |

### Graph
| File | Purpose |
|------|---------|
| `refnet/graph/expansion.py` | ExpansionEngine - main orchestrator, dendrimer logic |
| `refnet/graph/candidate_pool.py` | CandidatePool - wide pool with deduplication |
| `refnet/graph/working_graph.py` | WorkingGraph - narrow in-memory graph |

### Layers
| File | Purpose |
|------|---------|
| `refnet/layers/author.py` | AuthorLayer - expand key authors |
| `refnet/layers/trajectory.py` | TrajectoryLayer - JSD drift detection |

### Analysis
| File | Purpose |
|------|---------|
| `refnet/analysis/gap.py` | GapAnalyzer - bridges, missing links, unexplored |
| `refnet/analysis/hub.py` | HubDetector - suppress over-cited papers |

### Export
| File | Purpose |
|------|---------|
| `refnet/export/viewer.py` | HTML viewer with hover tooltips |
| `refnet/export/exporter.py` | JSON, GraphML, CSV export |

---

## How to Run

### Full Pipeline (All Layers + Dendrimer)
```bash
cd /storage/kiran-stuff/ref-network-app

# from DOI
python3 refnet.py --all-layers --bucket-mode --doi 10.1093/bioinformatics/btae061 -o output_dir --max-nodes 200

# from PDF
python3 refnet.py --all-layers --bucket-mode --pdf paper.pdf -o output_dir

# from collection
python3 refnet.py --all-layers --bucket-mode -c my_collection -o output_dir
```

### Output Files
```
output_dir/
├── candidates.db      # SQLite pool (all discovered papers)
├── graph.json         # Working graph + gap analysis
├── graph.graphml      # For Gephi/Cytoscape
├── nodes.csv          # Papers
├── edges.csv          # Citations
└── viewer.html        # Interactive visualization
```

### Check Status of Running Test
```bash
# process status
ps aux | grep refnet.py

# paper/edge counts
sqlite3 /tmp/aars_fixed/candidates.db "SELECT COUNT(*) FROM paper_candidates; SELECT COUNT(*) FROM edges;"

# log tail
tail -20 /tmp/aars_fixed.log
```

---

## Edge Types

| Type | Weight | Meaning |
|------|--------|---------|
| `intro_hint_cites` | 2.0 | First 25% of references (intro section) |
| `cites` | 1.0 | Regular citation |
| `authored_by` | 1.0 | Paper-author link |
| `co_authored` | varies | Author collaboration |

---

## Config Defaults

```python
# Expansion
max_api_calls_per_job = 2000
base_max_generations = 10
min_bucket_relevance = 0.15
intro_fraction = 0.25
intro_hint_weight = 2.0
max_refs_per_node = 100
max_cites_per_node = 50

# Author
enabled = True
max_authors_per_paper = 5
min_papers_for_key_author = 3

# Trajectory
enabled = True
max_trajectory_authors_auto = 10

# Gap Analysis
enabled = True
```

---

## Known Issues / TODOs

1. **Rate limiting**: Semantic Scholar API throttles heavily - messages like `[s2] rate limited` are normal
2. **API fallbacks**: When S2 fails, falls back to OpenAlex/PubMed
3. **Memory**: Large runs (>10k papers) may need monitoring

---

## Recent Changes (This Session)

1. **ID alignment fix** - edges now use correct paper IDs from pool
2. **Bidirectional edge sync** - `_sync_pool_edges` checks both FROM and TO directions
3. **Dendrimer model** - each paper creates own child bucket
4. **Branch pruning** - low-relevance branches die independently

---

## Verification Commands

```bash
# verify ID alignment in pool
sqlite3 /tmp/aars_fixed/candidates.db "
SELECT 'Edges with valid source' as metric, COUNT(*) as cnt
FROM edges e
WHERE EXISTS (SELECT 1 FROM paper_candidates p WHERE p.id = e.source_id);
"

# check for duplicate papers (should be 0)
sqlite3 /tmp/aars_fixed/candidates.db "
SELECT doi, COUNT(*) as cnt
FROM paper_candidates
WHERE doi IS NOT NULL
GROUP BY doi
HAVING cnt > 1;
"
```
