# RefNet Manual

## What is RefNet?

RefNet answers the question: **"Given a paper I care about, what else should I read?"**

It builds a citation network starting from seed papers, discovering related literature through citation relationships, author connections, and gap analysis.

---

## The Core Problem RefNet Solves

When you find an important paper, you want to know:
1. What papers did it cite? (backward - the foundations)
2. What papers cite it? (forward - the follow-up work)
3. What did the key authors write before/after?
4. What's the structure of this research field?
5. What am I missing?

RefNet automates this by crawling citation networks intelligently.

---

## The Dendrimer Model (Core Algorithm)

RefNet uses a **dendrimer expansion model** - named after dendrimer polymers in chemistry that branch exponentially but each branch can die independently.

### How It Works

```
SEED PAPER (your starting point)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Fetch seed's references + citations                │
│                                                              │
│  Seed Paper                                                  │
│      ├── Reference_1 (backward - paper the seed cites)       │
│      ├── Reference_2                                         │
│      ├── ...                                                 │
│      ├── Citation_1 (forward - paper that cites the seed)    │
│      └── Citation_2                                          │
│                                                              │
│  These become "Bucket_0" - the first generation              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Each paper in Bucket_0 creates its OWN child bucket │
│                                                              │
│  Reference_1 → Bucket_1A (refs + cites of Reference_1)       │
│  Reference_2 → Bucket_1B (refs + cites of Reference_2)       │
│  Citation_1  → Bucket_1C (refs + cites of Citation_1)        │
│  ...                                                         │
│                                                              │
│  Each bucket expands INDEPENDENTLY                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Branches PRUNE when relevance drops                 │
│                                                              │
│  Bucket_1A has avg_relevance = 0.45 → CONTINUE expanding     │
│  Bucket_1B has avg_relevance = 0.12 → PRUNE (stop this branch)│
│  Bucket_1C has avg_relevance = 0.38 → CONTINUE expanding     │
│                                                              │
│  Pruning prevents explosion into unrelated literature        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Process repeats until stopping conditions           │
│                                                              │
│  Stops when:                                                 │
│    - All branches pruned (relevance < 0.15)                  │
│    - API budget exhausted (2000 calls)                       │
│    - Max generations reached (10 levels deep)                │
│    - Natural exhaustion (no new papers to fetch)             │
└─────────────────────────────────────────────────────────────┘
```

### Why "Dendrimer"?

Like a dendrimer molecule:
- **Exponential branching**: Each node spawns multiple children
- **Independent branches**: Each branch lives or dies on its own
- **Natural pruning**: Weak branches stop growing automatically
- **Focused growth**: Strong (relevant) branches keep expanding

### Intro Hint Weighting

Papers typically cite foundational work in their **introduction** (first 25% of references). RefNet gives these 2x weight because:
- Intro refs establish context
- They're more likely to be topically central
- They help identify the "core" of the field

```
Reference list of a paper:
  [1-25]  → First 25% = "intro hints" → weight 2.0
  [26-100] → Rest = regular refs → weight 1.0
```

---

## Two-Tier Architecture

RefNet maintains TWO data structures:

```
┌─────────────────────────────────────────────────────────────┐
│                  CANDIDATE POOL (SQLite)                     │
│                                                              │
│  - Stores EVERY paper discovered (10,000 - 20,000+)          │
│  - Stores ALL edges (citations between papers)               │
│  - Persists to candidates.db                                 │
│  - Status: "candidate" (discovered) or "materialized" (used) │
│                                                              │
│  This is your WIDE pool - everything the dendrimer found     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Top N papers by relevance score
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  WORKING GRAPH (in-memory)                   │
│                                                              │
│  - Only the BEST 150-300 papers                              │
│  - Used for visualization and gap analysis                   │
│  - Exported to graph.json, viewer.html                       │
│                                                              │
│  This is your NARROW view - the papers worth reading         │
└─────────────────────────────────────────────────────────────┘
```

**Why two tiers?**
- You can't visualize 20,000 papers effectively
- But you want to KEEP all discovered papers for queries
- The pool is your searchable database; the graph is your reading list

---

## The 5 Analysis Layers

RefNet runs 5 sequential layers:

### Layer 1: Citation Expansion (Dendrimer)
```
What it does:
  - Fetches references (backward) and citations (forward) for each paper
  - Applies intro hint weighting (first 25% of refs get 2x weight)
  - Creates buckets for each paper, expands independently
  - Prunes branches when avg_relevance < 0.15

Why it matters:
  - Builds the core citation network
  - Discovers 10,000-20,000 papers
  - Identifies topically relevant literature through citation chains
```

### Layer 2: Author Expansion
```
What it does:
  - Identifies "key authors" (appear in 3+ papers in the network)
  - Fetches their other publications
  - Adds relevant papers to the pool

Why it matters:
  - Authors who appear often are field experts
  - Their other work is likely relevant
  - Captures papers that might not be directly cited
```

### Layer 3: Trajectory Analysis
```
What it does:
  - For top authors, analyzes their publication history over time
  - Uses Jensen-Shannon Divergence (JSD) to measure topic drift
  - Detects when authors pivot to new research areas

Why it matters:
  - Shows how the field evolved
  - Identifies when authors brought new ideas
  - Helps understand research lineages
```

### Layer 4: Clustering
```
What it does:
  - Runs community detection (Louvain algorithm) on the graph
  - Groups papers by citation patterns
  - Identifies distinct research clusters

Why it matters:
  - Shows the structure of the field
  - Reveals sub-communities (e.g., "structural biology" vs "evolution")
  - Helps navigate a large literature
```

### Layer 5: Gap Analysis
```
What it does:
  - Finds "bridges": papers connecting different clusters
  - Finds "missing links": gaps in your coverage
  - Finds "unexplored clusters": nearby areas you haven't covered

Why it matters:
  - Answers "what am I missing?"
  - Identifies interdisciplinary papers
  - Suggests next reading directions
```

---

## Critical Insight: Seeds Determine Everything

**The network you build depends ENTIRELY on your seed papers.**

Example:
```
Seed: "Reduced amino acid alphabets" paper
  → Network expands into bioinformatics methods
  → NOT an aaRS network, even though aaRS uses reduced alphabets

Seed: "Carter lab urzyme" paper
  → Network expands into aminoacyl-tRNA synthetase literature
  → Proper aaRS network with synthetase-specific papers
```

**Choose seeds carefully:**
- Use domain-specific papers, not methods papers
- Multiple seeds from the same field reinforce each other
- A review paper can be a good seed (cites many foundational works)

---

## Quick Start (30 seconds)

```bash
cd /storage/kiran-stuff/ref-network-app

# run with a DOI
python3 refnet.py --all-layers --bucket-mode --doi 10.1073/pnas.1818339116 -o my_network

# view results
firefox my_network/viewer.html
```

---

## Input Options

### Single DOI (recommended)
```bash
python3 refnet.py --all-layers --bucket-mode --doi 10.1093/bioinformatics/btae061 -o ./output
```

### Multiple DOIs
```bash
python3 refnet.py --all-layers --bucket-mode \
  --doi 10.1038/s41586-021-03819-2 \
  --doi 10.1126/science.abj8754 \
  -o ./output
```

### From PDF
```bash
python3 refnet.py --all-layers --bucket-mode --pdf paper.pdf -o ./output
```

### From BibTeX/RIS
```bash
python3 refnet.py --all-layers --bucket-mode --bib library.bib -o ./output
```

### From Collection
```bash
python3 refnet.py --all-layers --bucket-mode -c my_collection -o ./output --seed-limit 30
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-nodes` | 300 | Papers in final working graph |
| `--seed-limit` | 30 | Max seeds from collection/bib |
| `--no-authors` | off | Skip author expansion (faster) |
| `--no-trajectory` | off | Skip trajectory analysis |

### Size Examples
```bash
# small focused network (fast)
python3 refnet.py --all-layers --bucket-mode --doi <DOI> -o ./output --max-nodes 100

# medium (default)
python3 refnet.py --all-layers --bucket-mode --doi <DOI> -o ./output --max-nodes 200

# large comprehensive
python3 refnet.py --all-layers --bucket-mode --doi <DOI> -o ./output --max-nodes 500

# fast run (skip author layer)
python3 refnet.py --all-layers --bucket-mode --doi <DOI> -o ./output --no-authors
```

---

## Running Long Jobs

### Background (recommended for >100 nodes)
```bash
cd /storage/kiran-stuff/ref-network-app

nohup python3 refnet.py --all-layers --bucket-mode --doi <DOI> -o ./output > run.log 2>&1 &
echo "PID: $!"
```

### Check Progress
```bash
# is it running?
ps aux | grep refnet.py

# how many papers discovered?
sqlite3 ./output/candidates.db "SELECT COUNT(*) FROM paper_candidates"

# how many edges?
sqlite3 ./output/candidates.db "SELECT COUNT(*) FROM edges"

# tail the log
tail -20 run.log
```

### Kill if Needed
```bash
pkill -f "refnet.py"
```

---

## Output Files

```
output/
├── candidates.db      # SQLite - ALL discovered papers (wide pool, 10,000+)
├── graph.json         # Working graph + gap analysis (top N only)
├── graph.graphml      # For Gephi/Cytoscape
├── nodes.csv          # Papers (for Excel/R)
├── edges.csv          # Citations
└── viewer.html        # Interactive visualization
```

### View Results
```bash
# interactive viewer
firefox ./output/viewer.html

# quick stats
sqlite3 ./output/candidates.db "
SELECT 'Papers:' as metric, COUNT(*) as value FROM paper_candidates
UNION ALL
SELECT 'Edges:', COUNT(*) FROM edges
UNION ALL
SELECT 'Materialized:', COUNT(*) FROM paper_candidates WHERE status='materialized';
"
```

---

## Understanding the Output

### viewer.html
- **Nodes** = papers (size by citation count)
- **Edges** = citations (thick = intro_hint_cites, 2x weight)
- **Hover** = title, authors, year, citations
- **Drag** to rearrange

### Edge Types
| Type | Weight | Meaning |
|------|--------|---------|
| `intro_hint_cites` | 2.0 | First 25% of references (intro section) |
| `cites` | 1.0 | Regular citation |

### graph.json Structure
```json
{
  "nodes": [...],
  "edges": [...],
  "gap_analysis": {
    "bridges": [...],           // papers connecting clusters
    "missing_links": [...],     // gaps in coverage
    "unexplored_clusters": [...] // nearby unexplored areas
  }
}
```

---

## Querying the Full Pool

The working graph only shows ~150-300 papers, but ALL discovered papers (10,000-20,000) are in `candidates.db`.

### Top Cited Papers
```bash
sqlite3 ./output/candidates.db "
SELECT title, citation_count
FROM paper_candidates
ORDER BY citation_count DESC
LIMIT 10;
"
```

### Search by Keyword
```bash
sqlite3 ./output/candidates.db "
SELECT title, year, citation_count
FROM paper_candidates
WHERE title LIKE '%synthetase%'
ORDER BY citation_count DESC
LIMIT 10;
"
```

### Papers NOT in Graph (candidates only)
```bash
sqlite3 ./output/candidates.db "
SELECT title, citation_count, year
FROM paper_candidates
WHERE status='candidate'
ORDER BY citation_count DESC
LIMIT 20;
"
```

### Export All Papers to CSV
```bash
sqlite3 -header -csv ./output/candidates.db "
SELECT doi, title, year, citation_count, status
FROM paper_candidates
ORDER BY citation_count DESC;
" > all_papers.csv
```

### Count by Discovery Channel
```bash
sqlite3 ./output/candidates.db "
SELECT discovered_channel, COUNT(*) as cnt
FROM paper_candidates
GROUP BY discovered_channel;
"
```

---

## Troubleshooting

### "rate limited" messages
Normal. Semantic Scholar throttles API. System continues with OpenAlex/PubMed.

### "all providers failed"
Temporary API issue. System continues with other papers. Not every paper has data in all APIs.

### Low edge count (< 1 edge per paper)
Check STATUS_REPORT.md for the ID alignment fix.

### Runs too long
- Reduce `--max-nodes`
- Add `--no-authors`
- Kill and restart with smaller scope

---

## Recipes

### Literature Review for New Topic
```bash
# find a key review, get its DOI, then:
python3 refnet.py --all-layers --bucket-mode --doi 10.xxxx/review -o ./my_topic --max-nodes 200
firefox ./my_topic/viewer.html
```

### Expand Zotero Library
```bash
# export from Zotero as BibTeX, then:
python3 refnet.py --all-layers --bucket-mode --bib library.bib -o ./expanded --seed-limit 50
```

### Find Gaps in Reading
```bash
python3 refnet.py --all-layers --bucket-mode --bib library.bib -o ./gaps
# check gap_analysis in graph.json
```

### Quick Test
```bash
python3 refnet.py --all-layers --bucket-mode --doi 10.1093/bioinformatics/btae061 -o ./test --max-nodes 50 --no-authors
```

---

## Proven Working Examples

### aaRS Dendrimer (2026-01-17)
```bash
python3 refnet.py --all-layers --bucket-mode \
  --doi 10.1074/jbc.m113.496125 \
  --doi 10.1007/s00239-015-9672-1 \
  --doi 10.1261/rna.061069.117 \
  -o /tmp/aars_dendrimer_full \
  --max-nodes 200

# Results: 19,724 papers in pool, 186 in working graph
# 1,529 aaRS-specific papers discovered
# 10 bridges, 15 missing links identified
```

---

## Files Location

```
/storage/kiran-stuff/ref-network-app/
├── refnet.py            # main entry point
├── refnet/              # package
│   ├── core/            # models, config, db
│   ├── graph/           # expansion engine (dendrimer logic)
│   ├── layers/          # author, trajectory
│   ├── analysis/        # hub, gap
│   └── export/          # formats, viewer
├── MANUAL.md            # this file
├── STATUS_REPORT.md     # current state, bug fixes
└── ONBOARDING.md        # architecture overview
```

---

## Summary: The Full Logic

1. **You provide seed papers** (DOIs, PDF, BibTeX)

2. **Layer 1 (Dendrimer)** expands the citation network:
   - Fetch refs (backward) and citations (forward)
   - Weight intro refs 2x (first 25%)
   - Each paper creates its own bucket
   - Prune branches when relevance < 0.15
   - Result: 10,000-20,000 papers in candidate pool

3. **Layer 2 (Author)** finds key authors' other works

4. **Layer 3 (Trajectory)** analyzes author career evolution

5. **Layer 4 (Clustering)** groups papers by citation patterns

6. **Layer 5 (Gap)** identifies bridges, missing links, unexplored areas

7. **Output**:
   - `candidates.db` = ALL discovered papers (wide pool)
   - `graph.json` + `viewer.html` = TOP papers (narrow view)
   - Gap analysis tells you what to read next

**The key insight**: RefNet doesn't just find papers - it finds the STRUCTURE of a research field and shows you where you are in that structure.

---

## Extended Knowledge Mapping (New!)

RefNet now supports a multi-layer approach beyond just citation networks.

### The Four Layers

```
Layer 4: Review Curation   → TOC extraction, gap analysis
Layer 3: Google Scholar    → Global impact, PDF links
Layer 2: Concept Dendrimer → LLM semantic expansion
Layer 1: Paper Dendrimer   → Citation networks (original RefNet)
```

### Layer 2: Concept Dendrimer

Use LLM to expand topics semantically (complements citation-based expansion).

```bash
# via Ollama
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:27b",
  "prompt": "Given aminoacyl-tRNA synthetases, identify UPSTREAM, DOWNSTREAM, PARALLEL, and FRONTIER concepts."
}'
```

**Output**: Hierarchical concept map showing what connects to what.

### Layer 3: Google Scholar Integration

```python
from scholarly import scholarly

# search topic
results = scholarly.search_pubs("aminoacyl-tRNA synthetase review")

# extract papers with PDF links
for r in results:
    print(r['bib']['title'], r.get('eprint_url'))  # PDF link!
```

**Key insight**: 98% overlap with paper dendrimer validates our approach.

### Layer 4: Review TOC Extraction

Reviews have detailed Table of Contents - better than abstracts!

```python
# see aars_knowledge_map/toc_extractor.py
from toc_extractor import extract_toc_from_pmc
toc = extract_toc_from_pmc("PMC98992")
```

**Output**: Section structure showing exactly what a review covers.

---

## See Also

- `KNOWLEDGE_MAP.md` - Full multi-layer architecture
- `aars_knowledge_map/` - Complete aaRS case study with all layers
- `REVIEW_CURATION_PROTOCOL.md` - How to curate from reviews
