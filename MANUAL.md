# RefNet Manual — Running from Boron

## Quick Start (30 seconds)

```bash
# 1. go to the project
cd /storage/kiran-stuff/ref-network-app

# 2. run with a DOI
python -m refnet.cli --doi 10.1073/pnas.1818339116 -o my_network

# 3. view results
ls my_network/
# graph.json, graph.graphml, nodes.csv, edges.csv, viewer.html
```

---

## Full Usage Guide

### Step 1: Choose Your Input Method

**Option A: Start from DOI(s)** — RECOMMENDED
```bash
# single paper
python -m refnet.cli --doi 10.1073/pnas.1818339116

# multiple papers
python -m refnet.cli \
  --doi 10.1073/pnas.1818339116 \
  --doi 10.1126/science.aax3194 \
  --doi 10.1038/s41586-020-2649-2
```

**Option B: Start from paper title**
```bash
python -m refnet.cli --title "Asymmetric evolution of protein domains"
```

**Option C: Start from your literature collection**
```bash
# from a directory of JSON/CSV files
python -m refnet.cli --collection /storage/kiran-stuff/aaRS/literature/ --enrich

# limit seeds
python -m refnet.cli --collection /path/to/papers/ --seed-limit 20
```

**Option D: Start from BibTeX**
```bash
python -m refnet.cli --bib my_library.bib
```

**Option E: Topic bootstrap** (last resort)
```bash
python -m refnet.cli "aminoacyl-tRNA synthetase evolution"
```

### Step 2: Configure Expansion (Optional)

| Flag | Default | Description |
|------|---------|-------------|
| `--max-nodes` | 200 | How many papers in final graph |
| `--max-depth` | 3 | How many citation hops |
| `--size` | medium | Preset: small (100), medium (500), large (2000) |
| `--years` | 5 | For topic search: years back |

```bash
# large network
python -m refnet.cli --doi 10.1073/pnas.1818339116 \
  --max-nodes 500 \
  --max-depth 4 \
  --size large
```

### Step 3: Enable/Disable Features

```bash
# disable author expansion (faster)
python -m refnet.cli --doi 10.xxx --no-authors

# disable trajectory analysis
python -m refnet.cli --doi 10.xxx --no-trajectory

# disable gap analysis
python -m refnet.cli --doi 10.xxx --no-gap

# enable S2 fallback for resilience
python -m refnet.cli --doi 10.xxx --fallback
```

### Step 4: Choose Output

```bash
# specify output directory
python -m refnet.cli --doi 10.xxx -o /path/to/output

# choose format
python -m refnet.cli --doi 10.xxx --format json      # just JSON
python -m refnet.cli --doi 10.xxx --format graphml   # for Gephi
python -m refnet.cli --doi 10.xxx --format csv       # spreadsheets
python -m refnet.cli --doi 10.xxx --format html      # viewer only
python -m refnet.cli --doi 10.xxx --format all       # everything (default)
```

---

## Example: Your aaRS Literature

```bash
cd /storage/kiran-stuff/ref-network-app

# run on your aaRS collection
python -m refnet.cli \
  --collection /storage/kiran-stuff/aaRS/literature/ \
  --enrich \
  --seed-limit 20 \
  --max-nodes 500 \
  --fallback \
  -o /storage/kiran-stuff/aaRS/literature/network

# check results
ls /storage/kiran-stuff/aaRS/literature/network/
```

---

## Output Files Explained

| File | What It Contains |
|------|------------------|
| `graph.json` | Full network: papers, authors, trajectories, gap analysis |
| `graph.graphml` | For Gephi/yEd visualization |
| `nodes.csv` | Paper metadata (title, year, citations, DOI) |
| `edges.csv` | Citation relationships (source, target, type) |
| `viewer.html` | Interactive browser visualization |
| `candidates.db` | SQLite database of all discovered papers |

### JSON Structure
```json
{
  "meta": {"node_count": 487, "edge_count": 1823, ...},
  "nodes": [...],           // papers with scores
  "edges": [...],           // citation links
  "authors": [...],         // key researchers
  "trajectories": [...],    // research focus shifts
  "clusters": [...],        // topic clusters
  "gap_analysis": {
    "bridges": [...],       // bridge papers
    "missing_links": [...], // research gaps
    "unexplored": [...]     // nearby clusters not pulled in
  }
}
```

---

## Debugging

```bash
# verbose mode - see everything
python -m refnet.cli --doi 10.xxx -v

# quiet mode - minimal output
python -m refnet.cli --doi 10.xxx -q

# check provider stats after run
# (printed automatically if errors occurred)
```

---

## What SPEC2.md Features Are Implemented?

| Feature | Status |
|---------|--------|
| Scientist mode (citation-walk) | ✅ |
| DOI/title/BibTeX/collection input | ✅ |
| Topic triage (RED/YELLOW/GREEN) | ✅ |
| Bidirectional citations | ✅ |
| INTRO_HINT_CITES heuristic | ✅ |
| CandidatePool (SQLite) | ✅ |
| WorkingGraph (bounded) | ✅ |
| Hub suppression | ✅ |
| Author expansion | ✅ |
| Author trajectories (JSD drift) | ✅ |
| AuthorBridge detection | ✅ |
| Gap analysis | ✅ |
| OpenAlex provider | ✅ |
| Semantic Scholar provider | ✅ |
| Provider fallback | ✅ |
| Retry + circuit breaker | ✅ |
| JSON/GraphML/CSV/HTML export | ✅ |
| CLI with all options | ✅ |
| FastAPI backend | ❌ (CLI only) |
| REST API endpoints | ❌ (CLI only) |
| React/Next.js frontend | ❌ (static HTML) |
| PDF intro parsing (GROBID) | ❌ |
| Docker/docker-compose | ❌ |
| Automated tests | ❌ |

**Core functionality: 100% implemented**
**Web API/frontend: CLI-based alternative**

---

## Common Recipes

### Build network from your key papers
```bash
python -m refnet.cli \
  --doi 10.1073/pnas.1818339116 \
  --doi 10.1126/science.aax3194 \
  --max-nodes 300 \
  -o proRS_thrRS_network
```

### Build from existing collection with full features
```bash
python -m refnet.cli \
  --collection /path/to/papers/ \
  --enrich \
  --seed-limit 30 \
  --max-nodes 500 \
  --max-depth 4 \
  --fallback \
  -o full_network
```

### Quick small network for testing
```bash
python -m refnet.cli \
  --doi 10.1073/pnas.1818339116 \
  --max-nodes 50 \
  --max-depth 2 \
  --no-authors \
  --no-trajectory \
  -o test_network
```

### Export for Gephi
```bash
python -m refnet.cli --doi 10.xxx --format graphml -o gephi_export
# open gephi_export/graph.graphml in Gephi
```

---

## Troubleshooting

### "No seeds found"
- Check DOI format: should be `10.xxxx/xxxxx`
- Try exact title with `--title`
- Check network connectivity

### "Topic too broad"
- Use `--doi` instead of topic
- Add more specific terms
- Reduce `--years`

### API errors / timeouts
```bash
# enable fallback
python -m refnet.cli --doi 10.xxx --fallback

# check logs
python -m refnet.cli --doi 10.xxx -v
```

### Out of memory
```bash
# reduce network size
python -m refnet.cli --doi 10.xxx --max-nodes 100 --size small
```

---

## Environment Variables (Optional)

```bash
# Semantic Scholar API key (faster rate limits)
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"

# then run with fallback
python -m refnet.cli --doi 10.xxx --fallback
```

---

## Files Location

```
/storage/kiran-stuff/ref-network-app/
├── refnet/              # main package
│   ├── cli.py           # command-line interface
│   ├── core/            # models, config, db, resilience
│   ├── providers/       # OpenAlex, S2, composite
│   ├── graph/           # expansion engine
│   ├── layers/          # author, trajectory
│   ├── analysis/        # hub, gap
│   ├── scoring/         # relevance
│   └── export/          # formats, viewer
├── README.md            # overview
├── MANUAL.md            # this file
├── SPEC.md              # original spec
└── SPEC2.md             # detailed spec
```
