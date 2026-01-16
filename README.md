# RefNet - Scientist-Centric Citation Network Builder

RefNet builds citation networks from seed papers using a scientist-centric approach. Unlike keyword-based tools, RefNet follows citation trails to discover relevant literature organically.

## Features

- **3-Layer Architecture**: Papers, Authors, and Trajectories
- **Citation-Walk Expansion**: Forward citations + backward references
- **Author Trajectory Analysis**: Detect research focus shifts using Jensen-Shannon divergence
- **Hub Suppression**: Prevents methodology papers from dominating the network
- **Gap Analysis**: Identifies missing links and research opportunities
- **Resilient API Clients**: Retry logic, circuit breakers, provider fallback

## Installation

```bash
cd /storage/kiran-stuff/ref-network-app
pip install httpx bibtexparser  # dependencies
```

Set your Semantic Scholar API key (optional, for fallback):
```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key-here"
```

## Quick Start

```bash
# from a DOI (recommended)
python -m refnet.cli --doi 10.1038/s41586-021-03819-2

# from multiple DOIs
python -m refnet.cli --doi 10.1234/paper1 --doi 10.5678/paper2

# from a paper title
python -m refnet.cli --title "Highly accurate protein structure prediction"

# from a BibTeX file
python -m refnet.cli --bib my_library.bib

# from a collection (JSON/CSV directory)
python -m refnet.cli --collection /path/to/papers/ --enrich

# topic bootstrap (less preferred)
python -m refnet.cli "aminoacyl-tRNA synthetases evolution"
```

## CLI Options

### Input Modes
| Flag | Description |
|------|-------------|
| `--doi DOI` | Start from paper DOI (can specify multiple) |
| `--title TEXT` | Lookup paper by title |
| `--bib FILE` | Import seeds from BibTeX file |
| `--collection PATH` | Load from JSON/CSV collection |
| `TOPIC` | Topic search for bootstrap (last resort) |

### Expansion Settings
| Flag | Default | Description |
|------|---------|-------------|
| `--max-nodes`, `-n` | 200 | Maximum nodes in working graph |
| `--max-depth`, `-d` | 3 | Maximum expansion depth |
| `--years`, `-y` | 5 | Years back for seed search |
| `--size` | medium | Graph size preset (small/medium/large) |
| `--aggressiveness` | medium | Exploration aggressiveness (low/medium/high) |

### Feature Toggles
| Flag | Description |
|------|-------------|
| `--no-authors` | Disable author expansion |
| `--no-trajectory` | Disable trajectory analysis |
| `--no-gap` | Disable gap analysis |
| `--fallback` | Enable S2 fallback if OpenAlex fails |

### Output
| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | output | Output directory |
| `--format`, `-f` | all | Export format (json/graphml/csv/html/all) |
| `--no-viewer` | | Don't generate HTML viewer |

### Debugging
| Flag | Description |
|------|-------------|
| `--quiet`, `-q` | Minimal output |
| `--verbose`, `-v` | Debug output with full logging |

## Output Files

After running, you'll find in the output directory:

| File | Description |
|------|-------------|
| `graph.json` | Full network data with papers, authors, trajectories |
| `graph.graphml` | For Gephi/yEd visualization |
| `nodes.csv` | Paper metadata |
| `edges.csv` | Citation relationships |
| `viewer.html` | Interactive browser visualization |
| `candidates.db` | SQLite database of all discovered papers |

## Architecture

```
refnet/
├── core/           # models, config, database, resilience
├── providers/      # OpenAlex, Semantic Scholar, composite
├── graph/          # candidate pool, working graph, expansion engine
├── layers/         # author layer, trajectory analysis
├── analysis/       # hub detection, gap analysis
├── scoring/        # relevance scoring
├── inputs/         # collection loaders
└── export/         # JSON, GraphML, CSV, HTML viewer
```

### Three-Layer Model

1. **Papers Layer**: Citation network with seeds, references, citations
2. **Authors Layer**: Key researchers linked to their papers
3. **Trajectories Layer**: Research focus evolution over time

### Expansion Process

1. Start with seed papers (DOIs, titles, or BibTeX)
2. Fetch backward references (what seeds cite)
3. Fetch forward citations (what cites seeds)
4. Expand top authors' other relevant work
5. Score and materialize candidates into working graph
6. Repeat until depth/budget exhausted
7. Compute trajectories and gap analysis

## Resilience Features

RefNet is designed to be robust against API failures:

### Retry with Exponential Backoff
- Up to 3 retries per request
- Delays: 1s → 2s → 4s (with jitter)
- Handles rate limits (429) and server errors (5xx)

### Circuit Breaker
- Opens after 5 consecutive failures
- Stays open for 60s (OpenAlex) or 120s (S2)
- Tests recovery with 3 calls before closing

### Provider Fallback
```bash
# use --fallback to enable S2 as backup
python -m refnet.cli --doi 10.1234/example --fallback
```

### Graceful Degradation
- Continues if individual papers fail
- Saves partial results on interrupt (Ctrl+C)
- Reports all errors in summary

## Example: aaRS Literature Network

```bash
python -m refnet.cli \
  --collection /path/to/aars_papers/ \
  --enrich \
  --seed-limit 20 \
  --max-nodes 500 \
  --fallback \
  -o aars_network
```

Output:
```
--- SUMMARY ---
Nodes: 487 (papers: 450, authors: 37)
Edges: 1823
Seeds: 20
Duration: 45.2s
API calls: 127
```

## Programmatic Usage

```python
from refnet.providers import create_default_provider
from refnet.graph.expansion import ExpansionEngine
from refnet.core.config import RefnetConfig
from refnet.export.formats import GraphExporter

# create provider with fallback
provider = create_default_provider(
    email="your@email.com",
    s2_api_key="optional-key"
)

# get seed papers
seeds = [provider.get_paper("10.1234/your-doi")]

# configure and run
config = RefnetConfig()
config.expansion.max_depth = 3

engine = ExpansionEngine(provider, config)
job = engine.build(seeds)

# export
exporter = GraphExporter("output")
exporter.export_json(job.graph, "network.json", job.gap_analysis)

# check stats
print(f"Papers: {len(job.graph.papers)}")
print(f"Authors: {len(job.graph.authors)}")
print(f"Errors: {job.stats.errors}")
```

## Trajectory Analysis

RefNet detects when authors shift research focus:

```python
# trajectories are computed automatically
for author in job.graph.authors.values():
    if author.trajectory_computed:
        for event in author.drift_events:
            print(f"{author.name}: {event.from_focus} → {event.to_focus}")
            print(f"  Magnitude: {event.drift_magnitude:.2f}")
            if event.is_novelty_jump:
                print("  ** NOVELTY JUMP **")
```

Drift is measured using Jensen-Shannon divergence on topic distributions. A drift magnitude ≥ 0.55 is flagged as a "novelty jump".

## Gap Analysis

Identifies research opportunities:

```python
gap = job.gap_analysis

# bridge papers connecting clusters
for bridge in gap.bridges:
    print(f"Bridge: {bridge.paper_title}")
    print(f"  Connects: {bridge.cluster_a_id} ↔ {bridge.cluster_b_id}")

# unexplored areas
for cluster in gap.unexplored_clusters:
    print(f"Unexplored: {cluster.name} ({cluster.candidate_paper_count} papers)")
```

## Troubleshooting

### "No seeds found"
- Check DOI format (should be like `10.1234/...`)
- Try `--title` with exact paper title
- Ensure network connectivity

### "Topic too broad"
- Add anchor papers with `--doi`
- Narrow search terms
- Reduce `--years` parameter

### API errors
- Enable `--fallback` for S2 backup
- Check rate limits (OpenAlex: 10/sec, S2: 100/5min without key)
- Use `--verbose` to see detailed logs

### Circuit breaker open
- Wait 60-120s for recovery
- Check if API is actually down
- Errors are logged, partial results saved

## License

MIT

## Citation

If you use RefNet in your research:

```bibtex
@software{refnet2026,
  title = {RefNet: Scientist-Centric Citation Network Builder},
  year = {2026},
  url = {https://github.com/yourusername/ref-network-app}
}
```
