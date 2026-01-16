# RefNet - Universal Citation Network Builder

## One Command, Any Input

```bash
cd /storage/kiran-stuff/ref-network-app
python refnet.py YOUR_INPUT --all-layers
```

## Supported Inputs

| Input Type | Example |
|------------|---------|
| DOI | `python refnet.py --doi 10.1038/s41586-021-03819-2 --all-layers` |
| Multiple DOIs | `python refnet.py --doi DOI1 --doi DOI2 --all-layers` |
| PDF | `python refnet.py --pdf paper.pdf --all-layers` |
| BibTeX | `python refnet.py --bib library.bib --all-layers` |
| Zotero JSON | `python refnet.py -c zotero_export.json --all-layers` |
| CSV | `python refnet.py -c papers.csv --all-layers` |
| Topic | `python refnet.py "CRISPR evolution" --all-layers` |
| Topic (niche) | `python refnet.py "reduced amino acid alphabet" --all-layers --use-gscholar` |

## Decision Tree

```
INPUT (any format)
       │
       ▼
  ┌─────────────────────────────────────┐
  │          SEED DISCOVERY             │
  │                                     │
  │  DOI/PDF/BibTeX → direct lookup     │
  │  Collection → load + enrich         │
  │  Topic → OpenAlex search            │
  │  Topic + --use-gscholar →           │
  │    Google Scholar → OpenAlex DOI    │
  └────────────────┬────────────────────┘
                   │
            ┌──────▼──────┐
            │   ENRICH    │ ← OpenAlex metadata
            └──────┬──────┘
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
 REFERENCES   CITATIONS     AUTHORS
 (backward)   (forward)   (3rd axis)
      │            │            │
      └────────────┴────────────┘
                   │
            ┌──────▼──────┐
            │ GUARDRAILS  │
            │ • hub limit │
            │ • relevance │
            └──────┬──────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
CLUSTERING    TRAJECTORY      GAP
(Louvain)    (JSD drift)   ANALYSIS
     │             │             │
     └─────────────┴─────────────┘
                   │
                   ▼
                EXPORT
         (json/html/graphml/csv)
```

## Options

```bash
--all-layers          # use modern pipeline with all layers (REQUIRED)
--max-nodes N         # max papers in graph (default: 200)
--max-depth N         # citation depth (default: 2)
--use-gscholar        # enable Google Scholar for topic search (RECOMMENDED for niche fields)
--seed-limit N        # max seeds from collection (default: 30)
--no-authors          # disable author expansion
--no-trajectory       # disable JSD trajectory analysis
-o DIR                # output directory (default: refnet_output/)
```

## When to Use Google Scholar

**Use `--use-gscholar` when:**
- Searching for niche/specialized topics
- OpenAlex returns generic/unrelated papers
- You need papers from specific subfields

**How it works:**
1. Google Scholar finds papers by relevance (better for niche topics)
2. OpenAlex looks up DOIs via title matching
3. Only papers with verified DOIs proceed to expansion

```bash
# niche topic - use Google Scholar
python refnet.py "reduced amino acid alphabet" --all-layers --use-gscholar

# broad topic - OpenAlex is fine
python refnet.py "protein structure prediction" --all-layers
```

## Examples

### From a single paper (DOI)
```bash
python refnet.py --doi 10.1093/bioinformatics/btae061 --all-layers -o my_network
```

### From multiple seed papers
```bash
python refnet.py --doi 10.1093/bioinformatics/btae061 --doi 10.1016/j.csbj.2022.07.001 --all-layers
```

### From your Zotero library
```bash
# export from Zotero: File > Export Library > JSON
python refnet.py -c my_library.json --all-layers --seed-limit 20 --max-nodes 300
```

### From a PDF you're reading
```bash
python refnet.py --pdf interesting_paper.pdf --all-layers
```

### From a niche topic (with Google Scholar)
```bash
python refnet.py "reduced amino acid alphabet protein" --all-layers --use-gscholar -o reduced_aa_network
```

### From a broad topic (OpenAlex)
```bash
python refnet.py "CRISPR gene editing" --all-layers --max-nodes 200
```

## Output Files

```
refnet_output/
├── viewer.html      # interactive visualization (open in browser)
├── graph.json       # full network data + gap analysis
├── graph.graphml    # for Gephi/Cytoscape
├── nodes.csv        # papers table
├── edges.csv        # citations table
└── candidates.db    # SQLite pool (all discovered papers)
```

## Layers Explained

1. **Citation Expansion**: Follows references (backward) and citations (forward) from seeds
   - **INTRO_HINT_CITES**: First 25% of references (clamped 10-40, max 20) get boosted weight (2.0x)
   - Mimics how scientists read papers: intro refs define the problem space
2. **Author Expansion**: Fetches works by first/last authors of key papers
3. **Trajectory Analysis**: Uses Jensen-Shannon divergence to detect when authors shifted fields
4. **Clustering**: Groups papers into research communities (Louvain algorithm)
5. **Gap Analysis**: Identifies unexplored areas in the candidate pool

## Guardrails

- **Hub Suppression**: Methodology papers (BLAST, AlphaFold with >5k cites) don't explode the graph
- **Relevance Filter**: Papers drifting off-topic are deprioritized
- **Mega-Author Skip**: Authors with >1000 papers don't dominate expansion

## Alternative: Direct CLI

```bash
python -m refnet.cli --doi 10.1093/molbev/msaf197 -o output/
python -m refnet.cli --collection papers.json --max-nodes 300
```

---

## Changelog

### 2026-01-16
- **Fixed**: `run_modern_pipeline()` now uses correct API calls
  - `ExpansionEngine(provider, config)` instead of wrong 4-arg call
  - `engine.build()` instead of `engine.expand()`
  - Proper config initialization for max_nodes
- **Fixed**: INTRO_HINT_CITES logic now matches SPEC.md
  - First 25% of references (clamped 10-40, max 20) get boosted weight (2.0x)
  - Previously used flat threshold instead of percentage
  - Mimics how scientists read papers: intro refs define problem/context
- **Added**: Google Scholar + OpenAlex combo for niche topic searches
  - Google Scholar finds relevant papers by topic
  - OpenAlex looks up DOIs via title matching
  - Solves problem of OpenAlex returning generic papers for specialized fields
- **Added**: Multiple DOI support (`--doi DOI1 --doi DOI2`)
- **Added**: `--seed-limit` option for collections
- **Tested**: Successfully built "reduced amino acid alphabet" network (32 papers, 17 seeds)

### 2026-01-15
- Added Google Scholar provider with rate limiting
- Added author trajectory analysis (JSD drift detection)
- Created unified `refnet.py` wrapper with `--all-layers` flag
