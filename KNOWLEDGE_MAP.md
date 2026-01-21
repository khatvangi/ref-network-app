# RefNet Knowledge Map System

## Overview

RefNet has evolved from a simple citation network builder to a **multi-layer knowledge mapping system**. This document explains the full architecture.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       KNOWLEDGE MAP LAYERS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LAYER 5: REVIEW CURATION                                              │
│   ────────────────────────                                              │
│   TOC extraction → Topic coverage matrix → Gap analysis                 │
│   "WHAT DO EXPERTS SUMMARIZE?"                                          │
│                                                                          │
│   LAYER 4: AUTHOR NETWORK                                               │
│   ───────────────────────                                               │
│   Mega-authors → Collaborations → Other publications → Trajectories     │
│   "WHO ARE THE KEY PEOPLE?"                                             │
│                                                                          │
│   LAYER 3: GOOGLE SCHOLAR                                               │
│   ───────────────────────                                               │
│   Top papers → Citation counts → PDF links → Reviews                    │
│   "WHAT IS GLOBALLY IMPORTANT?"                                         │
│                                                                          │
│   LAYER 2: CONCEPT DENDRIMER (LLM-generated)                            │
│   ──────────────────────────────────────────                            │
│   Central topic → LLM expansion → Sub-concepts → Frontier topics        │
│   "WHAT CONNECTS TO WHAT?"                                              │
│                                                                          │
│   LAYER 1: PAPER DENDRIMER (Citation-based)                             │
│   ─────────────────────────────────────────                             │
│   Seed DOIs → refs + cites → Buckets → Network                          │
│   "WHO CITES WHOM?"                                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Paper Dendrimer

**Purpose**: Build citation networks from seed papers.

**How it works**:
1. Start with seed DOI(s)
2. For each paper, get references AND citations
3. Each paper creates its own "bucket"
4. Branches prune when relevance drops < 0.15
5. Result: comprehensive citation network

**Output**: SQLite database with papers + edges

**Key insight**: Seeds determine everything! Wrong seed = wrong network.

**Commands**:
```bash
./refnet.py build --seed DOI --name my_network
```

---

## Layer 2: Concept Dendrimer

**Purpose**: Map semantic relationships between topics using LLM.

**How it works**:
1. Give LLM a central concept (e.g., "aminoacyl-tRNA synthetases")
2. LLM identifies related concepts: upstream, downstream, parallel
3. Recursively expand sub-concepts ("turtles all the way down")
4. Result: hierarchical semantic map

**Output**: Text/JSON concept hierarchy

**Tools**:
- Gemma (27b) - fast, good quality
- Qwen (32b) - more detailed

**Example prompt**:
```
Given the concept "aminoacyl-tRNA synthetases", identify:
- UPSTREAM: What concepts enable/feed into this?
- DOWNSTREAM: What concepts does this enable?
- PARALLEL: What related concepts exist at same level?
- CROSS-CONNECTIONS: What unexpected links exist?
```

**Files**:
- `aars_knowledge_map/gemma_concept_dendrimer.txt` - Level 1 concept map
- `aars_knowledge_map/ncaa_expansion.txt` - Level 2-3 expansion

---

## Layer 3: Google Scholar Integration

**Purpose**: Validate networks + get global citation data + PDF links.

**How it works**:
1. Search Google Scholar for topic
2. Extract papers with citation counts + PDF URLs
3. Cross-reference with paper dendrimer (expect ~98% overlap)
4. Identify missing papers as potential new seeds
5. Extract REVIEWS separately (goldmine!)

**Output**:
- `google_scholar` table in candidates.db
- `google_scholar_reviews` table with PDF links

**Key insight**: Reviews are goldmines because they provide curated overviews.

**Files**:
- `aars_knowledge_map/review_pdfs.json` - 149 reviews with PDF links
- `aars_knowledge_map/minimum_viable_reviews.json` - top+recent per sub-field

---

## Layer 4: Review Curation Protocol

**Purpose**: Extract knowledge structure from review articles.

**Key insight**: TOC > Abstract. Table of Contents shows ALL topics covered.

**How it works**:
1. Extract TOC from PMC reviews (section headers)
2. Build topic coverage matrix
3. Identify gaps between reviews
4. LLM can analyze TOCs to recommend reading order

**Tools**:
- `toc_extractor.py` - extracts section headers from PMC

**Files**:
- `aars_knowledge_map/review_tocs.json` - 20 review TOCs
- `aars_knowledge_map/topic_coverage.json` - coverage matrix
- `aars_knowledge_map/REVIEW_CURATION_PROTOCOL.md` - full protocol

---

## The Fractal Vision: Turtles All The Way Down

Each concept can spawn its own dendrimer:

```
Level 0: "aaRS"
    │
    └─► Level 1: "Non-canonical amino acids"
            │
            └─► Level 2: "Orthogonal tRNA/aaRS pairs"
                    │
                    └─► Level 3: "Pyrrolysyl-tRNA synthetase"
                            │
                            └─► Level 4: "Active site mutations"
                                    │
                                    └─► (continues...)
```

At each level, you can:
1. Run paper dendrimer (find DOIs)
2. Run concept dendrimer (expand semantically)
3. Search Google Scholar (validate)
4. Extract review TOCs (curate)

---

## aaRS Knowledge Map (Complete Example)

**Paper Dendrimer**:
- 12 networks merged
- 145,447 papers
- 285,161 edges
- Database: `/tmp/aars_merged/candidates.db` (64 MB)

**Concept Dendrimer** (Gemma-generated):
- Core: aaRS accuracy, activation, structure
- Branch A: Fundamental biology
- Branch B: Evolution/origins
- Branch C: Cellular context
- Branch D: Disease/clinical
- Frontier: ncAA, cellular sensors, phase separation

**Google Scholar**:
- 500 papers extracted
- 98% overlap with dendrimer
- 200 reviews
- 149 with PDF links

**Review TOCs**:
- 20 reviews analyzed
- Coverage gaps identified: editing, inhibitors, structure

---

## Quick Reference

| Layer | Tool | Input | Output |
|-------|------|-------|--------|
| Paper | `refnet.py build` | Seed DOIs | Citation network |
| Concept | Ollama + Gemma/Qwen | Central topic | Semantic hierarchy |
| Scholar | `scholarly` library | Search query | Papers + PDFs |
| Reviews | `toc_extractor.py` | PMC IDs | Section structure |

---

## Files Structure

```
ref-network-app/
├── MANUAL.md              # How RefNet works
├── ONBOARDING.md          # Developer guide
├── KNOWLEDGE_MAP.md       # THIS FILE - full system overview
├── refnet.py              # Main CLI
├── aars_knowledge_map/    # aaRS case study
│   ├── COMPLETE_DENDRIMER_REPORT.md
│   ├── GOOGLE_SCHOLAR_INTEGRATION.md
│   ├── REVIEW_CURATION_PROTOCOL.md
│   ├── gemma_concept_dendrimer.txt
│   ├── ncaa_expansion.txt
│   ├── review_tocs.json
│   ├── review_pdfs.json
│   ├── topic_coverage.json
│   ├── minimum_viable_reviews.json
│   └── toc_extractor.py
└── outputs/               # Network outputs
```

---

## Layer 4: Author Network (NEW!)

**Purpose**: Map key researchers, collaborations, and hidden field connections.

**What it provides**:

### 1. Mega-Authors
The most prolific/influential researchers in the field:

| Author | Papers | Citations | h-index |
|--------|--------|-----------|---------|
| Dieter Söll | 636 | 36,742 | 96 |
| Paul Schimmel | 577 | 30,160 | 92 |
| Michael Ibba | 281 | 11,693 | 58 |

### 2. Collaboration Network
Who works with whom:
```
Söll ↔ Ibba: 4 papers (founding fathers!)
Guo ↔ Schimmel: 3 papers (non-translational)
Schimmel ↔ Yang: 3 papers (Scripps group)
```

### 3. Other Publications
What else do mega-authors work on?
- Reveals hidden connections to other fields
- Söll: selenocysteine, archaeal genomes
- Schimmel: biophysical chemistry, stem cells
- Ibba: cysteine biosynthesis, elongation factors

### 4. Author Trajectories
How has an author's focus evolved over time?

**Data sources**:
- Google Scholar (from our extraction)
- OpenAlex API (author IDs, h-index, institutions)

**Files**:
- `data/mega_authors_openalex.json` - enriched author profiles
- `data/author_collaboration_network.json` - collaboration graph
- Database tables: `authors`, `author_papers_gs`, `author_collaborations`

**Key queries**:
```sql
-- Top authors by papers
SELECT name, paper_count, total_citations 
FROM authors ORDER BY paper_count DESC;

-- Top collaborations
SELECT * FROM author_collaborations 
ORDER BY paper_count DESC LIMIT 20;
```

---

## The Five Layers Together

| Layer | Question | Data Source | Key Insight |
|-------|----------|-------------|-------------|
| 1. Paper | "Who cites whom?" | CrossRef/S2 | Citation structure |
| 2. Concept | "What connects?" | LLM (Gemma) | Semantic map |
| 3. Scholar | "Global impact?" | Google Scholar | Validation + PDFs |
| 4. Author | "Who matters?" | OpenAlex | People + collabs |
| 5. Review | "Expert view?" | PMC TOCs | Curated knowledge |

**Together**: Comprehensive field understanding from multiple angles.
