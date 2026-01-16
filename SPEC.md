# Reference Network App - V0 Specification

## 0) Non-negotiable Rules (Anti-slop)

- Do NOT invent API fields/endpoints. If unsure, consult official docs and leave short in-code comments pointing to the exact doc page used.
- Prefer OpenAlex + Semantic Scholar for graph edges. Use Crossref/Unpaywall only as helpers.
- No scraping Google directly. If "Google search" is required, implement a pluggable SearchProvider with a stub + optional SerpAPI (or other legal API) adapter. Default must run without paid keys.
- Ship runnable code + tests + README. If something can't be implemented without keys, it must degrade gracefully and still run.
- Keep scope disciplined: build v0 that produces a useful citation network for a topic and visualizes it.
- Enforce guardrails aggressively. The app must refuse to build graphs for over-broad topics until user narrows scope.

---

## 1) Product Goal (What the App Does)

User enters a topic string (e.g., "ancestral protein reconstruction").

The app:
1. Runs Topic Scope Triage (preflight) to detect if the query is too broad.
2. If too broad, forces a Narrowing Wizard that contracts the topic into a safe query.
3. Finds recent (last N years, default N=3) high-signal seed papers (prefer reviews + highly cited).
4. Expands a network iteratively using three signal channels:
   - A) Backward citations (references in the paper).
   - B) Forward citations (papers that cite this paper).
   - C) Intro signal:
     - Primary: INTRO_CITES extracted from actual Introduction section if OA PDF parsing succeeds.
     - Fallback: INTRO_HINT_CITES heuristic that boosts the first k references (10-40; default 25% of reference list) as intro-likely, with strict hub + relevance gates.
5. Repeats expansion ("rinse and repeat") until strict stop conditions are met.
6. Outputs:
   - Interactive graph visualization (nodes=papers, edges=typed relationships).
   - Filters: year range, relevance threshold, node type (review/primary), edge type.
   - Export: GraphML + CSV (nodes.csv, edges.csv) + JSON snapshot.

---

## 1.5) HARD GUARDRAIL: Topic Scope Triage + Narrowing Wizard

Before any graph build, the system MUST run scope triage. If the query is too broad, it must refuse to build until the user narrows it via a short wizard.

### 1.5.1 Scope Triage Endpoint + Data Model

Implement:

**POST /api/topic/triage**
- body: { topic: string, years_back: int }
- returns TopicTriage:
  - count_estimate (OpenAlex meta.count for works within last years_back)
  - scope_risk_score (0..1)
  - scope_band in {GREEN, YELLOW, RED}
  - top_concepts: [{name, score}]
  - top_venues: [{name, count}]
  - seed_preview: top 10 papers (title, year, doi/id, citation_count)
  - recommended_facets: {subdomain_options, method_options, organism_options} (derived from concepts/keywords)

Persist TopicProfile in DB:
- topic_original
- topic_refined
- triage_metrics JSON
- user_facets JSON

### 1.5.2 Scope Risk Scoring (v0 Heuristic, Deterministic)

Compute ScopeRiskScore using these signals:
- volume_score = clamp(log10(count_estimate)/5, 0, 1)
- specificity_score = clamp((num_specific_tokens + num_proper_nouns)/10, 0, 1)
- concept_entropy_score = normalized Shannon entropy over top_concepts (0..1)
- venue_dispersion_score = normalized unique venues / total seed_preview (0..1)

**Formula:**
```
ScopeRiskScore = 0.45*volume_score + 0.25*concept_entropy_score + 0.20*venue_dispersion_score + 0.10*(1 - specificity_score)
```

**Band rules:**
- RED if count_estimate > 20000 OR ScopeRiskScore > 0.75
- YELLOW if count_estimate in (5000..20000] OR ScopeRiskScore in (0.55..0.75]
- GREEN otherwise

**Hard rule:**
- The system MUST refuse POST /api/graph/build if latest triage band is RED.

### 1.5.3 Wizard UX + Refinement Endpoint (Bounded, Forced-choice)

Frontend must show a "Narrow Topic" wizard when band is RED (mandatory) or YELLOW (strongly recommended).

**Wizard steps (forced-choice + minimal text):**

1. Choose subdomain frame (pick 1-2):
   - Methods/algorithms
   - Benchmarking/accuracy
   - Case studies on a protein family
   - Evolutionary theory/assumptions
   - Experimental validation
   - Applications (enzymes/membranes/pathogens/etc.)

2. Choose object constraints (at least one required):
   - protein family OR molecular function (free text)
   - organism clade (choose or free text)

3. Choose method constraints (0-3) from suggested list + optional text

4. Choose year window (default years_back)

5. Optional exclusions (checkboxes): exclude review, exclude deep learning, exclude viral, etc.

**POST /api/topic/refine**
- body: { topic_original, facets, years_back }
- returns:
  - topic_refined_query_string (provider-ready boolean query)
  - updated triage metrics (same as /triage)
  - warnings if still too broad

**Additional contraction rule:**
- If user remains RED after 2 wizard passes, force them to add at least one: (protein family/entity), (organism clade), (method constraint), or (timeframe <= 5 years).

### 1.5.4 Query Construction Must Be Transparent

Construct refined query as boolean:
- MUST terms from object constraints
- SHOULD terms from methods
- add exclusion terms (NOT)

Show user the exact query string and the estimated count before building.

---

## 2) Core Model & Algorithm

### 2.1 Entities

**Paper:**
- internal_id (uuid)
- doi (nullable), openalex_id (nullable), s2_id (nullable)
- title, venue, year, authors (light)
- abstract (nullable)
- is_review (best-effort heuristic)
- citation_count (from provider, nullable)
- reference_count (nullable)
- url (nullable)
- oa_pdf_url (nullable)
- computed:
  - topic_relevance_score (0..1)
  - novelty_score (0..1) [penalize near-duplicates + too-general hubs]
  - priority_score = w1*relevance + w2*novelty + w3*recency + w4*citation_signal

**Edge:**
- src_paper_id, dst_paper_id
- type in {CITES, CITED_BY, INTRO_CITES, INTRO_HINT_CITES}
- weight (float; default 1.0; allow boosting intro edges)

### 2.2 Expansion Loop (Must Be Budgeted + Guarded)

Use a priority queue (max-heap) of candidate papers to expand.

**Inputs:**
- refined topic query string
- config: years_back, max_depth, max_nodes, max_edges, max_expand_per_node, min_relevance
- guardrails: per-channel budgets, degree caps, drift kill-switch

**Steps:**

1. **Seed selection:**
   - Query providers for papers matching refined query within years_back.
   - Prefer review articles; if missing flag, use heuristic keywords.
   - Pick top K seeds (default K=10) by combined (citation_count, recency, relevance).
   - Enforce seed-quality floor: citation_count >= 30 unless topic is extremely new (e.g., year_min >= current_year-1).

2. **For each expanded node:**
   - Fetch references (backward) + cited-by (forward) from providers.
   - Apply strict per-channel budgets (see 2.4).
   - Add INTRO_CITES if OA PDF parsing succeeds; else add INTRO_HINT_CITES fallback.
   - For each discovered paper:
     - Resolve identity (prefer DOI; else provider ids).
     - Fetch minimal metadata (title/year/citation_count/abstract if available).
     - Compute relevance score to topic using:
       - v0: BM25/keyword scoring over (title+abstract). Optional embeddings if installed.
     - Dedupe: canonicalize by DOI; else conservative fuzzy match (title+year+first_author).
     - Insert node/edge if passes relevance gate or "bridge allowance" (see 2.3.2).
   - Push high priority new nodes into PQ until limits reached.

3. **Stop conditions:**
   - max_nodes or max_edges reached
   - PQ empty
   - depth reached (store depth per node)
   - Drift kill-switch: if last M expansions add <X% nodes above relevance threshold, halt early.
   - Hub non-expansion: never expand nodes with degree >= degree_cap.

### 2.3 INTRO SIGNAL: Primary + Fallback

We distinguish:
- INTRO_CITES: extracted from actual Introduction section (only when OA PDF parsing succeeds).
- INTRO_HINT_CITES: fallback heuristic when intro text is unavailable.

#### 2.3.1 INTRO_HINT_CITES Heuristic (No PDF Required)

When OA PDF is missing or parsing fails:
- Obtain the reference list in provider-returned order.
- Let n = number of references.
- Compute k = clamp(round(n * intro_fraction), 10, 40), default intro_fraction=0.25.
- For i in [1..k], create edge type INTRO_HINT_CITES from paper -> ri with weight intro_hint_weight (default 2.0).
- For i > k, create normal CITES edges with weight 1.0.
- Also enforce a hard cap max_intro_hint_per_node (default 20) even if k > 20.

#### 2.3.2 Guardrails for INTRO_HINT_CITES

- Relevance gate: referenced paper must have relevance >= min_relevance_intro (default 0.20)
  OR (citation_count >= bridge_citation_min (default 500) AND relevance >= 0.15).
- Hub suppression: if citation_count >= hub_citation_threshold (default 50000) AND relevance < 0.30,
  do NOT boost; treat as normal CITES or skip entirely.
- Keep edge types distinct in exports and UI legend. Label INTRO_HINT_CITES explicitly as heuristic.

#### 2.3.3 INTRO_CITES via OA PDF Parsing (Best Effort)

If oa_pdf_url exists and pdf_parsing_enabled:
- Download + cache PDF.
- Parse with GROBID or a lightweight parser.
- Best-effort extract Introduction section boundaries.
- Identify cited works from Introduction; if mapping fails, skip INTRO_CITES safely (do not crash).

#### 2.3.4 Acceptance Tests

- When PDF parsing disabled/unavailable, graph still contains INTRO_HINT_CITES edges for nodes with >=10 references.
- UI can filter by edge type and shows counts per type.

### 2.4 Anti-explosion Guardrails (Must Be Enforced)

Implement these hard bounds even after narrowing:

**Per-node channel budgets:**
- max_refs_per_node = 50
- max_citedby_per_node = 30
- max_intro_hint_per_node = 20

**Edge budget per expansion:** never add more than (max_refs_per_node + max_citedby_per_node + max_intro_hint_per_node) edges per expanded node.

**Degree cap (hub avoidance):** if node degree >= degree_cap (default 80), do not expand this node (still allow it to exist).

**Hub filter on adding edges:** if candidate node has citation_count >= hub_citation_threshold and relevance is low, skip or do not boost.

**Drift kill-switch:** rolling window M=30 expansions; if % new nodes with relevance >= min_relevance falls below 10%, stop.

**Diversity guard (optional v0):** penalize additions that sharply increase concept entropy; deprioritize in PQ.

Always take top-N by priority_score, not random.

---

## 3) Tech Stack

**Backend: Python 3.11+**
- FastAPI for API server
- SQLite for v0 persistence (upgrade path to Postgres)
- SQLModel or SQLAlchemy
- httpx for API calls, tenacity for retries
- diskcache (or sqlite cache table) for caching provider responses + PDFs
- pydantic for configs

**Frontend:**
- Next.js (or Vite + React) with:
  - Cytoscape.js (or Sigma.js) for graph rendering
  - UI: topic input, triage meter (Green/Yellow/Red), wizard, run button, progress, graph view, filters, export buttons
  - Must show refined boolean query string before build starts

**DevOps:**
- docker-compose with backend + frontend
- .env.example containing API keys placeholders (OpenAlex usually no key; S2 may require key)
- Makefile with common commands

---

## 4) Provider Abstraction

Create /backend/providers/

**base.py: interface**
- search_papers(query, year_min, year_max, limit) -> list[PaperStub]
- get_paper(paper_id or doi) -> PaperFull
- get_references(paper_id or doi) -> list[PaperStub]
- get_citations(paper_id or doi) -> list[PaperStub]

**Implementations:**
- openalex.py
- semantic_scholar.py
- crossref.py (optional DOI resolution helper)
- unpaywall.py (optional OA PDF locator)

**Multi-provider merge:**
- Attempt DOI resolution; if DOI present, treat as canonical.
- If DOI absent, dedupe by fuzzy(title+year+first_author) with conservative thresholds.

---

## 5) API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/topic/triage | POST | Run scope triage on topic |
| /api/topic/refine | POST | Refine topic with facets |
| /api/graph/build | POST | Start graph build job (refuses if RED) |
| /api/graph/status/{job_id} | GET | Get job progress |
| /api/graph/result/{job_id} | GET | Get graph JSON |
| /api/graph/export/{job_id} | GET | Export (format=graphml/csv/json) |

---

## 6) Jobs + Progress

- Graph build runs as background task (FastAPI BackgroundTasks for v0)
- Persist job state in SQLite so refresh doesn't lose it
- Show progress bar and counters in UI

---

## 7) Acceptance Criteria

- Running the app starts successfully
- Entering 'chemistry' triggers RED and forces wizard; build is refused while RED
- After narrowing, count estimate drops; build proceeds
- Graph has typed edges (CITES, CITED_BY, INTRO_CITES, INTRO_HINT_CITES) and node metadata
- Hub nodes are not expanded (degree cap enforced)
- Channel budgets prevent edge explosion
- Export works
- Caching prevents repeated downloads
- Rate limiting respected (per-provider backoff)

**Unit tests must cover:**
- scope risk scoring + band rules
- query refinement construction
- relevance scoring
- dedupe logic
- provider normalization using fixtures (no live keys required)
- intro hint logic (k clamp + caps + hub suppression)

---

## 8) Deliverables (Repo Layout)

```
backend/
  main.py
  config.py
  models.py
  db.py
  jobs.py
  graph_builder.py
  topic_triage.py
  providers/
    __init__.py
    base.py
    openalex.py
    semantic_scholar.py
  scoring/
    __init__.py
    relevance.py
  export/
    __init__.py
    graphml.py
    csv_export.py
  tests/
    __init__.py
    test_triage.py
    test_scoring.py
    test_providers.py
frontend/
  src/ or app/
docker-compose.yml
README.md
.env.example
Makefile
```

---

## 9) Build Order

1. Backend skeleton + DB models + provider base + OpenAlex triage + refine + build loop (no UI yet) + return JSON.
2. Add Semantic Scholar provider and merge.
3. Add caching + rate limit + channel budgets + degree cap + drift kill-switch.
4. Add frontend triage meter + narrowing wizard + graph viewer.
5. Add export formats.
6. Add tests + fixtures.
7. Polish README.

---

## 10) Implementation Notes

- **DO NOT run infinite loops** - all expansion must respect hard limits
- **Ask clarifying questions** before major implementation decisions
- **Test incrementally** - verify each component works before moving on
- **Graceful degradation** - app must run without API keys (using cached fixtures or stubs)
