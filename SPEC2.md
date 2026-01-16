# Citation-Walk Literature Navigator (Scientist-Centric) — Claude Code / Ralph Loop Spec

/ralph-loop:ralph-loop "You are Claude Code operating as a senior full-stack engineer + research-infra engineer. Build a working v0 app, end-to-end, from an empty repo.

## 0) Non-negotiable rules (anti-slop)
- Do NOT invent API fields/endpoints. If unsure, consult official docs and leave short in-code comments pointing to the exact doc page used.
- Prefer OpenAlex + Semantic Scholar for graph edges. Use Crossref/Unpaywall only as helpers.
- No scraping Google directly. If “Google search” is required, implement a pluggable SearchProvider with a stub + optional SerpAPI (or other legal API) adapter. Default must run without paid keys.
- Ship runnable code + tests + README. If something can’t be implemented without keys, it must degrade gracefully and still run.
- Keep scope disciplined: ship a real v0 that works and is extensible.
- Default mode is SCIENTIST (citation-walk first). Keyword search is bootstrap only.

---

## 1) Product goal (what the app does)
The app mimics how a scientist actually surveys a field:

1) User starts with **an anchor** (preferred): DOI(s) / OpenAlex ID(s) / Semantic Scholar ID(s) / URL(s), or paper title+author/year.
2) If no anchor is supplied, the system uses topic string **only to find initial seeds** (bootstrap).
3) The map grows by **bidirectional citation-walk**:
   - what the paper cites (backward)
   - who cites the paper (forward)
4) Adds a **human layer**:
   - Author nodes and authored works in the same area
   - Author trajectories across time to explain cross-field bridges
5) Supports **controlled explosion**:
   - collect broadly into a CandidatePool
   - materialize a bounded WorkingGraph for UI
   - provide gap/bridge analysis from the CandidatePool
6) Outputs:
   - Interactive graph with typed nodes/edges
   - Filters by edge type, year, relevance, cluster
   - Exports: GraphML + CSV + JSON snapshot

---

## 1.2) Core philosophy: Scientist Mode = Citation-Walk Discovery (must implement)
This app is NOT a keyword-mapping engine.
In SCIENTIST mode (default):
- The topic string is used ONLY for seed acquisition when user provides no anchor.
- After seeds exist, expansion is driven primarily by graph signals (citations + authors), not keywords.
- Keywords/embeddings may be used as a weak tie-breaker only (<=10% weight).

Implement modes:
- mode = SCIENTIST (default)
- mode = KEYWORD (optional legacy)

---

## 1.3) Seed acquisition (bootstrap only; multiple entry methods)
Implement seed sources (in order):

A) User provides identifiers (preferred)
- DOI(s), OpenAlex ID(s), Semantic Scholar ID(s), or URLs

B) User provides anchor papers by title + author/year
- Resolve via OpenAlex/S2

C) Only if none provided: topic bootstrap
- Constrained search for top K seeds (default K=10), last N years (default 3)
- Prefer reviews + high-signal papers

Hard rule:
- After initial seed set is created, the system MUST stop issuing broad topic searches for expansion.
- Further growth is from citations/references/author works/portals.

---

## 1.5) Topic Scope Triage + Narrowing Wizard (bootstrap guardrail only)
This triage governs **seed search breadth**, not graph expansion.

### 1.5.1 Endpoints + model
POST `/api/topic/triage`
- body: `{ topic: string, years_back: int }`
- returns `TopicTriage`:
  - `count_estimate` (OpenAlex meta.count within years_back)
  - `scope_risk_score` (0..1)
  - `scope_band` ∈ {GREEN, YELLOW, RED}
  - `top_concepts`: [{name, score}]
  - `top_venues`: [{name, count}]
  - `seed_preview`: top 10 (title, year, doi/id, citation_count)
  - `recommended_facets`: {subdomain_options, method_options, organism_options}

Persist `TopicProfile`:
- topic_original, topic_refined
- triage_metrics JSON, user_facets JSON

### 1.5.2 Scope risk scoring
- volume_score = clamp(log10(count_estimate)/5, 0, 1)
- specificity_score = clamp((num_specific_tokens + num_proper_nouns)/10, 0, 1)
- concept_entropy_score = normalized Shannon entropy over top_concepts (0..1)
- venue_dispersion_score = normalized unique venues / total preview (0..1)

ScopeRiskScore =
- 0.45*volume_score +
- 0.25*concept_entropy_score +
- 0.20*venue_dispersion_score +
- 0.10*(1 - specificity_score)

Band rules:
- RED if count_estimate > 20000 OR ScopeRiskScore > 0.75
- YELLOW if count_estimate in (5000..20000] OR ScopeRiskScore in (0.55..0.75]
- GREEN otherwise

Hard rule:
- The system MUST refuse broad seed bootstrap if RED unless user provides an anchor DOI/title.
- For RED: force user to provide at least 1 anchor paper OR narrow via wizard.

### 1.5.3 Wizard (bounded forced-choice)
Frontend shows “Narrow Topic” wizard when RED (mandatory) or YELLOW (recommended).

Steps:
1) Subdomain frame (1–2)
2) Object constraint (at least one): protein family/function and/or organism clade
3) Method constraint (0–3)
4) Year window
5) Exclusions (checkboxes)

POST `/api/topic/refine`
- body: `{ topic_original, facets, years_back }`
- returns:
  - `topic_refined_query_string`
  - updated triage metrics
  - warnings if still too broad

Additional contraction:
- If still RED after 2 passes: force at least one of (entity/protein family), (organism), (method), or (timeframe ≤ 5 years)

### 1.5.4 Transparency
Show refined boolean query and estimated count before seeding.

---

## 2) Data model

### 2.1 Entities
**Paper**
- internal_id (uuid)
- doi (nullable), openalex_id (nullable), s2_id (nullable)
- title, venue, year, authors (light)
- abstract (nullable)
- is_review (best-effort heuristic)
- citation_count (nullable)
- reference_count (nullable)
- url (nullable)
- oa_pdf_url (nullable)

Computed:
- topic_relevance_score (weak tie-breaker only in SCIENTIST mode)
- novelty_score (penalize duplicates + generic hubs)
- priority_score (used internally)

**Author**
- internal_id (uuid)
- name
- orcid (nullable)
- affiliations (nullable)
- external_ids: {openalex_author_id?, s2_author_id?, orcid?}
Computed:
- author_topic_fit (0..1)
- author_centrality (0..1)
- author_priority

### 2.2 Edges
Paper-Paper:
- `CITES` (directed)
- `CITED_BY` (derived convenience edge for UI/export; do not double-count truth)
- `INTRO_CITES` (true intro extraction from PDF when possible)
- `INTRO_HINT_CITES` (fallback heuristic)

Paper-Author:
- `AUTHORED_BY` (Paper -> Author)
- `AUTHORED` (derived Author -> Paper)

Optional derived:
- `COAUTHOR` (Author -> Author; only if cheap)

---

## 2.3 Bidirectional citation-walk is mandatory
For any expanded paper P:

Backward (what P cites):
- references(P) → edges: `P -> R` type `CITES` (or intro variants)

Forward (who cites P):
- citing(P) → edges: `C -> P` type `CITES`
- additionally materialize derived `P -> C` type `CITED_BY` for UI/export convenience

Hard rule:
- Every expansion MUST attempt both:
  1) references(P)
  2) citing(P)
- If missing in provider, fallback provider (OpenAlex <-> S2). If still missing, proceed and log flags.

---

## 2.4 Intro signal: Primary + fallback

### 2.4.1 INTRO_HINT_CITES heuristic (no PDF required)
When OA PDF missing or parsing fails:
- reference list R = [r1..rn] in provider order
- k = clamp(round(n * intro_fraction), 10, 40), default intro_fraction=0.25
- For i in 1..k: edge type `INTRO_HINT_CITES`, weight intro_hint_weight (default 2.0)
- For i>k: edge type `CITES`, weight 1.0
- enforce cap: `max_intro_hint_per_node` (default 20) even if k > 20

### 2.4.2 Guardrails for INTRO_HINT_CITES
- relevance gate: referenced paper must have relevance ≥ min_relevance_intro (default 0.20)
  OR (citation_count ≥ bridge_citation_min (default 500) AND relevance ≥ 0.15)
- hub suppression: if citation_count ≥ hub_citation_threshold (default 50000) AND relevance < 0.30,
  do NOT boost; treat as normal CITES or skip entirely
- label clearly in UI/export as heuristic

### 2.4.3 INTRO_CITES via OA PDF parsing (best effort)
If oa_pdf_url exists and pdf_parsing_enabled:
- download + cache PDF
- parse (GROBID or lightweight)
- best-effort extract Introduction boundaries
- map citations to works; if mapping fails, skip safely

Acceptance:
- with PDF disabled, INTRO_HINT_CITES still present for papers with >=10 refs
- UI filters by edge type with counts

---

## 3) Scientist-centric scoring (graph-native, keyword weak)

### 3.1 Graph-native relevance (SCIENTIST mode)
Let SeedSet S = initial seeds + pinned trusted nodes.

**ProximityScore**
- d(p,S) = min hop distance (treat CITES + CITED_BY as undirected for distance only)
- ProximityScore = exp(-alpha*d), alpha default 0.9

**MultiPathScore**
- based on number of independent paths to seeds (approx)
- MultiPathScore = min(1.0, log(1+paths)/log(6))

**Coupling / Co-citation**
- BibCouplingScore: overlap refs(p) with refs(seeds/graph)
- CoCitationScore: overlap citers(p) with citers(seeds/graph)
Use Jaccard; approximate using top-N neighbors.

**Portal score**
- reviews/perspectives are PORTALS; cited by PORTAL and intro-hint edges add PortalSupport

**Keyword tie-breaker**
- <=10% weight only

GraphRelevance(p) =
- 0.45*ProximityScore +
- 0.20*MultiPathScore +
- 0.15*max(CoCitationScore, BibCouplingScore) +
- 0.10*PortalSupport +
- 0.10*KeywordTieBreaker

---

## 4) Controlled Explosion Architecture (explore wide, materialize narrow)

### 4.1 Two-tier storage (must implement)

**CandidatePool (wide)**
Lightweight store for discovered items not necessarily shown.
Store minimal metadata:
- candidate_id (doi/openalex/s2)
- type: PAPER | AUTHOR
- discovered_from: {src_node_id, channel, timestamp}
- cheap features: year, citation_count, venue, concept tags if available
- provisional scores: approx_relevance, approx_novelty, approx_hubness
- status: {CANDIDATE, MATERIALIZED, REJECTED}

CandidatePool may be large (10^5) but must be:
- deduped by IDs
- indexed in SQLite
- bounded by size with pruning (default max 200k by score+recency)

**WorkingGraph (narrow, user-visible)**
Hard bounds:
- max_working_nodes default 2000
- max_working_edges default 8000

### 4.2 Materialization vs Expansion (must separate)

**MaterializationScore (who enters WorkingGraph)**
MaterializationScore(p) =
- 0.35*GraphProximityToSeeds +
- 0.20*MultiPathSupport +
- 0.15*CouplingOrCoCitation +
- 0.10*PortalSupport +
- 0.10*BridgeScore +
- 0.10*RecencyOrImpact

BridgeScore rewards adjacent-field bridges:
- repeated discovery across channels (back+forward+author+portal)
- connects two clusters (two distinct concept/venue neighborhoods)
Hubness penalty:
- huge citation_count with low proximity/coupling reduces score

When WorkingGraph full:
- evict lowest-score peripheral nodes (never evict seeds/pinned/portals) using score+LRU policy.

**ExpansionScore (what to expand next)**
Favor nodes likely to reveal new structure:
- frontier/uncertainty/bridge potential
not only “most relevant”.

### 4.3 Frontier-based expansion
Maintain cluster assignments (concept-based or lightweight community detection).
Define Frontier nodes = boundary nodes with many CandidatePool edges but low explored ratio.

Expansion schedule:
- core_expand_fraction = 0.6
- frontier_expand_fraction = 0.4

### 4.4 User knobs (must implement)
- Exploration aggressiveness: LOW / MED / HIGH (controls how many items fetched into CandidatePool per expansion)
- WorkingGraph size: SMALL / MED / LARGE
- Bridge sensitivity: LOW / HIGH
Hard: WorkingGraph always bounded; CandidatePool bounded by max size + pruning.

### 4.5 Gap analysis outputs (must implement)
Separate views derived from CandidatePool + WorkingGraph:
- Nearby clusters not yet pulled in (top 5)
- Missing bridge candidates (highest BridgeScore not materialized)
- Field drift map (timeline of cluster emergence via forward citations)

### 4.6 System-level rails (shifted from “don’t fetch” to “don’t crash”)
- provider rate limiting + caching (mandatory)
- total API call budget per job (default max_calls=2000; adjustable)
- CandidatePool max size (default 200k) + pruning
- WorkingGraph hard bounds
- UI virtualization (render subset)

---

## 5) Author expansion layer (scientist-centric; budgeted)

### 5.1 Author identity resolution (ID-first)
When ingesting a paper:
- extract authors + corresponding author (if available)
Resolve identity via:
1) ORCID (if present)
2) OpenAlex author ID
3) Semantic Scholar author ID
4) fallback conservative match (name+affiliation+coauthor overlap within current graph only)

If ORCID available:
- use ORCID API to fetch works list (best-effort; cached; rate-limited)
Else:
- use OpenAlex author works endpoint (preferred), then S2 fallback

### 5.2 Author expansion as a third channel
When expanding focal paper P, do:
1) backward refs
2) forward citers
3) author works (selected authors)

Which authors to expand (strict):
- corresponding author if available, else first + last author
- plus up to (author_expand_k - 1) additional authors by current author_centrality
Defaults: author_expand_k=2 (max 3)

Which works to pull (strict):
- recent: last years_back, up to author_recent_cap=10
- foundational: up to author_foundational_cap=5, selected by:
  - impact AND
  - GraphRelevance to seeds AND
  - coupling/co-citation overlap with current graph
Total per author cap: author_total_cap <= 15
Total per expanded paper cap: author_edges_budget_per_node <= 20 (across all expanded authors)

AuthorTopicFit(A):
- average GraphRelevance(p) over top m=5 works for A (graph-first)
- centrality bonus if author appears across multiple high-relevance nodes

Author guardrails:
- do not expand coauthors recursively by default
- skip ambiguous authors without stable IDs when explosion likely
- skip mega-authors unless central in current graph

UI requirements:
- show/hide author nodes
- author panel: identifiers, top relevant works, manual “Expand author” with preview counts

---

## 6) Human-centric trajectories (author trajectories across fields) — must implement

### 6.1 AuthorYearTopicProfile (AYTP)
For each Author A and year Y:
- author_id, year
- works (bounded)
- topic_signature:
  - top_concepts [{concept, weight}]
  - optional venues
  - optional embedding centroid
- volume, impact

Store in DB table author_year_profiles.

### 6.2 TrajectoryEdge
Connect (A,Y) → (A,Y+1 or Y+Δ)
- type: TRAJECTORY_STEP
- attributes:
  - drift_magnitude (0..1)
  - drift_direction_summary (entering/exiting concepts)
  - novelty_jump (boolean)

Compute drift using normalized Jensen–Shannon divergence of concept distributions:
- drift_magnitude = JSD(profile_Y, profile_Y+1)
- novelty_jump if drift_magnitude >= drift_jump_threshold (default 0.55)

### 6.3 AuthorBridgeEdge (human-mediated bridge)
When an author shows a drift between two clusters:
- connect cluster_i <-> cluster_j with type AUTHOR_BRIDGE
- attributes:
  - author_id
  - years [y_start, y_end]
  - bridge_strength
  - exemplar_works [paper_ids]

### 6.4 Integrate trajectories into Controlled Explosion + Gap analysis
Update BridgeScore and author scoring:
- boost candidate papers near transition years for drifting authors
- boost bridges confirmed by multiple authors independently

Gap analysis: missing intermediates
Given an AUTHOR_BRIDGE (cluster_i -> cluster_j), search CandidatePool for papers that:
- are cited by cluster_i AND cite cluster_j (or vice versa)
- OR are authored by drifting authors during transition years
- OR have mixed concept signatures (overlap both clusters)
Rank and show top 20.

UI: Author trajectory view
- timeline
- drift event markers
- click jump → entering/exiting concepts + exemplar works
- explanation text must be generated from structured data only (no hallucination)

Compute trajectories only for:
- top N centrality authors OR user-pinned authors
Defaults:
- max_trajectory_authors_auto=30
- max_trajectory_authors_user=200 (still bounded by sampling caps)

---

## 7) Provider abstraction (must implement)
`/backend/providers/`
- base.py interface:
  - search_papers(query, year_min, year_max, limit)
  - get_paper(id/doi)
  - get_references(id/doi)
  - get_citations(id/doi)
  - get_author(author_id)
  - get_author_works(author_id, year_min, year_max, limit)
- openalex.py
- semantic_scholar.py
- crossref.py (optional DOI helper)
- unpaywall.py (optional OA PDF locator)
- orcid.py (optional; only if ORCID available)

Multi-provider merge:
- canonicalize by DOI when present
- else conservative fuzzy match (title+year+first_author)

---

## 8) Backend stack
Python 3.11+
- FastAPI
- SQLite (upgrade path to Postgres)
- SQLModel or SQLAlchemy
- httpx + tenacity
- diskcache or sqlite cache tables
- pydantic configs

Jobs:
- background tasks or asyncio registry
- persist job state in SQLite

---

## 9) Frontend stack
Next.js (or Vite+React) + Cytoscape.js (or Sigma.js)
UI must include:
- Seed input: DOI/URL/ID/title (preferred) + optional topic bootstrap
- Triage meter + wizard (only for bootstrap)
- Graph view with typed edges and filters
- Author toggles and author panel
- Trajectory timeline view
- Gap analysis views
- Export buttons

---

## 10) API endpoints
Topic bootstrap:
- POST /api/topic/triage
- POST /api/topic/refine

Graph build:
- POST /api/graph/build
  - body: { mode, seeds, topic_refined_query_string?, config }
  - If seeds provided: build WITHOUT broad topic search
  - If no seeds: must triage+refine before seeding

- GET /api/graph/status/{job_id}
- GET /api/graph/result/{job_id}
- GET /api/graph/export/{job_id}?format={graphml|csv|json}

Author/trajectory:
- GET /api/author/{author_id}
- GET /api/author/{author_id}/trajectory

Gap analysis:
- GET /api/gap/bridges/{job_id}
- GET /api/gap/missing_links/{job_id}?bridge_id=...

---

## 11) Acceptance criteria (must pass)
- `make up` starts app (docker-compose ok).
- With a single seed DOI, the system builds a meaningful map using only citations + author layers (no broad keyword expansion).
- Bidirectional citations are present (backward + forward) when provider data exists; missing directions are logged explicitly.
- CandidatePool can grow large; WorkingGraph remains bounded and responsive.
- Gap analysis shows adjacent clusters + bridge candidates + missing intermediates.
- Author identities resolved ID-first; ambiguous authors do not corrupt results.
- Trajectory view shows drift events and AUTHOR_BRIDGE explanations from structured data.
- Exports work (GraphML/CSV/JSON).
- Tests with fixtures (no live keys required) cover:
  - scope triage scoring/bands
  - seed resolution (DOI/title)
  - bidirectional citation normalization
  - intro hint logic (k clamp + caps + hub suppression)
  - CandidatePool/WorkingGraph gating + eviction
  - author expansion caps + ID resolution
  - trajectory drift computation

---

## 12) Deliverables (files)
Repo layout:
- backend/
  - main.py
  - config.py
  - models.py
  - db.py
  - jobs.py
  - graph_builder.py
  - candidate_pool.py
  - materialization.py
  - topic_triage.py
  - gap_analysis.py
  - author_layer.py
  - trajectory.py
  - providers/
  - scoring/
  - export/
  - tests/
- frontend/
- docker-compose.yml
- README.md
- .env.example
- Makefile

---

## 13) Build order (sequence)
1) Backend skeleton + DB + seed resolution + OpenAlex provider + bidirectional citations + CandidatePool/WorkingGraph gating (API returns JSON).
2) Add Semantic Scholar provider + merge.
3) Add caching + rate limit + call budgets.
4) Add author layer (ID-first) + caps.
5) Add trajectory computation + AUTHOR_BRIDGE + gap analysis endpoints.
6) Add frontend graph viewer + author panel + trajectory view + gap analysis.
7) Add exports.
8) Add tests + fixtures.
9) Polish README.

Start now. Create the repo structure, then implement v0. Print concise progress notes as you go, and do not skip running tests/linting at the end."
