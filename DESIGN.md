# RefNet Design Philosophy

## Core Principle: Castle, Not Sand

**Keywords are sand** - individual grains, no structure, blow away.
**Authors + citations are the castle** - structure, relationships, foundation.

Elicit, SciSpace, and other tools start with keywords and work backwards.
RefNet starts with **what scientists actually do**: follow the people and their work.

---

## The Hierarchy of Signal

```
1. AUTHORS (strongest signal)
   └── Their corpus reveals trajectory, methods, collaborators
   └── An author's body of work IS the context
   └── "Who is working on this?" > "What papers mention this?"

2. TITLE + ABSTRACT
   └── What is this paper actually about?
   └── The author's framing of their contribution

3. LLM EXTRACTION
   └── Understand meaning, not match strings
   └── Extract claims, methods, findings

4. KEYWORDS (weakest signal)
   └── Last resort, not first
   └── Emerge FROM understanding, not the entry point
```

**Keywords are where other tools START. Keywords are where we END.**

---

## How Scientists Actually Work

When a scientist finds an interesting paper, they don't search for keywords.
They ask:

1. **"Who wrote this?"** - Investigate the author
2. **"What else have they written?"** - Author's corpus reveals context
3. **"Who do they cite?"** - Intellectual lineage
4. **"Who cites them?"** - Who's building on this
5. **"Who do they work with?"** - Collaborator expertise

This builds a **mental model of the field**, not a list of papers.

---

## The Author Investigation Flow

```
Seed paper
    │
    ▼
Identify key authors
    │
    ▼
For each significant author:
    │
    ├──► CORPUS: All their papers
    │    └── What have they been working on?
    │    └── What methods do they use?
    │    └── What questions drive them?
    │
    ├──► TRAJECTORY: How has their focus evolved?
    │    └── Early work → current work
    │    └── Drift events (topic changes)
    │    └── Is this topic new for them or lifelong?
    │
    ├──► COLLABORATORS: Who do they work with?
    │    └── What expertise do collaborators bring?
    │    └── Cross-disciplinary connections
    │
    ├──► CITATIONS: Intellectual network
    │    └── Who they cite (lineage)
    │    └── Who cites them (influence)
    │    └── Bridge authors (connect fields)
    │
    └──► THEMES: What concepts recur?
         └── Extract from titles/abstracts across corpus
         └── These are the REAL keywords (emergent, not imposed)
```

---

## Why This Beats Keyword Search

| Keyword Search | Author-Centric |
|----------------|----------------|
| "aminoacyl tRNA synthetase" → 50,000 papers | Carter's network → 200 papers that matter |
| No structure, just matches | Field structure emerges |
| Miss papers with different terminology | Follow the people, not the words |
| Can't distinguish methods from topic | Author corpus reveals their angle |
| No sense of who matters | Leaders, followers, disruptors identified |

---

## Agent Architecture: One Task, One Agent

Each agent does ONE thing well. No god objects.

### Agent Inventory

| Agent | Task | Input | Output |
|-------|------|-------|--------|
| `SeedResolver` | Resolve DOI/title to full paper | DOI or title string | Paper object with metadata |
| `CitationWalker` | Fetch refs/cites for a paper | Paper ID | Classified refs (intro-hint vs regular) |
| `AuthorResolver` | Resolve author name to profile | Name or ID | Author object with IDs |
| `CorpusFetcher` | Get all papers by an author | Author ID | List of papers |
| `TrajectoryAnalyzer` | Compute author's research drift | Author + papers | Drift events, topic evolution |
| `CollaboratorMapper` | Find co-author network | Author ID | Collaborator graph |
| `TopicExtractor` | Extract themes from paper set | List of papers | Ranked concepts/themes |
| `RelevanceScorer` | Score paper relevance to query | Paper + context | Relevance score + explanation |
| `GapDetector` | Find missing connections | Graph | Bridge candidates, unexplored areas |

### Agent Principles

1. **Single responsibility** - One agent, one task
2. **Composable** - Agents call other agents
3. **Traceable** - Every decision logged with reason
4. **Fallible** - Handle failures gracefully, never crash pipeline
5. **Testable** - Each agent can be tested in isolation

### Pipeline Composition

```
User: "I want to understand the field around this paper"
    │
    ▼
SeedResolver(doi) → Paper
    │
    ▼
CitationWalker(paper) → refs[], cites[]
    │
    ├──► For each significant author in paper:
    │    AuthorResolver(name) → Author
    │    CorpusFetcher(author) → papers[]
    │    TrajectoryAnalyzer(author, papers) → trajectory
    │    CollaboratorMapper(author) → collaborators[]
    │
    ├──► TopicExtractor(all_papers) → themes[]
    │
    └──► GapDetector(graph) → gaps[]
    │
    ▼
LiteratureReport with:
  - Landscape (clusters, themes)
  - Leaders (authors, their roles)
  - Gaps (unexplored connections)
  - Reading list (papers that matter)
```

---

## What We're NOT Building

- **Not a search engine** - We don't index and match
- **Not a recommendation system** - We don't guess what you want
- **Not a chatbot** - We don't answer questions about papers

**We build the map. The scientist navigates.**

---

## Success Metrics

A scientist using RefNet should be able to:

1. **In 10 minutes**: Identify the 5 most important authors in a field
2. **In 30 minutes**: Have a reading list of 20-50 papers that actually matter
3. **In 1 hour**: Understand field structure, gaps, and trajectory
4. **For publication**: Export proper citations with author attribution

Compare to current tools:
- Elicit: 10 minutes → 50 keyword-matched papers, no structure
- SciSpace: 10 minutes → AI summary of one paper, no field context

---

## Next Steps

1. **Document each agent** - Input/output contracts, error handling
2. **Build agents one by one** - Test in isolation
3. **Compose pipeline** - Wire agents together
4. **Verify at each step** - Trace through real examples
5. **Iterate** - Refine based on actual scientist usage
