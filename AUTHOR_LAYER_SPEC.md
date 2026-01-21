# Author Layer Specification

## The Gap

Current RefNet has:
- Paper dendrimer (citations)
- Concept dendrimer (semantics)  
- Google Scholar (global impact)
- Review curation (TOC)

**Missing**: Author-centric layer

## What Author Layer Should Provide

### 1. Mega-Authors
- Who are the most prolific authors in this field?
- How many papers? How many citations?
- Which sub-networks do they appear in?

### 2. Author-Paper Mapping
```
author_papers table:
- author_id
- author_name
- paper_doi
- position (first, last, middle)
```

### 3. Collaboration Network
```
author_edges table:
- author_1
- author_2
- paper_count (co-authored)
- first_collab_year
- last_collab_year
```

### 4. Author Trajectories
- What topics has this author worked on over time?
- Early career → current focus
- Cross-field connections

### 5. Other Publications
For key authors, fetch ALL their papers (not just aaRS):
- What else do they work on?
- Hidden connections to other fields

## Data Sources

| Source | Author Data | API |
|--------|-------------|-----|
| CrossRef | Basic author list | Yes, free |
| OpenAlex | Author IDs, institutions, topics | Yes, free |
| Semantic Scholar | Author IDs, h-index, papers | Yes, free |
| Google Scholar | Profiles, citations | No (scrape) |

**Recommended**: OpenAlex - free, comprehensive, has author IDs

## Schema

```sql
CREATE TABLE authors (
    author_id TEXT PRIMARY KEY,  -- OpenAlex ID
    name TEXT,
    alt_names TEXT,              -- JSON array
    institution TEXT,
    h_index INTEGER,
    total_papers INTEGER,
    total_citations INTEGER,
    topics TEXT                  -- JSON array
);

CREATE TABLE author_papers (
    author_id TEXT,
    paper_doi TEXT,
    position TEXT,               -- first, last, middle
    year INTEGER,
    FOREIGN KEY (paper_doi) REFERENCES paper_candidates(doi)
);

CREATE TABLE author_collaborations (
    author_1 TEXT,
    author_2 TEXT,
    paper_count INTEGER,
    papers TEXT,                 -- JSON array of DOIs
    first_year INTEGER,
    last_year INTEGER
);
```

## Implementation Plan

1. **Phase 1**: Extract authors from existing Google Scholar data
2. **Phase 2**: Enrich via OpenAlex API (get author IDs, institutions)
3. **Phase 3**: Build collaboration network
4. **Phase 4**: Identify mega-authors + their other publications
5. **Phase 5**: Author trajectory analysis

## Key Queries

```sql
-- Top 20 mega-authors
SELECT author_id, name, COUNT(*) as papers, SUM(citation_count) as total_cites
FROM author_papers ap
JOIN paper_candidates p ON ap.paper_doi = p.doi
GROUP BY author_id
ORDER BY papers DESC LIMIT 20;

-- Author's collaborators
SELECT a2.name, COUNT(*) as collabs
FROM author_collaborations ac
JOIN authors a2 ON ac.author_2 = a2.author_id
WHERE ac.author_1 = 'target_author_id'
ORDER BY collabs DESC;

-- Author's topic trajectory
SELECT year, GROUP_CONCAT(DISTINCT topic) as topics
FROM author_papers ap
JOIN papers p ON ap.paper_doi = p.doi
WHERE author_id = 'target'
GROUP BY year ORDER BY year;
```
