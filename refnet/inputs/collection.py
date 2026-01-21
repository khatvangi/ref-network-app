"""
collection importer - load seeds from various research collection formats.
supports: JSON exports, CSV files, PMID lists, markdown with DOIs.
"""

import json
import csv
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..core.models import Paper, PaperStatus


def load_from_json(
    path: str,
    doi_field: str = "doi",
    title_field: str = "title",
    year_field: str = "year",
    authors_field: str = "authors",
    category_field: Optional[str] = "_category",
    limit: Optional[int] = None
) -> List[Paper]:
    """
    load papers from JSON file.
    supports various formats (scispace, pubmed exports, custom).
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # handle both list and dict formats
    if isinstance(data, dict):
        # might be wrapped in a key
        for key in ['papers', 'results', 'data', 'items']:
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        print(f"[collection] unexpected JSON format in {path}")
        return []

    papers = []
    for item in data:
        if limit and len(papers) >= limit:
            break

        # extract fields flexibly
        doi = _extract_field(item, [doi_field, 'DOI', 'Doi'])
        title = _extract_field(item, [title_field, 'Title', 'paper_title', 'Paper Title'])
        year = _extract_field(item, [year_field, 'Year', 'publication_year', 'Publication Year'])
        authors = _extract_field(item, [authors_field, 'Authors', 'author_names', 'Author Names'])

        # skip if no title or doi
        if not title and not doi:
            continue

        # normalize doi
        if doi:
            doi = _normalize_doi(doi)

        # parse year
        if year:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        # parse authors
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(',')]
        elif not isinstance(authors, list):
            authors = []

        # extract optional fields
        venue = _extract_field(item, ['journal', 'venue', 'Journal', 'Publication Title'])
        abstract = _extract_field(item, ['abstract', 'Abstract'])
        citations = _extract_field(item, ['citations', 'citation_count', 'cited_by_count'])
        category = _extract_field(item, [category_field, 'category', 'topic']) if category_field else None

        paper = Paper(
            doi=doi,
            title=title or "",
            year=year,
            authors=authors[:5],
            venue=venue,
            abstract=abstract,
            citation_count=int(citations) if citations else None,
            status=PaperStatus.SEED
        )

        # store category in concepts for now
        if category:
            paper.concepts = [{'name': category, 'score': 1.0}]

        papers.append(paper)

    print(f"[collection] loaded {len(papers)} papers from {path}")
    return papers


def load_from_csv(
    path: str,
    doi_column: Optional[str] = None,
    title_column: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Paper]:
    """
    load papers from CSV file.
    auto-detects column names if not specified.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    # auto-detect columns
    columns = rows[0].keys()

    if not doi_column:
        doi_column = _find_column(columns, ['doi', 'DOI', 'Doi'])
    if not title_column:
        title_column = _find_column(columns, ['title', 'Title', 'Paper Title', 'paper_title'])

    year_column = _find_column(columns, ['year', 'Year', 'Publication Year'])
    authors_column = _find_column(columns, ['authors', 'Authors', 'Author Names'])
    venue_column = _find_column(columns, ['journal', 'venue', 'Journal', 'Publication Title'])

    papers = []
    for row in rows:
        if limit and len(papers) >= limit:
            break

        doi = row.get(doi_column, '') if doi_column else ''
        title = row.get(title_column, '') if title_column else ''

        if not title and not doi:
            continue

        if doi:
            doi = _normalize_doi(doi)

        year = None
        if year_column and row.get(year_column):
            try:
                year = int(row[year_column])
            except (ValueError, TypeError):
                pass

        authors = []
        if authors_column and row.get(authors_column):
            authors = [a.strip() for a in row[authors_column].split(',')]

        venue = row.get(venue_column, '') if venue_column else ''

        paper = Paper(
            doi=doi if doi else None,
            title=title,
            year=year,
            authors=authors[:5],
            venue=venue,
            status=PaperStatus.SEED
        )
        papers.append(paper)

    print(f"[collection] loaded {len(papers)} papers from {path}")
    return papers


def load_from_pmid_list(path: str) -> List[Dict[str, str]]:
    """
    load PMIDs from text file (one per line).
    returns list of {pmid: ...} for provider lookup.
    """
    pmids = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                pmids.append(line)

    print(f"[collection] loaded {len(pmids)} PMIDs from {path}")
    return pmids


def load_from_markdown(path: str) -> List[Paper]:
    """
    extract papers from markdown file.
    looks for DOIs, titles in citations, etc.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    papers = []

    # find DOIs
    doi_pattern = r'10\.\d{4,}/[^\s\)\]\"\'<>]+'
    dois = set(re.findall(doi_pattern, content))

    for doi in dois:
        doi = _normalize_doi(doi)
        paper = Paper(doi=doi, status=PaperStatus.SEED)
        papers.append(paper)

    print(f"[collection] extracted {len(papers)} DOIs from {path}")
    return papers


def load_collection(
    path: str,
    limit: Optional[int] = None
) -> List[Paper]:
    """
    auto-detect format and load papers.
    """
    path = Path(path)

    if not path.exists():
        print(f"[collection] file not found: {path}")
        return []

    ext = path.suffix.lower()

    if ext == '.json':
        return load_from_json(str(path), limit=limit)
    elif ext == '.csv':
        return load_from_csv(str(path), limit=limit)
    elif ext in ['.md', '.markdown']:
        return load_from_markdown(str(path))
    elif ext == '.txt':
        # might be PMID list
        pmids = load_from_pmid_list(str(path))
        # return empty papers with pmids for lookup
        return [Paper(pmid=p, status=PaperStatus.SEED) for p in pmids]
    else:
        print(f"[collection] unknown format: {ext}")
        return []


def load_directory(
    path: str,
    extensions: List[str] = ['.json', '.csv'],
    limit_per_file: Optional[int] = None
) -> List[Paper]:
    """
    load papers from all matching files in a directory.
    """
    path = Path(path)
    if not path.is_dir():
        return load_collection(str(path), limit_per_file)

    all_papers = []
    seen_dois = set()

    for ext in extensions:
        for file in path.glob(f'*{ext}'):
            papers = load_collection(str(file), limit_per_file)

            # dedupe by DOI
            for p in papers:
                if p.doi and p.doi in seen_dois:
                    continue
                if p.doi:
                    seen_dois.add(p.doi)
                all_papers.append(p)

    print(f"[collection] loaded {len(all_papers)} unique papers from {path}")
    return all_papers


def enrich_papers_with_provider(
    papers: List[Paper],
    provider,
    batch_size: int = 10
) -> List[Paper]:
    """
    enrich paper metadata using provider lookups.
    fills in missing titles, citations, etc.
    """
    enriched = []
    missing_count = 0

    for i, paper in enumerate(papers):
        if paper.doi:
            # lookup by DOI
            full = provider.get_paper(paper.doi)
            if full:
                # merge
                full.status = paper.status
                if paper.concepts:
                    full.concepts = paper.concepts + full.concepts
                enriched.append(full)
            else:
                missing_count += 1
                enriched.append(paper)
        elif paper.pmid:
            # TODO: lookup by PMID
            enriched.append(paper)
        elif paper.title:
            # lookup by title
            results = provider.search_papers(f'"{paper.title}"', limit=3)
            if results:
                # find best match
                for r in results:
                    if r.title and paper.title.lower() in r.title.lower():
                        r.status = paper.status
                        enriched.append(r)
                        break
                else:
                    enriched.append(paper)
            else:
                enriched.append(paper)
        else:
            enriched.append(paper)

        if (i + 1) % batch_size == 0:
            print(f"[collection] enriched {i + 1}/{len(papers)} papers...")

    print(f"[collection] enriched {len(papers)} papers ({missing_count} not found)")
    return enriched


# helpers

def _extract_field(item: Dict, field_names: List[str]) -> Any:
    """try multiple field names."""
    for name in field_names:
        if name and name in item and item[name]:
            return item[name]
    return None


def _find_column(columns, candidates: List[str]) -> Optional[str]:
    """find matching column name."""
    columns_lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c.lower() in columns_lower:
            return columns_lower[c.lower()]
    return None


def _normalize_doi(doi: str) -> str:
    """normalize DOI format."""
    doi = doi.strip()
    if doi.startswith('https://doi.org/'):
        doi = doi[16:]
    if doi.startswith('http://doi.org/'):
        doi = doi[15:]
    if doi.startswith('doi:'):
        doi = doi[4:]
    # remove trailing punctuation
    doi = doi.rstrip('.,;:')
    return doi.lower()
