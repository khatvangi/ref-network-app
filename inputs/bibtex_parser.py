"""
bibtex and ris file parsers for importing zotero exports.
"""

from typing import List
from pathlib import Path

from providers.base import PaperStub


def parse_bibtex(file_path: str) -> List[PaperStub]:
    """
    parse bibtex file and return list of papers.
    uses bibtexparser library.
    """
    import bibtexparser

    if not Path(file_path).exists():
        print(f"[bibtex] file not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        bib_database = bibtexparser.load(f)

    papers = []

    for entry in bib_database.entries:
        # extract fields
        title = entry.get('title', '').strip('{}')

        # year
        year = None
        if entry.get('year'):
            try:
                year = int(entry.get('year'))
            except ValueError:
                pass

        # authors
        authors = []
        author_str = entry.get('author', '')
        if author_str:
            # bibtex uses " and " to separate authors
            for a in author_str.split(' and '):
                a = a.strip().strip('{}')
                if a:
                    authors.append(a)

        # venue
        venue = entry.get('journal') or entry.get('booktitle') or entry.get('publisher') or ''
        venue = venue.strip('{}')

        # doi
        doi = entry.get('doi', '').strip('{}')
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]

        # abstract
        abstract = entry.get('abstract', '').strip('{}')

        # entry type
        entry_type = entry.get('ENTRYTYPE', '').lower()
        is_review = 'review' in title.lower() or entry_type == 'review'

        paper = PaperStub(
            doi=doi if doi else None,
            title=title,
            year=year,
            venue=venue,
            authors=authors[:5],
            is_review=is_review,
            abstract=abstract if abstract else None
        )

        if paper.title:
            papers.append(paper)

    print(f"[bibtex] parsed {len(papers)} entries from {file_path}")
    return papers


def parse_ris(file_path: str) -> List[PaperStub]:
    """
    parse ris file and return list of papers.
    uses rispy library.
    """
    import rispy

    if not Path(file_path).exists():
        print(f"[ris] file not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        entries = rispy.load(f)

    papers = []

    for entry in entries:
        # extract fields
        title = entry.get('title') or entry.get('primary_title', '')

        # year
        year = None
        if entry.get('year'):
            try:
                year = int(entry.get('year'))
            except ValueError:
                pass
        elif entry.get('publication_year'):
            try:
                year = int(entry.get('publication_year'))
            except ValueError:
                pass

        # authors
        authors = entry.get('authors') or entry.get('first_authors', [])
        if isinstance(authors, str):
            authors = [authors]
        authors = authors[:5]

        # venue
        venue = (entry.get('journal_name') or
                 entry.get('secondary_title') or
                 entry.get('publisher', ''))

        # doi
        doi = entry.get('doi', '')
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]

        # abstract
        abstract = entry.get('abstract', '')

        # type
        entry_type = entry.get('type_of_reference', '').lower()
        is_review = 'review' in title.lower() or 'review' in entry_type

        paper = PaperStub(
            doi=doi if doi else None,
            title=title,
            year=year,
            venue=venue,
            authors=authors,
            is_review=is_review,
            abstract=abstract if abstract else None
        )

        if paper.title:
            papers.append(paper)

    print(f"[ris] parsed {len(papers)} entries from {file_path}")
    return papers


def parse_file(file_path: str) -> List[PaperStub]:
    """
    auto-detect file type and parse.
    supports: .bib, .bibtex, .ris
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in ['.bib', '.bibtex']:
        return parse_bibtex(file_path)
    elif ext == '.ris':
        return parse_ris(file_path)
    else:
        # try bibtex first, then ris
        papers = parse_bibtex(file_path)
        if not papers:
            papers = parse_ris(file_path)
        return papers


# simple test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        papers = parse_file(sys.argv[1])
        print(f"\nParsed {len(papers)} papers:")
        for p in papers[:10]:
            print(f"  - {p.title[:60]}... ({p.year})")
    else:
        print("usage: python bibtex_parser.py <bib_or_ris_file>")
