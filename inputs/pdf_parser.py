"""
pdf reference extraction.
extracts references from pdf files using multiple strategies:
1. look for "References" section and parse
2. extract DOIs from text
3. use regex patterns for common citation formats

for better results, consider using GROBID (https://github.com/kermitt2/grobid)
but that requires a server setup.
"""

import re
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from providers.base import PaperStub


def extract_text_from_pdf(pdf_path: str) -> str:
    """extract text from pdf using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"[pdf] error extracting text: {e}")
        return ""


def extract_dois_from_text(text: str) -> List[str]:
    """extract DOIs from text."""
    # doi pattern: 10.XXXX/...
    doi_pattern = r'10\.\d{4,}/[^\s\]\)>"\',]+'
    matches = re.findall(doi_pattern, text)

    # clean up matches
    dois = []
    for m in matches:
        # remove trailing punctuation
        m = m.rstrip('.,;:')
        if m not in dois:
            dois.append(m)

    return dois


def extract_references_section(text: str) -> str:
    """try to extract the references section from text."""
    # common patterns for references section header
    patterns = [
        r'\n\s*References?\s*\n',
        r'\n\s*REFERENCES?\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Works Cited\s*\n',
        r'\n\s*Literature Cited\s*\n',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # extract from match to end (or next section)
            start = match.end()
            # look for end markers
            end_patterns = [
                r'\n\s*Appendix',
                r'\n\s*APPENDIX',
                r'\n\s*Supplementary',
                r'\n\s*Supporting Information',
                r'\n\s*Acknowledgments?',
                r'\n\s*ACKNOWLEDGMENTS?',
            ]
            end = len(text)
            for ep in end_patterns:
                em = re.search(ep, text[start:], re.IGNORECASE)
                if em:
                    end = min(end, start + em.start())

            return text[start:end]

    return ""


def parse_reference_entries(ref_text: str) -> List[Dict[str, str]]:
    """
    parse individual reference entries from references section.
    this is a heuristic approach - not perfect.
    """
    entries = []

    # try to split by numbered references [1], [2], etc.
    numbered = re.split(r'\n\s*\[\d+\]\s*', ref_text)
    if len(numbered) > 2:
        for entry in numbered[1:]:
            entry = entry.strip()
            if entry:
                entries.append({'raw': entry})
        return entries

    # try to split by numbered references 1., 2., etc.
    numbered = re.split(r'\n\s*\d+\.\s+', ref_text)
    if len(numbered) > 2:
        for entry in numbered[1:]:
            entry = entry.strip()
            if entry:
                entries.append({'raw': entry})
        return entries

    # fall back to paragraph splitting
    paragraphs = ref_text.split('\n\n')
    for p in paragraphs:
        p = p.strip()
        if len(p) > 50:  # likely a reference, not just a page number
            entries.append({'raw': p})

    return entries


def parse_reference_text(ref_text: str) -> Optional[PaperStub]:
    """
    attempt to parse a reference text into structured data.
    very heuristic - works for common formats.
    """
    # try to extract year
    year_match = re.search(r'\((\d{4})\)|,\s*(\d{4})[,\.]|\b(19|20)\d{2}\b', ref_text)
    year = None
    if year_match:
        for g in year_match.groups():
            if g and len(g) == 4:
                try:
                    year = int(g)
                    break
                except ValueError:
                    pass

    # try to extract DOI
    doi_match = re.search(r'10\.\d{4,}/[^\s\]\)>"\',]+', ref_text)
    doi = doi_match.group(0).rstrip('.,;:') if doi_match else None

    # extract title (usually in quotes or after authors/year)
    title = ""
    # try quoted title
    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', ref_text)
    if quoted:
        title = quoted.group(1) or quoted.group(2)
    else:
        # try to extract after year
        if year_match:
            after_year = ref_text[year_match.end():].strip()
            # take first sentence-like chunk
            title_match = re.match(r'[^\.]+', after_year)
            if title_match:
                title = title_match.group(0).strip(' .,')

    if not title:
        # just use first 100 chars as title approximation
        title = ref_text[:100].strip()

    return PaperStub(
        doi=doi,
        title=title,
        year=year
    )


def extract_references_from_pdf(pdf_path: str, resolve_dois: bool = True) -> Tuple[Optional[PaperStub], List[PaperStub]]:
    """
    extract references from a pdf file.

    returns:
        (main_paper, references) where main_paper is the paper itself (if DOI found)
        and references is a list of cited papers.
    """
    print(f"[pdf] extracting references from: {pdf_path}")

    if not Path(pdf_path).exists():
        print(f"[pdf] file not found: {pdf_path}")
        return None, []

    # extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None, []

    print(f"[pdf] extracted {len(text)} characters")

    # extract all DOIs from the document
    all_dois = extract_dois_from_text(text)
    print(f"[pdf] found {len(all_dois)} DOIs in document")

    # the first DOI is often the paper itself
    main_doi = all_dois[0] if all_dois else None
    ref_dois = all_dois[1:] if len(all_dois) > 1 else []

    # try to extract references section
    ref_section = extract_references_section(text)
    if ref_section:
        print(f"[pdf] found references section ({len(ref_section)} chars)")

        # parse individual entries
        entries = parse_reference_entries(ref_section)
        print(f"[pdf] parsed {len(entries)} reference entries")
    else:
        entries = []

    # build reference list
    references = []

    # first, add papers from DOIs
    if resolve_dois and ref_dois:
        from providers.crossref import CrossrefProvider
        crossref = CrossrefProvider()

        for doi in ref_dois[:100]:  # limit to first 100
            paper = crossref.resolve_doi(doi)
            if paper:
                references.append(paper)

        print(f"[pdf] resolved {len(references)} DOIs via crossref")

    # then, add parsed entries that don't have DOIs
    for entry in entries:
        raw = entry.get('raw', '')

        # skip if we already have this via DOI
        entry_doi = None
        doi_match = re.search(r'10\.\d{4,}/[^\s\]\)>"\',]+', raw)
        if doi_match:
            entry_doi = doi_match.group(0).rstrip('.,;:')
            if entry_doi in ref_dois:
                continue  # already added

        # parse the entry
        paper = parse_reference_text(raw)
        if paper and paper.title:
            references.append(paper)

    # resolve main paper
    main_paper = None
    if main_doi and resolve_dois:
        from providers.crossref import CrossrefProvider
        crossref = CrossrefProvider()
        main_paper = crossref.resolve_doi(main_doi)

    print(f"[pdf] total references extracted: {len(references)}")
    return main_paper, references


# simple test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main, refs = extract_references_from_pdf(sys.argv[1])
        if main:
            print(f"\nMain paper: {main.title}")
        print(f"\nReferences ({len(refs)}):")
        for r in refs[:10]:
            print(f"  - {r.title[:60]}... ({r.year}) doi={r.doi}")
    else:
        print("usage: python pdf_parser.py <pdf_file>")
