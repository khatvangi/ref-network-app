"""
lookup a paper by title using multiple providers.
tries to find the best match across providers.
"""

from typing import Optional, List

from providers.base import PaperStub


def lookup_paper_by_title(
    title: str,
    use_openalex: bool = True,
    use_s2: bool = True,
    use_crossref: bool = True
) -> Optional[PaperStub]:
    """
    lookup a paper by title using multiple providers.
    returns the best match found.
    """
    from scoring.relevance import fuzzy_title_match

    candidates = []

    # try openalex
    if use_openalex:
        try:
            from providers.openalex import OpenAlexProvider
            provider = OpenAlexProvider()
            results = provider.search_papers(f'"{title}"', limit=5)
            candidates.extend(results)
        except Exception as e:
            print(f"[lookup] openalex error: {e}")

    # try semantic scholar
    if use_s2:
        try:
            from providers.semantic_scholar import SemanticScholarProvider
            provider = SemanticScholarProvider()
            results = provider.search_papers(title, limit=5)
            candidates.extend(results)
        except Exception as e:
            print(f"[lookup] s2 error: {e}")

    # try crossref
    if use_crossref:
        try:
            from providers.crossref import CrossrefProvider
            provider = CrossrefProvider()
            results = provider.search_papers(title, limit=5)
            candidates.extend(results)
        except Exception as e:
            print(f"[lookup] crossref error: {e}")

    if not candidates:
        return None

    # find best match
    best = None
    best_score = 0

    for paper in candidates:
        if not paper.title:
            continue

        # check title similarity
        if fuzzy_title_match(title, paper.title, threshold=0.6):
            # score based on metadata completeness
            score = 0
            if paper.doi:
                score += 3
            if paper.year:
                score += 1
            if paper.citation_count:
                score += min(paper.citation_count / 100, 2)
            if paper.abstract:
                score += 1

            if score > best_score:
                best_score = score
                best = paper

    return best


def lookup_paper_by_doi(doi: str) -> Optional[PaperStub]:
    """
    lookup a paper by DOI using crossref (primary) or other providers.
    """
    # normalize doi
    if doi.startswith('https://doi.org/'):
        doi = doi[16:]
    if doi.startswith('doi:'):
        doi = doi[4:]

    # try crossref first (most reliable for DOI resolution)
    try:
        from providers.crossref import CrossrefProvider
        provider = CrossrefProvider()
        paper = provider.get_paper(doi)
        if paper:
            return paper
    except Exception as e:
        print(f"[lookup] crossref error: {e}")

    # try semantic scholar
    try:
        from providers.semantic_scholar import SemanticScholarProvider
        provider = SemanticScholarProvider()
        paper = provider.get_paper(f"doi:{doi}")
        if paper:
            return paper
    except Exception as e:
        print(f"[lookup] s2 error: {e}")

    # try openalex
    try:
        from providers.openalex import OpenAlexProvider
        provider = OpenAlexProvider()
        paper = provider.get_paper(doi)
        if paper:
            return paper
    except Exception as e:
        print(f"[lookup] openalex error: {e}")

    return None


# simple test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])

        # check if it's a DOI
        if query.startswith('10.') or 'doi.org' in query:
            print(f"Looking up DOI: {query}")
            paper = lookup_paper_by_doi(query)
        else:
            print(f"Looking up title: {query}")
            paper = lookup_paper_by_title(query)

        if paper:
            print(f"\nFound: {paper.title}")
            print(f"  Year: {paper.year}")
            print(f"  DOI: {paper.doi}")
            print(f"  Citations: {paper.citation_count}")
        else:
            print("Paper not found")
    else:
        print("usage: python title_lookup.py <title or doi>")
