"""
seed resolver agent - resolve DOI/title/URL to full paper.

the entry point for the refnet pipeline.
takes various paper identifiers and returns a fully materialized Paper object.

input: DOI, title, URL, PMID, or OpenAlex ID
output: Paper object with full metadata
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, unquote

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper, PaperStatus
from ..providers.openalex import OpenAlexProvider


@dataclass
class ResolvedSeed:
    """
    result of seed resolution.
    """
    # the resolved paper
    paper: Paper

    # resolution info
    input_type: str  # "doi", "title", "url", "pmid", "openalex_id"
    input_value: str  # original input
    confidence: float = 1.0  # 1.0 for exact match, lower for title search

    # alternative matches (for title search)
    alternatives: List[Paper] = field(default_factory=list)

    # warnings
    warnings: List[str] = field(default_factory=list)


class SeedResolver(Agent):
    """
    resolve paper identifiers to full Paper objects.

    supported input types:
    - DOI: "10.1038/s41586-020-2649-2" or "https://doi.org/10.1038/..."
    - OpenAlex ID: "W2741809807" or "https://openalex.org/W2741809807"
    - PMID: "32908161" or "PMID:32908161"
    - arXiv: "2103.00020" or "https://arxiv.org/abs/2103.00020"
    - URL: any URL containing a DOI
    - Title: "Attention Is All You Need" (fuzzy search)

    usage:
        resolver = SeedResolver(provider)
        result = resolver.run(query="10.1038/s41586-020-2649-2")

        if result.ok:
            paper = result.data.paper
            print(f"Resolved: {paper.title}")
    """

    # regex patterns for identifier extraction
    DOI_PATTERN = re.compile(r'10\.\d{4,}/[^\s]+')
    PMID_PATTERN = re.compile(r'^(?:PMID:?)?(\d{7,8})$', re.IGNORECASE)
    OPENALEX_PATTERN = re.compile(r'^W\d+$')
    ARXIV_PATTERN = re.compile(r'(\d{4}\.\d{4,5}(?:v\d+)?)')

    def __init__(
        self,
        provider: OpenAlexProvider,
        title_search_limit: int = 5,
        min_title_similarity: float = 0.7,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.provider = provider
        self.title_search_limit = title_search_limit
        self.min_title_similarity = min_title_similarity

    @property
    def name(self) -> str:
        return "SeedResolver"

    def execute(
        self,
        query: str,
        hint_year: Optional[int] = None,
        hint_author: Optional[str] = None
    ) -> AgentResult[ResolvedSeed]:
        """
        resolve a paper identifier to a full Paper object.

        args:
            query: DOI, title, URL, PMID, or OpenAlex ID
            hint_year: optional publication year hint for title search
            hint_author: optional author name hint for title search

        returns:
            AgentResult with ResolvedSeed
        """
        result = AgentResult[ResolvedSeed](status=AgentStatus.SUCCESS)
        query = query.strip()
        result.add_trace(f"resolving: {query[:100]}...")

        if not query:
            result.status = AgentStatus.FAILED
            result.add_error("EMPTY_QUERY", "query cannot be empty")
            return result

        # detect input type and resolve
        input_type, extracted = self._detect_input_type(query)
        result.add_trace(f"detected type: {input_type}")

        if input_type == "doi":
            paper = self._resolve_doi(extracted, result)
        elif input_type == "openalex_id":
            paper = self._resolve_openalex(extracted, result)
        elif input_type == "pmid":
            paper = self._resolve_pmid(extracted, result)
        elif input_type == "arxiv":
            paper = self._resolve_arxiv(extracted, result)
        elif input_type == "title":
            paper, alternatives = self._resolve_title(
                extracted, hint_year, hint_author, result
            )
        else:
            result.status = AgentStatus.FAILED
            result.add_error("UNKNOWN_TYPE", f"could not detect input type for: {query}")
            return result

        if not paper:
            result.status = AgentStatus.FAILED
            result.add_error("NOT_FOUND", f"could not resolve: {query}")
            return result

        # set paper status to SEED
        paper.status = PaperStatus.SEED

        # build result
        confidence = 1.0 if input_type != "title" else self._compute_title_similarity(
            query, paper.title or ""
        )

        resolved = ResolvedSeed(
            paper=paper,
            input_type=input_type,
            input_value=query,
            confidence=confidence,
            alternatives=alternatives if input_type == "title" else []
        )

        # add warnings for low confidence
        if confidence < 0.9:
            resolved.warnings.append(
                f"Title match confidence: {confidence:.0%}. Please verify this is the correct paper."
            )

        result.data = resolved
        result.api_calls = 1
        result.add_trace(f"resolved to: {paper.title[:60] if paper.title else 'Unknown'}...")

        return result

    def _detect_input_type(self, query: str) -> tuple[str, str]:
        """
        detect the type of input and extract the identifier.

        returns: (input_type, extracted_value)
        """
        query = query.strip()

        # check for DOI patterns
        if query.startswith("10."):
            return ("doi", query)

        if "doi.org/" in query:
            # extract DOI from URL
            doi_match = self.DOI_PATTERN.search(query)
            if doi_match:
                return ("doi", unquote(doi_match.group()))

        # check for OpenAlex ID
        if self.OPENALEX_PATTERN.match(query):
            return ("openalex_id", query)

        if "openalex.org/" in query:
            parts = query.split("/")
            for part in parts:
                if self.OPENALEX_PATTERN.match(part):
                    return ("openalex_id", part)

        # check for PMID
        pmid_match = self.PMID_PATTERN.match(query)
        if pmid_match:
            return ("pmid", pmid_match.group(1))

        # check for arXiv
        arxiv_match = self.ARXIV_PATTERN.search(query)
        if arxiv_match:
            return ("arxiv", arxiv_match.group(1))

        # check for URL with embedded DOI
        if query.startswith("http"):
            doi_match = self.DOI_PATTERN.search(unquote(query))
            if doi_match:
                return ("doi", doi_match.group())

        # fallback to title search
        return ("title", query)

    def _resolve_doi(self, doi: str, result: AgentResult) -> Optional[Paper]:
        """resolve paper by DOI."""
        result.add_trace(f"resolving DOI: {doi}")

        paper = self.provider.get_paper(doi)
        if paper:
            result.add_trace(f"found via DOI: {paper.title[:50] if paper.title else 'Unknown'}...")
        else:
            result.add_warning(f"DOI not found in OpenAlex: {doi}")

        return paper

    def _resolve_openalex(self, openalex_id: str, result: AgentResult) -> Optional[Paper]:
        """resolve paper by OpenAlex ID."""
        result.add_trace(f"resolving OpenAlex ID: {openalex_id}")

        paper = self.provider.get_paper(openalex_id)
        if paper:
            result.add_trace(f"found via OpenAlex: {paper.title[:50] if paper.title else 'Unknown'}...")
        else:
            result.add_warning(f"OpenAlex ID not found: {openalex_id}")

        return paper

    def _resolve_pmid(self, pmid: str, result: AgentResult) -> Optional[Paper]:
        """resolve paper by PMID via OpenAlex."""
        result.add_trace(f"resolving PMID: {pmid}")

        # OpenAlex supports PMID lookup
        paper = self.provider.get_paper(f"pmid:{pmid}")
        if paper:
            result.add_trace(f"found via PMID: {paper.title[:50] if paper.title else 'Unknown'}...")
        else:
            result.add_warning(f"PMID not found in OpenAlex: {pmid}")

        return paper

    def _resolve_arxiv(self, arxiv_id: str, result: AgentResult) -> Optional[Paper]:
        """resolve paper by arXiv ID."""
        result.add_trace(f"resolving arXiv: {arxiv_id}")

        # try to find via DOI (arXiv DOIs follow pattern 10.48550/arXiv.XXXX.XXXXX)
        arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
        paper = self.provider.get_paper(arxiv_doi)

        if not paper:
            # fallback: search by arXiv URL in OpenAlex
            # OpenAlex indexes arXiv papers with their IDs
            result.add_trace("DOI lookup failed, trying title search...")
            # this would need a search API call

        if paper:
            result.add_trace(f"found via arXiv: {paper.title[:50] if paper.title else 'Unknown'}...")
        else:
            result.add_warning(f"arXiv ID not found: {arxiv_id}")

        return paper

    def _resolve_title(
        self,
        title: str,
        hint_year: Optional[int],
        hint_author: Optional[str],
        result: AgentResult
    ) -> tuple[Optional[Paper], List[Paper]]:
        """
        resolve paper by title search.

        returns: (best_match, alternatives)
        """
        result.add_trace(f"searching by title: {title[:50]}...")

        # build search query
        papers = self._search_papers(title, hint_year, hint_author, result)

        if not papers:
            result.add_warning(f"no papers found matching: {title}")
            return (None, [])

        # rank by title similarity
        scored = []
        for paper in papers:
            similarity = self._compute_title_similarity(title, paper.title or "")
            scored.append((similarity, paper))

        scored.sort(key=lambda x: -x[0])

        # filter by minimum similarity
        good_matches = [(s, p) for s, p in scored if s >= self.min_title_similarity]

        if not good_matches:
            result.add_warning(
                f"best match has low similarity ({scored[0][0]:.0%}): {scored[0][1].title}"
            )
            # return best match anyway with warning
            return (scored[0][1], [p for _, p in scored[1:self.title_search_limit]])

        best = good_matches[0][1]
        alternatives = [p for _, p in good_matches[1:self.title_search_limit]]

        result.add_trace(f"best match ({good_matches[0][0]:.0%}): {best.title[:50]}...")

        return (best, alternatives)

    def _search_papers(
        self,
        title: str,
        hint_year: Optional[int],
        hint_author: Optional[str],
        result: AgentResult
    ) -> List[Paper]:
        """search OpenAlex for papers by title."""
        # build filter
        params = {
            "search": title,
            "per_page": self.title_search_limit * 2  # get extra for filtering
        }

        # add year filter if provided
        if hint_year:
            params["filter"] = f"publication_year:{hint_year}"

        # make request
        data = self.provider._request("works", params)
        if not data or "results" not in data:
            return []

        papers = []
        for work in data["results"]:
            paper = self.provider._parse_work(work)
            if paper:
                # apply author hint filtering
                if hint_author:
                    author_match = any(
                        hint_author.lower() in a.lower()
                        for a in (paper.authors or [])
                    )
                    if not author_match:
                        continue
                papers.append(paper)

        return papers[:self.title_search_limit]

    def _compute_title_similarity(self, query: str, title: str) -> float:
        """
        compute similarity between query and title.
        uses simple word overlap (jaccard-like).
        """
        if not query or not title:
            return 0.0

        # normalize
        q_words = set(self._normalize_title(query).split())
        t_words = set(self._normalize_title(title).split())

        if not q_words or not t_words:
            return 0.0

        # jaccard similarity
        intersection = len(q_words & t_words)
        union = len(q_words | t_words)

        return intersection / union if union > 0 else 0.0

    def _normalize_title(self, title: str) -> str:
        """normalize title for comparison."""
        # lowercase
        title = title.lower()
        # remove punctuation
        title = re.sub(r'[^\w\s]', ' ', title)
        # collapse whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        # remove common words
        stopwords = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'and', 'or', 'to', 'with'}
        words = [w for w in title.split() if w not in stopwords]
        return ' '.join(words)
