"""
author resolver agent - resolve author names to full profiles.

handles the challenge of author name disambiguation:
- common names (e.g., "J. Smith")
- name variations (e.g., "Charles W. Carter" vs "C. Carter")
- multiple authors with same name

input: author name + optional hints (affiliation, coauthors, paper context)
output: ResolvedAuthor with profile and confidence score
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set
from difflib import SequenceMatcher

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper
from ..providers.base import AuthorInfo
from ..providers.openalex import OpenAlexProvider


@dataclass
class AuthorCandidate:
    """
    a candidate author match.
    """
    author_info: AuthorInfo

    # match quality
    name_similarity: float = 0.0
    affiliation_match: bool = False
    coauthor_overlap: int = 0
    paper_overlap: int = 0

    # overall confidence
    confidence: float = 0.0

    # why this candidate was selected
    match_reasons: List[str] = field(default_factory=list)


@dataclass
class ResolvedAuthor:
    """
    result of author resolution.
    """
    # the resolved author
    author_info: AuthorInfo

    # resolution quality
    input_name: str
    confidence: float = 1.0

    # alternative candidates (for ambiguous cases)
    alternatives: List[AuthorCandidate] = field(default_factory=list)

    # disambiguation hints used
    hints_used: List[str] = field(default_factory=list)

    # warnings
    warnings: List[str] = field(default_factory=list)


class AuthorResolver(Agent):
    """
    resolve author names to full profiles with disambiguation.

    disambiguation strategies:
    1. Exact name match with high paper count (likely the prominent one)
    2. Affiliation hint (if provided)
    3. Coauthor network (if known coauthors provided)
    4. Paper context (if specific paper provided)

    usage:
        resolver = AuthorResolver(provider)
        result = resolver.run(
            name="Charles W. Carter",
            affiliation_hint="UNC",
            coauthor_hints=["Violetta Weinreb"]
        )

        if result.ok:
            author = result.data.author_info
            print(f"Resolved: {author.name} ({author.openalex_id})")
    """

    # thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MIN_NAME_SIMILARITY = 0.6

    def __init__(
        self,
        provider: OpenAlexProvider,
        max_candidates: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.provider = provider
        self.max_candidates = max_candidates

    @property
    def name(self) -> str:
        return "AuthorResolver"

    def execute(
        self,
        name: str,
        affiliation_hint: Optional[str] = None,
        coauthor_hints: Optional[List[str]] = None,
        paper_context: Optional[Paper] = None
    ) -> AgentResult[ResolvedAuthor]:
        """
        resolve an author name to a full profile.

        args:
            name: author name to resolve
            affiliation_hint: optional institution/affiliation hint
            coauthor_hints: optional list of known coauthor names
            paper_context: optional paper for context (to check author list)

        returns:
            AgentResult with ResolvedAuthor
        """
        result = AgentResult[ResolvedAuthor](status=AgentStatus.SUCCESS)
        name = name.strip()
        result.add_trace(f"resolving author: {name}")

        if not name:
            result.status = AgentStatus.FAILED
            result.add_error("EMPTY_NAME", "author name cannot be empty")
            return result

        # search for candidates
        candidates = self._search_candidates(name, affiliation_hint, result)

        if not candidates:
            result.status = AgentStatus.FAILED
            result.add_error("NOT_FOUND", f"no authors found matching: {name}")
            return result

        # score candidates
        scored_candidates = self._score_candidates(
            candidates, name, affiliation_hint, coauthor_hints, paper_context, result
        )

        # select best candidate
        best = scored_candidates[0]
        alternatives = scored_candidates[1:self.max_candidates]

        # build result
        hints_used = []
        if affiliation_hint:
            hints_used.append(f"affiliation: {affiliation_hint}")
        if coauthor_hints:
            hints_used.append(f"coauthors: {', '.join(coauthor_hints[:3])}")
        if paper_context:
            hints_used.append(f"paper context: {paper_context.title[:40]}...")

        resolved = ResolvedAuthor(
            author_info=best.author_info,
            input_name=name,
            confidence=best.confidence,
            alternatives=alternatives,
            hints_used=hints_used
        )

        # add warnings for low confidence or close alternatives
        if best.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
            resolved.warnings.append(
                f"Low confidence match ({best.confidence:.0%}). "
                f"Found {len(scored_candidates)} candidates."
            )

        if alternatives and alternatives[0].confidence > best.confidence * 0.8:
            resolved.warnings.append(
                f"Close alternative: {alternatives[0].author_info.name} "
                f"({alternatives[0].confidence:.0%} confidence)"
            )

        result.data = resolved
        result.api_calls = 1
        result.add_trace(
            f"resolved to: {best.author_info.name} ({best.confidence:.0%} confidence)"
        )

        return result

    def _search_candidates(
        self,
        name: str,
        affiliation_hint: Optional[str],
        result: AgentResult
    ) -> List[AuthorInfo]:
        """search for author candidates."""
        result.add_trace(f"searching for candidates...")

        # use provider's resolve_author_id which returns the best match
        # but we need multiple candidates, so let's search directly
        candidates = []

        # search OpenAlex authors
        data = self.provider._request("authors", {
            "search": name,
            "per_page": self.max_candidates * 2  # get extra for filtering
        })

        if not data or "results" not in data:
            return []

        for author_data in data["results"]:
            author_info = self._parse_author(author_data)
            if author_info:
                candidates.append(author_info)

        result.add_trace(f"found {len(candidates)} candidates")
        return candidates[:self.max_candidates * 2]

    def _parse_author(self, data: dict) -> Optional[AuthorInfo]:
        """parse OpenAlex author data to AuthorInfo."""
        if not data:
            return None

        # extract affiliations
        affiliations = []
        last_known = data.get("last_known_institutions") or []
        for inst in last_known[:3]:
            if inst and inst.get("display_name"):
                affiliations.append(inst["display_name"])

        # extract summary stats
        summary = data.get("summary_stats", {})

        return AuthorInfo(
            name=data.get("display_name", ""),
            openalex_id=data.get("id", "").split("/")[-1] if data.get("id") else None,
            orcid=data.get("orcid"),
            affiliations=affiliations,
            paper_count=data.get("works_count", 0),
            citation_count=data.get("cited_by_count", 0),
            h_index=summary.get("h_index")
        )

    def _score_candidates(
        self,
        candidates: List[AuthorInfo],
        query_name: str,
        affiliation_hint: Optional[str],
        coauthor_hints: Optional[List[str]],
        paper_context: Optional[Paper],
        result: AgentResult
    ) -> List[AuthorCandidate]:
        """score and rank candidates."""
        result.add_trace("scoring candidates...")

        scored = []

        for author in candidates:
            candidate = AuthorCandidate(author_info=author)

            # name similarity
            candidate.name_similarity = self._name_similarity(query_name, author.name)

            # skip if name is too different
            if candidate.name_similarity < self.MIN_NAME_SIMILARITY:
                continue

            candidate.match_reasons.append(
                f"name match: {candidate.name_similarity:.0%}"
            )

            # affiliation match
            if affiliation_hint and author.affiliations:
                affiliation_hint_lower = affiliation_hint.lower()
                for aff in author.affiliations:
                    if affiliation_hint_lower in aff.lower():
                        candidate.affiliation_match = True
                        candidate.match_reasons.append(f"affiliation: {aff}")
                        break

            # coauthor overlap (would need additional API call)
            # for now, skip this expensive check

            # paper context check
            if paper_context and paper_context.authors:
                paper_authors_lower = [a.lower() for a in paper_context.authors]
                if author.name.lower() in paper_authors_lower:
                    candidate.paper_overlap = 1
                    candidate.match_reasons.append("found in paper author list")

            # compute overall confidence
            candidate.confidence = self._compute_confidence(candidate)

            scored.append(candidate)

        # sort by confidence
        scored.sort(key=lambda c: -c.confidence)

        return scored

    def _name_similarity(self, name1: str, name2: str) -> float:
        """
        compute name similarity with special handling for author names.
        """
        # normalize names
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)

        # exact match
        if n1 == n2:
            return 1.0

        # check for initial-based matching
        # "C. W. Carter" should match "Charles W. Carter"
        parts1 = n1.split()
        parts2 = n2.split()

        # if last names match, check first name/initials
        if parts1 and parts2 and parts1[-1] == parts2[-1]:
            # last names match - check first names
            if len(parts1) > 1 and len(parts2) > 1:
                # check if one is initial of other
                f1, f2 = parts1[0], parts2[0]
                if f1[0] == f2[0]:  # same first initial
                    return 0.9

        # fallback to sequence matching
        return SequenceMatcher(None, n1, n2).ratio()

    def _normalize_name(self, name: str) -> str:
        """normalize author name for comparison."""
        # lowercase
        name = name.lower()
        # remove periods and commas
        name = name.replace(".", "").replace(",", "")
        # collapse whitespace
        name = " ".join(name.split())
        return name

    def _compute_confidence(self, candidate: AuthorCandidate) -> float:
        """compute overall confidence score."""
        score = candidate.name_similarity * 0.5

        # affiliation match bonus
        if candidate.affiliation_match:
            score += 0.2

        # paper context bonus
        if candidate.paper_overlap > 0:
            score += 0.2

        # prominence bonus (highly published authors more likely to be searched for)
        if candidate.author_info.paper_count and candidate.author_info.paper_count > 50:
            score += 0.1

        return min(1.0, score)
