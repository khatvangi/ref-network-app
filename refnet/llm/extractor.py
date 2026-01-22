"""
paper extractor - LLM-based extraction of structured info from papers.

extracts:
- key findings
- methods/techniques
- relationships to other work
- contribution type

usage:
    from refnet.llm import PaperExtractor, OllamaProvider

    provider = OllamaProvider(model="qwen3:32b")
    extractor = PaperExtractor(provider)

    # extract from a single paper
    info = extractor.extract(paper)
    print(info.key_findings)
    print(info.methods)

    # extract relationship to seed
    rel = extractor.extract_relationship(paper, seed_paper)
    print(rel.relationship_type)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .provider import LLMProvider
from ..core.models import Paper

logger = logging.getLogger("refnet.llm.extractor")


@dataclass
class ExtractedInfo:
    """structured information extracted from a paper."""
    paper_id: str
    paper_title: str

    # extracted content
    key_findings: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    contribution_type: str = ""  # "empirical", "theoretical", "methodological", "review"
    domain: str = ""  # primary research domain
    summary: str = ""  # 1-2 sentence summary

    # metadata
    extraction_model: str = ""
    extraction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "key_findings": self.key_findings,
            "methods": self.methods,
            "contribution_type": self.contribution_type,
            "domain": self.domain,
            "summary": self.summary
        }


@dataclass
class PaperRelationship:
    """relationship between two papers."""
    source_id: str
    target_id: str

    relationship_type: str = ""  # "builds_on", "contradicts", "extends", "applies", "reviews"
    relationship_strength: float = 0.0  # 0-1
    shared_concepts: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "relationship_strength": self.relationship_strength,
            "shared_concepts": self.shared_concepts,
            "description": self.description
        }


# extraction prompts
EXTRACTION_SYSTEM = """You are a scientific paper analyzer. Extract structured information from paper metadata.
Be precise and concise. Focus on the actual content, not generic statements.
Always respond with valid JSON."""

EXTRACTION_PROMPT = """Analyze this scientific paper and extract key information.

Title: {title}
Year: {year}
Venue: {venue}
Abstract: {abstract}
Concepts: {concepts}

Extract and return as JSON:
{{
    "key_findings": ["finding 1", "finding 2", ...],  // 2-4 main findings or contributions
    "methods": ["method 1", "method 2", ...],  // techniques, approaches, tools used
    "contribution_type": "empirical|theoretical|methodological|review|computational",
    "domain": "primary research field",
    "summary": "1-2 sentence summary of the paper's main contribution"
}}

Be specific to THIS paper. If abstract is missing, infer from title and concepts."""


RELATIONSHIP_SYSTEM = """You are analyzing relationships between scientific papers.
Identify how papers relate to each other in terms of ideas, methods, and findings.
Always respond with valid JSON."""

RELATIONSHIP_PROMPT = """Analyze the relationship between these two papers.

SEED PAPER (the reference point):
Title: {seed_title}
Year: {seed_year}
Abstract: {seed_abstract}

TARGET PAPER (to analyze relationship):
Title: {target_title}
Year: {target_year}
Abstract: {target_abstract}

Determine how the TARGET paper relates to the SEED paper.

Return as JSON:
{{
    "relationship_type": "builds_on|extends|applies|contradicts|reviews|parallel|foundational",
    "relationship_strength": 0.0-1.0,  // how strongly related
    "shared_concepts": ["concept1", "concept2"],  // concepts shared between papers
    "description": "one sentence explaining the relationship"
}}

Relationship types:
- builds_on: directly builds on the seed's work
- extends: extends seed's ideas to new areas
- applies: applies seed's methods to new problems
- contradicts: presents conflicting findings
- reviews: reviews/summarizes including the seed
- parallel: works on similar problems independently
- foundational: provides foundation that seed builds on"""


class PaperExtractor:
    """
    extracts structured information from papers using LLM.

    usage:
        provider = OllamaProvider(model="qwen3:32b")
        extractor = PaperExtractor(provider)

        info = extractor.extract(paper)
        relationship = extractor.extract_relationship(paper, seed)
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def extract(self, paper: Paper) -> ExtractedInfo:
        """extract structured info from a paper."""
        result = ExtractedInfo(
            paper_id=paper.id,
            paper_title=paper.title or "",
            extraction_model=self.provider.name
        )

        # build prompt
        concepts_str = ", ".join(
            c.get("name", str(c)) if isinstance(c, dict) else str(c)
            for c in (paper.concepts or [])[:10]
        )

        prompt = EXTRACTION_PROMPT.format(
            title=paper.title or "Unknown",
            year=paper.year or "Unknown",
            venue=paper.venue or "Unknown",
            abstract=paper.abstract or "Not available",
            concepts=concepts_str or "None"
        )

        # call LLM
        data = self.provider.generate_json(
            prompt=prompt,
            system=EXTRACTION_SYSTEM,
            temperature=0.2
        )

        if not data:
            logger.warning(f"extraction failed for paper {paper.id}")
            result.extraction_confidence = 0.0
            return result

        # populate result
        result.key_findings = data.get("key_findings", [])
        result.methods = data.get("methods", [])
        result.contribution_type = data.get("contribution_type", "")
        result.domain = data.get("domain", "")
        result.summary = data.get("summary", "")
        result.extraction_confidence = 0.8 if result.key_findings else 0.3

        return result

    def extract_relationship(
        self,
        target_paper: Paper,
        seed_paper: Paper
    ) -> PaperRelationship:
        """extract relationship between target and seed paper."""
        result = PaperRelationship(
            source_id=seed_paper.id,
            target_id=target_paper.id
        )

        prompt = RELATIONSHIP_PROMPT.format(
            seed_title=seed_paper.title or "Unknown",
            seed_year=seed_paper.year or "Unknown",
            seed_abstract=seed_paper.abstract or "Not available",
            target_title=target_paper.title or "Unknown",
            target_year=target_paper.year or "Unknown",
            target_abstract=target_paper.abstract or "Not available"
        )

        data = self.provider.generate_json(
            prompt=prompt,
            system=RELATIONSHIP_SYSTEM,
            temperature=0.2
        )

        if not data:
            logger.warning(f"relationship extraction failed for {target_paper.id}")
            return result

        result.relationship_type = data.get("relationship_type", "")
        result.relationship_strength = float(data.get("relationship_strength", 0.0))
        result.shared_concepts = data.get("shared_concepts", [])
        result.description = data.get("description", "")

        return result

    def extract_batch(
        self,
        papers: List[Paper],
        max_papers: int = 20
    ) -> List[ExtractedInfo]:
        """extract info from multiple papers."""
        results = []

        for paper in papers[:max_papers]:
            try:
                info = self.extract(paper)
                results.append(info)
                logger.debug(f"extracted: {paper.title[:40]}...")
            except Exception as e:
                logger.error(f"extraction failed for {paper.id}: {e}")
                results.append(ExtractedInfo(
                    paper_id=paper.id,
                    paper_title=paper.title or "",
                    extraction_confidence=0.0
                ))

        return results

    def extract_relationships_batch(
        self,
        papers: List[Paper],
        seed_paper: Paper,
        max_papers: int = 10
    ) -> List[PaperRelationship]:
        """extract relationships for multiple papers relative to seed."""
        results = []

        for paper in papers[:max_papers]:
            if paper.id == seed_paper.id:
                continue  # skip self
            try:
                rel = self.extract_relationship(paper, seed_paper)
                results.append(rel)
            except Exception as e:
                logger.error(f"relationship extraction failed: {e}")

        return results
