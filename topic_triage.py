"""
topic scope triage - determines if a topic is too broad.
implements the scope risk scoring from SPEC.md section 1.5.2.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from providers.openalex import OpenAlexProvider
from providers.base import PaperStub


class ScopeBand(Enum):
    GREEN = "GREEN"    # safe to proceed
    YELLOW = "YELLOW"  # recommend narrowing
    RED = "RED"        # must narrow before proceeding


@dataclass
class TopicTriage:
    """result of topic scope triage."""
    topic: str
    years_back: int
    count_estimate: int
    scope_risk_score: float
    scope_band: ScopeBand

    # breakdown of score components (for debugging/transparency)
    volume_score: float = 0.0
    specificity_score: float = 0.0
    concept_entropy_score: float = 0.0
    venue_dispersion_score: float = 0.0

    # additional context
    top_concepts: List[Dict[str, Any]] = field(default_factory=list)
    top_venues: List[Dict[str, Any]] = field(default_factory=list)
    seed_preview: List[PaperStub] = field(default_factory=list)

    # suggested refinements
    recommended_facets: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary for JSON output."""
        return {
            'topic': self.topic,
            'years_back': self.years_back,
            'count_estimate': self.count_estimate,
            'scope_risk_score': round(self.scope_risk_score, 3),
            'scope_band': self.scope_band.value,
            'score_breakdown': {
                'volume_score': round(self.volume_score, 3),
                'specificity_score': round(self.specificity_score, 3),
                'concept_entropy_score': round(self.concept_entropy_score, 3),
                'venue_dispersion_score': round(self.venue_dispersion_score, 3)
            },
            'top_concepts': self.top_concepts[:10],
            'top_venues': self.top_venues[:10],
            'seed_preview': [
                {
                    'title': p.title,
                    'year': p.year,
                    'doi': p.doi,
                    'citation_count': p.citation_count
                }
                for p in self.seed_preview[:10]
            ],
            'recommended_facets': self.recommended_facets
        }


def clamp(value: float, min_val: float, max_val: float) -> float:
    """clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def count_specific_tokens(topic: str) -> int:
    """
    count tokens that suggest specificity.
    includes: proper nouns (capitalized), technical terms, multi-word phrases.
    """
    # simple heuristic: count capitalized words and long technical-looking words
    words = topic.split()
    count = 0

    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if not clean:
            continue
        # capitalized (proper noun)
        if clean[0].isupper() and len(clean) > 1:
            count += 1
        # long technical word (>8 chars)
        elif len(clean) > 8:
            count += 0.5
        # contains numbers (e.g., "GPCR", "mRNA")
        elif any(c.isdigit() for c in clean):
            count += 0.5

    return int(count)


def shannon_entropy(counts: List[int]) -> float:
    """calculate normalized shannon entropy of a distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    # normalize by max entropy (uniform distribution)
    if len(counts) > 1:
        max_entropy = math.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    return 0.0


class TopicTriageEngine:
    """
    runs topic scope triage.
    determines if a topic is too broad and needs narrowing.
    """

    def __init__(self, provider: Optional[OpenAlexProvider] = None):
        self.provider = provider or OpenAlexProvider()

    def triage(self, topic: str, years_back: int = 3) -> TopicTriage:
        """
        run scope triage on a topic.
        returns TopicTriage with band (GREEN/YELLOW/RED) and metrics.
        """
        from datetime import datetime
        current_year = datetime.now().year
        year_min = current_year - years_back

        print(f"[triage] analyzing topic: '{topic}' (last {years_back} years)")

        # 1. get count estimate
        count_estimate = self.provider.get_count_estimate(topic, year_min=year_min)
        print(f"[triage] count estimate: {count_estimate:,}")

        # 2. get top concepts
        top_concepts = self.provider.get_top_concepts(topic, year_min=year_min, limit=20)

        # 3. get top venues
        top_venues = self.provider.get_top_venues(topic, year_min=year_min, limit=20)

        # 4. get seed preview (top 10 papers)
        seed_preview = self.provider.search_papers(topic, year_min=year_min, limit=10)

        # 5. compute scope risk score
        # volume_score = clamp(log10(count_estimate)/5, 0, 1)
        volume_score = clamp(math.log10(max(count_estimate, 1)) / 5, 0, 1)

        # specificity_score = clamp((num_specific_tokens)/10, 0, 1)
        num_specific = count_specific_tokens(topic)
        specificity_score = clamp(num_specific / 5, 0, 1)  # using /5 for reasonable scaling

        # concept_entropy_score = normalized shannon entropy over top_concepts
        concept_counts = [c['count'] for c in top_concepts[:10]] if top_concepts else []
        concept_entropy_score = shannon_entropy(concept_counts) if concept_counts else 0.5

        # venue_dispersion_score = unique venues / total seed_preview
        unique_venues = len(set(p.venue for p in seed_preview if p.venue))
        venue_dispersion_score = unique_venues / max(len(seed_preview), 1)

        # combine into scope risk score
        # ScopeRiskScore = 0.45*volume + 0.25*concept_entropy + 0.20*venue_dispersion + 0.10*(1 - specificity)
        scope_risk_score = (
            0.45 * volume_score +
            0.25 * concept_entropy_score +
            0.20 * venue_dispersion_score +
            0.10 * (1 - specificity_score)
        )

        print(f"[triage] risk score: {scope_risk_score:.3f}")
        print(f"[triage]   volume: {volume_score:.3f}, specificity: {specificity_score:.3f}")
        print(f"[triage]   concept_entropy: {concept_entropy_score:.3f}, venue_disp: {venue_dispersion_score:.3f}")

        # 6. determine band
        # RED if count > 20000 OR score > 0.75
        # YELLOW if count in (5000..20000] OR score in (0.55..0.75]
        # GREEN otherwise
        if count_estimate > 20000 or scope_risk_score > 0.75:
            scope_band = ScopeBand.RED
        elif count_estimate > 5000 or scope_risk_score > 0.55:
            scope_band = ScopeBand.YELLOW
        else:
            scope_band = ScopeBand.GREEN

        print(f"[triage] band: {scope_band.value}")

        # 7. generate recommended facets from concepts
        recommended_facets = self._extract_facets(top_concepts)

        return TopicTriage(
            topic=topic,
            years_back=years_back,
            count_estimate=count_estimate,
            scope_risk_score=scope_risk_score,
            scope_band=scope_band,
            volume_score=volume_score,
            specificity_score=specificity_score,
            concept_entropy_score=concept_entropy_score,
            venue_dispersion_score=venue_dispersion_score,
            top_concepts=top_concepts,
            top_venues=top_venues,
            seed_preview=seed_preview,
            recommended_facets=recommended_facets
        )

    def _extract_facets(self, concepts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        extract suggested facets from concepts.
        groups concepts into categories for the narrowing wizard.
        """
        # simple heuristic: just return top concepts as suggestions
        method_keywords = ['method', 'algorithm', 'technique', 'analysis', 'model', 'simulation']
        organism_keywords = ['human', 'mouse', 'bacteria', 'virus', 'plant', 'yeast', 'cell']

        methods = []
        organisms = []
        domains = []

        for c in concepts:
            name = c.get('name', '').lower()
            if any(kw in name for kw in method_keywords):
                methods.append(c['name'])
            elif any(kw in name for kw in organism_keywords):
                organisms.append(c['name'])
            else:
                domains.append(c['name'])

        return {
            'subdomain_options': domains[:5],
            'method_options': methods[:5],
            'organism_options': organisms[:5]
        }


# simple test
if __name__ == "__main__":
    engine = TopicTriageEngine()

    # test with a specific topic (should be GREEN)
    print("\n=== Test 1: specific topic ===")
    result = engine.triage("ancestral protein reconstruction", years_back=3)
    print(f"Result: {result.scope_band.value}")

    # test with a broad topic (should be RED)
    print("\n=== Test 2: broad topic ===")
    result = engine.triage("chemistry", years_back=3)
    print(f"Result: {result.scope_band.value}")
