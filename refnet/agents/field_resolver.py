"""
field resolver - identifies research field and provides domain knowledge.

usage:
    resolver = FieldResolver()
    result = resolver.run(seed_paper=paper)
    # or
    result = resolver.run(query="palladium catalysis")

    if result.ok:
        field = result.data.primary_field
        print(f"Field: {field.name}")
        print(f"Tier 1 journals: {field.tier1_journals}")
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from difflib import SequenceMatcher

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper

logger = logging.getLogger("refnet.agents.FieldResolver")


@dataclass
class FieldProfile:
    """domain knowledge for a research field."""
    name: str                    # "Organic Chemistry"
    aliases: List[str] = field(default_factory=list)  # ["orgo", "synthetic chemistry"]
    parent_field: str = ""       # "Chemistry"

    # journal tiers
    tier1_journals: List[str] = field(default_factory=list)  # high-impact
    tier2_journals: List[str] = field(default_factory=list)  # specialty
    tier3_sources: List[str] = field(default_factory=list)   # general

    # bootstrap knowledge
    known_leaders: List[str] = field(default_factory=list)   # famous researchers
    key_concepts: List[str] = field(default_factory=list)    # domain terminology

    # OpenAlex mapping
    openalex_concepts: List[str] = field(default_factory=list)  # concept IDs

    # search hints
    exclude_terms: List[str] = field(default_factory=list)   # terms indicating wrong field
    include_terms: List[str] = field(default_factory=list)   # terms that should appear


@dataclass
class FieldResolution:
    """result of field identification."""
    primary_field: FieldProfile
    confidence: float                        # 0.0 - 1.0
    secondary_fields: List[FieldProfile] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # why we think this
    suggested_strategy: str = "balanced"     # "tier1_first", "author_centric", "broad"


# =============================================================================
# Field Profiles Database
# =============================================================================

FIELD_PROFILES: Dict[str, FieldProfile] = {

    "organic_chemistry": FieldProfile(
        name="Organic Chemistry",
        aliases=["synthetic chemistry", "orgo", "organic synthesis"],
        parent_field="Chemistry",
        tier1_journals=[
            "Journal of the American Chemical Society",
            "Angewandte Chemie International Edition",
            "Nature Chemistry",
            "Science",
            "Nature",
            "Chemical Science"
        ],
        tier2_journals=[
            "Journal of Organic Chemistry",
            "Organic Letters",
            "Chemistry - A European Journal",
            "ACS Catalysis",
            "Organic & Biomolecular Chemistry",
            "Tetrahedron",
            "Tetrahedron Letters"
        ],
        known_leaders=[
            "John F. Hartwig",
            "Stephen L. Buchwald",
            "Jin-Quan Yu",
            "Melanie S. Sanford",
            "Phil S. Baran",
            "David W. C. MacMillan",
            "Benjamin List"
        ],
        key_concepts=[
            "catalysis", "synthesis", "cross-coupling", "C-H activation",
            "asymmetric", "ligand", "palladium", "reaction", "substrate",
            "stereochemistry", "enantioselective", "total synthesis"
        ],
        exclude_terms=["in vivo", "clinical trial", "patient"],
        include_terms=["synthesis", "reaction", "catalyst"]
    ),

    "biochemistry": FieldProfile(
        name="Biochemistry",
        aliases=["biological chemistry", "molecular biology"],
        parent_field="Biology",
        tier1_journals=[
            "Nature",
            "Science",
            "Cell",
            "Nature Chemical Biology",
            "Nature Structural & Molecular Biology",
            "Molecular Cell"
        ],
        tier2_journals=[
            "Journal of Biological Chemistry",
            "Biochemistry",
            "PNAS",
            "eLife",
            "Journal of Molecular Biology",
            "Nucleic Acids Research"
        ],
        known_leaders=[
            "Jennifer Doudna",
            "Feng Zhang",
            "David Baker",
            "Frances Arnold",
            "Carolyn Bertozzi"
        ],
        key_concepts=[
            "protein", "enzyme", "DNA", "RNA", "gene", "cell",
            "binding", "structure", "mechanism", "pathway",
            "substrate", "kinetics", "folding"
        ],
        exclude_terms=["synthesis route", "total synthesis"],
        include_terms=["protein", "enzyme", "biological"]
    ),

    "origins_of_life": FieldProfile(
        name="Origins of Life",
        aliases=["prebiotic chemistry", "abiogenesis", "chemical evolution"],
        parent_field="Chemistry",
        tier1_journals=[
            "Nature",
            "Science",
            "Nature Chemistry",
            "PNAS",
            "Journal of the American Chemical Society"
        ],
        tier2_journals=[
            "Origins of Life and Evolution of Biospheres",
            "Astrobiology",
            "Life",
            "ChemBioChem",
            "Angewandte Chemie"
        ],
        known_leaders=[
            "Jack W. Szostak",
            "John Sutherland",
            "Gerald Joyce",
            "Ramanarayanan Krishnamurthy",
            "Nicholas Hud",
            "Matthew Powner"
        ],
        key_concepts=[
            "RNA", "ribozyme", "prebiotic", "origin", "life", "protocell",
            "replication", "nucleotide", "amino acid", "primitive",
            "abiotic", "nonenzymatic", "self-assembly"
        ],
        exclude_terms=["clinical", "therapeutic", "drug"],
        include_terms=["prebiotic", "origin", "primitive"]
    ),

    "structural_biology": FieldProfile(
        name="Structural Biology",
        aliases=["protein structure", "molecular structure"],
        parent_field="Biology",
        tier1_journals=[
            "Nature",
            "Science",
            "Cell",
            "Nature Structural & Molecular Biology",
            "Structure"
        ],
        tier2_journals=[
            "Journal of Molecular Biology",
            "Protein Science",
            "Acta Crystallographica Section D",
            "Journal of Structural Biology",
            "IUCrJ"
        ],
        known_leaders=[
            "David Baker",
            "John Jumper",
            "Venki Ramakrishnan",
            "Ada Yonath",
            "Roger Kornberg"
        ],
        key_concepts=[
            "structure", "crystal", "cryo-EM", "X-ray", "NMR",
            "AlphaFold", "protein", "complex", "resolution",
            "conformation", "binding", "domain"
        ],
        exclude_terms=["synthesis", "reaction mechanism"],
        include_terms=["structure", "crystal", "cryo-EM"]
    ),

    "machine_learning": FieldProfile(
        name="Machine Learning",
        aliases=["deep learning", "AI", "artificial intelligence"],
        parent_field="Computer Science",
        tier1_journals=[
            "Nature",
            "Science",
            "Nature Machine Intelligence",
            "NeurIPS",
            "ICML",
            "ICLR"
        ],
        tier2_journals=[
            "Journal of Machine Learning Research",
            "IEEE TPAMI",
            "Neural Networks",
            "Machine Learning",
            "AAAI"
        ],
        known_leaders=[
            "Geoffrey Hinton",
            "Yann LeCun",
            "Yoshua Bengio",
            "Demis Hassabis",
            "Ilya Sutskever",
            "Andrej Karpathy"
        ],
        key_concepts=[
            "neural network", "deep learning", "transformer", "attention",
            "training", "model", "prediction", "classification",
            "representation", "embedding", "gradient"
        ],
        exclude_terms=["synthesis", "wet lab", "in vivo"],
        include_terms=["model", "training", "neural"]
    ),

    "medicinal_chemistry": FieldProfile(
        name="Medicinal Chemistry",
        aliases=["drug discovery", "pharmaceutical chemistry"],
        parent_field="Chemistry",
        tier1_journals=[
            "Nature",
            "Science",
            "Journal of Medicinal Chemistry",
            "Nature Medicine",
            "Cell"
        ],
        tier2_journals=[
            "Journal of Medicinal Chemistry",
            "Bioorganic & Medicinal Chemistry",
            "European Journal of Medicinal Chemistry",
            "ACS Medicinal Chemistry Letters",
            "Drug Discovery Today"
        ],
        known_leaders=[
            "Patrick Baeuerle",
            "Christopher Lipinski"
        ],
        key_concepts=[
            "drug", "target", "inhibitor", "binding", "IC50",
            "potency", "selectivity", "ADMET", "pharmacokinetics",
            "lead optimization", "SAR", "clinical"
        ],
        exclude_terms=["total synthesis", "methodology"],
        include_terms=["drug", "inhibitor", "therapeutic"]
    ),

    "genomics": FieldProfile(
        name="Genomics",
        aliases=["genome science", "sequencing"],
        parent_field="Biology",
        tier1_journals=[
            "Nature",
            "Science",
            "Nature Genetics",
            "Cell",
            "Nature Methods"
        ],
        tier2_journals=[
            "Genome Research",
            "Genome Biology",
            "Nucleic Acids Research",
            "BMC Genomics",
            "PLOS Genetics"
        ],
        known_leaders=[
            "Eric Lander",
            "J. Craig Venter",
            "Francis Collins",
            "Svante Pääbo"
        ],
        key_concepts=[
            "genome", "sequencing", "gene", "variant", "mutation",
            "GWAS", "expression", "chromosome", "DNA", "RNA-seq",
            "single-cell", "epigenetic"
        ],
        exclude_terms=["synthesis", "catalyst"],
        include_terms=["genome", "sequencing", "gene"]
    ),

    "neuroscience": FieldProfile(
        name="Neuroscience",
        aliases=["brain science", "neural science"],
        parent_field="Biology",
        tier1_journals=[
            "Nature",
            "Science",
            "Cell",
            "Neuron",
            "Nature Neuroscience"
        ],
        tier2_journals=[
            "Journal of Neuroscience",
            "eLife",
            "Current Biology",
            "PNAS",
            "Cerebral Cortex"
        ],
        known_leaders=[
            "Karl Deisseroth",
            "Ed Boyden",
            "May-Britt Moser",
            "Edvard Moser"
        ],
        key_concepts=[
            "neuron", "brain", "synapse", "cortex", "circuit",
            "behavior", "cognitive", "neural", "imaging",
            "optogenetics", "connectome"
        ],
        exclude_terms=["synthesis", "catalyst", "reaction"],
        include_terms=["brain", "neuron", "neural"]
    ),

    "aminoacyl_trna_synthetases": FieldProfile(
        name="Aminoacyl-tRNA Synthetases",
        aliases=["aaRS", "tRNA synthetases", "genetic code"],
        parent_field="Biochemistry",
        tier1_journals=[
            "Nature",
            "Science",
            "PNAS",
            "Cell",
            "Nature Chemical Biology"
        ],
        tier2_journals=[
            "Journal of Biological Chemistry",
            "Nucleic Acids Research",
            "RNA",
            "Journal of Molecular Biology",
            "Biochemistry"
        ],
        known_leaders=[
            "Charles W. Carter",
            "Paul Schimmel",
            "Dieter Söll",
            "Michael Ibba",
            "Karin Musier-Forsyth"
        ],
        key_concepts=[
            "aminoacyl-tRNA synthetase", "tRNA", "genetic code",
            "aminoacylation", "codon", "anticodon", "class I", "class II",
            "protein synthesis", "translation", "amino acid"
        ],
        exclude_terms=["drug", "clinical", "patient"],
        include_terms=["tRNA", "synthetase", "aminoacyl"]
    )
}


def _text_similarity(text1: str, text2: str) -> float:
    """compute similarity between two strings."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


class FieldResolver(Agent):
    """
    identifies research field from seed paper or query.

    provides domain knowledge including:
    - journal tiers (high-impact first)
    - known leaders in the field
    - key concepts/terminology
    - search strategy recommendations
    """

    def __init__(self, custom_profiles: Optional[Dict[str, FieldProfile]] = None):
        """
        initialize with optional custom field profiles.

        args:
            custom_profiles: additional field profiles to use
        """
        super().__init__()
        self.profiles = {**FIELD_PROFILES}
        if custom_profiles:
            self.profiles.update(custom_profiles)

    @property
    def name(self) -> str:
        return "FieldResolver"

    def execute(
        self,
        seed_paper: Optional[Paper] = None,
        query: Optional[str] = None
    ) -> AgentResult[FieldResolution]:
        """
        identify research field.

        args:
            seed_paper: paper to analyze
            query: text query to analyze

        returns:
            AgentResult with FieldResolution
        """
        result = AgentResult[FieldResolution](status=AgentStatus.SUCCESS)

        if not seed_paper and not query:
            result.status = AgentStatus.FAILED
            result.add_error("MISSING_INPUT", "either seed_paper or query required")
            return result

        # extract signals
        signals = self._extract_signals(seed_paper, query)
        result.add_trace(f"extracted signals from input")

        # match against known fields
        candidates = self._match_fields(signals)

        if not candidates:
            # "ocean mode" - no specific field, go broad
            fallback = FieldProfile(
                name="General Science",
                aliases=["multidisciplinary", "broad science"],
                parent_field="Science",
                # broad tier1 includes top multidisciplinary journals
                tier1_journals=[
                    "Nature", "Science", "PNAS", "Cell",
                    "Nature Communications", "Science Advances",
                    "eLife", "PLoS ONE"
                ],
                # tier2 is empty - in ocean mode, all venues are equal
                tier2_journals=[],
                tier3_sources=["any"],
                known_leaders=[],  # no field-specific leaders
                key_concepts=signals.get("concepts", [])[:10],
                exclude_terms=[],
                include_terms=[]
            )
            result.data = FieldResolution(
                primary_field=fallback,
                confidence=0.0,  # zero confidence = ocean mode
                evidence=["No specific field matched; using broad search (ocean mode)"],
                suggested_strategy="ocean"  # special strategy for broad search
            )
            result.add_trace("no specific field matched, entering ocean mode")
            return result

        # rank by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        primary = candidates[0]
        secondary = [c[0] for c in candidates[1:3]]

        confidence = min(primary[1], 1.0)

        # determine strategy based on confidence
        if confidence >= 0.8:
            strategy = "tier1_first"
        elif confidence >= 0.5:
            strategy = "balanced"
        else:
            strategy = "broad"

        result.data = FieldResolution(
            primary_field=primary[0],
            confidence=confidence,
            secondary_fields=secondary,
            evidence=primary[2],
            suggested_strategy=strategy
        )

        result.add_trace(f"identified field: {primary[0].name} (confidence: {confidence:.0%})")

        return result

    def _extract_signals(
        self,
        seed_paper: Optional[Paper],
        query: Optional[str]
    ) -> Dict[str, any]:
        """extract signals for field matching."""
        signals = {
            "title": "",
            "abstract": "",
            "concepts": [],
            "venue": "",
            "authors": [],
            "query": query or ""
        }

        if seed_paper:
            signals["title"] = seed_paper.title or ""
            signals["abstract"] = seed_paper.abstract or ""
            # concepts is List[Dict[str, Any]] with {name, score}
            if seed_paper.concepts:
                signals["concepts"] = [
                    c.get("name", "") if isinstance(c, dict) else str(c)
                    for c in seed_paper.concepts
                ]
            signals["venue"] = seed_paper.venue or ""
            signals["authors"] = seed_paper.authors or []

        return signals

    def _match_fields(
        self,
        signals: Dict[str, any]
    ) -> List[tuple]:
        """match signals against known fields, return (profile, score, evidence)."""
        candidates = []

        text = " ".join([
            signals["title"],
            signals["abstract"],
            signals["query"],
            " ".join(signals["concepts"])
        ]).lower()

        for field_id, profile in self.profiles.items():
            score = 0.0
            evidence = []

            # check key concepts (most important)
            concept_matches = 0
            for concept in profile.key_concepts:
                if concept.lower() in text:
                    concept_matches += 1

            if profile.key_concepts:
                concept_score = concept_matches / len(profile.key_concepts)
                score += concept_score * 0.4
                if concept_matches > 0:
                    evidence.append(f"matched {concept_matches} key concepts")

            # check known leaders
            leader_matches = 0
            for leader in profile.known_leaders:
                if any(_text_similarity(leader, a) > 0.8 for a in signals["authors"]):
                    leader_matches += 1

            if leader_matches > 0:
                score += 0.3
                evidence.append(f"matched {leader_matches} known leaders")

            # check venue (journal/conference)
            venue = signals["venue"].lower()
            if venue:
                for j in profile.tier1_journals:
                    if _text_similarity(venue, j) > 0.7:
                        score += 0.2
                        evidence.append(f"tier 1 venue: {j}")
                        break
                else:
                    for j in profile.tier2_journals:
                        if _text_similarity(venue, j) > 0.7:
                            score += 0.1
                            evidence.append(f"tier 2 venue: {j}")
                            break

            # check field name/aliases in text
            for alias in [profile.name.lower()] + [a.lower() for a in profile.aliases]:
                if alias in text:
                    score += 0.1
                    evidence.append(f"field name mentioned: {alias}")
                    break

            # check exclude terms (reduce score)
            for term in profile.exclude_terms:
                if term.lower() in text:
                    score -= 0.15
                    evidence.append(f"exclude term found: {term}")

            # check include terms (boost score)
            include_matches = sum(1 for t in profile.include_terms if t.lower() in text)
            if include_matches > 0 and profile.include_terms:
                boost = 0.1 * (include_matches / len(profile.include_terms))
                score += boost

            if score > 0.1:  # minimum threshold
                candidates.append((profile, score, evidence))

        return candidates

    def get_profile(self, field_name: str) -> Optional[FieldProfile]:
        """get a specific field profile by name or alias."""
        name_lower = field_name.lower()

        # direct match
        if name_lower in self.profiles:
            return self.profiles[name_lower]

        # alias match
        for profile in self.profiles.values():
            if name_lower == profile.name.lower():
                return profile
            if name_lower in [a.lower() for a in profile.aliases]:
                return profile

        return None

    def list_fields(self) -> List[str]:
        """list all known field names."""
        return [p.name for p in self.profiles.values()]
