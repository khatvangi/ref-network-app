"""
checkpoint core types - user interaction points in the pipeline.

checkpoints allow the pipeline to pause and ask for user confirmation
at critical decision points:
- field identification
- seed paper selection
- expansion direction

usage:
    checkpoint = create_field_checkpoint(field_resolution)
    response = handler.present(checkpoint)

    if response.confirmed:
        # proceed with identified field
    else:
        # use user's correction
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..agents.field_resolver import FieldResolution
from ..core.models import Paper


class CheckpointType(Enum):
    """types of checkpoints."""
    FIELD = "field"              # verify identified field
    SEED = "seed"                # verify seed papers
    DIRECTION = "direction"      # verify expansion direction
    SCOPE = "scope"              # verify search scope
    AUTHOR = "author"            # verify author identification


@dataclass
class CheckpointResponse:
    """user response to a checkpoint."""
    confirmed: bool                          # user confirmed the suggestion
    correction: Optional[str] = None         # user's correction if not confirmed
    skip_future: bool = False                # skip similar checkpoints
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """base checkpoint for user verification."""
    checkpoint_type: CheckpointType
    question: str                            # main question to ask
    context: str                             # background info
    suggestion: str                          # what we're suggesting
    confidence: float                        # how confident we are
    evidence: List[str] = field(default_factory=list)  # why we think this
    options: List[str] = field(default_factory=list)   # alternative options
    details: Dict[str, Any] = field(default_factory=dict)  # additional data


@dataclass
class FieldCheckpoint(Checkpoint):
    """checkpoint for field identification."""
    identified_field: str = ""
    suggested_journals: List[str] = field(default_factory=list)
    suggested_authors: List[str] = field(default_factory=list)
    alternative_fields: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.checkpoint_type = CheckpointType.FIELD


@dataclass
class SeedCheckpoint(Checkpoint):
    """checkpoint for seed paper verification."""
    seed_papers: List[Paper] = field(default_factory=list)
    paper_summaries: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.checkpoint_type = CheckpointType.SEED


@dataclass
class DirectionCheckpoint(Checkpoint):
    """checkpoint for expansion direction."""
    current_topics: List[str] = field(default_factory=list)
    proposed_directions: List[str] = field(default_factory=list)
    papers_in_direction: int = 0

    def __post_init__(self):
        self.checkpoint_type = CheckpointType.DIRECTION


# =============================================================================
# Factory Functions
# =============================================================================

def create_field_checkpoint(resolution: FieldResolution) -> FieldCheckpoint:
    """create a field verification checkpoint from a field resolution."""
    field = resolution.primary_field

    # build question
    journal_list = ", ".join(field.tier1_journals[:3])
    question = (
        f"I identified this as '{field.name}'. "
        f"Key journals include: {journal_list}. "
        f"Is this the correct research field?"
    )

    # build context
    context = (
        f"Confidence: {resolution.confidence:.0%}\n"
        f"Parent field: {field.parent_field or 'General'}"
    )

    # build options (alternative fields)
    alternatives = [f.name for f in resolution.secondary_fields[:3]]

    return FieldCheckpoint(
        checkpoint_type=CheckpointType.FIELD,
        question=question,
        context=context,
        suggestion=field.name,
        confidence=resolution.confidence,
        evidence=resolution.evidence,
        options=["Confirm", "Choose alternative", "Specify different field"],
        details={"strategy": resolution.suggested_strategy},
        identified_field=field.name,
        suggested_journals=field.tier1_journals[:5],
        suggested_authors=field.known_leaders[:5],
        alternative_fields=alternatives
    )


def create_seed_checkpoint(
    papers: List[Paper],
    query: str,
    confidence: float = 0.8
) -> SeedCheckpoint:
    """create a seed paper verification checkpoint."""
    # build summaries
    summaries = []
    for p in papers[:5]:
        year_str = f" ({p.year})" if p.year else ""
        cites_str = f", {p.citation_count} citations" if p.citation_count else ""
        summary = f"• {p.title[:60]}...{year_str}{cites_str}"
        summaries.append(summary)

    # build question
    question = (
        f"Found {len(papers)} seed paper(s) for query '{query[:50]}'. "
        f"Are these the papers you're looking for?"
    )

    # context
    context = f"Showing top {min(5, len(papers))} results by citation count."

    return SeedCheckpoint(
        checkpoint_type=CheckpointType.SEED,
        question=question,
        context=context,
        suggestion=papers[0].title if papers else "No papers found",
        confidence=confidence,
        evidence=[f"Query: {query}"],
        options=["Confirm all", "Select specific papers", "Search again"],
        details={"query": query, "total_found": len(papers)},
        seed_papers=papers[:5],
        paper_summaries=summaries
    )


def create_direction_checkpoint(
    topics: List[str],
    proposed_directions: List[str],
    papers_found: int,
    confidence: float = 0.7
) -> DirectionCheckpoint:
    """create an expansion direction checkpoint."""
    # build question
    direction_list = ", ".join(proposed_directions[:3])
    question = (
        f"Current focus: {', '.join(topics[:3])}. "
        f"I suggest expanding towards: {direction_list}. "
        f"Should I proceed in this direction?"
    )

    # context
    context = f"Found {papers_found} papers in the proposed direction."

    return DirectionCheckpoint(
        checkpoint_type=CheckpointType.DIRECTION,
        question=question,
        context=context,
        suggestion=proposed_directions[0] if proposed_directions else "No direction",
        confidence=confidence,
        evidence=[f"Based on {papers_found} papers"],
        options=["Proceed", "Explore different direction", "Stay focused"],
        details={"papers_in_direction": papers_found},
        current_topics=topics,
        proposed_directions=proposed_directions,
        papers_in_direction=papers_found
    )
