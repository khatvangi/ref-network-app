"""
trajectory analyzer agent - understand an author's research evolution.

answers key questions:
- what has this author been working on?
- how has their focus evolved over time?
- is this paper a new direction or lifelong interest?
- where are they heading?

input: AuthorCorpus (from CorpusFetcher)
output: TrajectoryAnalysis with phases, drift events, and insights
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple

from .base import Agent, AgentResult, AgentStatus
from .corpus_fetcher import AuthorCorpus
from ..core.models import Paper


@dataclass
class ResearchPhase:
    """
    a distinct phase in an author's research trajectory.

    authors typically have 2-5 major phases in their career,
    each characterized by dominant concepts and collaborators.
    """
    phase_id: int
    start_year: int
    end_year: int
    duration_years: int

    # what defined this phase
    dominant_concepts: List[str]
    top_papers: List[str]  # paper IDs
    paper_count: int
    total_citations: int

    # collaborators in this phase
    key_collaborators: List[str]

    # phase characteristics
    label: str = ""  # auto-generated or user-provided label
    is_formative: bool = False  # early career foundation
    is_current: bool = False  # ongoing phase


@dataclass
class DriftEvent:
    """
    a significant shift in research focus.

    drift events mark transitions between phases.
    high drift = author moved to new territory.
    """
    year: int
    drift_magnitude: float  # 0-1, higher = bigger shift

    # what changed
    concepts_entering: List[str]  # new topics
    concepts_exiting: List[str]  # abandoned topics

    # trigger (if detectable)
    trigger_paper: Optional[str] = None  # paper ID that started new direction
    trigger_type: str = "gradual"  # "gradual", "sudden", "collaboration"

    # is this a significant jump?
    is_novelty_jump: bool = False  # true if magnitude > threshold


@dataclass
class TrajectoryAnalysis:
    """
    complete trajectory analysis for an author.

    tells the story of their research career.
    """
    # identity
    author_id: str
    author_name: str

    # career span
    career_start: int
    career_end: int
    career_years: int
    total_papers: int

    # phases
    phases: List[ResearchPhase] = field(default_factory=list)
    current_phase: Optional[ResearchPhase] = None

    # drift events
    drift_events: List[DriftEvent] = field(default_factory=list)
    total_drift: float = 0.0  # cumulative drift
    avg_drift_per_year: float = 0.0

    # trajectory characteristics
    trajectory_type: str = "unknown"  # "focused", "explorer", "bridger", "shifter"
    stability_score: float = 0.0  # 0-1, higher = more stable focus

    # current direction
    emerging_concepts: List[str] = field(default_factory=list)
    declining_concepts: List[str] = field(default_factory=list)

    # insights
    insights: List[str] = field(default_factory=list)

    def add_insight(self, insight: str):
        """add human-readable insight about trajectory."""
        self.insights.append(insight)


class TrajectoryAnalyzer(Agent):
    """
    analyze author's research trajectory over time.

    takes a corpus and identifies:
    - research phases (major focus areas)
    - drift events (significant topic changes)
    - trajectory type (focused vs. explorer)
    - current direction (where they're heading)

    usage:
        analyzer = TrajectoryAnalyzer()
        result = analyzer.run(corpus)

        if result.ok:
            trajectory = result.data
            print(f"Trajectory type: {trajectory.trajectory_type}")
            for phase in trajectory.phases:
                print(f"  {phase.start_year}-{phase.end_year}: {phase.label}")
    """

    # parameters for analysis
    MIN_PAPERS_FOR_PHASE = 3  # need at least this many to form a phase
    DRIFT_THRESHOLD = 0.3  # drift above this is "significant"
    NOVELTY_JUMP_THRESHOLD = 0.5  # drift above this is a "novelty jump"
    WINDOW_SIZE = 3  # years to group for concept analysis

    def __init__(
        self,
        window_size: int = 3,
        drift_threshold: float = 0.3,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.window_size = window_size
        self.drift_threshold = drift_threshold

    @property
    def name(self) -> str:
        return "TrajectoryAnalyzer"

    def execute(
        self,
        corpus: AuthorCorpus,
        reference_concepts: Optional[Set[str]] = None
    ) -> AgentResult[TrajectoryAnalysis]:
        """
        analyze author's research trajectory.

        args:
            corpus: AuthorCorpus from CorpusFetcher
            reference_concepts: optional set of concepts to track
                               (e.g., concepts from seed papers)

        returns:
            AgentResult with TrajectoryAnalysis
        """
        result = AgentResult[TrajectoryAnalysis](status=AgentStatus.SUCCESS)
        result.add_trace(f"analyzing trajectory for: {corpus.name}")

        # need papers to analyze
        if not corpus.papers:
            result.status = AgentStatus.FAILED
            result.add_error("NO_PAPERS", "corpus has no papers to analyze")
            return result

        # filter papers with years
        papers_with_years = [p for p in corpus.papers if p.year]
        if len(papers_with_years) < 3:
            result.status = AgentStatus.PARTIAL
            result.add_warning(f"only {len(papers_with_years)} papers with years, analysis may be limited")

        # create analysis object
        years = [p.year for p in papers_with_years]
        analysis = TrajectoryAnalysis(
            author_id=corpus.author_id,
            author_name=corpus.name,
            career_start=min(years) if years else 0,
            career_end=max(years) if years else 0,
            career_years=(max(years) - min(years) + 1) if years else 0,
            total_papers=len(corpus.papers)
        )

        result.add_trace(f"career span: {analysis.career_start}-{analysis.career_end} ({analysis.career_years} years)")

        # step 1: build concept timeline
        concept_timeline = self._build_concept_timeline(papers_with_years, result)

        # step 2: detect drift events
        drift_events = self._detect_drift_events(concept_timeline, result)
        analysis.drift_events = drift_events

        # compute drift statistics
        if drift_events:
            analysis.total_drift = sum(d.drift_magnitude for d in drift_events)
            analysis.avg_drift_per_year = analysis.total_drift / max(analysis.career_years, 1)

        # step 3: identify phases (based on drift events)
        phases = self._identify_phases(papers_with_years, drift_events, result)
        analysis.phases = phases
        if phases:
            analysis.current_phase = phases[-1]

        # step 4: determine trajectory type
        analysis.trajectory_type = self._classify_trajectory(analysis, result)
        analysis.stability_score = self._compute_stability(analysis)

        # step 5: find current direction
        emerging, declining = self._find_direction(concept_timeline, result)
        analysis.emerging_concepts = emerging
        analysis.declining_concepts = declining

        # step 6: generate insights
        self._generate_insights(analysis, corpus, reference_concepts, result)

        result.data = analysis
        result.add_trace(f"analysis complete: {len(phases)} phases, {len(drift_events)} drift events")

        return result

    def _build_concept_timeline(
        self,
        papers: List[Paper],
        result: AgentResult
    ) -> Dict[int, Dict[str, float]]:
        """
        build year -> {concept: weight} mapping.

        groups papers by year and aggregates concept weights.
        """
        result.add_trace("building concept timeline")

        timeline: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for paper in papers:
            year = paper.year
            if not year:
                continue

            for concept in (paper.concepts or [])[:7]:
                name = concept.get('name', '').lower()
                score = concept.get('score', 0.5)
                if name:
                    timeline[year][name] += score

        result.add_trace(f"timeline built: {len(timeline)} years")
        return dict(timeline)

    def _detect_drift_events(
        self,
        timeline: Dict[int, Dict[str, float]],
        result: AgentResult
    ) -> List[DriftEvent]:
        """
        detect significant changes in research focus.

        compares concept distributions between adjacent time windows.
        """
        result.add_trace("detecting drift events")

        if len(timeline) < 2:
            return []

        years = sorted(timeline.keys())
        drift_events = []

        for i in range(1, len(years)):
            year = years[i]
            prev_year = years[i - 1]

            # get concept sets for current and previous
            curr_concepts = set(timeline[year].keys())
            prev_concepts = set(timeline[prev_year].keys())

            # compute drift as jaccard distance
            intersection = curr_concepts & prev_concepts
            union = curr_concepts | prev_concepts

            if not union:
                continue

            jaccard_sim = len(intersection) / len(union)
            drift = 1.0 - jaccard_sim

            # identify entering and exiting concepts
            entering = list(curr_concepts - prev_concepts)[:5]
            exiting = list(prev_concepts - curr_concepts)[:5]

            # only record significant drift
            if drift >= self.drift_threshold:
                event = DriftEvent(
                    year=year,
                    drift_magnitude=drift,
                    concepts_entering=entering,
                    concepts_exiting=exiting,
                    is_novelty_jump=(drift >= self.NOVELTY_JUMP_THRESHOLD)
                )
                drift_events.append(event)

        result.add_trace(f"detected {len(drift_events)} drift events")
        return drift_events

    def _identify_phases(
        self,
        papers: List[Paper],
        drift_events: List[DriftEvent],
        result: AgentResult
    ) -> List[ResearchPhase]:
        """
        identify distinct research phases based on drift events.

        phases are periods of relative stability between drift events.
        """
        result.add_trace("identifying research phases")

        if not papers:
            return []

        years = sorted(set(p.year for p in papers if p.year))
        if not years:
            return []

        # determine phase boundaries from drift events
        boundaries = [years[0]]
        for event in drift_events:
            if event.is_novelty_jump:
                boundaries.append(event.year)
        boundaries.append(years[-1] + 1)

        # create phases
        phases = []
        for i in range(len(boundaries) - 1):
            start_year = boundaries[i]
            end_year = boundaries[i + 1] - 1

            # get papers in this phase
            phase_papers = [
                p for p in papers
                if p.year and start_year <= p.year <= end_year
            ]

            if len(phase_papers) < self.MIN_PAPERS_FOR_PHASE:
                continue

            # compute phase characteristics
            concept_counts: Dict[str, float] = defaultdict(float)
            collaborator_counts: Dict[str, int] = defaultdict(int)

            for p in phase_papers:
                for c in (p.concepts or [])[:5]:
                    concept_counts[c.get('name', '')] += c.get('score', 0.5)
                for author in (p.authors or []):
                    collaborator_counts[author] += 1

            # sort concepts and collaborators
            top_concepts = sorted(concept_counts.keys(), key=lambda x: -concept_counts[x])[:5]
            top_collaborators = sorted(
                [c for c in collaborator_counts.keys()],
                key=lambda x: -collaborator_counts[x]
            )[:5]

            # get top papers by citations
            phase_papers_sorted = sorted(phase_papers, key=lambda p: p.citation_count or 0, reverse=True)
            top_paper_ids = [p.id for p in phase_papers_sorted[:3]]

            phase = ResearchPhase(
                phase_id=len(phases),
                start_year=start_year,
                end_year=end_year,
                duration_years=end_year - start_year + 1,
                dominant_concepts=top_concepts,
                top_papers=top_paper_ids,
                paper_count=len(phase_papers),
                total_citations=sum(p.citation_count or 0 for p in phase_papers),
                key_collaborators=top_collaborators,
                label=top_concepts[0] if top_concepts else "unknown",
                is_formative=(i == 0),
                is_current=(i == len(boundaries) - 2)
            )
            phases.append(phase)

        result.add_trace(f"identified {len(phases)} phases")
        return phases

    def _classify_trajectory(
        self,
        analysis: TrajectoryAnalysis,
        result: AgentResult
    ) -> str:
        """
        classify trajectory type based on drift patterns.

        types:
        - focused: low drift, single main topic
        - explorer: high drift, many topics
        - bridger: moderate drift, connects fields
        - shifter: sudden major changes
        """
        if analysis.career_years < 5:
            return "early_career"

        drift_per_year = analysis.avg_drift_per_year
        novelty_jumps = sum(1 for d in analysis.drift_events if d.is_novelty_jump)
        phase_count = len(analysis.phases)

        if drift_per_year < 0.05 and phase_count <= 2:
            return "focused"
        elif novelty_jumps >= 3:
            return "shifter"
        elif drift_per_year > 0.15 and phase_count >= 3:
            return "explorer"
        elif 0.05 <= drift_per_year <= 0.15 and phase_count >= 2:
            return "bridger"
        else:
            return "steady"

    def _compute_stability(self, analysis: TrajectoryAnalysis) -> float:
        """compute stability score (0-1, higher = more stable)."""
        if analysis.career_years == 0:
            return 0.5

        # based on inverse of drift rate
        drift_rate = analysis.avg_drift_per_year
        stability = 1.0 / (1.0 + drift_rate * 5)  # scale factor

        return min(max(stability, 0.0), 1.0)

    def _find_direction(
        self,
        timeline: Dict[int, Dict[str, float]],
        result: AgentResult
    ) -> Tuple[List[str], List[str]]:
        """
        identify emerging and declining concepts.

        compares recent years to earlier years.
        """
        if len(timeline) < 4:
            return [], []

        years = sorted(timeline.keys())
        mid_point = len(years) // 2

        # aggregate early and late concepts
        early_concepts: Dict[str, float] = defaultdict(float)
        late_concepts: Dict[str, float] = defaultdict(float)

        for year in years[:mid_point]:
            for concept, weight in timeline[year].items():
                early_concepts[concept] += weight

        for year in years[mid_point:]:
            for concept, weight in timeline[year].items():
                late_concepts[concept] += weight

        # find emerging (in late, not in early or much higher)
        emerging = []
        for concept, late_weight in late_concepts.items():
            early_weight = early_concepts.get(concept, 0)
            if late_weight > early_weight * 1.5 or early_weight == 0:
                emerging.append((concept, late_weight - early_weight))

        # find declining (in early, not in late or much lower)
        declining = []
        for concept, early_weight in early_concepts.items():
            late_weight = late_concepts.get(concept, 0)
            if early_weight > late_weight * 1.5 or late_weight == 0:
                declining.append((concept, early_weight - late_weight))

        # sort by magnitude of change
        emerging.sort(key=lambda x: -x[1])
        declining.sort(key=lambda x: -x[1])

        return [c[0] for c in emerging[:5]], [c[0] for c in declining[:5]]

    def _generate_insights(
        self,
        analysis: TrajectoryAnalysis,
        corpus: AuthorCorpus,
        reference_concepts: Optional[Set[str]],
        result: AgentResult
    ):
        """generate human-readable insights about trajectory."""
        # career span
        analysis.add_insight(
            f"{analysis.author_name} has been publishing for {analysis.career_years} years "
            f"({analysis.career_start}-{analysis.career_end}), with {analysis.total_papers} papers."
        )

        # trajectory type
        type_descriptions = {
            "focused": "maintains a focused research program with consistent themes",
            "explorer": "explores diverse topics, frequently venturing into new areas",
            "bridger": "connects different research areas, bridging disciplines",
            "shifter": "has made significant pivots in research direction",
            "steady": "follows a steady research trajectory",
            "early_career": "is early in their career, trajectory still forming"
        }
        analysis.add_insight(
            f"Trajectory type: {analysis.trajectory_type} - {type_descriptions.get(analysis.trajectory_type, '')}"
        )

        # phases
        if analysis.phases:
            if len(analysis.phases) == 1:
                phase = analysis.phases[0]
                analysis.add_insight(
                    f"Single main research focus: {phase.dominant_concepts[0] if phase.dominant_concepts else 'unknown'}"
                )
            else:
                phase_summary = ", ".join(
                    f"{p.label} ({p.start_year}-{p.end_year})"
                    for p in analysis.phases
                )
                analysis.add_insight(f"Research phases: {phase_summary}")

        # drift events
        if analysis.drift_events:
            jumps = [d for d in analysis.drift_events if d.is_novelty_jump]
            if jumps:
                jump_years = ", ".join(str(d.year) for d in jumps)
                analysis.add_insight(f"Major research pivots in: {jump_years}")

        # current direction
        if analysis.emerging_concepts:
            analysis.add_insight(
                f"Currently expanding into: {', '.join(analysis.emerging_concepts[:3])}"
            )

        # reference concepts (if provided)
        if reference_concepts:
            # check how the author's work relates to reference concepts
            author_concepts = set()
            for phase in analysis.phases:
                author_concepts.update(c.lower() for c in phase.dominant_concepts)

            overlap = author_concepts & {c.lower() for c in reference_concepts}
            if overlap:
                analysis.add_insight(
                    f"Relevant expertise: {', '.join(list(overlap)[:3])}"
                )
            else:
                # check if emerging
                emerging_overlap = set(c.lower() for c in analysis.emerging_concepts) & {c.lower() for c in reference_concepts}
                if emerging_overlap:
                    analysis.add_insight(
                        f"Recently moving into relevant area: {', '.join(list(emerging_overlap)[:2])}"
                    )
