"""
trajectory analyzer agent - understand an author's research evolution.

answers key questions:
- what has this author been working on?
- how has their focus evolved over time?
- is this paper a new direction or lifelong interest?
- where are they heading?

input: AuthorCorpus (from CorpusFetcher)
output: TrajectoryAnalysis with phases, drift events, and insights

tuning notes (v2):
- filters generic/noisy concepts from OpenAlex
- uses sliding windows for drift detection (not year-to-year)
- merges short phases into coherent research periods
- uses cosine similarity for smoother drift measurement
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple

from .base import Agent, AgentResult, AgentStatus
from .corpus_fetcher import AuthorCorpus
from ..core.models import Paper


# generic concepts to filter out - these don't reveal research focus
GENERIC_CONCEPTS = {
    # very broad fields
    'biology', 'chemistry', 'physics', 'medicine', 'computer science',
    'mathematics', 'engineering', 'science', 'technology',
    # broad bio
    'biochemistry', 'molecular biology', 'cell biology', 'genetics',
    'bioinformatics', 'computational biology', 'biophysics',
    # broad chem
    'organic chemistry', 'inorganic chemistry', 'physical chemistry',
    # meta/noise
    'research', 'study', 'analysis', 'method', 'experiment',
    'data', 'model', 'algorithm', 'system', 'process',
    # geographic/institutional noise
    'south carolina', 'north carolina', 'california', 'new york',
    'family medicine', 'continuing education', 'medical education',
    # artifacts from OpenAlex misclassification
    'context (archaeology)', 'crew', 'real estate', 'nursing',
    'paleontology', 'trojan', 'zoology', 'archaeology',
    'path (computing)', 'reliability (semiconductor)',
    'substitution (logic)', 'interpolation (computer graphics)',
    'cleavage (geology)', 'denaturation (fissile materials)',
    'core (optical fiber)', 'phase (matter)', 'polar',
    'sketch', 'cuff', 'transcription (linguistics)',
    'groundwater', 'ethylene glycol', 'digital signal processing',
    'signal processing', 'mean absolute percentage error',
    'yield (engineering)', 'resource (disambiguation)',
    'overfitting', 'quality management', 'intensive care medicine',
}

# concepts that indicate methodology papers (not topic papers)
METHODOLOGY_CONCEPTS = {
    'machine learning', 'deep learning', 'neural network',
    'algorithm', 'software', 'database', 'web service',
    'python', 'r programming', 'statistics',
}


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
class EmploymentPeriod:
    """employment/affiliation period from ORCID."""
    organization: str
    role: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None  # None = current
    department: Optional[str] = None


@dataclass
class TrajectoryAnalysis:
    """
    complete trajectory analysis for an author.

    tells the story of their research career.
    """
    # identity
    author_id: str
    author_name: str
    orcid: Optional[str] = None

    # career span
    career_start: int = 0
    career_end: int = 0
    career_years: int = 0
    total_papers: int = 0

    # employment history (from ORCID)
    employment_history: List[EmploymentPeriod] = field(default_factory=list)

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

    # core concepts (appear throughout career)
    core_concepts: List[str] = field(default_factory=list)

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

    v2 improvements:
    - filters generic concepts (Biology, Chemistry, etc.)
    - uses 3-year sliding windows for smoother drift detection
    - merges short phases (< 3 years)
    - uses cosine similarity instead of jaccard distance
    - can integrate ORCID employment history

    usage:
        analyzer = TrajectoryAnalyzer()
        result = analyzer.run(corpus)

        if result.ok:
            trajectory = result.data
            print(f"Trajectory type: {trajectory.trajectory_type}")
            for phase in trajectory.phases:
                print(f"  {phase.start_year}-{phase.end_year}: {phase.label}")
    """

    # tuned parameters
    MIN_PAPERS_FOR_PHASE = 5  # increased from 3
    MIN_PHASE_YEARS = 3  # minimum years for a phase
    DRIFT_THRESHOLD = 0.4  # increased from 0.3 (less sensitive)
    NOVELTY_JUMP_THRESHOLD = 0.6  # increased from 0.5
    WINDOW_SIZE = 3  # years to group for concept analysis

    def __init__(
        self,
        window_size: int = 3,
        drift_threshold: float = 0.4,
        min_phase_years: int = 3,
        orcid_provider=None,  # optional ORCID provider
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_phase_years = min_phase_years
        self.orcid_provider = orcid_provider

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
        if len(papers_with_years) < 5:
            result.status = AgentStatus.PARTIAL
            result.add_warning(f"only {len(papers_with_years)} papers with years, analysis may be limited")

        # create analysis object
        years = [p.year for p in papers_with_years]
        analysis = TrajectoryAnalysis(
            author_id=corpus.author_id,
            author_name=corpus.name,
            orcid=corpus.orcid,
            career_start=min(years) if years else 0,
            career_end=max(years) if years else 0,
            career_years=(max(years) - min(years) + 1) if years else 0,
            total_papers=len(corpus.papers)
        )

        result.add_trace(f"career span: {analysis.career_start}-{analysis.career_end} ({analysis.career_years} years)")

        # step 0: fetch ORCID employment history if available
        if self.orcid_provider and corpus.orcid:
            employment = self._fetch_employment_history(corpus.orcid, result)
            analysis.employment_history = employment

        # step 1: build concept timeline with filtering
        concept_timeline = self._build_concept_timeline(papers_with_years, result)

        # step 2: find core concepts (appear in >50% of years)
        analysis.core_concepts = self._find_core_concepts(concept_timeline, result)

        # step 3: detect drift events using windowed comparison
        drift_events = self._detect_drift_events_windowed(concept_timeline, result)
        analysis.drift_events = drift_events

        # compute drift statistics
        if drift_events:
            analysis.total_drift = sum(d.drift_magnitude for d in drift_events)
            analysis.avg_drift_per_year = analysis.total_drift / max(analysis.career_years, 1)

        # step 4: identify phases (based on drift events) and merge short ones
        phases = self._identify_phases(papers_with_years, drift_events, result)
        phases = self._merge_short_phases(phases, papers_with_years, result)
        analysis.phases = phases
        if phases:
            analysis.current_phase = phases[-1]

        # step 5: determine trajectory type
        analysis.trajectory_type = self._classify_trajectory(analysis, result)
        analysis.stability_score = self._compute_stability(analysis)

        # step 6: find current direction
        emerging, declining = self._find_direction(concept_timeline, result)
        analysis.emerging_concepts = emerging
        analysis.declining_concepts = declining

        # step 7: generate insights
        self._generate_insights(analysis, corpus, reference_concepts, result)

        result.data = analysis
        result.add_trace(f"analysis complete: {len(phases)} phases, {len(drift_events)} drift events")

        return result

    def _filter_concept(self, concept_name: str) -> bool:
        """return True if concept should be kept (not generic)."""
        name_lower = concept_name.lower().strip()

        # filter generic concepts
        if name_lower in GENERIC_CONCEPTS:
            return False

        # filter very short names (usually noise)
        if len(name_lower) < 3:
            return False

        return True

    def _build_concept_timeline(
        self,
        papers: List[Paper],
        result: AgentResult
    ) -> Dict[int, Dict[str, float]]:
        """
        build year -> {concept: weight} mapping with filtering.

        filters out generic concepts and normalizes weights.
        """
        result.add_trace("building concept timeline with filtering")

        timeline: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        filtered_count = 0

        for paper in papers:
            year = paper.year
            if not year:
                continue

            for concept in (paper.concepts or [])[:10]:
                name = concept.get('name', '')
                score = concept.get('score', 0.5)

                if not name:
                    continue

                # filter generic concepts
                if not self._filter_concept(name):
                    filtered_count += 1
                    continue

                # use original case for display but lowercase for dedup
                name_key = name.lower().strip()
                timeline[year][name_key] += score

        # normalize weights per year
        for year in timeline:
            total = sum(timeline[year].values())
            if total > 0:
                for concept in timeline[year]:
                    timeline[year][concept] /= total

        result.add_trace(f"timeline built: {len(timeline)} years, filtered {filtered_count} generic concepts")
        return dict(timeline)

    def _find_core_concepts(
        self,
        timeline: Dict[int, Dict[str, float]],
        result: AgentResult
    ) -> List[str]:
        """find concepts that appear in >50% of active years."""
        if not timeline:
            return []

        concept_years: Dict[str, int] = defaultdict(int)
        total_years = len(timeline)

        for year_concepts in timeline.values():
            for concept in year_concepts:
                concept_years[concept] += 1

        # concepts in >50% of years
        threshold = total_years * 0.5
        core = [c for c, count in concept_years.items() if count >= threshold]

        # sort by frequency
        core.sort(key=lambda c: -concept_years[c])

        result.add_trace(f"found {len(core)} core concepts (appear in >50% of years)")
        return core[:10]

    def _detect_drift_events_windowed(
        self,
        timeline: Dict[int, Dict[str, float]],
        result: AgentResult
    ) -> List[DriftEvent]:
        """
        detect drift using sliding windows instead of year-to-year.

        compares concept distributions between adjacent windows
        using cosine similarity for smoother measurement.
        """
        result.add_trace(f"detecting drift events (window={self.window_size})")

        if len(timeline) < self.window_size * 2:
            result.add_trace("not enough years for windowed analysis")
            return []

        years = sorted(timeline.keys())
        drift_events = []

        # slide window through years
        for i in range(self.window_size, len(years)):
            # window 1: years[i-window_size : i]
            # window 2: years[i : i+window_size] or remaining

            window1_years = years[max(0, i - self.window_size):i]
            window2_years = years[i:min(len(years), i + self.window_size)]

            if len(window2_years) < 2:
                continue

            # aggregate concepts for each window
            window1_concepts = self._aggregate_window(timeline, window1_years)
            window2_concepts = self._aggregate_window(timeline, window2_years)

            # compute cosine distance (1 - similarity)
            drift = 1.0 - self._cosine_similarity(window1_concepts, window2_concepts)

            # only record significant drift
            if drift >= self.drift_threshold:
                entering = [c for c in window2_concepts if c not in window1_concepts][:5]
                exiting = [c for c in window1_concepts if c not in window2_concepts][:5]

                event = DriftEvent(
                    year=years[i],
                    drift_magnitude=drift,
                    concepts_entering=entering,
                    concepts_exiting=exiting,
                    is_novelty_jump=(drift >= self.NOVELTY_JUMP_THRESHOLD)
                )
                drift_events.append(event)

        result.add_trace(f"detected {len(drift_events)} drift events")
        return drift_events

    def _aggregate_window(
        self,
        timeline: Dict[int, Dict[str, float]],
        years: List[int]
    ) -> Dict[str, float]:
        """aggregate concept weights across multiple years."""
        aggregated: Dict[str, float] = defaultdict(float)

        for year in years:
            if year in timeline:
                for concept, weight in timeline[year].items():
                    aggregated[concept] += weight

        # normalize
        total = sum(aggregated.values())
        if total > 0:
            for concept in aggregated:
                aggregated[concept] /= total

        return dict(aggregated)

    def _cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float]
    ) -> float:
        """compute cosine similarity between two concept vectors."""
        if not vec1 or not vec2:
            return 0.0

        # get all concepts
        all_concepts = set(vec1.keys()) | set(vec2.keys())

        # compute dot product and magnitudes
        dot_product = 0.0
        mag1 = 0.0
        mag2 = 0.0

        for concept in all_concepts:
            v1 = vec1.get(concept, 0.0)
            v2 = vec2.get(concept, 0.0)
            dot_product += v1 * v2
            mag1 += v1 * v1
            mag2 += v2 * v2

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (math.sqrt(mag1) * math.sqrt(mag2))

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

        # determine phase boundaries from novelty jumps only
        boundaries = [years[0]]
        for event in drift_events:
            if event.is_novelty_jump:
                boundaries.append(event.year)
        boundaries.append(years[-1] + 1)

        # remove duplicate boundaries
        boundaries = sorted(set(boundaries))

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

            # compute phase characteristics with filtered concepts
            concept_counts: Dict[str, float] = defaultdict(float)
            collaborator_counts: Dict[str, int] = defaultdict(int)

            for p in phase_papers:
                for c in (p.concepts or [])[:7]:
                    name = c.get('name', '')
                    if name and self._filter_concept(name):
                        concept_counts[name.lower()] += c.get('score', 0.5)
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

    def _merge_short_phases(
        self,
        phases: List[ResearchPhase],
        papers: List[Paper],
        result: AgentResult
    ) -> List[ResearchPhase]:
        """merge phases shorter than min_phase_years."""
        if len(phases) <= 1:
            return phases

        result.add_trace(f"merging short phases (min {self.min_phase_years} years)")

        merged = []
        i = 0

        while i < len(phases):
            current = phases[i]

            # if too short and not first/last, try to merge
            if current.duration_years < self.min_phase_years:
                # merge with next phase if possible
                if i + 1 < len(phases):
                    next_phase = phases[i + 1]
                    merged_phase = self._combine_phases(current, next_phase, papers)
                    phases[i + 1] = merged_phase
                    i += 1
                    continue
                # else merge with previous if exists
                elif merged:
                    prev = merged.pop()
                    merged_phase = self._combine_phases(prev, current, papers)
                    merged.append(merged_phase)
                    i += 1
                    continue

            merged.append(current)
            i += 1

        # renumber phases
        for idx, phase in enumerate(merged):
            phase.phase_id = idx
            phase.is_formative = (idx == 0)
            phase.is_current = (idx == len(merged) - 1)

        result.add_trace(f"after merging: {len(merged)} phases")
        return merged

    def _combine_phases(
        self,
        phase1: ResearchPhase,
        phase2: ResearchPhase,
        papers: List[Paper]
    ) -> ResearchPhase:
        """combine two phases into one."""
        start_year = min(phase1.start_year, phase2.start_year)
        end_year = max(phase1.end_year, phase2.end_year)

        # get all papers in combined range
        phase_papers = [
            p for p in papers
            if p.year and start_year <= p.year <= end_year
        ]

        # recompute concepts
        concept_counts: Dict[str, float] = defaultdict(float)
        collaborator_counts: Dict[str, int] = defaultdict(int)

        for p in phase_papers:
            for c in (p.concepts or [])[:7]:
                name = c.get('name', '')
                if name and self._filter_concept(name):
                    concept_counts[name.lower()] += c.get('score', 0.5)
            for author in (p.authors or []):
                collaborator_counts[author] += 1

        top_concepts = sorted(concept_counts.keys(), key=lambda x: -concept_counts[x])[:5]
        top_collaborators = sorted(
            collaborator_counts.keys(),
            key=lambda x: -collaborator_counts[x]
        )[:5]

        return ResearchPhase(
            phase_id=0,  # will be renumbered
            start_year=start_year,
            end_year=end_year,
            duration_years=end_year - start_year + 1,
            dominant_concepts=top_concepts,
            top_papers=phase1.top_papers + phase2.top_papers,
            paper_count=len(phase_papers),
            total_citations=sum(p.citation_count or 0 for p in phase_papers),
            key_collaborators=top_collaborators,
            label=top_concepts[0] if top_concepts else "unknown"
        )

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
        - steady: consistent gradual evolution
        """
        if analysis.career_years < 5:
            return "early_career"

        drift_per_year = analysis.avg_drift_per_year
        novelty_jumps = sum(1 for d in analysis.drift_events if d.is_novelty_jump)
        phase_count = len(analysis.phases)

        # core concepts = focused researcher
        core_count = len(analysis.core_concepts)

        # classification logic
        if core_count >= 3 and drift_per_year < 0.1:
            return "focused"
        elif novelty_jumps >= 2 and phase_count >= 3:
            return "shifter"
        elif drift_per_year > 0.2 and phase_count >= 3:
            return "explorer"
        elif 0.05 <= drift_per_year <= 0.15 and core_count >= 2:
            return "bridger"
        else:
            return "steady"

    def _compute_stability(self, analysis: TrajectoryAnalysis) -> float:
        """compute stability score (0-1, higher = more stable)."""
        if analysis.career_years == 0:
            return 0.5

        # based on inverse of drift rate and core concept count
        drift_rate = analysis.avg_drift_per_year
        core_bonus = min(len(analysis.core_concepts) * 0.1, 0.3)

        stability = 1.0 / (1.0 + drift_rate * 3) + core_bonus

        return min(max(stability, 0.0), 1.0)

    def _find_direction(
        self,
        timeline: Dict[int, Dict[str, float]],
        result: AgentResult
    ) -> Tuple[List[str], List[str]]:
        """
        identify emerging and declining concepts.

        compares recent 5 years to earlier career.
        """
        if len(timeline) < 6:
            return [], []

        years = sorted(timeline.keys())

        # recent 5 years vs earlier
        recent_years = years[-5:]
        early_years = years[:-5]

        # aggregate concepts
        recent_concepts = self._aggregate_window(timeline, recent_years)
        early_concepts = self._aggregate_window(timeline, early_years)

        # find emerging (higher in recent)
        emerging = []
        for concept, recent_weight in recent_concepts.items():
            early_weight = early_concepts.get(concept, 0)
            if recent_weight > early_weight * 1.5 or early_weight == 0:
                emerging.append((concept, recent_weight - early_weight))

        # find declining (higher in early)
        declining = []
        for concept, early_weight in early_concepts.items():
            recent_weight = recent_concepts.get(concept, 0)
            if early_weight > recent_weight * 1.5 or recent_weight == 0:
                declining.append((concept, early_weight - recent_weight))

        # sort by magnitude of change
        emerging.sort(key=lambda x: -x[1])
        declining.sort(key=lambda x: -x[1])

        return [c[0] for c in emerging[:5]], [c[0] for c in declining[:5]]

    def _fetch_employment_history(
        self,
        orcid: str,
        result: AgentResult
    ) -> List[EmploymentPeriod]:
        """fetch employment history from ORCID."""
        if not self.orcid_provider:
            return []

        result.add_trace(f"fetching ORCID employment for {orcid}")

        try:
            author_info = self.orcid_provider.get_author_by_orcid(orcid)
            if not author_info:
                return []

            # ORCID affiliations don't have full details in basic API
            # but we can create periods from available data
            periods = []
            for affil in (author_info.affiliations or []):
                periods.append(EmploymentPeriod(
                    organization=affil
                ))

            result.add_trace(f"found {len(periods)} employment periods")
            return periods

        except Exception as e:
            result.add_warning(f"could not fetch ORCID: {e}")
            return []

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

        # core concepts
        if analysis.core_concepts:
            analysis.add_insight(
                f"Core research themes: {', '.join(analysis.core_concepts[:3])}"
            )

        # trajectory type
        type_descriptions = {
            "focused": "maintains a focused research program with consistent themes",
            "explorer": "explores diverse topics, frequently venturing into new areas",
            "bridger": "connects different research areas, bridging disciplines",
            "shifter": "has made significant pivots in research direction",
            "steady": "follows a steady research trajectory with gradual evolution",
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
            author_concepts = set(analysis.core_concepts)
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
