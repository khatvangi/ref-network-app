"""
trajectory layer - author trajectories across fields.
detects drift, novelty jumps, and author bridges between clusters.
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
import math
from collections import defaultdict

from ..core.models import (
    Author, AuthorYearProfile, Paper, Edge, EdgeType, Cluster
)
from ..core.config import TrajectoryConfig
from ..providers.base import PaperProvider
from ..graph.working_graph import WorkingGraph


@dataclass
class DriftEvent:
    """a significant drift in an author's research focus."""
    author_id: str
    year_from: int
    year_to: int
    drift_magnitude: float  # 0-1 (JSD)
    is_novelty_jump: bool
    entering_concepts: List[str]
    exiting_concepts: List[str]
    exemplar_work_ids: List[str]


@dataclass
class AuthorBridge:
    """a bridge between clusters mediated by an author's trajectory."""
    id: str = ""
    author_id: str = ""
    cluster_a_id: str = ""
    cluster_b_id: str = ""
    years: Tuple[int, int] = (0, 0)
    bridge_strength: float = 0.0
    exemplar_work_ids: List[str] = field(default_factory=list)
    drift_events: List[DriftEvent] = field(default_factory=list)


class TrajectoryLayer:
    """
    computes and analyzes author trajectories.
    detects field transitions and bridges between clusters.
    """

    def __init__(
        self,
        provider: PaperProvider,
        config: Optional[TrajectoryConfig] = None
    ):
        self.provider = provider
        self.config = config or TrajectoryConfig()

        # caches
        self._profiles: Dict[str, Dict[int, AuthorYearProfile]] = {}  # author_id -> {year -> profile}
        self._bridges: List[AuthorBridge] = []

    def build_author_profile(
        self,
        author: Author,
        year: int,
        papers: List[Paper]
    ) -> AuthorYearProfile:
        """
        build a year profile for an author based on their papers.
        """
        # filter papers for this year
        year_papers = [p for p in papers if p.year == year]

        if not year_papers:
            return AuthorYearProfile(
                author_id=author.id,
                year=year,
                work_ids=[],
                work_count=0
            )

        # limit papers
        year_papers = year_papers[:self.config.max_works_per_year_profile]

        # aggregate concepts
        concept_counts: Dict[str, float] = defaultdict(float)
        venues: List[str] = []
        total_citations = 0

        for paper in year_papers:
            total_citations += paper.citation_count or 0

            if paper.venue:
                venues.append(paper.venue)

            for concept in paper.concepts[:5]:
                name = concept.get('name', '')
                score = concept.get('score', 1.0)
                if name:
                    concept_counts[name] += score

        # normalize and sort concepts
        total_concept_weight = sum(concept_counts.values())
        top_concepts = []
        if total_concept_weight > 0:
            for name, weight in sorted(
                concept_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_concepts_per_profile]:
                top_concepts.append({
                    'name': name,
                    'weight': weight / total_concept_weight
                })

        # unique venues
        unique_venues = list(dict.fromkeys(venues))[:5]

        # average relevance
        relevances = [p.relevance_score for p in year_papers if p.relevance_score > 0]
        avg_relevance = sum(relevances) / len(relevances) if relevances else 0.0

        return AuthorYearProfile(
            author_id=author.id,
            year=year,
            work_ids=[p.id for p in year_papers],
            work_count=len(year_papers),
            top_concepts=top_concepts,
            venues=unique_venues,
            total_citations=total_citations,
            avg_relevance=avg_relevance
        )

    def compute_trajectory(
        self,
        author: Author,
        papers: List[Paper],
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> List[DriftEvent]:
        """
        compute trajectory for an author across years.
        returns list of drift events.
        """
        if not papers:
            return []

        # determine year range
        years = [p.year for p in papers if p.year]
        if not years:
            return []

        year_min = min_year or min(years)
        year_max = max_year or max(years)

        if year_max - year_min < self.config.min_years_for_trajectory:
            return []

        # build profiles for each year
        profiles: Dict[int, AuthorYearProfile] = {}
        for year in range(year_min, year_max + 1):
            profile = self.build_author_profile(author, year, papers)
            if profile.work_count > 0:
                profiles[year] = profile

        # cache profiles
        self._profiles[author.id] = profiles

        # detect drift events
        drift_events = []
        sorted_years = sorted(profiles.keys())

        for i in range(len(sorted_years) - 1):
            y1 = sorted_years[i]
            y2 = sorted_years[i + 1]

            # skip if gap is too large
            if y2 - y1 > self.config.year_window_size + 1:
                continue

            p1 = profiles[y1]
            p2 = profiles[y2]

            # compute drift
            drift = self._compute_drift(p1, p2)

            if drift > 0.2:  # minimum threshold for any drift
                entering, exiting = self._get_concept_changes(p1, p2)

                is_jump = drift >= self.config.drift_jump_threshold

                event = DriftEvent(
                    author_id=author.id,
                    year_from=y1,
                    year_to=y2,
                    drift_magnitude=drift,
                    is_novelty_jump=is_jump,
                    entering_concepts=entering,
                    exiting_concepts=exiting,
                    exemplar_work_ids=p2.work_ids[:3]
                )
                drift_events.append(event)

        # store on author
        author.drift_events = [
            {
                'year_from': e.year_from,
                'year_to': e.year_to,
                'magnitude': e.drift_magnitude,
                'is_jump': e.is_novelty_jump,
                'entering': e.entering_concepts,
                'exiting': e.exiting_concepts
            }
            for e in drift_events
        ]
        author.trajectory_computed = True

        return drift_events

    def detect_author_bridges(
        self,
        author: Author,
        graph: WorkingGraph,
        drift_events: List[DriftEvent]
    ) -> List[AuthorBridge]:
        """
        detect bridges between clusters based on author drift.
        """
        if not drift_events or not graph.clusters:
            return []

        bridges = []

        for event in drift_events:
            if not event.is_novelty_jump:
                continue

            # find which clusters the author connects
            # look at papers before and after drift
            before_papers = set()
            after_papers = set()

            for paper_id, paper in graph.papers.items():
                if author.openalex_id in paper.author_ids or \
                   author.s2_id in paper.author_ids:
                    if paper.year and paper.year <= event.year_from:
                        before_papers.add(paper_id)
                    elif paper.year and paper.year >= event.year_to:
                        after_papers.add(paper_id)

            # find cluster assignments
            before_clusters = set()
            after_clusters = set()

            for pid in before_papers:
                if pid in graph.node_cluster_map:
                    before_clusters.add(graph.node_cluster_map[pid])

            for pid in after_papers:
                if pid in graph.node_cluster_map:
                    after_clusters.add(graph.node_cluster_map[pid])

            # check for cross-cluster movement
            for c1 in before_clusters:
                for c2 in after_clusters:
                    if c1 != c2:
                        bridge = AuthorBridge(
                            author_id=author.id,
                            cluster_a_id=c1,
                            cluster_b_id=c2,
                            years=(event.year_from, event.year_to),
                            bridge_strength=event.drift_magnitude,
                            exemplar_work_ids=event.exemplar_work_ids,
                            drift_events=[event]
                        )
                        bridges.append(bridge)

        self._bridges.extend(bridges)
        return bridges

    def get_trajectory_summary(self, author: Author) -> Dict[str, Any]:
        """
        get summary of author's trajectory for UI.
        """
        profiles = self._profiles.get(author.id, {})
        if not profiles:
            return {"error": "no trajectory computed"}

        years = sorted(profiles.keys())

        # build timeline
        timeline = []
        for year in years:
            p = profiles[year]
            timeline.append({
                'year': year,
                'work_count': p.work_count,
                'top_concepts': [c['name'] for c in p.top_concepts[:3]],
                'avg_relevance': p.avg_relevance
            })

        # summarize drift events
        drift_summary = []
        for event in author.drift_events:
            drift_summary.append({
                'years': f"{event['year_from']}-{event['year_to']}",
                'magnitude': round(event['magnitude'], 2),
                'is_jump': event['is_jump'],
                'direction': f"entering: {', '.join(event['entering'][:2])}; exiting: {', '.join(event['exiting'][:2])}"
            })

        return {
            'author_id': author.id,
            'author_name': author.name,
            'year_range': f"{years[0]}-{years[-1]}" if years else "",
            'timeline': timeline,
            'drift_events': drift_summary,
            'total_novelty_jumps': sum(1 for e in author.drift_events if e.get('is_jump', False))
        }

    # private helpers

    def _compute_drift(
        self,
        p1: AuthorYearProfile,
        p2: AuthorYearProfile
    ) -> float:
        """
        compute Jensen-Shannon divergence between two profiles.
        returns value in [0, 1].
        """
        dist1 = p1.concept_distribution()
        dist2 = p2.concept_distribution()

        if not dist1 or not dist2:
            return 0.0

        # get all concepts
        all_concepts = set(dist1.keys()) | set(dist2.keys())

        if not all_concepts:
            return 0.0

        # build probability arrays
        eps = 1e-10
        p = []
        q = []

        for c in all_concepts:
            p.append(dist1.get(c, 0.0) + eps)
            q.append(dist2.get(c, 0.0) + eps)

        # normalize
        p_sum = sum(p)
        q_sum = sum(q)
        p = [x / p_sum for x in p]
        q = [x / q_sum for x in q]

        # compute JSD
        m = [(p[i] + q[i]) / 2 for i in range(len(p))]

        def kl_div(a, b):
            return sum(a[i] * math.log(a[i] / b[i]) for i in range(len(a)) if a[i] > 0)

        jsd = (kl_div(p, m) + kl_div(q, m)) / 2

        # normalize to [0, 1]
        # JSD is bounded by log(2) ≈ 0.693
        return min(jsd / 0.693, 1.0)

    def _get_concept_changes(
        self,
        p1: AuthorYearProfile,
        p2: AuthorYearProfile
    ) -> Tuple[List[str], List[str]]:
        """
        identify entering and exiting concepts between profiles.
        """
        dist1 = p1.concept_distribution()
        dist2 = p2.concept_distribution()

        entering = []
        exiting = []

        # entering: in p2 but not in p1 (or much higher)
        for c, w2 in dist2.items():
            w1 = dist1.get(c, 0.0)
            if w2 > w1 + 0.1:  # significant increase
                entering.append(c)

        # exiting: in p1 but not in p2 (or much lower)
        for c, w1 in dist1.items():
            w2 = dist2.get(c, 0.0)
            if w1 > w2 + 0.1:  # significant decrease
                exiting.append(c)

        # sort by magnitude of change
        entering.sort(key=lambda c: dist2.get(c, 0) - dist1.get(c, 0), reverse=True)
        exiting.sort(key=lambda c: dist1.get(c, 0) - dist2.get(c, 0), reverse=True)

        return entering[:5], exiting[:5]
