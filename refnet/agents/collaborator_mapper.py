"""
collaborator mapper agent - map an author's collaboration network.

answers key questions:
- who does this author work with?
- are collaborations long-term or one-off?
- what expertise do collaborators bring?
- are there cross-disciplinary bridges?

input: AuthorCorpus (from CorpusFetcher)
output: CollaborationNetwork with collaborators, clusters, and insights
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple

from .base import Agent, AgentResult, AgentStatus
from .corpus_fetcher import AuthorCorpus
from ..core.models import Paper


@dataclass
class Collaborator:
    """
    a collaborator in an author's network.

    collaborators are ranked by:
    - frequency (how many papers together)
    - recency (recent collaborations matter more)
    - duration (long-term vs one-off)
    """
    name: str
    author_id: Optional[str] = None  # OpenAlex/ORCID if known

    # collaboration metrics
    paper_count: int = 0
    first_year: Optional[int] = None
    last_year: Optional[int] = None
    collaboration_years: int = 0  # span of collaboration

    # what they work on together
    shared_concepts: List[str] = field(default_factory=list)
    shared_venues: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)

    # collaboration type
    is_long_term: bool = False  # >3 years of collaboration
    is_frequent: bool = False   # >5 papers together
    is_recent: bool = False     # collaborated in last 3 years

    # scores
    strength_score: float = 0.0  # overall collaboration strength

    def __post_init__(self):
        if self.first_year and self.last_year:
            self.collaboration_years = self.last_year - self.first_year + 1
            self.is_long_term = self.collaboration_years >= 3
        self.is_frequent = self.paper_count >= 5


@dataclass
class CollaboratorCluster:
    """
    a cluster of related collaborators.

    collaborators often cluster by:
    - lab/institution
    - research topic
    - time period
    """
    cluster_id: int
    name: str  # auto-generated from shared concepts

    collaborator_names: List[str] = field(default_factory=list)
    shared_concepts: List[str] = field(default_factory=list)
    year_range: Tuple[int, int] = (0, 0)
    total_papers: int = 0

    # cluster type
    cluster_type: str = "topic"  # "topic", "institution", "time_period"


@dataclass
class CollaborationNetwork:
    """
    complete collaboration network for an author.
    """
    # identity
    author_id: str
    author_name: str

    # network stats
    total_collaborators: int = 0
    total_collaborative_papers: int = 0
    solo_papers: int = 0
    avg_authors_per_paper: float = 0.0

    # collaborators (sorted by strength)
    collaborators: List[Collaborator] = field(default_factory=list)

    # clusters
    clusters: List[CollaboratorCluster] = field(default_factory=list)

    # key collaborators
    top_collaborators: List[str] = field(default_factory=list)  # by frequency
    long_term_collaborators: List[str] = field(default_factory=list)
    recent_collaborators: List[str] = field(default_factory=list)

    # collaboration patterns
    collaboration_style: str = "unknown"  # "solo", "small_team", "large_network", "stable_group"
    network_density: float = 0.0  # how interconnected

    # insights
    insights: List[str] = field(default_factory=list)

    def add_insight(self, insight: str):
        self.insights.append(insight)


class CollaboratorMapper(Agent):
    """
    map an author's collaboration network.

    takes a corpus and identifies:
    - who they collaborate with
    - collaboration patterns (long-term, frequent, recent)
    - collaborator clusters (by topic or time)
    - collaboration style (solo vs team)

    usage:
        mapper = CollaboratorMapper()
        result = mapper.run(corpus)

        if result.ok:
            network = result.data
            print(f"Top collaborators: {network.top_collaborators[:5]}")
    """

    # thresholds
    LONG_TERM_YEARS = 3
    FREQUENT_PAPERS = 5
    RECENT_YEARS = 3
    MIN_CLUSTER_SIZE = 2

    def __init__(
        self,
        recent_years: int = 3,
        min_papers_for_top: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.recent_years = recent_years
        self.min_papers_for_top = min_papers_for_top

    @property
    def name(self) -> str:
        return "CollaboratorMapper"

    def execute(
        self,
        corpus: AuthorCorpus,
        current_year: Optional[int] = None
    ) -> AgentResult[CollaborationNetwork]:
        """
        map collaboration network from author corpus.

        args:
            corpus: AuthorCorpus from CorpusFetcher
            current_year: year to use for "recent" calculations (default: max year in corpus)

        returns:
            AgentResult with CollaborationNetwork
        """
        result = AgentResult[CollaborationNetwork](status=AgentStatus.SUCCESS)
        result.add_trace(f"mapping collaborations for: {corpus.name}")

        if not corpus.papers:
            result.status = AgentStatus.FAILED
            result.add_error("NO_PAPERS", "corpus has no papers to analyze")
            return result

        # determine current year
        if current_year is None:
            years = [p.year for p in corpus.papers if p.year]
            current_year = max(years) if years else 2025

        # create network object
        network = CollaborationNetwork(
            author_id=corpus.author_id,
            author_name=corpus.name
        )

        # step 1: extract all collaborators
        collaborators = self._extract_collaborators(corpus, current_year, result)
        network.collaborators = collaborators
        network.total_collaborators = len(collaborators)

        # step 2: compute network stats
        self._compute_network_stats(corpus, network, result)

        # step 3: identify key collaborators
        self._identify_key_collaborators(network, result)

        # step 4: cluster collaborators by topic
        clusters = self._cluster_collaborators(collaborators, corpus, result)
        network.clusters = clusters

        # step 5: determine collaboration style
        network.collaboration_style = self._classify_collaboration_style(network, result)

        # step 6: generate insights
        self._generate_insights(network, corpus, result)

        result.data = network
        result.add_trace(f"mapped {len(collaborators)} collaborators, {len(clusters)} clusters")

        return result

    def _extract_collaborators(
        self,
        corpus: AuthorCorpus,
        current_year: int,
        result: AgentResult
    ) -> List[Collaborator]:
        """extract all collaborators from corpus papers."""
        result.add_trace("extracting collaborators")

        # aggregate collaborator data
        collab_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'paper_count': 0,
            'years': [],
            'concepts': defaultdict(float),
            'venues': defaultdict(int),
            'paper_ids': [],
            'author_ids': set()
        })

        author_name_lower = corpus.name.lower()

        for paper in corpus.papers:
            if not paper.authors:
                continue

            # get concepts for this paper
            paper_concepts = {
                c.get('name', '').lower(): c.get('score', 0.5)
                for c in (paper.concepts or [])[:5]
            }

            for i, author in enumerate(paper.authors):
                # skip the main author
                if author.lower() == author_name_lower:
                    continue
                # also skip if name is very similar (middle initial differences)
                if self._names_similar(author, corpus.name):
                    continue

                # get author id if available
                author_id = None
                if paper.author_ids and i < len(paper.author_ids):
                    author_id = paper.author_ids[i]

                # aggregate
                data = collab_data[author]
                data['paper_count'] += 1
                if paper.year:
                    data['years'].append(paper.year)
                for concept, score in paper_concepts.items():
                    data['concepts'][concept] += score
                if paper.venue:
                    data['venues'][paper.venue] += 1
                data['paper_ids'].append(paper.id)
                if author_id:
                    data['author_ids'].add(author_id)

        # convert to Collaborator objects
        collaborators = []
        for name, data in collab_data.items():
            years = data['years']
            first_year = min(years) if years else None
            last_year = max(years) if years else None

            # top concepts
            sorted_concepts = sorted(
                data['concepts'].items(),
                key=lambda x: -x[1]
            )
            top_concepts = [c[0] for c in sorted_concepts[:5]]

            # top venues
            sorted_venues = sorted(
                data['venues'].items(),
                key=lambda x: -x[1]
            )
            top_venues = [v[0] for v in sorted_venues[:3]]

            # get author id (most common if multiple)
            author_id = None
            if data['author_ids']:
                author_id = list(data['author_ids'])[0]

            collab = Collaborator(
                name=name,
                author_id=author_id,
                paper_count=data['paper_count'],
                first_year=first_year,
                last_year=last_year,
                shared_concepts=top_concepts,
                shared_venues=top_venues,
                paper_ids=data['paper_ids']
            )

            # set flags
            if first_year and last_year:
                collab.collaboration_years = last_year - first_year + 1
                collab.is_long_term = collab.collaboration_years >= self.LONG_TERM_YEARS

            collab.is_frequent = collab.paper_count >= self.FREQUENT_PAPERS

            if last_year:
                collab.is_recent = (current_year - last_year) <= self.recent_years

            # compute strength score
            collab.strength_score = self._compute_strength(collab, current_year)

            collaborators.append(collab)

        # sort by strength
        collaborators.sort(key=lambda c: -c.strength_score)

        result.add_trace(f"found {len(collaborators)} collaborators")
        return collaborators

    def _names_similar(self, name1: str, name2: str) -> bool:
        """check if two names are similar (same person, different formatting)."""
        # normalize
        n1 = name1.lower().replace('.', '').replace(',', '').split()
        n2 = name2.lower().replace('.', '').replace(',', '').split()

        if not n1 or not n2:
            return False

        # check last names match
        if n1[-1] != n2[-1]:
            return False

        # check first name/initial matches
        if n1[0] == n2[0]:
            return True
        if n1[0][0] == n2[0][0]:  # same first initial
            return True

        return False

    def _compute_strength(self, collab: Collaborator, current_year: int) -> float:
        """compute collaboration strength score."""
        score = 0.0

        # frequency component (log scale)
        import math
        score += math.log(1 + collab.paper_count) * 2

        # duration component
        if collab.collaboration_years:
            score += math.log(1 + collab.collaboration_years)

        # recency component
        if collab.last_year:
            years_ago = current_year - collab.last_year
            recency_factor = 1.0 / (1 + years_ago * 0.2)
            score *= (0.5 + 0.5 * recency_factor)

        return score

    def _compute_network_stats(
        self,
        corpus: AuthorCorpus,
        network: CollaborationNetwork,
        result: AgentResult
    ):
        """compute overall network statistics."""
        result.add_trace("computing network stats")

        solo_count = 0
        total_authors = 0
        collaborative_count = 0

        for paper in corpus.papers:
            if not paper.authors:
                continue

            num_authors = len(paper.authors)
            total_authors += num_authors

            if num_authors <= 1:
                solo_count += 1
            else:
                collaborative_count += 1

        network.solo_papers = solo_count
        network.total_collaborative_papers = collaborative_count

        if corpus.papers:
            network.avg_authors_per_paper = total_authors / len(corpus.papers)

    def _identify_key_collaborators(
        self,
        network: CollaborationNetwork,
        result: AgentResult
    ):
        """identify top, long-term, and recent collaborators."""
        result.add_trace("identifying key collaborators")

        # top by frequency (already sorted by strength)
        network.top_collaborators = [
            c.name for c in network.collaborators[:10]
            if c.paper_count >= self.min_papers_for_top
        ]

        # long-term
        network.long_term_collaborators = [
            c.name for c in network.collaborators
            if c.is_long_term
        ][:10]

        # recent
        network.recent_collaborators = [
            c.name for c in network.collaborators
            if c.is_recent
        ][:10]

    def _cluster_collaborators(
        self,
        collaborators: List[Collaborator],
        corpus: AuthorCorpus,
        result: AgentResult
    ) -> List[CollaboratorCluster]:
        """cluster collaborators by shared concepts."""
        result.add_trace("clustering collaborators")

        if len(collaborators) < 3:
            return []

        # simple clustering by shared concepts
        # group collaborators who share top concepts

        concept_to_collabs: Dict[str, List[str]] = defaultdict(list)

        for collab in collaborators:
            for concept in collab.shared_concepts[:3]:
                concept_to_collabs[concept].append(collab.name)

        # find clusters (concepts with multiple collaborators)
        clusters = []
        used_collabs: Set[str] = set()

        for concept, collab_names in sorted(
            concept_to_collabs.items(),
            key=lambda x: -len(x[1])
        ):
            # filter to unused collaborators
            available = [n for n in collab_names if n not in used_collabs]

            if len(available) >= self.MIN_CLUSTER_SIZE:
                cluster = CollaboratorCluster(
                    cluster_id=len(clusters),
                    name=concept,
                    collaborator_names=available[:10],
                    shared_concepts=[concept],
                    cluster_type="topic"
                )

                # compute year range and paper count
                years = []
                paper_count = 0
                for name in available:
                    collab = next((c for c in collaborators if c.name == name), None)
                    if collab:
                        if collab.first_year:
                            years.append(collab.first_year)
                        if collab.last_year:
                            years.append(collab.last_year)
                        paper_count += collab.paper_count

                if years:
                    cluster.year_range = (min(years), max(years))
                cluster.total_papers = paper_count

                clusters.append(cluster)
                used_collabs.update(available)

                if len(clusters) >= 5:  # limit clusters
                    break

        result.add_trace(f"found {len(clusters)} clusters")
        return clusters

    def _classify_collaboration_style(
        self,
        network: CollaborationNetwork,
        result: AgentResult
    ) -> str:
        """classify the author's collaboration style."""
        total_papers = network.total_collaborative_papers + network.solo_papers

        if total_papers == 0:
            return "unknown"

        solo_ratio = network.solo_papers / total_papers

        # solo author (>50% solo)
        if solo_ratio > 0.5:
            return "solo"

        # check collaboration patterns
        long_term_count = len(network.long_term_collaborators)
        total_collabs = network.total_collaborators

        if total_collabs < 10:
            return "small_team"

        # stable group (many long-term)
        if long_term_count >= 5 and long_term_count / total_collabs > 0.3:
            return "stable_group"

        # large network
        if total_collabs > 50:
            return "large_network"

        return "collaborative"

    def _generate_insights(
        self,
        network: CollaborationNetwork,
        corpus: AuthorCorpus,
        result: AgentResult
    ):
        """generate human-readable insights."""
        # collaboration style
        style_desc = {
            "solo": "primarily works independently",
            "small_team": "works with a small, focused team",
            "stable_group": "maintains long-term collaborations with a stable group",
            "large_network": "has an extensive collaboration network",
            "collaborative": "regularly collaborates with others"
        }

        network.add_insight(
            f"{network.author_name} {style_desc.get(network.collaboration_style, 'collaborates with others')} "
            f"({network.total_collaborators} collaborators across {network.total_collaborative_papers} papers)."
        )

        # top collaborator
        if network.top_collaborators:
            top = network.top_collaborators[0]
            top_collab = next((c for c in network.collaborators if c.name == top), None)
            if top_collab:
                network.add_insight(
                    f"Most frequent collaborator: {top} ({top_collab.paper_count} papers, "
                    f"{top_collab.collaboration_years} years)."
                )

        # long-term collaborations
        if len(network.long_term_collaborators) > 0:
            network.add_insight(
                f"Long-term collaborators (3+ years): {', '.join(network.long_term_collaborators[:5])}"
            )

        # topic clusters
        if network.clusters:
            cluster_summary = ", ".join(
                f"{c.name} ({len(c.collaborator_names)} people)"
                for c in network.clusters[:3]
            )
            network.add_insight(f"Collaboration clusters by topic: {cluster_summary}")

        # recent activity
        if network.recent_collaborators:
            recent_new = [
                n for n in network.recent_collaborators
                if n not in network.long_term_collaborators
            ]
            if recent_new:
                network.add_insight(
                    f"New recent collaborators: {', '.join(recent_new[:3])}"
                )
