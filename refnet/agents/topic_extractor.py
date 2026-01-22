"""
topic extractor agent - extract themes from a collection of papers.

answers key questions:
- what are the main topics in this collection?
- which topics are emerging vs established?
- what's the topic hierarchy?
- how do topics relate to each other?

input: List of Papers (from any source)
output: TopicAnalysis with themes, trends, and hierarchy
"""

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple

from .base import Agent, AgentResult, AgentStatus
from ..core.models import Paper


# generic concepts to filter (same as trajectory_analyzer)
GENERIC_CONCEPTS = {
    'biology', 'chemistry', 'physics', 'medicine', 'computer science',
    'mathematics', 'engineering', 'science', 'technology',
    'biochemistry', 'molecular biology', 'cell biology', 'genetics',
    'bioinformatics', 'computational biology', 'biophysics',
    'organic chemistry', 'inorganic chemistry', 'physical chemistry',
    'research', 'study', 'analysis', 'method', 'experiment',
    'data', 'model', 'algorithm', 'system', 'process',
}

# stopwords for title/abstract extraction
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
    'we', 'our', 'us', 'i', 'my', 'me', 'you', 'your', 'he', 'she',
    'his', 'her', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    'then', 'once', 'new', 'using', 'based', 'via', 'between', 'after',
    'before', 'during', 'into', 'through', 'about', 'above', 'below',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
    'however', 'although', 'thus', 'therefore', 'hence', 'moreover',
}


@dataclass
class Topic:
    """
    a topic/theme extracted from papers.

    topics have:
    - a canonical name
    - related terms (synonyms, variants)
    - metrics (frequency, citation weight, trend)
    """
    name: str
    canonical_name: str  # normalized version

    # frequency metrics
    paper_count: int = 0
    total_weight: float = 0.0  # weighted by citations

    # related terms
    related_terms: List[str] = field(default_factory=list)

    # temporal info
    first_year: Optional[int] = None
    last_year: Optional[int] = None
    year_distribution: Dict[int, int] = field(default_factory=dict)

    # trend
    trend: str = "stable"  # "emerging", "stable", "declining", "new"
    trend_score: float = 0.0  # positive = growing, negative = declining

    # specificity
    specificity: str = "medium"  # "broad", "medium", "specific"

    # sample papers
    top_paper_ids: List[str] = field(default_factory=list)


@dataclass
class TopicCluster:
    """
    a cluster of related topics forming a theme.
    """
    cluster_id: int
    name: str  # descriptive name for the cluster

    topics: List[str] = field(default_factory=list)
    total_papers: int = 0
    total_citations: int = 0

    # cluster characteristics
    is_core: bool = False  # central to the collection
    is_peripheral: bool = False  # tangential


@dataclass
class TopicAnalysis:
    """
    complete topic analysis for a paper collection.
    """
    # collection info
    total_papers: int = 0
    year_range: Tuple[int, int] = (0, 0)

    # topics (sorted by relevance)
    topics: List[Topic] = field(default_factory=list)

    # categorized topics
    core_topics: List[str] = field(default_factory=list)  # central themes
    emerging_topics: List[str] = field(default_factory=list)  # growing
    declining_topics: List[str] = field(default_factory=list)  # shrinking
    specific_topics: List[str] = field(default_factory=list)  # narrow/specialized

    # clusters
    topic_clusters: List[TopicCluster] = field(default_factory=list)

    # topic connections
    topic_cooccurrence: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # insights
    insights: List[str] = field(default_factory=list)

    def add_insight(self, insight: str):
        self.insights.append(insight)


class TopicExtractor(Agent):
    """
    extract topics and themes from a paper collection.

    analyzes:
    - OpenAlex concepts
    - title keywords
    - abstract terms (if available)

    identifies:
    - core topics (central to collection)
    - emerging topics (growing in recent years)
    - declining topics (shrinking)
    - topic clusters (related themes)

    usage:
        extractor = TopicExtractor()
        result = extractor.run(papers)

        if result.ok:
            analysis = result.data
            print(f"Core topics: {analysis.core_topics}")
            print(f"Emerging: {analysis.emerging_topics}")
    """

    # parameters
    MIN_PAPERS_FOR_TOPIC = 3
    MIN_WEIGHT_FOR_TOPIC = 2.0
    TOP_TOPICS_LIMIT = 50

    def __init__(
        self,
        min_papers: int = 3,
        use_titles: bool = True,
        use_abstracts: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.min_papers = min_papers
        self.use_titles = use_titles
        self.use_abstracts = use_abstracts

    @property
    def name(self) -> str:
        return "TopicExtractor"

    def execute(
        self,
        papers: List[Paper],
        reference_topics: Optional[Set[str]] = None
    ) -> AgentResult[TopicAnalysis]:
        """
        extract topics from paper collection.

        args:
            papers: list of Paper objects
            reference_topics: optional set of topics to prioritize

        returns:
            AgentResult with TopicAnalysis
        """
        result = AgentResult[TopicAnalysis](status=AgentStatus.SUCCESS)
        result.add_trace(f"extracting topics from {len(papers)} papers")

        if not papers:
            result.status = AgentStatus.FAILED
            result.add_error("NO_PAPERS", "no papers to analyze")
            return result

        # create analysis object
        years = [p.year for p in papers if p.year]
        analysis = TopicAnalysis(
            total_papers=len(papers),
            year_range=(min(years), max(years)) if years else (0, 0)
        )

        # step 1: extract raw topics from all sources
        topic_data = self._extract_topics(papers, result)

        # step 2: aggregate and score topics
        topics = self._aggregate_topics(topic_data, papers, result)
        analysis.topics = topics

        # step 3: compute topic trends
        self._compute_trends(topics, analysis.year_range, result)

        # step 4: categorize topics
        self._categorize_topics(topics, analysis, result)

        # step 5: find topic co-occurrence
        analysis.topic_cooccurrence = self._find_cooccurrence(topic_data, result)

        # step 6: cluster topics
        clusters = self._cluster_topics(topics, analysis.topic_cooccurrence, result)
        analysis.topic_clusters = clusters

        # step 7: generate insights
        self._generate_insights(analysis, reference_topics, result)

        result.data = analysis
        result.add_trace(f"extracted {len(topics)} topics, {len(clusters)} clusters")

        return result

    def _extract_topics(
        self,
        papers: List[Paper],
        result: AgentResult
    ) -> Dict[str, Dict[str, Any]]:
        """extract topics from all sources (concepts, titles, abstracts)."""
        result.add_trace("extracting topics from papers")

        # topic_name -> {papers: set, weight: float, paper_years: dict, sources: set}
        # paper_years maps paper_id -> year to avoid counting same paper multiple times
        topics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'papers': set(),
            'weight': 0.0,
            'paper_years': {},  # paper_id -> year (deduplicates year counts)
            'sources': set(),
            'variants': set()
        })

        for paper in papers:
            paper_weight = math.log(1 + (paper.citation_count or 0)) + 1

            # 1. extract from OpenAlex concepts
            for concept in (paper.concepts or [])[:10]:
                name = concept.get('name', '')
                score = concept.get('score', 0.5)

                if not self._is_valid_topic(name):
                    continue

                canonical = self._canonicalize(name)
                data = topics[canonical]
                data['papers'].add(paper.id)
                data['weight'] += score * paper_weight
                if paper.year:
                    data['paper_years'][paper.id] = paper.year
                data['sources'].add('concept')
                data['variants'].add(name)

            # 2. extract from title
            if self.use_titles and paper.title:
                title_terms = self._extract_terms_from_text(paper.title)
                for term in title_terms:
                    canonical = self._canonicalize(term)
                    data = topics[canonical]
                    data['papers'].add(paper.id)
                    data['weight'] += 0.5 * paper_weight  # lower weight for title terms
                    if paper.year:
                        data['paper_years'][paper.id] = paper.year
                    data['sources'].add('title')
                    data['variants'].add(term)

            # 3. extract from abstract (if available)
            if self.use_abstracts and paper.abstract:
                abstract_terms = self._extract_terms_from_text(paper.abstract)
                for term in abstract_terms[:20]:  # limit abstract terms
                    canonical = self._canonicalize(term)
                    data = topics[canonical]
                    data['papers'].add(paper.id)
                    data['weight'] += 0.3 * paper_weight  # lower weight for abstract
                    if paper.year:
                        data['paper_years'][paper.id] = paper.year
                    data['sources'].add('abstract')
                    data['variants'].add(term)

        result.add_trace(f"found {len(topics)} raw topics")
        return dict(topics)

    def _is_valid_topic(self, name: str) -> bool:
        """check if topic name is valid (not generic/stopword)."""
        if not name or len(name) < 3:
            return False

        name_lower = name.lower().strip()

        if name_lower in GENERIC_CONCEPTS:
            return False

        if name_lower in STOPWORDS:
            return False

        # filter single letters/numbers
        if len(name_lower) <= 2:
            return False

        return True

    def _canonicalize(self, name: str) -> str:
        """canonicalize topic name for deduplication."""
        # lowercase and strip
        canonical = name.lower().strip()

        # remove trailing 's' for simple plural handling
        if canonical.endswith('s') and len(canonical) > 4:
            canonical = canonical[:-1]

        return canonical

    def _extract_terms_from_text(self, text: str) -> List[str]:
        """extract meaningful terms from title/abstract text."""
        if not text:
            return []

        # simple tokenization
        text = text.lower()
        # keep alphanumeric and hyphens
        tokens = re.findall(r'[a-z][a-z0-9-]*[a-z0-9]|[a-z]', text)

        # filter stopwords and short tokens
        terms = [
            t for t in tokens
            if t not in STOPWORDS and len(t) > 2
        ]

        # also extract bigrams for multi-word concepts
        bigrams = []
        for i in range(len(terms) - 1):
            bigram = f"{terms[i]} {terms[i+1]}"
            if len(bigram) > 5:
                bigrams.append(bigram)

        return terms + bigrams[:10]

    def _aggregate_topics(
        self,
        topic_data: Dict[str, Dict[str, Any]],
        papers: List[Paper],
        result: AgentResult
    ) -> List[Topic]:
        """aggregate and filter topics."""
        result.add_trace("aggregating topics")

        topics = []

        for canonical, data in topic_data.items():
            paper_count = len(data['papers'])

            # filter by minimum papers
            if paper_count < self.min_papers:
                continue

            # filter by minimum weight
            if data['weight'] < self.MIN_WEIGHT_FOR_TOPIC:
                continue

            # get display name (most common variant)
            variants = list(data['variants'])
            display_name = max(variants, key=len) if variants else canonical

            # year distribution (from deduplicated paper_years)
            year_dist = defaultdict(int)
            years_list = list(data['paper_years'].values())
            for year in years_list:
                year_dist[year] += 1

            # top papers
            paper_ids = list(data['papers'])[:5]

            topic = Topic(
                name=display_name,
                canonical_name=canonical,
                paper_count=paper_count,
                total_weight=data['weight'],
                related_terms=variants[:5],
                first_year=min(years_list) if years_list else None,
                last_year=max(years_list) if years_list else None,
                year_distribution=dict(year_dist),
                top_paper_ids=paper_ids
            )

            # determine specificity
            if paper_count > len(papers) * 0.3:
                topic.specificity = "broad"
            elif paper_count < len(papers) * 0.05:
                topic.specificity = "specific"
            else:
                topic.specificity = "medium"

            topics.append(topic)

        # sort by weight
        topics.sort(key=lambda t: -t.total_weight)

        # limit
        topics = topics[:self.TOP_TOPICS_LIMIT]

        result.add_trace(f"aggregated to {len(topics)} topics")
        return topics

    def _compute_trends(
        self,
        topics: List[Topic],
        year_range: Tuple[int, int],
        result: AgentResult
    ):
        """compute trend for each topic."""
        result.add_trace("computing topic trends")

        if year_range[1] - year_range[0] < 3:
            return  # not enough years for trends

        mid_year = (year_range[0] + year_range[1]) // 2

        for topic in topics:
            if not topic.year_distribution:
                continue

            # count papers in early vs late period
            early_count = sum(
                count for year, count in topic.year_distribution.items()
                if year <= mid_year
            )
            late_count = sum(
                count for year, count in topic.year_distribution.items()
                if year > mid_year
            )

            total = early_count + late_count
            if total == 0:
                continue

            # compute trend score
            if early_count == 0 and late_count > 0:
                topic.trend = "new"
                topic.trend_score = 1.0
            elif late_count == 0 and early_count > 0:
                topic.trend = "declining"
                topic.trend_score = -1.0
            else:
                # ratio-based trend
                early_ratio = early_count / total
                late_ratio = late_count / total

                topic.trend_score = late_ratio - early_ratio

                if topic.trend_score > 0.2:
                    topic.trend = "emerging"
                elif topic.trend_score < -0.2:
                    topic.trend = "declining"
                else:
                    topic.trend = "stable"

    def _categorize_topics(
        self,
        topics: List[Topic],
        analysis: TopicAnalysis,
        result: AgentResult
    ):
        """categorize topics into core, emerging, declining, specific."""
        result.add_trace("categorizing topics")

        for topic in topics:
            # core topics (high weight, broad)
            if topic.specificity == "broad" or topic.paper_count > analysis.total_papers * 0.1:
                analysis.core_topics.append(topic.name)

            # emerging
            if topic.trend == "emerging" or topic.trend == "new":
                analysis.emerging_topics.append(topic.name)

            # declining
            if topic.trend == "declining":
                analysis.declining_topics.append(topic.name)

            # specific
            if topic.specificity == "specific":
                analysis.specific_topics.append(topic.name)

        # limit lists
        analysis.core_topics = analysis.core_topics[:10]
        analysis.emerging_topics = analysis.emerging_topics[:10]
        analysis.declining_topics = analysis.declining_topics[:10]
        analysis.specific_topics = analysis.specific_topics[:10]

    def _find_cooccurrence(
        self,
        topic_data: Dict[str, Dict[str, Any]],
        result: AgentResult
    ) -> Dict[str, Dict[str, int]]:
        """find topic co-occurrence (topics appearing in same papers)."""
        result.add_trace("computing topic co-occurrence")

        # invert: paper -> topics
        paper_topics: Dict[str, Set[str]] = defaultdict(set)
        for topic, data in topic_data.items():
            for paper_id in data['papers']:
                paper_topics[paper_id].add(topic)

        # count co-occurrences
        cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for paper_id, topics in paper_topics.items():
            topic_list = list(topics)
            for i in range(len(topic_list)):
                for j in range(i + 1, len(topic_list)):
                    t1, t2 = topic_list[i], topic_list[j]
                    cooccurrence[t1][t2] += 1
                    cooccurrence[t2][t1] += 1

        return {k: dict(v) for k, v in cooccurrence.items()}

    def _cluster_topics(
        self,
        topics: List[Topic],
        cooccurrence: Dict[str, Dict[str, int]],
        result: AgentResult
    ) -> List[TopicCluster]:
        """cluster topics by co-occurrence."""
        result.add_trace("clustering topics")

        if len(topics) < 5:
            return []

        # simple greedy clustering
        topic_names = {t.canonical_name for t in topics}
        used = set()
        clusters = []

        for topic in topics[:20]:  # start with top topics
            if topic.canonical_name in used:
                continue

            # find co-occurring topics
            related = []
            if topic.canonical_name in cooccurrence:
                for other, count in sorted(
                    cooccurrence[topic.canonical_name].items(),
                    key=lambda x: -x[1]
                ):
                    if other in topic_names and other not in used:
                        related.append(other)

            if len(related) >= 2:
                cluster_topics = [topic.canonical_name] + related[:5]

                cluster = TopicCluster(
                    cluster_id=len(clusters),
                    name=topic.name,  # use top topic as name
                    topics=cluster_topics,
                    total_papers=topic.paper_count
                )

                # mark as core if contains core topic
                if topic.specificity == "broad":
                    cluster.is_core = True

                clusters.append(cluster)
                used.update(cluster_topics)

            if len(clusters) >= 5:
                break

        return clusters

    def _generate_insights(
        self,
        analysis: TopicAnalysis,
        reference_topics: Optional[Set[str]],
        result: AgentResult
    ):
        """generate human-readable insights."""
        # overview
        analysis.add_insight(
            f"Analyzed {analysis.total_papers} papers ({analysis.year_range[0]}-{analysis.year_range[1]}), "
            f"found {len(analysis.topics)} significant topics."
        )

        # core topics
        if analysis.core_topics:
            analysis.add_insight(
                f"Core themes: {', '.join(analysis.core_topics[:5])}"
            )

        # emerging
        if analysis.emerging_topics:
            analysis.add_insight(
                f"Emerging topics: {', '.join(analysis.emerging_topics[:3])}"
            )

        # declining
        if analysis.declining_topics:
            analysis.add_insight(
                f"Declining topics: {', '.join(analysis.declining_topics[:3])}"
            )

        # specific/niche
        if analysis.specific_topics:
            analysis.add_insight(
                f"Specialized topics: {', '.join(analysis.specific_topics[:3])}"
            )

        # reference overlap
        if reference_topics:
            topic_names = {t.canonical_name for t in analysis.topics}
            overlap = topic_names & {t.lower() for t in reference_topics}
            if overlap:
                analysis.add_insight(
                    f"Matches reference topics: {', '.join(list(overlap)[:3])}"
                )
