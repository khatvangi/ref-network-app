"""
pipeline orchestrator - coordinates agents for end-to-end literature analysis.

usage:
    pipeline = Pipeline(provider)
    result = pipeline.analyze_paper("10.1038/s41586-020-2649-2")
    # or
    result = pipeline.analyze_author("Charles W. Carter")

    print(result.summary())
    for paper in result.reading_list[:10]:
        print(f"  {paper.paper.title}")
"""

import logging
import time
from typing import List, Optional, Set
from collections import Counter

from ..providers.openalex import OpenAlexProvider
from ..providers.base import ORCIDProvider

from ..agents import (
    SeedResolver, CitationWalker, AuthorResolver, CorpusFetcher,
    TrajectoryAnalyzer, CollaboratorMapper, TopicExtractor,
    GapDetector, RelevanceScorer
)
from ..agents.relevance_scorer import ScoringContext
from ..core.models import Paper

from .config import PipelineConfig
from .results import (
    LiteratureAnalysis, AuthorProfile, ReadingListItem,
    FieldLandscape, ResearchGaps
)


logger = logging.getLogger("refnet.pipeline")


class Pipeline:
    """
    orchestrates literature analysis from seed to complete report.

    modes:
    - analyze_paper(doi/title): start from a paper, explore citations and authors
    - analyze_author(name): start from an author, explore their work and network
    - analyze_topic(terms): start from topic terms, find key papers and authors

    each mode follows the same general flow:
    1. resolve seed (paper/author)
    2. expand graph (citations, author corpora)
    3. analyze (trajectories, topics, gaps)
    4. score and rank
    5. generate insights
    """

    def __init__(
        self,
        provider: OpenAlexProvider,
        orcid_provider: Optional[ORCIDProvider] = None,
        config: Optional[PipelineConfig] = None
    ):
        self.provider = provider
        self.orcid_provider = orcid_provider or ORCIDProvider()
        self.config = config or PipelineConfig()

        # initialize agents
        self._init_agents()

    def _init_agents(self):
        """initialize all agents."""
        self.seed_resolver = SeedResolver(self.provider)
        self.citation_walker = CitationWalker(
            self.provider,
            max_references=self.config.max_references,
            max_citations=self.config.max_citations
        )
        self.author_resolver = AuthorResolver(self.provider)
        self.corpus_fetcher = CorpusFetcher(
            self.provider,
            max_papers=self.config.max_papers_per_author
        )
        self.trajectory_analyzer = TrajectoryAnalyzer(
            orcid_provider=self.orcid_provider
        )
        self.collaborator_mapper = CollaboratorMapper()
        self.topic_extractor = TopicExtractor()
        self.gap_detector = GapDetector()
        self.relevance_scorer = RelevanceScorer()

    def analyze_paper(
        self,
        query: str,
        config: Optional[PipelineConfig] = None
    ) -> LiteratureAnalysis:
        """
        analyze literature starting from a paper.

        args:
            query: DOI, title, or other paper identifier
            config: optional config override

        returns:
            LiteratureAnalysis with complete results
        """
        config = config or self.config
        start_time = time.time()

        result = LiteratureAnalysis(
            seed_query=query,
            seed_type="paper"
        )

        logger.info(f"starting paper analysis: {query[:50]}...")

        # step 1: resolve seed paper
        logger.info("step 1: resolving seed paper")
        seed_result = self.seed_resolver.run(query=query)
        result.api_calls += seed_result.api_calls

        if not seed_result.ok:
            result.add_error(f"could not resolve seed: {seed_result.errors}")
            return result

        seed_paper = seed_result.data.paper
        result.seed_paper = seed_paper
        result.all_papers.append(seed_paper)

        logger.info(f"  resolved: {seed_paper.title[:50]}...")

        # step 2: walk citations
        logger.info("step 2: walking citations")
        all_papers, key_author_names = self._expand_citations(
            seed_paper, config, result
        )
        result.all_papers = list({p.id: p for p in result.all_papers + all_papers}.values())

        # step 3: identify and analyze key authors
        logger.info("step 3: analyzing key authors")
        key_authors = self._analyze_key_authors(
            key_author_names, seed_paper, config, result
        )
        result.key_authors = key_authors

        # add author papers to collection
        for author in key_authors:
            if author.trajectory and hasattr(author.trajectory, 'papers'):
                # papers are in the corpus, not trajectory directly
                pass  # papers already added during corpus fetch

        # deduplicate papers before analysis (prefer DOI, fallback to ID)
        result.all_papers = self._deduplicate_papers(result.all_papers)
        result.paper_count = len(result.all_papers)

        # step 4: analyze field landscape
        logger.info("step 4: analyzing field landscape")
        if config.analyze_topics:
            result.landscape = self._analyze_landscape(result.all_papers, config, result)

        # step 5: detect gaps
        logger.info("step 5: detecting gaps")
        if config.analyze_gaps and len(result.all_papers) >= 10:
            result.gaps = self._detect_gaps(result.all_papers, config, result)

        # step 6: score and rank papers
        logger.info("step 6: scoring papers")
        if config.score_relevance:
            result.reading_list = self._build_reading_list(
                result.all_papers, seed_paper, key_authors, config, result
            )

        # step 7: generate insights
        logger.info("step 7: generating insights")
        self._generate_insights(result, config)

        result.duration_seconds = time.time() - start_time

        logger.info(f"analysis complete: {result.paper_count} papers, {len(result.key_authors)} authors")

        return result

    def analyze_author(
        self,
        name: str,
        affiliation_hint: Optional[str] = None,
        config: Optional[PipelineConfig] = None
    ) -> LiteratureAnalysis:
        """
        analyze literature starting from an author.

        args:
            name: author name
            affiliation_hint: optional affiliation for disambiguation
            config: optional config override

        returns:
            LiteratureAnalysis with complete results
        """
        config = config or self.config
        start_time = time.time()

        result = LiteratureAnalysis(
            seed_query=name,
            seed_type="author"
        )

        logger.info(f"starting author analysis: {name}")

        # step 1: resolve author
        logger.info("step 1: resolving author")
        author_result = self.author_resolver.run(
            name=name,
            affiliation_hint=affiliation_hint
        )
        result.api_calls += author_result.api_calls

        if not author_result.ok:
            result.add_error(f"could not resolve author: {author_result.errors}")
            return result

        author_info = author_result.data.author_info
        logger.info(f"  resolved: {author_info.name} ({author_info.openalex_id})")

        # step 2: fetch author corpus
        logger.info("step 2: fetching author corpus")
        corpus_result = self.corpus_fetcher.run(author_id=author_info.openalex_id)
        result.api_calls += corpus_result.api_calls

        if not corpus_result.ok:
            result.add_error(f"could not fetch corpus: {corpus_result.errors}")
            return result

        corpus = corpus_result.data
        result.all_papers = list(corpus.papers)

        # step 3: analyze seed author
        logger.info("step 3: analyzing seed author")
        seed_author = self._build_author_profile(
            author_info, corpus, config, result
        )
        result.seed_author = seed_author
        result.key_authors = [seed_author]

        # step 4: find and analyze collaborators
        logger.info("step 4: analyzing collaborators")
        if config.analyze_collaborations and seed_author.collaboration_network:
            top_collabs = seed_author.collaboration_network.top_collaborators[:config.max_authors_to_follow]
            for collab_name in top_collabs:
                collab_result = self.author_resolver.run(name=collab_name)
                if collab_result.ok:
                    collab_info = collab_result.data.author_info
                    # fetch a smaller corpus for collaborators
                    collab_corpus = self.corpus_fetcher.run(author_id=collab_info.openalex_id)
                    if collab_corpus.ok:
                        collab_profile = AuthorProfile(
                            name=collab_info.name,
                            author_id=collab_info.openalex_id,
                            paper_count=collab_info.paper_count,
                            citation_count=collab_info.citation_count,
                            affiliations=collab_info.affiliations or []
                        )
                        result.key_authors.append(collab_profile)
                        # add unique papers
                        existing_ids = {p.id for p in result.all_papers}
                        for p in collab_corpus.data.papers:
                            if p.id not in existing_ids:
                                result.all_papers.append(p)
                                existing_ids.add(p.id)

        # deduplicate papers (preprint vs published versions)
        result.all_papers = self._deduplicate_papers(result.all_papers)
        result.paper_count = len(result.all_papers)

        # step 5: analyze field landscape
        logger.info("step 5: analyzing field landscape")
        if config.analyze_topics:
            result.landscape = self._analyze_landscape(result.all_papers, config, result)

        # step 6: detect gaps
        logger.info("step 6: detecting gaps")
        if config.analyze_gaps and len(result.all_papers) >= 10:
            result.gaps = self._detect_gaps(result.all_papers, config, result)

        # step 7: build reading list
        logger.info("step 7: building reading list")
        if config.score_relevance:
            # use seed author's top papers as context
            seed_papers = sorted(
                corpus.papers,
                key=lambda p: p.citation_count or 0,
                reverse=True
            )[:5]
            result.reading_list = self._build_reading_list(
                result.all_papers, seed_papers[0] if seed_papers else None,
                result.key_authors, config, result
            )

        # finalize paper count before insights
        result.paper_count = len(result.all_papers)

        # step 8: generate insights
        logger.info("step 8: generating insights")
        self._generate_insights(result, config)

        result.duration_seconds = time.time() - start_time

        logger.info(f"analysis complete: {result.paper_count} papers")

        return result

    def _expand_citations(
        self,
        seed_paper: Paper,
        config: PipelineConfig,
        result: LiteratureAnalysis
    ) -> tuple[List[Paper], List[str]]:
        """expand graph via citations, return papers and key author names."""
        all_papers = []
        author_counts = Counter()

        # walk citations from seed
        cite_result = self.citation_walker.run(paper=seed_paper)
        result.api_calls += cite_result.api_calls

        if cite_result.ok:
            citations = cite_result.data

            # add references
            for ref in citations.references:
                all_papers.append(ref.paper)
                for author in (ref.paper.authors or []):
                    author_counts[author] += 1

            # add citations
            for cite in citations.citations:
                all_papers.append(cite.paper)
                for author in (cite.paper.authors or []):
                    author_counts[author] += 1

            # add seed paper authors
            for author in (seed_paper.authors or []):
                author_counts[author] += 5  # boost seed authors

        # identify key authors (most frequent, excluding very common names)
        key_authors = [
            name for name, count in author_counts.most_common(config.max_authors_to_follow * 2)
            if count >= 2
        ][:config.max_authors_to_follow]

        return all_papers, key_authors

    def _analyze_key_authors(
        self,
        author_names: List[str],
        seed_paper: Paper,
        config: PipelineConfig,
        result: LiteratureAnalysis
    ) -> List[AuthorProfile]:
        """resolve and analyze key authors."""
        profiles = []

        for name in author_names:
            # resolve author
            resolve_result = self.author_resolver.run(
                name=name,
                paper_context=seed_paper
            )
            result.api_calls += resolve_result.api_calls

            if not resolve_result.ok:
                continue

            author_info = resolve_result.data.author_info

            # fetch corpus
            corpus_result = self.corpus_fetcher.run(author_id=author_info.openalex_id)
            result.api_calls += corpus_result.api_calls

            if not corpus_result.ok:
                # create basic profile without corpus
                profiles.append(AuthorProfile(
                    name=author_info.name,
                    author_id=author_info.openalex_id,
                    paper_count=author_info.paper_count,
                    citation_count=author_info.citation_count,
                    affiliations=author_info.affiliations or []
                ))
                continue

            corpus = corpus_result.data

            # add corpus papers to collection
            existing_ids = {p.id for p in result.all_papers}
            for p in corpus.papers:
                if p.id not in existing_ids:
                    result.all_papers.append(p)
                    existing_ids.add(p.id)

            # build full profile
            profile = self._build_author_profile(author_info, corpus, config, result)
            profiles.append(profile)

        return profiles

    def _build_author_profile(
        self,
        author_info,
        corpus,
        config: PipelineConfig,
        result: LiteratureAnalysis
    ) -> AuthorProfile:
        """build complete author profile with analysis."""
        profile = AuthorProfile(
            name=author_info.name,
            author_id=author_info.openalex_id,
            paper_count=author_info.paper_count,
            citation_count=author_info.citation_count,
            h_index=author_info.h_index,
            affiliations=author_info.affiliations or []
        )

        # trajectory analysis
        if config.analyze_trajectories:
            traj_result = self.trajectory_analyzer.run(corpus=corpus)
            result.api_calls += traj_result.api_calls
            if traj_result.ok:
                profile.trajectory = traj_result.data

        # collaboration analysis
        if config.analyze_collaborations:
            collab_result = self.collaborator_mapper.run(corpus=corpus)
            if collab_result.ok:
                profile.collaboration_network = collab_result.data

        # key papers
        top_papers = sorted(
            corpus.papers,
            key=lambda p: p.citation_count or 0,
            reverse=True
        )
        profile.key_papers = [p.id for p in top_papers[:10]]

        return profile

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        deduplicate papers preferring DOI, then title+year, then ID.

        OpenAlex often has multiple records for the same paper (preprint
        and published versions). This merges them, keeping the version
        with more citations.
        """
        # first pass: group by DOI (most reliable)
        doi_groups = {}
        no_doi = []

        for paper in papers:
            if paper.doi:
                key = paper.doi.lower().strip()
                if key not in doi_groups:
                    doi_groups[key] = []
                doi_groups[key].append(paper)
            else:
                no_doi.append(paper)

        # pick best from each DOI group (highest citations)
        result = []
        for doi, group in doi_groups.items():
            best = max(group, key=lambda p: p.citation_count or 0)
            result.append(best)

        # second pass: dedupe no-DOI papers by title+year
        title_groups = {}
        for paper in no_doi:
            if paper.title:
                # normalize title for comparison
                key = (paper.title.lower()[:50], paper.year or 0)
                if key not in title_groups:
                    title_groups[key] = []
                title_groups[key].append(paper)
            else:
                result.append(paper)  # keep papers without title

        for key, group in title_groups.items():
            best = max(group, key=lambda p: p.citation_count or 0)
            result.append(best)

        return result

    def _analyze_landscape(
        self,
        papers: List[Paper],
        config: PipelineConfig,
        result: LiteratureAnalysis
    ) -> FieldLandscape:
        """analyze the field landscape."""
        landscape = FieldLandscape()

        # basic stats
        landscape.total_papers = len(papers)
        authors = set()
        for p in papers:
            for a in (p.authors or []):
                authors.add(a)
        landscape.total_authors = len(authors)

        # year distribution
        years = [p.year for p in papers if p.year]
        if years:
            landscape.year_range = (min(years), max(years))
            year_counts = Counter(years)
            landscape.papers_per_year = dict(year_counts)
            landscape.peak_years = [
                y for y, c in year_counts.most_common(3)
            ]

        # topic analysis
        topic_result = self.topic_extractor.run(papers=papers)
        if topic_result.ok:
            analysis = topic_result.data
            landscape.topic_analysis = analysis
            landscape.core_topics = analysis.core_topics[:10]
            landscape.emerging_topics = analysis.emerging_topics[:10]
            landscape.declining_topics = analysis.declining_topics[:10]

        return landscape

    def _detect_gaps(
        self,
        papers: List[Paper],
        config: PipelineConfig,
        result: LiteratureAnalysis
    ) -> ResearchGaps:
        """detect research gaps."""
        gaps = ResearchGaps()

        gap_result = self.gap_detector.run(
            papers=papers,
            focus_concepts=config.focus_concepts if config.focus_concepts else None
        )

        if gap_result.ok:
            analysis = gap_result.data
            gaps.gap_analysis = analysis
            gaps.concept_gaps = analysis.concept_gaps[:config.top_gaps_count]
            gaps.bridge_papers = analysis.bridge_papers[:config.top_gaps_count]
            gaps.method_gaps = [
                {"method": g.method, "domain": g.domain, "score": g.gap_score}
                for g in analysis.method_gaps[:config.top_gaps_count]
            ]
            gaps.unexplored_areas = [
                {"name": a.name, "description": a.description, "papers": a.existing_papers}
                for a in analysis.unexplored_areas[:config.top_gaps_count]
            ]

        return gaps

    def _build_reading_list(
        self,
        papers: List[Paper],
        seed_paper: Optional[Paper],
        key_authors: List[AuthorProfile],
        config: PipelineConfig,
        result: LiteratureAnalysis
    ) -> List[ReadingListItem]:
        """build prioritized reading list."""
        # build scoring context
        seed_papers = [seed_paper] if seed_paper else []
        target_authors = [a.name for a in key_authors]

        # get target concepts from landscape
        target_concepts = []
        if result.landscape and result.landscape.core_topics:
            target_concepts = result.landscape.core_topics[:5]

        context = ScoringContext(
            seed_papers=seed_papers,
            target_concepts=target_concepts + config.focus_concepts,
            target_authors=target_authors + config.focus_authors,
            min_year=config.min_year,
            max_year=config.max_year
        )

        # score papers
        score_result = self.relevance_scorer.score_batch(papers, context)

        if not score_result.ok:
            return []

        # build reading list (deduplicated)
        reading_list = []
        seen_paper_ids = set()
        key_author_names = {a.name.lower() for a in key_authors}

        for score in score_result.data:
            # skip already-added papers
            if score.paper_id in seen_paper_ids:
                continue

            # stop when we have enough
            if len(reading_list) >= config.top_papers_count:
                break

            paper = next((p for p in papers if p.id == score.paper_id), None)
            if not paper:
                continue

            seen_paper_ids.add(paper.id)

            # determine category
            if seed_paper and paper.id == seed_paper.id:
                category = "seed"
                priority = 1
            elif score.is_highly_relevant:
                # check if by key author
                paper_authors = {a.lower() for a in (paper.authors or [])}
                if paper_authors & key_author_names:
                    category = "key_author"
                else:
                    category = "highly_relevant"
                priority = 1
            elif score.citation_score > 0.7:
                category = "foundational"
                priority = 2
            elif paper.year and paper.year >= 2023:
                category = "recent"
                priority = 2
            else:
                category = "general"
                priority = 3

            reading_list.append(ReadingListItem(
                paper=paper,
                relevance_score=score.score,
                category=category,
                reason=score.explanation,
                priority=priority
            ))

        # sort by priority, then score
        reading_list.sort(key=lambda x: (x.priority, -x.relevance_score))

        return reading_list

    def _generate_insights(
        self,
        result: LiteratureAnalysis,
        config: PipelineConfig
    ):
        """generate human-readable insights."""
        # field size
        result.add_insight(
            f"Analyzed {result.paper_count} papers "
            f"from {len(result.key_authors)} key authors."
        )

        # landscape insights
        if result.landscape:
            if result.landscape.core_topics:
                result.add_insight(
                    f"Core topics: {', '.join(result.landscape.core_topics[:3])}"
                )
            if result.landscape.emerging_topics:
                result.add_insight(
                    f"Emerging topics: {', '.join(result.landscape.emerging_topics[:3])}"
                )
            if result.landscape.year_range[0]:
                result.add_insight(
                    f"Publication years: {result.landscape.year_range[0]}-{result.landscape.year_range[1]}"
                )

        # author insights
        if result.key_authors:
            top_author = result.key_authors[0]
            result.add_insight(
                f"Key author: {top_author.name} ({top_author.paper_count} papers, "
                f"{top_author.citation_count} citations)"
            )

            if top_author.trajectory:
                traj = top_author.trajectory
                if traj.phases:
                    phases = [f"{p.dominant_concepts[0] if p.dominant_concepts else '?'}" for p in traj.phases[:3]]
                    result.add_insight(
                        f"Research evolution: {' → '.join(phases)}"
                    )

        # gap insights
        if result.gaps and result.gaps.concept_gaps:
            top_gap = result.gaps.concept_gaps[0]
            result.add_insight(
                f"Research opportunity: '{top_gap.concept_a}' and '{top_gap.concept_b}' "
                f"are rarely studied together (gap score: {top_gap.gap_score:.2f})"
            )

        # reading list summary
        if result.reading_list:
            must_read = [r for r in result.reading_list if r.priority == 1]
            result.add_insight(
                f"Reading list: {len(must_read)} must-read papers, "
                f"{len(result.reading_list)} total recommendations"
            )
