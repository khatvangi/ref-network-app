"""
expansion engine - orchestrates the citation-walk discovery process.
scientist-centric: citations drive expansion, not keywords.
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import heapq
import logging
import traceback

from ..core.models import Paper, Author, Edge, EdgeType, PaperStatus, AuthorStatus
from ..core.config import RefnetConfig, ExpansionMode
from ..core.resilience import safe_execute
from ..providers.base import PaperProvider
from ..graph.candidate_pool import CandidatePool
from ..graph.working_graph import WorkingGraph
from ..layers.author import AuthorLayer
from ..layers.trajectory import TrajectoryLayer
from ..analysis.hub import HubDetector
from ..analysis.gap import GapAnalyzer
from ..scoring.graph_relevance import GraphRelevanceScorer

logger = logging.getLogger("refnet.expansion")


@dataclass
class ExpansionStats:
    """statistics for an expansion run."""
    papers_discovered: int = 0
    papers_materialized: int = 0
    papers_expanded: int = 0
    authors_discovered: int = 0
    authors_expanded: int = 0
    edges_created: int = 0
    api_calls: int = 0
    hubs_suppressed: int = 0
    duration_seconds: float = 0.0
    # error tracking
    errors: int = 0
    retries: int = 0
    papers_failed: int = 0


@dataclass
class ExpansionJob:
    """a complete expansion job."""
    id: str = ""
    seeds: List[Paper] = field(default_factory=list)
    topic: Optional[str] = None
    config: RefnetConfig = field(default_factory=RefnetConfig)

    # state
    pool: CandidatePool = None
    graph: WorkingGraph = None
    stats: ExpansionStats = field(default_factory=ExpansionStats)

    # flags
    is_complete: bool = False
    error: Optional[str] = None


class ExpansionEngine:
    """
    main engine for citation-walk graph expansion.
    orchestrates providers, scoring, and layers.
    """

    def __init__(
        self,
        provider: PaperProvider,
        config: Optional[RefnetConfig] = None
    ):
        self.provider = provider
        self.config = config or RefnetConfig()

        # components
        self.hub_detector = HubDetector(self.config.expansion)
        self.scorer = GraphRelevanceScorer(self.config.scoring)
        self.author_layer = AuthorLayer(provider, self.config.author)
        self.trajectory_layer = TrajectoryLayer(provider, self.config.trajectory)
        self.gap_analyzer = GapAnalyzer(self.config.gap_analysis)

        # state
        self.api_call_count = 0

    def build(
        self,
        seeds: List[Paper],
        topic: Optional[str] = None,
        pool: Optional[CandidatePool] = None,
        graph: Optional[WorkingGraph] = None
    ) -> ExpansionJob:
        """
        build citation network from seeds.
        main entry point for expansion.
        """
        start_time = datetime.now()

        # initialize storage
        if pool is None:
            pool = CandidatePool(self.config.candidate_pool)
        if graph is None:
            graph = WorkingGraph(self.config.working_graph)

        job = ExpansionJob(
            id=f"job_{int(start_time.timestamp())}",
            seeds=seeds,
            topic=topic,
            config=self.config,
            pool=pool,
            graph=graph
        )

        # add seeds to graph
        for seed in seeds:
            seed.status = PaperStatus.SEED
            seed.relevance_score = 1.0
            pool.add_paper(seed)
            graph.add_seed(seed)
            job.stats.papers_materialized += 1

        logger.info(f"[expansion] starting with {len(seeds)} seeds")

        # main expansion loop with error recovery
        depth = 0
        expanded_ids: Set[str] = set()
        critical_error = None

        try:
            while depth < self.config.expansion.max_depth:
                if self.api_call_count >= self.config.expansion.max_api_calls_per_job:
                    logger.info(f"[expansion] api call budget exhausted ({self.api_call_count})")
                    break

                # get papers to expand
                try:
                    to_expand = self._get_expansion_queue(graph, expanded_ids)
                except Exception as e:
                    logger.error(f"[expansion] failed to get expansion queue: {e}")
                    job.stats.errors += 1
                    break

                if not to_expand:
                    logger.info(f"[expansion] no more papers to expand at depth {depth}")
                    break

                logger.info(f"[expansion] depth {depth}: expanding {len(to_expand)} papers")

                for paper in to_expand:
                    if self.api_call_count >= self.config.expansion.max_api_calls_per_job:
                        break

                    try:
                        self._expand_paper(paper, pool, graph, expanded_ids, job.stats)
                        expanded_ids.add(paper.id)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logger.error(f"[expansion] paper expansion failed: {e}")
                        job.stats.errors += 1
                        job.stats.papers_failed += 1
                        # continue with other papers

                # materialize top candidates into graph
                try:
                    self._materialize_candidates(pool, graph)
                except Exception as e:
                    logger.error(f"[expansion] materialization failed: {e}")
                    job.stats.errors += 1

                depth += 1

        except KeyboardInterrupt:
            logger.warning("[expansion] interrupted by user, saving partial results")
            job.error = "interrupted by user"
        except Exception as e:
            logger.error(f"[expansion] critical error: {e}\n{traceback.format_exc()}")
            critical_error = str(e)
            job.error = critical_error

        # always try to finalize, even on error
        try:
            # sync all edges from pool to graph (catches edges between seeds)
            self._sync_pool_edges(pool, graph)
        except Exception as e:
            logger.error(f"[expansion] edge sync failed: {e}")

        # compute trajectories for top authors
        if self.config.trajectory.enabled:
            try:
                self._compute_trajectories(graph)
            except Exception as e:
                logger.error(f"[expansion] trajectory computation failed: {e}")

        # run gap analysis
        if self.config.gap_analysis.enabled:
            try:
                job.gap_analysis = self.gap_analyzer.analyze(graph, pool)
            except Exception as e:
                logger.error(f"[expansion] gap analysis failed: {e}")

        # finalize
        job.stats.duration_seconds = (datetime.now() - start_time).total_seconds()
        job.stats.api_calls = self.api_call_count
        job.is_complete = critical_error is None

        # log summary
        logger.info(
            f"[expansion] {'complete' if job.is_complete else 'partial'}: "
            f"{graph.stats()}, errors={job.stats.errors}"
        )

        return job

    def _get_expansion_queue(
        self,
        graph: WorkingGraph,
        expanded_ids: Set[str]
    ) -> List[Paper]:
        """get papers to expand next, prioritized."""
        candidates = []

        for paper_id, paper in graph.papers.items():
            if paper_id in expanded_ids:
                continue
            if paper.status == PaperStatus.EXPANDED:
                continue

            # compute expansion priority
            priority = self._compute_expansion_priority(paper, graph)
            candidates.append((priority, paper))

        # sort by priority
        candidates.sort(key=lambda x: x[0], reverse=True)

        # take top papers for expansion
        batch_size = min(
            10,
            self.config.expansion.max_api_calls_per_job - self.api_call_count
        )

        return [p for _, p in candidates[:batch_size]]

    def _expand_paper(
        self,
        paper: Paper,
        pool: CandidatePool,
        graph: WorkingGraph,
        expanded_ids: Set[str],
        stats: ExpansionStats
    ):
        """expand a single paper (refs + cites + authors)."""
        print(f"[expansion] expanding: {paper.title[:50]}...")

        # check hub status (but always expand seeds)
        is_seed = paper.status == PaperStatus.SEED
        hub_analysis = self.hub_detector.analyze_paper(paper)
        limits = self.hub_detector.get_expansion_limits(paper)

        if hub_analysis.suppress_expansion and not is_seed:
            print(f"[expansion] hub suppressed: {paper.title[:30]} ({hub_analysis.suppress_reason})")
            stats.hubs_suppressed += 1
            paper.is_methodology = hub_analysis.is_hub
            paper.status = PaperStatus.EXPANDED
            return

        if hub_analysis.is_hub and is_seed:
            # reduce limits for hub seeds but still expand
            print(f"[expansion] hub seed (reduced limits): {paper.title[:30]}")
            limits = {"max_refs": 20, "max_cites": 10, "max_author_works": 5}

        paper_id = paper.doi or paper.openalex_id
        if not paper_id:
            logger.warning(f"[expansion] no id for paper: {paper.title[:30]}")
            stats.papers_failed += 1
            return

        # 1. backward references (what this paper cites)
        if limits["max_refs"] > 0:
            try:
                refs = self.provider.get_references(paper_id, limit=limits["max_refs"])
                self.api_call_count += 1

                if refs:
                    for i, ref in enumerate(refs):
                        ref.discovered_from = paper.id
                        ref.discovered_channel = "backward"
                        ref.depth = paper.depth + 1

                        if pool.add_paper(ref):
                            stats.papers_discovered += 1

                            # add edge
                            edge_type = EdgeType.INTRO_HINT_CITES if i < self.config.expansion.max_intro_hint_per_node else EdgeType.CITES
                            pool.add_edge(paper.id, ref.id, edge_type)
                            stats.edges_created += 1
                    paper.refs_fetched = True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"[expansion] refs failed for {paper.title[:30]}: {e}")
                stats.errors += 1

        # 2. forward citations (what cites this paper)
        if limits["max_cites"] > 0:
            try:
                cites = self.provider.get_citations(paper_id, limit=limits["max_cites"])
                self.api_call_count += 1

                if cites:
                    for cite in cites:
                        cite.discovered_from = paper.id
                        cite.discovered_channel = "forward"
                        cite.depth = paper.depth + 1

                        if pool.add_paper(cite):
                            stats.papers_discovered += 1

                            # add edge (citing paper -> this paper)
                            pool.add_edge(cite.id, paper.id, EdgeType.CITES)
                            stats.edges_created += 1
                    paper.cites_fetched = True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"[expansion] cites failed for {paper.title[:30]}: {e}")
                stats.errors += 1

        # 3. author expansion
        if self.config.author.enabled and limits["max_author_works"] > 0:
            try:
                expanded_author_ids = {a.id for a in graph.authors.values()}
                authors_to_expand = self.author_layer.get_authors_to_expand(
                    paper, expanded_author_ids
                )

                current_year = datetime.now().year
                author_papers_added = 0

                for author in authors_to_expand[:2]:  # max 2 authors per paper
                    if author_papers_added >= limits["max_author_works"]:
                        break

                    try:
                        result = self.author_layer.expand_author(
                            author, pool, set(graph.seed_ids), current_year
                        )
                        self.api_call_count += 1

                        author_papers_added += result.papers_added
                        stats.papers_discovered += result.papers_added
                        stats.authors_expanded += 1

                        # add author to graph if relevant
                        if result.papers_added > 0:
                            pool.add_author(author)
                            # also add to working graph directly
                            graph.add_author(author)
                            stats.authors_discovered += 1
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logger.warning(f"[expansion] author expand failed: {e}")
                        stats.errors += 1

                paper.authors_fetched = True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"[expansion] author layer failed for {paper.title[:30]}: {e}")
                stats.errors += 1

        paper.status = PaperStatus.EXPANDED
        stats.papers_expanded += 1

    def _materialize_candidates(
        self,
        pool: CandidatePool,
        graph: WorkingGraph
    ):
        """move top candidates from pool into working graph."""
        # get top candidates by materialization score
        candidates = pool.get_top_candidates(limit=100, by="materialization_score")

        for paper in candidates:
            if paper.id in graph.papers:
                continue

            # recompute scores with graph context
            breakdown = self.scorer.compute_relevance(paper, graph)
            paper.relevance_score = breakdown.final_score

            bridge = self.scorer.compute_bridge_score(paper, graph)
            paper.bridge_score = bridge

            mat_score = self.scorer.compute_materialization_score(paper, graph, bridge)
            paper.materialization_score = mat_score

            # update in pool
            pool.update_paper_scores(
                paper.id,
                relevance=paper.relevance_score,
                bridge=paper.bridge_score,
                materialization=mat_score
            )

            # add to graph if score is high enough
            if mat_score > 0.2:
                paper.status = PaperStatus.MATERIALIZED
                if graph.add_paper(paper):
                    pool.update_paper_status(paper.id, PaperStatus.MATERIALIZED)

                    # add edges to graph
                    neighbors = pool.get_neighbors(paper.id)
                    for etype, targets in neighbors.items():
                        if etype.startswith("reverse_"):
                            continue
                        try:
                            edge_type = EdgeType(etype)
                        except ValueError:
                            continue
                        for target in targets:
                            if target in graph.papers:
                                graph.add_edge(paper.id, target, edge_type)

    def _compute_expansion_priority(
        self,
        paper: Paper,
        graph: WorkingGraph
    ) -> float:
        """compute priority for expansion (not just relevance)."""
        # base relevance
        relevance = paper.relevance_score

        # boost frontier nodes (at edge of explored region)
        neighbors = graph.get_neighbors(paper.id)
        unexplored_neighbors = sum(
            1 for n in neighbors
            if n in graph.papers and graph.papers[n].status != PaperStatus.EXPANDED
        )
        frontier_bonus = min(unexplored_neighbors / 10, 0.3)

        # boost bridge candidates
        bridge_bonus = paper.bridge_score * 0.2

        # penalize very old papers
        from datetime import datetime
        age_penalty = 0.0
        if paper.year:
            age = datetime.now().year - paper.year
            if age > 10:
                age_penalty = 0.1

        return relevance + frontier_bonus + bridge_bonus - age_penalty

    def _sync_pool_edges(
        self,
        pool: CandidatePool,
        graph: WorkingGraph
    ):
        """
        sync all edges from pool to graph.
        catches edges that were missed during materialization,
        especially edges between seeds.
        """
        edges_added = 0

        # for each paper in the graph, check for edges in the pool
        for paper_id in list(graph.papers.keys()):
            neighbors = pool.get_neighbors(paper_id)

            for etype, targets in neighbors.items():
                if etype.startswith("reverse_"):
                    continue

                try:
                    edge_type = EdgeType(etype)
                except ValueError:
                    continue

                for target in targets:
                    if target in graph.papers:
                        # check if edge already exists
                        if not graph.has_edge(paper_id, target):
                            graph.add_edge(paper_id, target, edge_type)
                            edges_added += 1

        if edges_added > 0:
            print(f"[expansion] synced {edges_added} edges from pool to graph")

    def _compute_trajectories(self, graph: WorkingGraph):
        """compute trajectories for top authors."""
        # get authors with enough presence
        author_papers: Dict[str, List[Paper]] = {}

        for paper in graph.papers.values():
            for aid in paper.author_ids:
                if aid not in author_papers:
                    author_papers[aid] = []
                author_papers[aid].append(paper)

        # find authors with enough papers
        top_authors = [
            (aid, papers) for aid, papers in author_papers.items()
            if len(papers) >= 3
        ]

        # sort by paper count
        top_authors.sort(key=lambda x: len(x[1]), reverse=True)

        # compute trajectories for top N
        for aid, papers in top_authors[:self.config.trajectory.max_trajectory_authors_auto]:
            # try to get author object
            author = graph.authors.get(aid)
            if not author:
                # try to find author name from papers
                author_name = ""
                for p in papers:
                    if p.author_ids and p.authors:
                        try:
                            idx = list(p.author_ids).index(aid)
                            if idx < len(p.authors):
                                author_name = p.authors[idx]
                                break
                        except (ValueError, IndexError):
                            pass
                author = Author(
                    id=aid,
                    name=author_name,
                    openalex_id=aid if aid.startswith("A") else None
                )

            # compute trajectory
            drift_events = self.trajectory_layer.compute_trajectory(author, papers)

            # store on author for export
            if drift_events:
                author.drift_events = drift_events
                author.trajectory_computed = True

                # add author to graph if not already there
                if author.id not in graph.authors:
                    graph.add_author(author)
                else:
                    # update existing author with trajectory
                    graph.authors[author.id].drift_events = drift_events
                    graph.authors[author.id].trajectory_computed = True

                # detect bridges
                self.trajectory_layer.detect_author_bridges(author, graph, drift_events)
