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

from ..core.models import (
    Paper, Author, Edge, EdgeType, PaperStatus, AuthorStatus,
    Bucket, BucketExpansionState
)
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
        # INTRO_HINT_CITES heuristic: first 25% of refs (clamped 10-40, max 20) are
        # likely from the introduction section where problem/context is defined
        if limits["max_refs"] > 0:
            try:
                refs = self.provider.get_references(paper_id, limit=limits["max_refs"])
                self.api_call_count += 1

                if refs:
                    # compute intro hint threshold per spec:
                    # k = clamp(round(n * intro_fraction), 10, 40)
                    # then cap at max_intro_hint_per_node
                    n_refs = len(refs)
                    intro_fraction = self.config.expansion.intro_fraction
                    max_intro = self.config.expansion.max_intro_hint_per_node
                    intro_weight = self.config.expansion.intro_hint_weight

                    # calculate k: 25% of refs, minimum 10 (if enough refs), max 40
                    k = int(n_refs * intro_fraction)
                    if n_refs >= 10:
                        k = max(k, 10)  # at least 10 if paper has 10+ refs
                    k = min(k, 40)  # cap at 40
                    k = min(k, max_intro)  # then cap at config limit (default 20)

                    for i, ref in enumerate(refs):
                        ref.discovered_from = paper.id
                        ref.discovered_channel = "backward"
                        ref.depth = paper.depth + 1

                        # use returned paper to handle deduplication correctly
                        added_ref = pool.add_paper(ref)
                        if added_ref:
                            stats.papers_discovered += 1

                            # apply intro hint: first k refs get boosted weight
                            is_intro_hint = i < k
                            edge_type = EdgeType.INTRO_HINT_CITES if is_intro_hint else EdgeType.CITES
                            edge_weight = intro_weight if is_intro_hint else 1.0

                            pool.add_edge(paper.id, added_ref.id, edge_type, weight=edge_weight)
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

                        # use returned paper to handle deduplication correctly
                        added_cite = pool.add_paper(cite)
                        if added_cite:
                            stats.papers_discovered += 1

                            # add edge (citing paper -> this paper)
                            pool.add_edge(added_cite.id, paper.id, EdgeType.CITES)
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
        catches edges that were missed during materialization.
        checks BOTH directions: edges FROM and edges TO each paper.
        preserves edge weights (e.g., intro_hint_cites have weight 2.0).

        this is critical for dendrimer-style expansion where:
        - paper A discovers paper B (edge A->B in pool)
        - if B is in graph but A is not, we still need the edge
        """
        edges_added = 0
        seen_edges = set()  # track (source, target) to avoid duplicates

        # for each paper in the graph, check BOTH directions
        for paper_id in list(graph.papers.keys()):
            # 1. edges FROM this paper (this paper cites others)
            edges_from = pool.get_edges_from_with_weight(paper_id)
            for target, etype, weight, conf in edges_from:
                if target in graph.papers:
                    edge_key = (paper_id, target)
                    if edge_key not in seen_edges and not graph.has_edge(paper_id, target):
                        try:
                            edge_type = EdgeType(etype)
                            graph.add_edge(paper_id, target, edge_type, weight=weight)
                            edges_added += 1
                            seen_edges.add(edge_key)
                        except ValueError:
                            continue

            # 2. edges TO this paper (others cite this paper)
            edges_to = pool.get_edges_to_with_weight(paper_id)
            for source, etype, weight, conf in edges_to:
                if source in graph.papers:
                    edge_key = (source, paper_id)
                    if edge_key not in seen_edges and not graph.has_edge(source, paper_id):
                        try:
                            edge_type = EdgeType(etype)
                            graph.add_edge(source, paper_id, edge_type, weight=weight)
                            edges_added += 1
                            seen_edges.add(edge_key)
                        except ValueError:
                            continue

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

    # =========================================================================
    # BUCKET-BASED EXPANSION (CITATION WALKING)
    # =========================================================================

    def build_with_buckets(
        self,
        seeds: List[Paper],
        topic: Optional[str] = None,
        pool: Optional[CandidatePool] = None,
        graph: Optional[WorkingGraph] = None
    ) -> ExpansionJob:
        """
        build citation network using DENDRIMER bucket expansion.

        dendrimer model:
        - each paper creates its OWN child bucket (not shared)
        - forms a tree structure where branches can be pruned independently
        - like a dendrimer polymer: exponential branching

        example:
          seed (1 paper) -> bucket_0 (20 papers)
          each of 20 papers -> 20 separate buckets at gen 1
          each of those -> more buckets at gen 2
          branches die independently when relevance decays

        layers (all automatic):
        1. citation expansion - refs (backward) + cites (forward)
        2. author expansion - key authors' other works
        3. trajectory - JSD drift detection
        4. clustering - community detection
        5. gap analysis - unexplored areas
        """
        start_time = datetime.now()

        # initialize storage
        if pool is None:
            pool = CandidatePool(self.config.candidate_pool)
        if graph is None:
            graph = WorkingGraph(self.config.working_graph)

        job = ExpansionJob(
            id=f"dendrimer_job_{int(start_time.timestamp())}",
            seeds=seeds,
            topic=topic,
            config=self.config,
            pool=pool,
            graph=graph
        )

        # initialize bucket expansion state
        state = BucketExpansionState(topic=topic)

        # add seeds to graph and pool
        for seed in seeds:
            seed.status = PaperStatus.SEED
            seed.relevance_score = 1.0
            pool.add_paper(seed)
            graph.add_seed(seed)
            state.seen_paper_ids.add(seed.id)
            job.stats.papers_materialized += 1

        logger.info(f"[dendrimer] starting with {len(seeds)} seeds, topic='{topic}'")

        # create root bucket (bucket_0) containing seeds' refs + cites
        root_bucket = self._create_initial_bucket(seeds, state, pool, job.stats)
        state.all_buckets[root_bucket.id] = root_bucket
        state.root_bucket_id = root_bucket.id
        state.total_buckets_created = 1
        state.active_branches = 1

        # compute adaptive max generations
        state.max_generations = self._compute_adaptive_max_generation(root_bucket)
        logger.info(
            f"[dendrimer] root bucket: {len(root_bucket.papers)} papers, "
            f"max_generations={state.max_generations}"
        )

        # DENDRIMER EXPANSION LOOP
        # each paper in a bucket creates its OWN child bucket
        current_buckets = [root_bucket]

        try:
            while current_buckets:
                # global stopping conditions
                if self._check_api_budget(state):
                    state.stopped_reason = "api_budget_exhausted"
                    logger.info(f"[dendrimer] stopping: api budget ({state.total_api_calls} calls)")
                    break

                if self._check_topic_drift(state):
                    state.stopped_reason = "topic_drift"
                    logger.info(f"[dendrimer] stopping: topic drift detected")
                    break

                if state.current_generation >= state.max_generations:
                    state.stopped_reason = "max_generations_reached"
                    logger.info(f"[dendrimer] stopping: max generations ({state.max_generations})")
                    break

                alive_count = state.count_alive_branches()
                logger.info(
                    f"[dendrimer] gen {state.current_generation}: "
                    f"{len(current_buckets)} buckets, {alive_count} alive branches"
                )

                next_generation_buckets = []

                for parent_bucket in current_buckets:
                    # check if this bucket was pruned
                    if not parent_bucket.is_alive:
                        continue

                    # per-bucket relevance decay check
                    if self._check_relevance_decay(parent_bucket):
                        parent_bucket.prune("relevance_decay")
                        state.pruned_branches += 1
                        logger.info(
                            f"[dendrimer] pruned branch: {parent_bucket.id} "
                            f"(avg_rel={parent_bucket.avg_relevance:.3f})"
                        )
                        continue

                    # DENDRIMER: each paper in bucket creates its OWN child bucket
                    for paper in parent_bucket.papers:
                        if self._check_api_budget(state):
                            break

                        # expand this single paper -> create child bucket
                        child_bucket = self._expand_single_paper(
                            paper, parent_bucket, state, pool, job.stats, topic
                        )

                        if child_bucket and len(child_bucket.papers) > 0:
                            # link parent <-> child
                            parent_bucket.child_bucket_ids.append(child_bucket.id)
                            state.all_buckets[child_bucket.id] = child_bucket
                            state.total_buckets_created += 1
                            next_generation_buckets.append(child_bucket)
                        # else: natural exhaustion for this paper (no new discoveries)

                current_buckets = next_generation_buckets
                state.current_generation += 1
                state.active_branches = len([b for b in current_buckets if b.is_alive])

            # check if naturally exhausted
            if not current_buckets and not state.stopped_reason:
                state.stopped_reason = "naturally_exhausted"
                state.is_exhausted = True
                logger.info("[dendrimer] naturally exhausted - topic covered!")

        except KeyboardInterrupt:
            logger.warning("[dendrimer] interrupted by user")
            state.stopped_reason = "interrupted"
            job.error = "interrupted by user"
        except Exception as e:
            logger.error(f"[dendrimer] critical error: {e}\n{traceback.format_exc()}")
            state.stopped_reason = f"error: {e}"
            job.error = str(e)

        # materialize top candidates into graph
        try:
            self._materialize_from_buckets(state, pool, graph)
        except Exception as e:
            logger.error(f"[dendrimer] materialization failed: {e}")

        # sync edges (bidirectional)
        try:
            self._sync_pool_edges(pool, graph)
        except Exception as e:
            logger.error(f"[dendrimer] edge sync failed: {e}")

        # LAYER 2: author expansion (if enabled)
        if self.config.author.enabled:
            try:
                self._run_author_layer(graph, pool, job.stats)
            except Exception as e:
                logger.error(f"[dendrimer] author layer failed: {e}")

        # LAYER 3: trajectory analysis (if enabled)
        if self.config.trajectory.enabled:
            try:
                self._compute_trajectories(graph)
            except Exception as e:
                logger.error(f"[dendrimer] trajectory failed: {e}")

        # LAYER 4: clustering (automatic in graph)
        # already done during materialization

        # LAYER 5: gap analysis (if enabled)
        if self.config.gap_analysis.enabled:
            try:
                job.gap_analysis = self.gap_analyzer.analyze(graph, pool)
            except Exception as e:
                logger.error(f"[dendrimer] gap analysis failed: {e}")

        # finalize
        job.stats.duration_seconds = (datetime.now() - start_time).total_seconds()
        job.stats.api_calls = state.total_api_calls
        job.is_complete = job.error is None

        # summary
        logger.info(
            f"[dendrimer] {'complete' if job.is_complete else 'partial'}: "
            f"generations={state.current_generation}, "
            f"total_buckets={state.total_buckets_created}, "
            f"pruned={state.pruned_branches}, "
            f"papers={state.total_papers_discovered}, "
            f"stopped={state.stopped_reason}, "
            f"graph={graph.stats()}"
        )

        return job

    def _create_initial_bucket(
        self,
        seeds: List[Paper],
        state: BucketExpansionState,
        pool: CandidatePool,
        stats: ExpansionStats
    ) -> Bucket:
        """
        create bucket_0 from seeds.
        for each seed, fetch intro refs + citations bidirectionally.
        """
        bucket_papers = []

        for seed in seeds:
            seed_id = seed.doi or seed.openalex_id or seed.s2_id
            if not seed_id:
                continue

            # backward: get intro refs (first 25%)
            try:
                refs = self.provider.get_references(seed_id, limit=self.config.expansion.max_refs_per_node)
                state.total_api_calls += 1

                if refs:
                    n_refs = len(refs)
                    intro_fraction = self.config.expansion.intro_fraction
                    k = int(n_refs * intro_fraction)
                    if n_refs >= 10:
                        k = max(k, 10)
                    k = min(k, 40)
                    k = min(k, self.config.expansion.max_intro_hint_per_node)

                    for i, ref in enumerate(refs[:k]):  # only intro refs for initial bucket
                        if ref.id in state.seen_paper_ids:
                            continue

                        # compute basic relevance
                        ref.relevance_score = self._compute_basic_relevance(ref, state.topic)
                        state.relevance_history.append(ref.relevance_score)

                        if ref.relevance_score >= self.config.expansion.min_relevance:
                            ref.discovered_from = seed.id
                            ref.discovered_channel = "backward"
                            ref.depth = 1

                            # use returned paper to handle deduplication correctly
                            added_ref = pool.add_paper(ref)
                            if added_ref:
                                pool.add_edge(seed.id, added_ref.id, EdgeType.INTRO_HINT_CITES, weight=2.0)
                                bucket_papers.append(added_ref)
                                state.seen_paper_ids.add(added_ref.id)
                                state.total_papers_discovered += 1
                                stats.papers_discovered += 1
                                stats.edges_created += 1
            except Exception as e:
                logger.warning(f"[bucket] refs failed for seed {seed.title[:30]}: {e}")
                stats.errors += 1

            # forward: get citations
            try:
                cites = self.provider.get_citations(seed_id, limit=self.config.expansion.max_cites_per_node)
                state.total_api_calls += 1

                if cites:
                    for cite in cites:
                        if cite.id in state.seen_paper_ids:
                            continue

                        # compute basic relevance
                        cite.relevance_score = self._compute_basic_relevance(cite, state.topic)
                        state.relevance_history.append(cite.relevance_score)

                        if cite.relevance_score >= self.config.expansion.min_relevance:
                            cite.discovered_from = seed.id
                            cite.discovered_channel = "forward"
                            cite.depth = 1

                            # use returned paper to handle deduplication correctly
                            added_cite = pool.add_paper(cite)
                            if added_cite:
                                pool.add_edge(added_cite.id, seed.id, EdgeType.CITES, weight=1.0)
                                bucket_papers.append(added_cite)
                                state.seen_paper_ids.add(added_cite.id)
                                state.total_papers_discovered += 1
                                stats.papers_discovered += 1
                                stats.edges_created += 1
            except Exception as e:
                logger.warning(f"[bucket] cites failed for seed {seed.title[:30]}: {e}")
                stats.errors += 1

        # create bucket
        bucket = Bucket(
            generation=0,
            papers=bucket_papers
        )
        bucket.compute_avg_relevance()

        logger.info(f"[bucket] created bucket_0: {len(bucket_papers)} papers, avg_relevance={bucket.avg_relevance:.3f}")

        return bucket

    def _expand_bucket(
        self,
        bucket: Bucket,
        state: BucketExpansionState,
        pool: CandidatePool,
        stats: ExpansionStats,
        topic: Optional[str]
    ) -> Bucket:
        """
        expand all papers in bucket -> create next generation bucket.
        bidirectional: intro refs + citations for each paper.
        """
        new_papers = []

        for paper in bucket.papers:
            # check api budget
            if state.total_api_calls >= self.config.expansion.max_api_calls_per_job:
                break

            paper_id = paper.doi or paper.openalex_id or paper.s2_id
            if not paper_id:
                continue

            # backward: intro refs (first 25%)
            try:
                refs = self.provider.get_references(paper_id, limit=self.config.expansion.max_refs_per_node)
                state.total_api_calls += 1
                bucket.total_refs_fetched += 1

                if refs:
                    n_refs = len(refs)
                    intro_fraction = self.config.expansion.intro_fraction
                    k = int(n_refs * intro_fraction)
                    if n_refs >= 10:
                        k = max(k, 10)
                    k = min(k, 40)
                    k = min(k, self.config.expansion.max_intro_hint_per_node)

                    for i, ref in enumerate(refs[:k]):
                        if ref.id in state.seen_paper_ids:
                            continue

                        ref.relevance_score = self._compute_basic_relevance(ref, topic)
                        state.relevance_history.append(ref.relevance_score)

                        if ref.relevance_score >= self.config.expansion.min_relevance:
                            ref.discovered_from = paper.id
                            ref.discovered_channel = "backward"
                            ref.depth = bucket.generation + 2  # +2 because bucket gen starts at 0

                            # use returned paper to handle deduplication correctly
                            added_ref = pool.add_paper(ref)
                            if added_ref:
                                pool.add_edge(paper.id, added_ref.id, EdgeType.INTRO_HINT_CITES, weight=2.0)
                                new_papers.append(added_ref)
                                state.seen_paper_ids.add(added_ref.id)
                                state.total_papers_discovered += 1
                                stats.papers_discovered += 1
                                stats.edges_created += 1
            except Exception as e:
                logger.warning(f"[bucket] refs failed: {e}")
                stats.errors += 1

            # forward: citations
            try:
                cites = self.provider.get_citations(paper_id, limit=self.config.expansion.max_cites_per_node)
                state.total_api_calls += 1
                bucket.total_cites_fetched += 1

                if cites:
                    for cite in cites:
                        if cite.id in state.seen_paper_ids:
                            continue

                        cite.relevance_score = self._compute_basic_relevance(cite, topic)
                        state.relevance_history.append(cite.relevance_score)

                        if cite.relevance_score >= self.config.expansion.min_relevance:
                            cite.discovered_from = paper.id
                            cite.discovered_channel = "forward"
                            cite.depth = bucket.generation + 2

                            # use returned paper to handle deduplication correctly
                            added_cite = pool.add_paper(cite)
                            if added_cite:
                                pool.add_edge(added_cite.id, paper.id, EdgeType.CITES, weight=1.0)
                                new_papers.append(added_cite)
                                state.seen_paper_ids.add(added_cite.id)
                                state.total_papers_discovered += 1
                                stats.papers_discovered += 1
                                stats.edges_created += 1
            except Exception as e:
                logger.warning(f"[bucket] cites failed: {e}")
                stats.errors += 1

        # create new bucket
        new_bucket = Bucket(
            generation=bucket.generation + 1,
            papers=new_papers,
            source_bucket_id=bucket.id
        )
        new_bucket.compute_avg_relevance()

        logger.info(
            f"[bucket] expanded {bucket.id} -> gen {new_bucket.generation}: "
            f"{len(new_papers)} papers, avg_relevance={new_bucket.avg_relevance:.3f}"
        )

        return new_bucket

    def _compute_basic_relevance(self, paper: Paper, topic: Optional[str]) -> float:
        """
        compute basic relevance score for a paper.
        used during bucket expansion before full graph context is available.
        """
        if not topic:
            # no topic filter - everything is potentially relevant
            return 0.5

        score = 0.0
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

        # title match (most important)
        if paper.title:
            title_lower = paper.title.lower()
            title_words = set(title_lower.split())

            # exact phrase match
            if topic_lower in title_lower:
                score += 0.6

            # word overlap
            overlap = len(topic_words & title_words)
            if overlap > 0:
                score += 0.2 * (overlap / len(topic_words))

        # abstract match
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            if topic_lower in abstract_lower:
                score += 0.3
            else:
                # partial word match
                overlap = sum(1 for w in topic_words if w in abstract_lower)
                if overlap > 0:
                    score += 0.1 * (overlap / len(topic_words))

        # concepts match
        if paper.concepts:
            concept_names = [c.get('name', '').lower() for c in paper.concepts]
            for concept in concept_names:
                if any(w in concept for w in topic_words):
                    score += 0.1
                    break

        return min(score, 1.0)

    def _compute_adaptive_max_generation(self, bucket_0: Bucket) -> int:
        """
        compute adaptive max generations based on initial bucket size.
        larger initial bucket suggests more to explore.
        """
        base_depth = self.config.expansion.base_max_generations
        bucket_size = len(bucket_0.papers)

        if bucket_size < 10:
            # small initial bucket - may exhaust quickly
            return min(base_depth, 5)
        elif bucket_size < 50:
            # moderate - use base depth
            return base_depth
        else:
            # large initial bucket - allow more generations
            # +1 per 50 papers
            extra = int(bucket_size / 50)
            return base_depth + extra

    def _check_relevance_decay(self, bucket: Bucket) -> bool:
        """
        check if bucket's average relevance is too low to continue.
        per-bucket stopping condition.
        """
        return bucket.avg_relevance < self.config.expansion.min_bucket_relevance

    def _check_topic_drift(self, state: BucketExpansionState) -> bool:
        """
        check if overall discovery quality has dropped too low.
        global emergency brake.
        """
        window = self.config.expansion.drift_window
        threshold = self.config.expansion.drift_kill_threshold

        if len(state.relevance_history) < window:
            return False  # not enough data yet

        recent = state.relevance_history[-window:]
        relevant_count = sum(1 for r in recent if r >= self.config.expansion.min_relevance)
        relevant_ratio = relevant_count / len(recent)

        return relevant_ratio < threshold

    def _check_api_budget(self, state: BucketExpansionState) -> bool:
        """check if api budget is exhausted."""
        return state.total_api_calls >= self.config.expansion.max_api_calls_per_job

    def _check_natural_exhaustion(self, bucket: Bucket) -> bool:
        """check if bucket is empty (no new papers found)."""
        return len(bucket.papers) == 0

    def _expand_single_paper(
        self,
        paper: Paper,
        parent_bucket: Bucket,
        state: BucketExpansionState,
        pool: CandidatePool,
        stats: ExpansionStats,
        topic: Optional[str]
    ) -> Optional[Bucket]:
        """
        DENDRIMER: expand a single paper to create its own child bucket.
        this is the core of dendrimer expansion - each paper spawns one bucket.

        returns:
            Bucket with discovered papers, or None if expansion failed
        """
        paper_id = paper.doi or paper.openalex_id or paper.s2_id
        if not paper_id:
            return None

        child_papers = []

        # backward: intro refs (first 25%)
        try:
            refs = self.provider.get_references(paper_id, limit=self.config.expansion.max_refs_per_node)
            state.total_api_calls += 1

            if refs:
                n_refs = len(refs)
                intro_fraction = self.config.expansion.intro_fraction
                k = int(n_refs * intro_fraction)
                if n_refs >= 10:
                    k = max(k, 10)
                k = min(k, 40)
                k = min(k, self.config.expansion.max_intro_hint_per_node)

                for ref in refs[:k]:
                    if ref.id in state.seen_paper_ids:
                        continue

                    ref.relevance_score = self._compute_basic_relevance(ref, topic)
                    state.relevance_history.append(ref.relevance_score)

                    if ref.relevance_score >= self.config.expansion.min_relevance:
                        ref.discovered_from = paper.id
                        ref.discovered_channel = "backward"
                        ref.depth = parent_bucket.generation + 2

                        # use returned paper to handle deduplication correctly
                        added_ref = pool.add_paper(ref)
                        if added_ref:
                            pool.add_edge(paper.id, added_ref.id, EdgeType.INTRO_HINT_CITES, weight=2.0)
                            child_papers.append(added_ref)
                            state.seen_paper_ids.add(added_ref.id)
                            state.total_papers_discovered += 1
                            stats.papers_discovered += 1
                            stats.edges_created += 1
        except Exception as e:
            logger.warning(f"[dendrimer] refs failed for {paper.title[:30] if paper.title else '?'}: {e}")
            stats.errors += 1

        # forward: citations
        try:
            cites = self.provider.get_citations(paper_id, limit=self.config.expansion.max_cites_per_node)
            state.total_api_calls += 1

            if cites:
                for cite in cites:
                    if cite.id in state.seen_paper_ids:
                        continue

                    cite.relevance_score = self._compute_basic_relevance(cite, topic)
                    state.relevance_history.append(cite.relevance_score)

                    if cite.relevance_score >= self.config.expansion.min_relevance:
                        cite.discovered_from = paper.id
                        cite.discovered_channel = "forward"
                        cite.depth = parent_bucket.generation + 2

                        # use returned paper to handle deduplication correctly
                        added_cite = pool.add_paper(cite)
                        if added_cite:
                            pool.add_edge(added_cite.id, paper.id, EdgeType.CITES, weight=1.0)
                            child_papers.append(added_cite)
                            state.seen_paper_ids.add(added_cite.id)
                            state.total_papers_discovered += 1
                            stats.papers_discovered += 1
                            stats.edges_created += 1
        except Exception as e:
            logger.warning(f"[dendrimer] cites failed for {paper.title[:30] if paper.title else '?'}: {e}")
            stats.errors += 1

        # create child bucket
        child_bucket = Bucket(
            generation=parent_bucket.generation + 1,
            papers=child_papers,
            source_paper_id=paper.id,
            source_paper_title=paper.title[:50] if paper.title else None,
            parent_bucket_id=parent_bucket.id
        )
        child_bucket.compute_avg_relevance()

        return child_bucket

    def _run_author_layer(
        self,
        graph: WorkingGraph,
        pool: CandidatePool,
        stats: ExpansionStats
    ):
        """
        LAYER 2: author expansion.
        for top authors in the graph, fetch their other works.
        """
        # collect author publication counts from graph
        author_counts: Dict[str, int] = {}
        for paper in graph.papers.values():
            for aid in (paper.author_ids or []):
                author_counts[aid] = author_counts.get(aid, 0) + 1

        # get top authors (appear in multiple papers)
        top_authors = sorted(
            [(aid, cnt) for aid, cnt in author_counts.items() if cnt >= 2],
            key=lambda x: x[1],
            reverse=True
        )[:10]  # top 10 authors

        if not top_authors:
            return

        logger.info(f"[author layer] expanding {len(top_authors)} key authors")

        expanded_author_ids: Set[str] = set()
        current_year = datetime.now().year

        for author_id, paper_count in top_authors:
            if author_id in expanded_author_ids:
                continue

            # get author object
            author = graph.authors.get(author_id)
            if not author:
                # try to create from paper metadata
                for paper in graph.papers.values():
                    if author_id in (paper.author_ids or []):
                        try:
                            idx = list(paper.author_ids).index(author_id)
                            if paper.authors and idx < len(paper.authors):
                                from ..core.models import Author
                                author = Author(
                                    id=author_id,
                                    name=paper.authors[idx],
                                    openalex_id=author_id if author_id.startswith('A') else None
                                )
                                break
                        except (ValueError, IndexError):
                            pass

            if not author:
                continue

            # expand author
            try:
                result = self.author_layer.expand_author(
                    author, pool, set(graph.seed_ids), current_year
                )
                self.api_call_count += 1

                if result.papers_added > 0:
                    stats.papers_discovered += result.papers_added
                    stats.authors_expanded += 1
                    expanded_author_ids.add(author_id)

                    # add author to graph
                    if author_id not in graph.authors:
                        graph.add_author(author)
            except Exception as e:
                logger.warning(f"[author layer] failed for {author.name if author else author_id}: {e}")
                stats.errors += 1

    def _materialize_from_buckets(
        self,
        state: BucketExpansionState,
        pool: CandidatePool,
        graph: WorkingGraph
    ):
        """
        materialize papers from buckets into working graph.
        prioritize by relevance score.
        """
        # collect all papers from all buckets
        all_papers = []
        for bucket in state.all_buckets.values():
            for paper in bucket.papers:
                if paper.id not in graph.papers:
                    all_papers.append(paper)

        # sort by relevance
        all_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        # materialize top papers up to graph limit
        max_nodes = self.config.get_max_nodes()
        materialized = 0

        for paper in all_papers:
            if len(graph.papers) >= max_nodes:
                break

            paper.status = PaperStatus.MATERIALIZED
            if graph.add_paper(paper):
                pool.update_paper_status(paper.id, PaperStatus.MATERIALIZED)
                materialized += 1

        logger.info(f"[bucket] materialized {materialized} papers into working graph")
