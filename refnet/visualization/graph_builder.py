"""
graph builder - constructs NetworkX graphs from literature analysis.

creates:
- citation graphs (papers as nodes, citations as edges)
- author graphs (authors as nodes, collaborations as edges)
- topic graphs (topics as nodes, co-occurrence as edges)

usage:
    from refnet.visualization import GraphBuilder

    builder = GraphBuilder(analysis_result)
    citation_graph = builder.build_citation_graph()
    author_graph = builder.build_author_graph()
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from ..core.models import Paper
from ..pipeline.results import LiteratureAnalysis, AuthorProfile

logger = logging.getLogger("refnet.visualization")


@dataclass
class CitationGraph:
    """citation network - papers as nodes, citations as directed edges."""
    graph: Any  # nx.DiGraph
    paper_count: int = 0
    edge_count: int = 0
    seed_id: Optional[str] = None

    # stats
    most_cited: List[str] = field(default_factory=list)
    hub_papers: List[str] = field(default_factory=list)


@dataclass
class AuthorGraph:
    """author collaboration network - authors as nodes, co-authorships as edges."""
    graph: Any  # nx.Graph
    author_count: int = 0
    edge_count: int = 0

    # stats
    most_connected: List[str] = field(default_factory=list)
    communities: List[List[str]] = field(default_factory=list)


@dataclass
class TopicGraph:
    """topic co-occurrence network - topics as nodes, co-occurrence as edges."""
    graph: Any  # nx.Graph
    topic_count: int = 0
    edge_count: int = 0

    # stats
    central_topics: List[str] = field(default_factory=list)
    topic_clusters: List[List[str]] = field(default_factory=list)


class GraphBuilder:
    """
    builds graphs from literature analysis results.

    usage:
        builder = GraphBuilder(analysis)

        # build citation network
        citation_graph = builder.build_citation_graph()
        print(f"nodes: {citation_graph.paper_count}, edges: {citation_graph.edge_count}")

        # build author network
        author_graph = builder.build_author_graph()
    """

    def __init__(self, analysis: LiteratureAnalysis):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for graph building: pip install networkx")

        self.analysis = analysis
        self.papers_by_id: Dict[str, Paper] = {p.id: p for p in analysis.all_papers}

    def build_citation_graph(
        self,
        include_references: bool = True,
        min_citations: int = 0
    ) -> CitationGraph:
        """
        build a directed citation graph.

        nodes: papers (with attributes: title, year, citations, venue)
        edges: citation relationships (citing -> cited)
        """
        G = nx.DiGraph()

        # add papers as nodes
        for paper in self.analysis.all_papers:
            if paper.citation_count and paper.citation_count < min_citations:
                continue

            G.add_node(
                paper.id,
                title=paper.title or "",
                year=paper.year or 0,
                citations=paper.citation_count or 0,
                venue=paper.venue or "",
                doi=paper.doi or "",
                is_seed=(paper.id == self.analysis.seed_paper.id if self.analysis.seed_paper else False)
            )

        # infer edges from reading list categories (we don't have explicit citation data)
        # use author overlap and concept overlap as proxy for relatedness
        if self.analysis.seed_paper:
            seed_id = self.analysis.seed_paper.id
            seed_authors = set(self.analysis.seed_paper.authors or [])
            seed_concepts = self._get_concept_names(self.analysis.seed_paper)

            for paper in self.analysis.all_papers:
                if paper.id == seed_id:
                    continue
                if paper.id not in G:
                    continue

                # connect papers that share authors with seed (likely citations)
                paper_authors = set(paper.authors or [])
                if seed_authors & paper_authors:
                    # author overlap - likely related
                    if paper.year and self.analysis.seed_paper.year:
                        if paper.year < self.analysis.seed_paper.year:
                            # older paper -> seed cites it (reference)
                            G.add_edge(seed_id, paper.id, type="reference")
                        elif paper.year > self.analysis.seed_paper.year:
                            # newer paper -> cites seed (citation)
                            G.add_edge(paper.id, seed_id, type="citation")
                else:
                    # concept overlap - potentially related
                    paper_concepts = self._get_concept_names(paper)
                    overlap = seed_concepts & paper_concepts
                    if len(overlap) >= 3:  # significant overlap
                        if paper.year and self.analysis.seed_paper.year:
                            if paper.year <= self.analysis.seed_paper.year:
                                G.add_edge(seed_id, paper.id, type="reference", concepts=list(overlap)[:3])
                            else:
                                G.add_edge(paper.id, seed_id, type="citation", concepts=list(overlap)[:3])

        # add edges based on extracted relationships if available
        for rel in self.analysis.paper_relationships:
            if rel.source_id in G and rel.target_id in G:
                G.add_edge(
                    rel.target_id,  # target cites source
                    rel.source_id,
                    type=rel.relationship_type,
                    strength=rel.relationship_strength,
                    concepts=rel.shared_concepts[:3]
                )

        result = CitationGraph(
            graph=G,
            paper_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            seed_id=self.analysis.seed_paper.id if self.analysis.seed_paper else None
        )

        # find most cited papers (highest in-degree)
        if G.number_of_nodes() > 0:
            in_degrees = dict(G.in_degree())
            result.most_cited = sorted(in_degrees, key=in_degrees.get, reverse=True)[:10]

            # find hub papers (highest out-degree - cite many papers)
            out_degrees = dict(G.out_degree())
            result.hub_papers = sorted(out_degrees, key=out_degrees.get, reverse=True)[:10]

        return result

    def build_author_graph(
        self,
        min_collaborations: int = 1,
        max_authors: int = 200
    ) -> AuthorGraph:
        """
        build an undirected author collaboration graph.

        nodes: authors (with attributes: paper_count, citation_count)
        edges: co-authorship (weight = number of papers together)
        """
        G = nx.Graph()

        # count author stats
        author_papers = defaultdict(list)
        author_citations = defaultdict(int)

        for paper in self.analysis.all_papers:
            for author in (paper.authors or []):
                author_papers[author].append(paper.id)
                author_citations[author] += paper.citation_count or 0

        # filter to top authors by productivity
        top_authors = sorted(
            author_papers.keys(),
            key=lambda a: len(author_papers[a]),
            reverse=True
        )[:max_authors]
        top_author_set = set(top_authors)

        # add author nodes
        for author in top_authors:
            G.add_node(
                author,
                paper_count=len(author_papers[author]),
                citation_count=author_citations[author],
                is_key_author=any(a.name == author for a in self.analysis.key_authors)
            )

        # add collaboration edges
        collaboration_counts = defaultdict(int)

        for paper in self.analysis.all_papers:
            authors = [a for a in (paper.authors or []) if a in top_author_set]
            # count each pair of co-authors
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    pair = tuple(sorted([a1, a2]))
                    collaboration_counts[pair] += 1

        for (a1, a2), count in collaboration_counts.items():
            if count >= min_collaborations:
                G.add_edge(a1, a2, weight=count, papers=count)

        result = AuthorGraph(
            graph=G,
            author_count=G.number_of_nodes(),
            edge_count=G.number_of_edges()
        )

        # find most connected authors
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            result.most_connected = sorted(degrees, key=degrees.get, reverse=True)[:10]

            # detect communities (if graph is connected enough)
            if G.number_of_edges() > 5:
                try:
                    from networkx.algorithms.community import louvain_communities
                    communities = louvain_communities(G, resolution=1.0)
                    result.communities = [list(c) for c in communities if len(c) >= 2][:5]
                except Exception:
                    pass  # community detection failed

        return result

    def build_topic_graph(
        self,
        min_cooccurrence: int = 2,
        max_topics: int = 50
    ) -> TopicGraph:
        """
        build a topic co-occurrence graph.

        nodes: topics/concepts
        edges: co-occurrence in papers (weight = number of papers)
        """
        G = nx.Graph()

        # count topic occurrences
        topic_papers = defaultdict(set)
        cooccurrence = defaultdict(int)

        for paper in self.analysis.all_papers:
            topics = self._get_concept_names(paper)

            for topic in topics:
                topic_papers[topic].add(paper.id)

            # count co-occurrences
            topic_list = list(topics)
            for i, t1 in enumerate(topic_list):
                for t2 in topic_list[i+1:]:
                    pair = tuple(sorted([t1, t2]))
                    cooccurrence[pair] += 1

        # filter to top topics by frequency
        top_topics = sorted(
            topic_papers.keys(),
            key=lambda t: len(topic_papers[t]),
            reverse=True
        )[:max_topics]
        top_topic_set = set(top_topics)

        # add topic nodes
        for topic in top_topics:
            G.add_node(
                topic,
                paper_count=len(topic_papers[topic]),
                is_core=(topic in (self.analysis.landscape.core_topics if self.analysis.landscape else [])),
                is_emerging=(topic in (self.analysis.landscape.emerging_topics if self.analysis.landscape else []))
            )

        # add co-occurrence edges
        for (t1, t2), count in cooccurrence.items():
            if count >= min_cooccurrence and t1 in top_topic_set and t2 in top_topic_set:
                G.add_edge(t1, t2, weight=count, papers=count)

        result = TopicGraph(
            graph=G,
            topic_count=G.number_of_nodes(),
            edge_count=G.number_of_edges()
        )

        # find central topics (high betweenness)
        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            try:
                centrality = nx.betweenness_centrality(G)
                result.central_topics = sorted(centrality, key=centrality.get, reverse=True)[:10]
            except Exception:
                # fallback to degree centrality
                degrees = dict(G.degree())
                result.central_topics = sorted(degrees, key=degrees.get, reverse=True)[:10]

            # detect topic clusters
            if G.number_of_edges() > 5:
                try:
                    from networkx.algorithms.community import louvain_communities
                    communities = louvain_communities(G, resolution=1.0)
                    result.topic_clusters = [list(c) for c in communities if len(c) >= 2][:5]
                except Exception:
                    pass

        return result

    def _get_concept_names(self, paper: Paper) -> Set[str]:
        """extract concept names from paper."""
        names = set()
        for c in (paper.concepts or []):
            if isinstance(c, dict):
                name = c.get("name") or c.get("display_name")
                if name:
                    names.add(name.lower())
            elif isinstance(c, str):
                names.add(c.lower())
        return names
