"""
export formats - GraphML, CSV, JSON, and visualization data.
"""

import json
import csv
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from io import StringIO

from ..core.models import Paper, Author, EdgeType
from ..graph.working_graph import WorkingGraph
from ..analysis.gap import GapAnalysisResult


class GraphExporter:
    """
    exports working graph to various formats.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        graph: WorkingGraph,
        filename: str = "graph.json",
        include_gap_analysis: Optional[GapAnalysisResult] = None
    ) -> str:
        """export to JSON format."""
        data = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "seed_count": len(graph.seed_ids),
                "author_count": len(graph.authors)
            },
            "stats": graph.stats(),
            "nodes": self._nodes_to_list(graph),
            "edges": self._edges_to_list(graph),
            "clusters": self._clusters_to_list(graph),
            "authors": self._authors_to_list(graph),
            "trajectories": self._trajectories_to_list(graph)
        }

        if include_gap_analysis:
            data["gap_analysis"] = {
                "bridges": [
                    {
                        "paper_id": b.paper_id,
                        "paper_title": b.paper_title,
                        "cluster_a": b.cluster_a_id,
                        "cluster_b": b.cluster_b_id,
                        "score": b.bridge_score
                    }
                    for b in include_gap_analysis.bridges
                ],
                "missing_links": [
                    {
                        "type": m.gap_type,
                        "source": m.source_id,
                        "target": m.target_id,
                        "explanation": m.explanation
                    }
                    for m in include_gap_analysis.missing_links
                ],
                "unexplored": [
                    {
                        "name": u.name,
                        "count": u.candidate_paper_count
                    }
                    for u in include_gap_analysis.unexplored_clusters
                ],
                "summary": include_gap_analysis.summary
            }

        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(path)

    def export_graphml(
        self,
        graph: WorkingGraph,
        filename: str = "graph.graphml"
    ) -> str:
        """export to GraphML format for Gephi/yEd."""
        lines = []

        # header
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"')
        lines.append('    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
        lines.append('    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns')
        lines.append('    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">')

        # node attributes
        lines.append('  <key id="title" for="node" attr.name="title" attr.type="string"/>')
        lines.append('  <key id="year" for="node" attr.name="year" attr.type="int"/>')
        lines.append('  <key id="citations" for="node" attr.name="citations" attr.type="int"/>')
        lines.append('  <key id="type" for="node" attr.name="type" attr.type="string"/>')
        lines.append('  <key id="relevance" for="node" attr.name="relevance" attr.type="double"/>')
        lines.append('  <key id="is_seed" for="node" attr.name="is_seed" attr.type="boolean"/>')
        lines.append('  <key id="cluster" for="node" attr.name="cluster" attr.type="string"/>')

        # edge attributes
        lines.append('  <key id="edge_type" for="edge" attr.name="edge_type" attr.type="string"/>')
        lines.append('  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>')

        lines.append('  <graph id="citation_network" edgedefault="directed">')

        # nodes
        for node_id, node in graph.nodes.items():
            paper = graph.papers.get(node_id)
            if paper:
                title_escaped = self._escape_xml(paper.title or "")
                cluster = graph.node_cluster_map.get(node_id, "")

                lines.append(f'    <node id="{node_id}">')
                lines.append(f'      <data key="title">{title_escaped}</data>')
                if paper.year:
                    lines.append(f'      <data key="year">{paper.year}</data>')
                if paper.citation_count:
                    lines.append(f'      <data key="citations">{paper.citation_count}</data>')
                lines.append(f'      <data key="type">paper</data>')
                lines.append(f'      <data key="relevance">{paper.relevance_score:.3f}</data>')
                lines.append(f'      <data key="is_seed">{str(node.is_seed).lower()}</data>')
                if cluster:
                    lines.append(f'      <data key="cluster">{cluster}</data>')
                lines.append('    </node>')

        # edges
        for edge_id, edge in graph.edges.items():
            lines.append(f'    <edge id="{edge_id}" source="{edge.source_id}" target="{edge.target_id}">')
            lines.append(f'      <data key="edge_type">{edge.edge_type.value}</data>')
            lines.append(f'      <data key="weight">{edge.weight}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')

        path = self.output_dir / filename
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

        return str(path)

    def export_csv(
        self,
        graph: WorkingGraph,
        nodes_filename: str = "nodes.csv",
        edges_filename: str = "edges.csv"
    ) -> tuple:
        """export to CSV format (nodes and edges)."""
        # nodes
        nodes_path = self.output_dir / nodes_filename
        with open(nodes_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'doi', 'title', 'year', 'venue', 'authors',
                'citation_count', 'is_review', 'is_seed', 'relevance_score',
                'cluster', 'concepts'
            ])

            for node_id, node in graph.nodes.items():
                paper = graph.papers.get(node_id)
                if paper:
                    cluster = graph.node_cluster_map.get(node_id, "")
                    concepts = "; ".join([c.get('name', '') for c in paper.concepts[:3]])
                    authors = "; ".join(paper.authors[:3])

                    writer.writerow([
                        node_id,
                        paper.doi or "",
                        paper.title or "",
                        paper.year or "",
                        paper.venue or "",
                        authors,
                        paper.citation_count or 0,
                        paper.is_review,
                        node.is_seed,
                        f"{paper.relevance_score:.3f}",
                        cluster,
                        concepts
                    ])

        # edges
        edges_path = self.output_dir / edges_filename
        with open(edges_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'type', 'weight'])

            for edge in graph.edges.values():
                writer.writerow([
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type.value,
                    edge.weight
                ])

        return str(nodes_path), str(edges_path)

    def export_cytoscape(
        self,
        graph: WorkingGraph,
        filename: str = "cytoscape.json"
    ) -> str:
        """export to Cytoscape.js compatible JSON."""
        elements = {
            "nodes": [],
            "edges": []
        }

        # nodes
        for node_id, node in graph.nodes.items():
            paper = graph.papers.get(node_id)
            if paper:
                cluster = graph.node_cluster_map.get(node_id, "default")

                node_data = {
                    "data": {
                        "id": node_id,
                        "label": (paper.title or "")[:40],
                        "title": paper.title or "",
                        "year": paper.year,
                        "citations": paper.citation_count or 0,
                        "relevance": paper.relevance_score,
                        "cluster": cluster,
                        "is_seed": node.is_seed,
                        "is_review": paper.is_review
                    },
                    "classes": self._get_node_classes(node, paper)
                }
                elements["nodes"].append(node_data)

        # edges
        for edge in graph.edges.values():
            edge_data = {
                "data": {
                    "id": edge.id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type.value,
                    "weight": edge.weight
                },
                "classes": edge.edge_type.value
            }
            elements["edges"].append(edge_data)

        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(elements, f, indent=2)

        return str(path)

    def export_flourish(
        self,
        graph: WorkingGraph,
        filename: str = "flourish_network.csv"
    ) -> str:
        """
        export to Flourish-compatible format.
        Flourish uses CSV with specific columns.
        """
        path = self.output_dir / filename
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Flourish network template columns
            writer.writerow([
                'Source', 'Target', 'Value', 'Link color'
            ])

            for edge in graph.edges.values():
                # get source/target labels
                source_paper = graph.papers.get(edge.source_id)
                target_paper = graph.papers.get(edge.target_id)

                source_label = (source_paper.title[:30] + "...") if source_paper else edge.source_id
                target_label = (target_paper.title[:30] + "...") if target_paper else edge.target_id

                # color by edge type
                color = self._edge_type_color(edge.edge_type)

                writer.writerow([
                    source_label,
                    target_label,
                    edge.weight,
                    color
                ])

        # also export points data for node sizing
        points_path = self.output_dir / filename.replace('.csv', '_points.csv')
        with open(points_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Group', 'Size', 'Color'])

            for node_id, node in graph.nodes.items():
                paper = graph.papers.get(node_id)
                if paper:
                    label = (paper.title[:30] + "...") if paper.title else node_id
                    cluster = graph.node_cluster_map.get(node_id, "Other")
                    size = (paper.citation_count or 1) ** 0.5  # sqrt scale
                    color = "#ff6b6b" if node.is_seed else "#4ecdc4"

                    writer.writerow([label, cluster, size, color])

        return str(path)

    # helpers

    def _nodes_to_list(self, graph: WorkingGraph) -> List[Dict]:
        """convert nodes to list of dicts."""
        nodes = []
        for node_id, node in graph.nodes.items():
            paper = graph.papers.get(node_id)
            if paper:
                nodes.append({
                    "id": node_id,
                    "doi": paper.doi,
                    "title": paper.title,
                    "year": paper.year,
                    "venue": paper.venue,
                    "authors": paper.authors[:5],
                    "citation_count": paper.citation_count,
                    "is_review": paper.is_review,
                    "is_methodology": paper.is_methodology,
                    "relevance_score": paper.relevance_score,
                    "bridge_score": paper.bridge_score,
                    "is_seed": node.is_seed,
                    "is_pinned": node.is_pinned,
                    "cluster": graph.node_cluster_map.get(node_id),
                    "concepts": [c.get('name') for c in paper.concepts[:5]]
                })
        return nodes

    def _edges_to_list(self, graph: WorkingGraph) -> List[Dict]:
        """convert edges to list of dicts."""
        return [
            {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.value,
                "weight": e.weight
            }
            for e in graph.edges.values()
        ]

    def _clusters_to_list(self, graph: WorkingGraph) -> List[Dict]:
        """convert clusters to list of dicts."""
        return [
            {
                "id": c.id,
                "name": c.name,
                "size": c.size,
                "top_concepts": c.top_concepts[:5]
            }
            for c in graph.clusters.values()
        ]

    def _authors_to_list(self, graph: WorkingGraph) -> List[Dict]:
        """convert authors to list of dicts."""
        authors = []
        for author_id, author in graph.authors.items():
            # count papers by this author in the graph
            paper_count = sum(
                1 for p in graph.papers.values()
                if author.openalex_id in (p.author_ids or []) or
                   author.s2_id in (p.author_ids or [])
            )

            authors.append({
                "id": author_id,
                "name": author.name,
                "orcid": author.orcid,
                "openalex_id": author.openalex_id,
                "affiliations": author.affiliations[:3],
                "topic_fit": author.topic_fit,
                "centrality": author.centrality,
                "paper_count": paper_count,
                "trajectory_computed": author.trajectory_computed,
                "drift_event_count": len(author.drift_events) if author.drift_events else 0
            })
        return authors

    def _trajectories_to_list(self, graph: WorkingGraph) -> List[Dict]:
        """convert author trajectories to list of dicts."""
        trajectories = []
        for author_id, author in graph.authors.items():
            if not author.trajectory_computed or not author.drift_events:
                continue

            # convert drift events to serializable format
            events = []
            for event in author.drift_events:
                if hasattr(event, '__dict__'):
                    # DriftEvent dataclass
                    events.append({
                        "year_from": event.year_from,
                        "year_to": event.year_to,
                        "drift_magnitude": event.drift_magnitude,
                        "is_novelty_jump": event.is_novelty_jump,
                        "entering_concepts": event.entering_concepts[:5],
                        "exiting_concepts": event.exiting_concepts[:5]
                    })
                elif isinstance(event, dict):
                    events.append(event)

            trajectories.append({
                "author_id": author_id,
                "author_name": author.name,
                "drift_events": events
            })
        return trajectories

    def _escape_xml(self, text: str) -> str:
        """escape XML special characters."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))

    def _get_node_classes(self, node, paper) -> str:
        """get CSS classes for node."""
        classes = []
        if node.is_seed:
            classes.append("seed")
        if node.is_pinned:
            classes.append("pinned")
        if paper.is_review:
            classes.append("review")
        if paper.is_methodology:
            classes.append("methodology")
        return " ".join(classes) if classes else "default"

    def _edge_type_color(self, edge_type: EdgeType) -> str:
        """get color for edge type."""
        colors = {
            EdgeType.CITES: "#888888",
            EdgeType.CITED_BY: "#aaaaaa",
            EdgeType.INTRO_CITES: "#e74c3c",
            EdgeType.INTRO_HINT_CITES: "#e67e22",
            EdgeType.AUTHORED_BY: "#3498db",
            EdgeType.AUTHORED: "#2980b9",
            EdgeType.COAUTHOR: "#9b59b6",
            EdgeType.TRAJECTORY_STEP: "#1abc9c",
            EdgeType.AUTHOR_BRIDGE: "#f1c40f"
        }
        return colors.get(edge_type, "#cccccc")
