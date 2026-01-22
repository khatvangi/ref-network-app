"""
graph exporter - exports graphs to various formats.

supports:
- JSON (for web visualization)
- GraphML (for Gephi, Cytoscape)
- GEXF (for Gephi)
- DOT (for Graphviz)

usage:
    from refnet.visualization import GraphBuilder, GraphExporter

    builder = GraphBuilder(analysis)
    citation_graph = builder.build_citation_graph()

    exporter = GraphExporter()
    exporter.to_json(citation_graph.graph, "citation_network.json")
    exporter.to_graphml(citation_graph.graph, "citation_network.graphml")
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger("refnet.visualization")


class GraphExporter:
    """exports NetworkX graphs to various formats."""

    def to_json(
        self,
        graph,
        filepath: Optional[str] = None,
        include_layout: bool = True
    ) -> Dict[str, Any]:
        """
        export graph to JSON format suitable for d3.js visualization.

        format:
        {
            "nodes": [{"id": "...", "label": "...", "x": 0, "y": 0, ...}],
            "edges": [{"source": "...", "target": "...", "weight": 1, ...}],
            "metadata": {"node_count": N, "edge_count": M}
        }
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required for graph export")

        # compute layout if requested
        positions = {}
        if include_layout and graph.number_of_nodes() > 0:
            try:
                if graph.number_of_nodes() < 500:
                    positions = nx.spring_layout(graph, k=2, iterations=50)
                else:
                    # faster layout for large graphs
                    positions = nx.kamada_kawai_layout(graph)
            except Exception:
                pass  # layout failed, skip positions

        # build nodes list
        nodes = []
        for node_id in graph.nodes():
            attrs = dict(graph.nodes[node_id])
            node_data = {
                "id": str(node_id),
                "label": attrs.get("title", str(node_id))[:60],
                **{k: v for k, v in attrs.items() if self._is_json_serializable(v)}
            }

            # add position if available
            if node_id in positions:
                pos = positions[node_id]
                node_data["x"] = float(pos[0])
                node_data["y"] = float(pos[1])

            nodes.append(node_data)

        # build edges list
        edges = []
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                "source": str(source),
                "target": str(target),
                **{k: v for k, v in attrs.items() if self._is_json_serializable(v)}
            }
            edges.append(edge_data)

        result = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "directed": graph.is_directed()
            }
        }

        # write to file if path provided
        if filepath:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"exported JSON to {filepath}")

        return result

    def to_graphml(self, graph, filepath: str):
        """
        export graph to GraphML format (for Gephi, Cytoscape).
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required for graph export")

        # convert non-serializable attributes to strings
        G = graph.copy()
        for node_id in G.nodes():
            for key, value in list(G.nodes[node_id].items()):
                if isinstance(value, (list, dict)):
                    G.nodes[node_id][key] = json.dumps(value)

        for u, v in G.edges():
            for key, value in list(G.edges[u, v].items()):
                if isinstance(value, (list, dict)):
                    G.edges[u, v][key] = json.dumps(value)

        nx.write_graphml(G, filepath)
        logger.info(f"exported GraphML to {filepath}")

    def to_gexf(self, graph, filepath: str):
        """
        export graph to GEXF format (for Gephi).
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required for graph export")

        # convert non-serializable attributes
        G = graph.copy()
        for node_id in G.nodes():
            for key, value in list(G.nodes[node_id].items()):
                if isinstance(value, (list, dict)):
                    G.nodes[node_id][key] = json.dumps(value)
                elif isinstance(value, bool):
                    G.nodes[node_id][key] = str(value).lower()

        for u, v in G.edges():
            for key, value in list(G.edges[u, v].items()):
                if isinstance(value, (list, dict)):
                    G.edges[u, v][key] = json.dumps(value)
                elif isinstance(value, bool):
                    G.edges[u, v][key] = str(value).lower()

        nx.write_gexf(G, filepath)
        logger.info(f"exported GEXF to {filepath}")

    def to_dot(self, graph, filepath: str, max_label_length: int = 30):
        """
        export graph to DOT format (for Graphviz).
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required for graph export")

        try:
            from networkx.drawing.nx_pydot import write_dot

            # create copy with simplified labels for readability
            G = graph.copy()
            mapping = {}
            for node_id in G.nodes():
                label = G.nodes[node_id].get("title", str(node_id))
                short_label = label[:max_label_length]
                if len(label) > max_label_length:
                    short_label += "..."
                G.nodes[node_id]["label"] = short_label

            write_dot(G, filepath)
            logger.info(f"exported DOT to {filepath}")

        except ImportError:
            logger.error("pydot required for DOT export: pip install pydot")
            raise

    def to_cytoscape_json(
        self,
        graph,
        filepath: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        export graph to Cytoscape.js JSON format.

        format:
        {
            "elements": {
                "nodes": [{"data": {"id": "...", ...}}],
                "edges": [{"data": {"source": "...", "target": "...", ...}}]
            }
        }
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx required for graph export")

        nodes = []
        for node_id in graph.nodes():
            attrs = dict(graph.nodes[node_id])
            node_data = {
                "data": {
                    "id": str(node_id),
                    "label": attrs.get("title", str(node_id))[:50],
                    **{k: v for k, v in attrs.items() if self._is_json_serializable(v)}
                }
            }
            nodes.append(node_data)

        edges = []
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                "data": {
                    "id": f"{source}-{target}",
                    "source": str(source),
                    "target": str(target),
                    **{k: v for k, v in attrs.items() if self._is_json_serializable(v)}
                }
            }
            edges.append(edge_data)

        result = {
            "elements": {
                "nodes": nodes,
                "edges": edges
            }
        }

        if filepath:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"exported Cytoscape JSON to {filepath}")

        return result

    def _is_json_serializable(self, value) -> bool:
        """check if value is JSON serializable."""
        return isinstance(value, (str, int, float, bool, type(None), list, dict))
