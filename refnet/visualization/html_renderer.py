"""
html renderer - generates standalone HTML visualizations.

creates interactive force-directed graphs using d3.js (CDN loaded).

usage:
    from refnet.visualization import GraphBuilder, HTMLRenderer

    builder = GraphBuilder(analysis)
    citation_graph = builder.build_citation_graph()

    renderer = HTMLRenderer()
    renderer.render_citation_graph(citation_graph, "citation_network.html")
"""

import json
import logging
from typing import Optional
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .graph_builder import CitationGraph, AuthorGraph, TopicGraph

logger = logging.getLogger("refnet.visualization")


# d3.js force-directed graph template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f8f9fa;
        }}
        #header {{
            background: #2c3e50;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #header h1 {{
            margin: 0;
            font-size: 20px;
            font-weight: 500;
        }}
        #stats {{
            font-size: 14px;
            opacity: 0.8;
        }}
        #container {{
            display: flex;
            height: calc(100vh - 60px);
        }}
        #graph {{
            flex: 1;
            background: white;
            border-right: 1px solid #ddd;
        }}
        #sidebar {{
            width: 300px;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }}
        #sidebar h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        #node-info {{
            display: none;
        }}
        #node-info.active {{
            display: block;
        }}
        #node-info h4 {{
            margin: 0 0 10px 0;
            word-wrap: break-word;
        }}
        #node-info p {{
            margin: 5px 0;
            font-size: 14px;
            color: #555;
        }}
        .node {{
            cursor: pointer;
        }}
        .node:hover {{
            stroke: #000;
            stroke-width: 2px;
        }}
        .link {{
            stroke-opacity: 0.4;
        }}
        .link:hover {{
            stroke-opacity: 1;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            max-width: 300px;
            word-wrap: break-word;
        }}
        #legend {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 13px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        #controls {{
            margin-bottom: 20px;
        }}
        #controls label {{
            display: block;
            margin: 10px 0 5px 0;
            font-size: 13px;
            color: #555;
        }}
        #controls input[type="range"] {{
            width: 100%;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <div id="stats">{node_count} nodes, {edge_count} edges</div>
    </div>
    <div id="container">
        <div id="graph"></div>
        <div id="sidebar">
            <div id="controls">
                <h3>Controls</h3>
                <label>Link Distance</label>
                <input type="range" id="distance" min="30" max="200" value="80">
                <label>Charge Strength</label>
                <input type="range" id="charge" min="-500" max="-50" value="-200">
            </div>
            <div id="node-info">
                <h3>Selected Node</h3>
                <h4 id="info-title"></h4>
                <div id="info-details"></div>
            </div>
            <div id="legend">
                <h3>Legend</h3>
                {legend_items}
            </div>
        </div>
    </div>
    <div class="tooltip" style="display: none;"></div>

    <script>
    const graphData = {graph_data};
    const config = {config};

    // set up SVG
    const container = document.getElementById('graph');
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select('#graph')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // zoom behavior
    const g = svg.append('g');
    svg.call(d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => g.attr('transform', event.transform)));

    // color scale
    const color = d3.scaleOrdinal()
        .domain(config.categories)
        .range(config.colors);

    // size scale for nodes
    const sizeScale = d3.scaleSqrt()
        .domain([0, d3.max(graphData.nodes, d => d[config.sizeField] || 1)])
        .range([5, 25]);

    // simulation
    const simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.edges)
            .id(d => d.id)
            .distance(80))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => sizeScale(d[config.sizeField] || 1) + 2));

    // links
    const link = g.append('g')
        .selectAll('line')
        .data(graphData.edges)
        .join('line')
        .attr('class', 'link')
        .attr('stroke', '#999')
        .attr('stroke-width', d => Math.sqrt(d.weight || 1));

    // nodes
    const node = g.append('g')
        .selectAll('circle')
        .data(graphData.nodes)
        .join('circle')
        .attr('class', 'node')
        .attr('r', d => sizeScale(d[config.sizeField] || 1))
        .attr('fill', d => color(d[config.colorField] || 'default'))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .call(drag(simulation));

    // tooltip
    const tooltip = d3.select('.tooltip');

    node.on('mouseover', (event, d) => {{
        tooltip.style('display', 'block')
            .html(d.label || d.id)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    }})
    .on('mouseout', () => tooltip.style('display', 'none'))
    .on('click', (event, d) => showNodeInfo(d));

    // update positions on tick
    simulation.on('tick', () => {{
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    }});

    // drag behavior
    function drag(simulation) {{
        return d3.drag()
            .on('start', (event) => {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }})
            .on('drag', (event) => {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }})
            .on('end', (event) => {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }});
    }}

    // show node info
    function showNodeInfo(d) {{
        document.getElementById('node-info').classList.add('active');
        document.getElementById('info-title').textContent = d.label || d.id;

        let details = '';
        for (const [key, value] of Object.entries(d)) {{
            if (['id', 'x', 'y', 'vx', 'vy', 'fx', 'fy', 'index', 'label'].includes(key)) continue;
            if (value === null || value === undefined) continue;
            details += `<p><strong>${{key}}:</strong> ${{value}}</p>`;
        }}
        document.getElementById('info-details').innerHTML = details;
    }}

    // controls
    document.getElementById('distance').addEventListener('input', (e) => {{
        simulation.force('link').distance(+e.target.value);
        simulation.alpha(0.3).restart();
    }});

    document.getElementById('charge').addEventListener('input', (e) => {{
        simulation.force('charge').strength(+e.target.value);
        simulation.alpha(0.3).restart();
    }});
    </script>
</body>
</html>'''


class HTMLRenderer:
    """renders interactive HTML visualizations using d3.js."""

    def render_citation_graph(
        self,
        citation_graph: CitationGraph,
        filepath: str,
        title: str = "Citation Network"
    ):
        """
        render citation graph as interactive HTML.

        nodes colored by: year
        node size by: citation count
        """
        from .exporter import GraphExporter
        exporter = GraphExporter()

        # export to JSON format
        graph_data = exporter.to_json(citation_graph.graph, include_layout=False)

        # add year-based coloring
        for node in graph_data["nodes"]:
            year = node.get("year", 0)
            if year >= 2020:
                node["era"] = "recent (2020+)"
            elif year >= 2010:
                node["era"] = "modern (2010-2019)"
            elif year >= 2000:
                node["era"] = "mature (2000-2009)"
            else:
                node["era"] = "classic (<2000)"

            # mark seed
            if node.get("is_seed"):
                node["era"] = "seed"

        config = {
            "categories": ["seed", "recent (2020+)", "modern (2010-2019)", "mature (2000-2009)", "classic (<2000)"],
            "colors": ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#95a5a6"],
            "sizeField": "citations",
            "colorField": "era"
        }

        legend_items = self._build_legend(config["categories"], config["colors"])

        html = HTML_TEMPLATE.format(
            title=title,
            node_count=citation_graph.paper_count,
            edge_count=citation_graph.edge_count,
            graph_data=json.dumps(graph_data),
            config=json.dumps(config),
            legend_items=legend_items
        )

        with open(filepath, "w") as f:
            f.write(html)
        logger.info(f"rendered citation graph to {filepath}")

    def render_author_graph(
        self,
        author_graph: AuthorGraph,
        filepath: str,
        title: str = "Author Collaboration Network"
    ):
        """
        render author collaboration graph as interactive HTML.

        nodes colored by: key author status
        node size by: paper count
        """
        from .exporter import GraphExporter
        exporter = GraphExporter()

        graph_data = exporter.to_json(author_graph.graph, include_layout=False)

        # add author type coloring
        for node in graph_data["nodes"]:
            if node.get("is_key_author"):
                node["type"] = "key author"
            else:
                node["type"] = "collaborator"

        config = {
            "categories": ["key author", "collaborator"],
            "colors": ["#e74c3c", "#3498db"],
            "sizeField": "paper_count",
            "colorField": "type"
        }

        legend_items = self._build_legend(config["categories"], config["colors"])

        html = HTML_TEMPLATE.format(
            title=title,
            node_count=author_graph.author_count,
            edge_count=author_graph.edge_count,
            graph_data=json.dumps(graph_data),
            config=json.dumps(config),
            legend_items=legend_items
        )

        with open(filepath, "w") as f:
            f.write(html)
        logger.info(f"rendered author graph to {filepath}")

    def render_topic_graph(
        self,
        topic_graph: TopicGraph,
        filepath: str,
        title: str = "Topic Co-occurrence Network"
    ):
        """
        render topic co-occurrence graph as interactive HTML.

        nodes colored by: core/emerging/other
        node size by: paper count
        """
        from .exporter import GraphExporter
        exporter = GraphExporter()

        graph_data = exporter.to_json(topic_graph.graph, include_layout=False)

        # add topic type coloring
        for node in graph_data["nodes"]:
            if node.get("is_core"):
                node["type"] = "core topic"
            elif node.get("is_emerging"):
                node["type"] = "emerging topic"
            else:
                node["type"] = "other topic"

        config = {
            "categories": ["core topic", "emerging topic", "other topic"],
            "colors": ["#e74c3c", "#2ecc71", "#95a5a6"],
            "sizeField": "paper_count",
            "colorField": "type"
        }

        legend_items = self._build_legend(config["categories"], config["colors"])

        html = HTML_TEMPLATE.format(
            title=title,
            node_count=topic_graph.topic_count,
            edge_count=topic_graph.edge_count,
            graph_data=json.dumps(graph_data),
            config=json.dumps(config),
            legend_items=legend_items
        )

        with open(filepath, "w") as f:
            f.write(html)
        logger.info(f"rendered topic graph to {filepath}")

    def _build_legend(self, categories: list, colors: list) -> str:
        """build HTML for legend items."""
        items = []
        for cat, color in zip(categories, colors):
            items.append(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background: {color}"></div>'
                f'{cat}</div>'
            )
        return "\n".join(items)
