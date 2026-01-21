"""
html viewer generator - creates interactive cytoscape.js visualization.
"""

from typing import Optional
from pathlib import Path
import json

from ..graph.working_graph import WorkingGraph
from ..analysis.gap import GapAnalysisResult


class HTMLViewer:
    """
    generates standalone HTML file with Cytoscape.js visualization.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        graph: WorkingGraph,
        filename: str = "graph_viewer.html",
        title: str = "Citation Network",
        gap_analysis: Optional[GapAnalysisResult] = None
    ) -> str:
        """generate interactive HTML viewer."""

        # prepare data
        elements = self._prepare_elements(graph)
        stats = graph.stats()
        clusters = list(graph.clusters.values())

        gap_data = None
        if gap_analysis:
            gap_data = {
                "bridges": len(gap_analysis.bridges),
                "missing_links": len(gap_analysis.missing_links),
                "unexplored": len(gap_analysis.unexplored_clusters),
                "summary": gap_analysis.summary
            }

        html = self._generate_html(elements, stats, clusters, gap_data, title)

        path = self.output_dir / filename
        with open(path, 'w') as f:
            f.write(html)

        return str(path)

    def _prepare_elements(self, graph: WorkingGraph) -> dict:
        """prepare cytoscape elements."""
        nodes = []
        edges = []

        # nodes
        for node_id, node in graph.nodes.items():
            paper = graph.papers.get(node_id)
            if paper:
                cluster = graph.node_cluster_map.get(node_id, "other")

                # get authors for hover display
                authors_str = "Unknown"
                if paper.authors:
                    authors_str = ", ".join(paper.authors[:3])
                    if len(paper.authors) > 3:
                        authors_str += f" +{len(paper.authors) - 3} more"

                nodes.append({
                    "data": {
                        "id": node_id,
                        "label": self._truncate(paper.title, 30),
                        "title": paper.title,
                        "authors": authors_str,
                        "year": paper.year or 0,
                        "citations": paper.citation_count or 0,
                        "relevance": round(paper.relevance_score, 2),
                        "cluster": cluster,
                        "is_seed": node.is_seed,
                        "is_review": paper.is_review,
                        "doi": paper.doi or "",
                        "node_type": "paper"
                    }
                })
            else:
                # check if it's an author node
                author = graph.authors.get(node_id)
                if author:
                    nodes.append({
                        "data": {
                            "id": node_id,
                            "label": self._truncate(author.name, 20) if author.name else "Author",
                            "title": author.name or "Unknown Author",
                            "authors": "",
                            "year": 0,
                            "citations": author.paper_count or 0,
                            "relevance": 0.5,
                            "cluster": "authors",
                            "is_seed": False,
                            "is_review": False,
                            "doi": "",
                            "node_type": "author"
                        }
                    })

        # edges
        for edge in graph.edges.values():
            edges.append({
                "data": {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type.value,
                    "weight": edge.weight
                }
            })

        return {"nodes": nodes, "edges": edges}

    def _generate_html(
        self,
        elements: dict,
        stats: dict,
        clusters: list,
        gap_data: Optional[dict],
        title: str
    ) -> str:
        """generate complete HTML."""

        elements_json = json.dumps(elements)
        stats_json = json.dumps(stats)
        clusters_json = json.dumps([{"id": c.id, "name": c.name, "size": c.size} for c in clusters])

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-cola/2.5.1/cytoscape-cola.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}

        #container {{ display: flex; height: 100vh; }}

        #sidebar {{
            width: 300px;
            background: #1a1a2e;
            color: white;
            padding: 20px;
            overflow-y: auto;
        }}

        #cy {{
            flex: 1;
            background: #0f0f1a;
        }}

        h1 {{ font-size: 18px; margin-bottom: 20px; color: #4ecdc4; }}
        h2 {{ font-size: 14px; margin: 15px 0 10px; color: #888; text-transform: uppercase; }}

        .stat {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #333; }}
        .stat-value {{ color: #4ecdc4; }}

        .controls {{ margin: 20px 0; }}
        .btn {{
            background: #4ecdc4;
            color: #1a1a2e;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            margin: 2px;
            font-size: 12px;
        }}
        .btn:hover {{ background: #3dbdb5; }}
        .btn.active {{ background: #ff6b6b; }}

        #filters {{ margin: 15px 0; }}
        .filter-group {{ margin: 10px 0; }}
        label {{ display: block; font-size: 12px; color: #888; margin-bottom: 5px; }}
        select, input[type="range"] {{ width: 100%; padding: 5px; background: #2a2a4a; color: white; border: 1px solid #444; border-radius: 4px; }}

        #node-info {{
            background: #2a2a4a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }}
        #node-info h3 {{ color: #4ecdc4; margin-bottom: 10px; font-size: 14px; }}
        #node-info p {{ font-size: 12px; color: #ccc; margin: 5px 0; }}
        #node-info a {{ color: #4ecdc4; }}

        .legend {{ margin: 20px 0; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; font-size: 12px; }}
        .legend-color {{ width: 15px; height: 15px; border-radius: 50%; margin-right: 10px; }}

        .cluster-list {{ max-height: 150px; overflow-y: auto; }}
        .cluster-item {{
            padding: 5px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
        }}
        .cluster-item:hover {{ background: #2a2a4a; }}

        #gap-summary {{
            background: #2a2a4a;
            padding: 10px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 12px;
            color: #ccc;
        }}

        /* hover tooltip */
        #tooltip {{
            position: absolute;
            display: none;
            background: rgba(26, 26, 46, 0.95);
            border: 1px solid #4ecdc4;
            border-radius: 8px;
            padding: 12px 15px;
            max-width: 350px;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }}
        #tooltip .tooltip-title {{
            font-size: 13px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        #tooltip .tooltip-authors {{
            font-size: 11px;
            color: #4ecdc4;
            margin-bottom: 4px;
            font-style: italic;
        }}
        #tooltip .tooltip-meta {{
            font-size: 11px;
            color: #888;
        }}
        #tooltip .tooltip-meta span {{
            margin-right: 10px;
        }}
        #tooltip .seed-badge {{
            display: inline-block;
            background: #ff6b6b;
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 5px;
        }}
        #tooltip .review-badge {{
            display: inline-block;
            background: #f1c40f;
            color: #1a1a2e;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 5px;
        }}
    </style>
</head>
<body>
    <!-- hover tooltip -->
    <div id="tooltip">
        <div class="tooltip-title"></div>
        <div class="tooltip-authors"></div>
        <div class="tooltip-meta"></div>
    </div>

    <div id="container">
        <div id="sidebar">
            <h1>📚 {title}</h1>

            <h2>Statistics</h2>
            <div id="stats"></div>

            <h2>View Controls</h2>
            <div class="controls">
                <button class="btn" onclick="resetView()">Reset View</button>
                <button class="btn" onclick="toggleLabels()">Toggle Labels</button>
                <button class="btn" onclick="highlightSeeds()">Show Seeds</button>
                <button class="btn" onclick="runLayout()">Re-layout</button>
            </div>

            <h2>Filters</h2>
            <div id="filters">
                <div class="filter-group">
                    <label>Edge Type</label>
                    <select id="edgeFilter" onchange="applyFilters()">
                        <option value="all">All Edges</option>
                        <option value="cites">Citations Only</option>
                        <option value="intro_hint_cites">Intro Hints</option>
                        <option value="authored_by">Author Links</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Min Citations: <span id="citVal">0</span></label>
                    <input type="range" id="citFilter" min="0" max="1000" value="0" onchange="applyFilters()">
                </div>
            </div>

            <h2>Clusters</h2>
            <div class="cluster-list" id="clusters"></div>

            {"<h2>Gap Analysis</h2><div id='gap-summary'>" + (gap_data['summary'] if gap_data else "") + "</div>" if gap_data else ""}

            <h2>Legend</h2>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #ff6b6b;"></div> Seed Paper</div>
                <div class="legend-item"><div class="legend-color" style="background: #4ecdc4;"></div> Regular Paper</div>
                <div class="legend-item"><div class="legend-color" style="background: #f1c40f;"></div> Review Paper</div>
            </div>

            <div id="node-info">
                <h3 id="info-title"></h3>
                <p id="info-year"></p>
                <p id="info-citations"></p>
                <p id="info-doi"></p>
            </div>
        </div>

        <div id="cy"></div>
    </div>

    <script>
        const elements = {elements_json};
        const stats = {stats_json};
        const clusters = {clusters_json};

        // populate stats
        const statsDiv = document.getElementById('stats');
        statsDiv.innerHTML = Object.entries(stats)
            .filter(([k, v]) => typeof v === 'number' || typeof v === 'string')
            .map(([k, v]) => `<div class="stat"><span>${{k}}</span><span class="stat-value">${{v}}</span></div>`)
            .join('');

        // populate clusters
        const clustersDiv = document.getElementById('clusters');
        clustersDiv.innerHTML = clusters.map(c =>
            `<div class="cluster-item" onclick="highlightCluster('${{c.id}}')">${{c.name}} (${{c.size}})</div>`
        ).join('');

        // initialize cytoscape
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: [...elements.nodes, ...elements.edges],
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': ele => ele.data('is_seed') ? '#ff6b6b' : (ele.data('is_review') ? '#f1c40f' : '#4ecdc4'),
                        'label': 'data(label)',
                        'width': ele => Math.max(20, Math.sqrt(ele.data('citations') || 1) * 3),
                        'height': ele => Math.max(20, Math.sqrt(ele.data('citations') || 1) * 3),
                        'font-size': '8px',
                        'color': '#fff',
                        'text-valign': 'bottom',
                        'text-margin-y': 5,
                        'text-outline-color': '#0f0f1a',
                        'text-outline-width': 2
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': ele => Math.max(1, ele.data('weight') * 0.5),
                        'line-color': ele => {{
                            const type = ele.data('type');
                            if (type === 'intro_hint_cites') return '#e67e22';
                            if (type === 'intro_cites') return '#e74c3c';
                            if (type.includes('author')) return '#3498db';
                            return '#555';
                        }},
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'target-arrow-color': '#555',
                        'opacity': 0.6
                    }}
                }},
                {{
                    selector: ':selected',
                    style: {{
                        'border-width': 3,
                        'border-color': '#fff'
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#e74c3c',
                        'border-width': 3,
                        'border-color': '#fff'
                    }}
                }},
                {{
                    selector: '.dimmed',
                    style: {{
                        'opacity': 0.2
                    }}
                }}
            ],
            layout: {{
                name: 'cose',
                animate: false,
                nodeDimensionsIncludeLabels: true,
                nodeRepulsion: 8000,
                idealEdgeLength: 100
            }}
        }});

        // show labels toggle
        let labelsVisible = true;
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            cy.style().selector('node').style('label', labelsVisible ? 'data(label)' : '').update();
        }}

        // reset view
        function resetView() {{
            cy.fit();
            cy.elements().removeClass('highlighted dimmed');
        }}

        // highlight seeds
        function highlightSeeds() {{
            cy.elements().removeClass('highlighted dimmed');
            const seeds = cy.nodes().filter(n => n.data('is_seed'));
            cy.elements().addClass('dimmed');
            seeds.removeClass('dimmed').addClass('highlighted');
            seeds.neighborhood().removeClass('dimmed');
        }}

        // highlight cluster
        function highlightCluster(clusterId) {{
            cy.elements().removeClass('highlighted dimmed');
            const inCluster = cy.nodes().filter(n => n.data('cluster') === clusterId);
            cy.elements().addClass('dimmed');
            inCluster.removeClass('dimmed').addClass('highlighted');
            inCluster.edgesWith(inCluster).removeClass('dimmed');
        }}

        // run layout
        function runLayout() {{
            cy.layout({{ name: 'cose', animate: true }}).run();
        }}

        // apply filters
        function applyFilters() {{
            const edgeType = document.getElementById('edgeFilter').value;
            const minCit = parseInt(document.getElementById('citFilter').value);
            document.getElementById('citVal').textContent = minCit;

            // filter edges
            cy.edges().forEach(e => {{
                let show = true;
                if (edgeType !== 'all' && e.data('type') !== edgeType) show = false;
                e.style('display', show ? 'element' : 'none');
            }});

            // filter nodes
            cy.nodes().forEach(n => {{
                let show = true;
                if ((n.data('citations') || 0) < minCit) show = false;
                n.style('display', show ? 'element' : 'none');
            }});
        }}

        // node click handler
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();

            document.getElementById('node-info').style.display = 'block';
            document.getElementById('info-title').textContent = data.title || 'Unknown';
            document.getElementById('info-year').textContent = 'Year: ' + (data.year || 'N/A');
            document.getElementById('info-citations').textContent = 'Citations: ' + (data.citations || 0);
            document.getElementById('info-doi').innerHTML = data.doi ?
                `<a href="https://doi.org/${{data.doi}}" target="_blank">DOI: ${{data.doi}}</a>` : '';
        }});

        // background click
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                document.getElementById('node-info').style.display = 'none';
                cy.elements().removeClass('highlighted dimmed');
            }}
        }});

        // tooltip element
        const tooltip = document.getElementById('tooltip');

        // node hover - show tooltip
        cy.on('mouseover', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();

            // build tooltip content
            let titleHtml = data.title || 'Unknown';
            if (data.is_seed) {{
                titleHtml += '<span class="seed-badge">SEED</span>';
            }}
            if (data.is_review) {{
                titleHtml += '<span class="review-badge">REVIEW</span>';
            }}

            tooltip.querySelector('.tooltip-title').innerHTML = titleHtml;
            tooltip.querySelector('.tooltip-authors').textContent = data.authors || 'Unknown authors';

            let metaHtml = '';
            if (data.year) metaHtml += `<span>📅 ${{data.year}}</span>`;
            if (data.citations) metaHtml += `<span>📚 ${{data.citations}} citations</span>`;
            if (data.relevance) metaHtml += `<span>⭐ ${{data.relevance}} rel</span>`;
            tooltip.querySelector('.tooltip-meta').innerHTML = metaHtml;

            tooltip.style.display = 'block';
        }});

        // node hover - hide tooltip
        cy.on('mouseout', 'node', function() {{
            tooltip.style.display = 'none';
        }});

        // track mouse position for tooltip
        document.getElementById('cy').addEventListener('mousemove', function(evt) {{
            if (tooltip.style.display === 'block') {{
                // position tooltip near cursor but ensure it stays on screen
                let x = evt.clientX + 15;
                let y = evt.clientY + 15;

                // prevent tooltip from going off right edge
                if (x + tooltip.offsetWidth > window.innerWidth) {{
                    x = evt.clientX - tooltip.offsetWidth - 15;
                }}
                // prevent tooltip from going off bottom edge
                if (y + tooltip.offsetHeight > window.innerHeight) {{
                    y = evt.clientY - tooltip.offsetHeight - 15;
                }}

                tooltip.style.left = x + 'px';
                tooltip.style.top = y + 'px';
            }}
        }});
    </script>
</body>
</html>'''

    def _truncate(self, text: str, length: int) -> str:
        """truncate text with ellipsis."""
        if not text:
            return ""
        if len(text) <= length:
            return text
        return text[:length-3] + "..."
