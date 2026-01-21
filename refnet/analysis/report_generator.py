"""
report generator - extract landscape, leaders, gaps, topic flow from citation DB.
uses natural breakpoints instead of arbitrary limits.
designed for literature review workflow.
"""

import json
import sqlite3
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import logging

logger = logging.getLogger("refnet.report")


@dataclass
class TopicalPaper:
    """paper with computed topical relevance to seeds."""
    id: str
    title: str
    year: Optional[int]
    citation_count: int
    depth: int
    topical_score: float  # concept overlap with seeds
    connectivity: int  # edges to other topical papers
    concepts: List[str]
    is_seed: bool = False
    is_hub: bool = False  # methodology hub vs topical paper


@dataclass
class Leader:
    """research leader with topical significance."""
    name: str
    paper_count: int
    topical_papers: int  # papers actually on-topic
    total_citations: int
    topical_citations: int  # citations from topical papers only
    first_year: int
    last_year: int
    active_years: int
    significance_score: float  # combines all factors


@dataclass
class ConceptFlow:
    """concept evolution over time."""
    concept: str
    appearances_by_year: Dict[int, int]
    peak_year: int
    trend: str  # "emerging", "stable", "declining"
    total_papers: int


@dataclass
class LiteratureReport:
    """complete literature review report."""
    # landscape
    topical_papers: List[TopicalPaper]
    clusters: Dict[str, List[str]]  # cluster_name -> paper_ids
    methodology_hubs: List[TopicalPaper]  # identified hubs to exclude/note

    # leaders
    leaders: List[Leader]
    breakpoint_info: Dict[str, Any]  # how breakpoints were computed

    # flow
    concept_flow: List[ConceptFlow]
    emerging_concepts: List[str]
    declining_concepts: List[str]

    # gaps
    sparse_connections: List[Tuple[str, str]]  # cluster pairs with few links
    suggested_reading: List[str]  # paper_ids that would fill gaps

    # metadata
    seed_count: int
    total_papers_analyzed: int
    natural_breakpoints: Dict[str, float]


class ReportGenerator:
    """
    generates literature review reports from citation database.

    key design: uses natural breakpoints computed from data distribution,
    not arbitrary limits like "top 10" or "top 50".
    """

    # concepts that indicate methodology papers, not topical papers
    METHODOLOGY_CONCEPTS = {
        'deep learning', 'machine learning', 'neural network', 'protein structure prediction',
        'sequence alignment', 'database', 'algorithm', 'software', 'web server',
        'bioinformatics', 'computational biology', 'statistics', 'data analysis',
        'computer science', 'artificial intelligence', 'language model', 'source code',
        'cluster analysis', 'inference', 'encoding (memory)', 'alphabet'
    }

    # generic concepts that are too broad (low weight)
    GENERIC_CONCEPTS = {
        'biology', 'chemistry', 'biochemistry', 'genetics', 'medicine',
        'cell biology', 'molecular biology', 'sequence (biology)', 'gene', 'protein'
    }

    # minimum concept score to consider (filters out low-confidence concepts)
    MIN_CONCEPT_SCORE = 0.5

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"database not found: {db_path}")

        # cache seed concepts for topical scoring
        self._seed_concepts: Set[str] = set()
        self._load_seed_concepts()

    def _load_seed_concepts(self):
        """load concepts from seed papers to define topic.

        filters by:
        - concept score >= MIN_CONCEPT_SCORE (to get high-confidence concepts)
        - excludes methodology concepts (to focus on topic, not methods)
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT concepts_json FROM paper_candidates
            WHERE status = 'seed'
        """)

        # track concept weights across all seeds
        concept_weights: Dict[str, float] = defaultdict(float)

        for row in c.fetchall():
            if row[0]:
                try:
                    concepts = json.loads(row[0])
                    for concept in concepts:
                        name = concept.get('name', '').lower()
                        score = concept.get('score', 0.0)

                        # only include high-confidence, non-methodology concepts
                        if (name and
                            score >= self.MIN_CONCEPT_SCORE and
                            name not in self.METHODOLOGY_CONCEPTS):
                            concept_weights[name] += score
                except (json.JSONDecodeError, TypeError):
                    pass

        conn.close()

        # use concepts that appear in multiple seeds or have high combined weight
        self._seed_concepts = {
            name for name, weight in concept_weights.items()
            if weight >= 0.5  # at least 0.5 combined weight
        }

        logger.info(f"loaded {len(self._seed_concepts)} seed concepts: {sorted(self._seed_concepts)[:10]}...")

    def compute_topical_score(self, concepts_json: str) -> Tuple[float, bool]:
        """
        compute topical relevance score based on concept overlap with seeds.
        returns (score, is_methodology_hub).

        scoring strategy:
        - specific concepts (like "aminoacyl trna synthetase") get full weight
        - generic concepts (like "biology") get reduced weight (0.2x)
        - normalized by number of CORE seed concepts (excluding generic)
        """
        if not concepts_json or not self._seed_concepts:
            return 0.0, False

        try:
            concepts = json.loads(concepts_json)
            methodology_weight = 0.0
            topical_weight = 0.0
            has_specific_topical = False

            for c in concepts:
                name = c.get('name', '').lower()
                score = c.get('score', 0.5)

                # track methodology concept weight
                if name in self.METHODOLOGY_CONCEPTS:
                    methodology_weight += score

                # track topical concept weight (matches seed concepts)
                if name in self._seed_concepts:
                    if name in self.GENERIC_CONCEPTS:
                        # generic concepts get reduced weight
                        topical_weight += score * 0.2
                    else:
                        # specific concepts get full weight
                        topical_weight += score
                        has_specific_topical = True

            # count core (non-generic) seed concepts for normalization
            core_seed_count = sum(
                1 for c in self._seed_concepts
                if c not in self.GENERIC_CONCEPTS
            )
            core_seed_count = max(core_seed_count, 1)  # avoid division by zero

            # normalize by core seed concepts, cap at 1.0
            topical_score = min(topical_weight / core_seed_count, 1.0)

            # is this a methodology hub?
            # high methodology weight AND no specific topical overlap
            is_hub = methodology_weight > 1.5 and not has_specific_topical

            return topical_score, is_hub

        except (json.JSONDecodeError, TypeError):
            return 0.0, False

    def find_natural_breakpoint(
        self,
        values: List[float],
        method: str = "elbow"
    ) -> Tuple[float, int]:
        """
        find natural breakpoint in a sorted distribution.

        methods:
        - elbow: find knee in sorted values (where derivative changes most)
        - std: mean + 1 standard deviation
        - gap: largest gap between consecutive values

        returns (threshold_value, index_at_threshold)
        """
        if not values:
            return 0.0, 0

        sorted_vals = sorted(values, reverse=True)
        n = len(sorted_vals)

        if method == "std":
            mean = sum(values) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
            threshold = mean + std
            idx = sum(1 for v in sorted_vals if v >= threshold)
            return threshold, idx

        elif method == "gap":
            # find largest gap
            max_gap = 0
            gap_idx = 0
            for i in range(1, min(n, 100)):  # check first 100
                gap = sorted_vals[i-1] - sorted_vals[i]
                if gap > max_gap:
                    max_gap = gap
                    gap_idx = i
            return sorted_vals[gap_idx] if gap_idx < n else 0.0, gap_idx

        else:  # elbow method
            # normalize to 0-1 range
            if n < 3:
                return sorted_vals[0], 1

            max_v, min_v = sorted_vals[0], sorted_vals[-1]
            if max_v == min_v:
                return max_v, n

            # find point of maximum curvature
            # using second derivative approximation
            max_curve = 0
            elbow_idx = 1

            for i in range(1, min(n - 1, 100)):
                # second derivative: f''(i) ≈ f(i+1) - 2f(i) + f(i-1)
                second_deriv = abs(
                    sorted_vals[i+1] - 2 * sorted_vals[i] + sorted_vals[i-1]
                )
                if second_deriv > max_curve:
                    max_curve = second_deriv
                    elbow_idx = i

            return sorted_vals[elbow_idx], elbow_idx

    def generate_landscape(self) -> Tuple[List[TopicalPaper], List[TopicalPaper], Dict[str, List[str]]]:
        """
        generate landscape: topical papers, methodology hubs, clusters.
        uses topical scoring to separate signal from noise.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # get all papers with concepts
        c.execute("""
            SELECT id, title, year, citation_count, depth, status, concepts_json
            FROM paper_candidates
            WHERE concepts_json IS NOT NULL AND concepts_json != '[]'
        """)

        papers = []
        hubs = []

        for row in c.fetchall():
            paper_id, title, year, cites, depth, status, concepts_json = row

            topical_score, is_hub = self.compute_topical_score(concepts_json)

            # extract concept names
            try:
                concepts = json.loads(concepts_json)
                concept_names = [c.get('name', '') for c in concepts[:5]]
            except:
                concept_names = []

            paper = TopicalPaper(
                id=paper_id,
                title=title or "",
                year=year,
                citation_count=cites or 0,
                depth=depth or 99,
                topical_score=topical_score,
                connectivity=0,  # compute later
                concepts=concept_names,
                is_seed=(status == 'seed'),
                is_hub=is_hub
            )

            if is_hub:
                hubs.append(paper)
            else:
                papers.append(paper)

        conn.close()

        # find natural breakpoint for topical papers using gap method
        # (elbow can be too aggressive when scores are clustered)
        topical_scores = [p.topical_score for p in papers if p.topical_score > 0]

        if topical_scores:
            # try multiple methods and use the one that gives reasonable coverage
            elbow_thresh, elbow_count = self.find_natural_breakpoint(topical_scores, method="elbow")
            gap_thresh, gap_count = self.find_natural_breakpoint(topical_scores, method="gap")
            std_thresh, std_count = self.find_natural_breakpoint(topical_scores, method="std")

            # prefer the method that gives reasonable coverage
            candidates = [
                (elbow_thresh, elbow_count, "elbow"),
                (gap_thresh, gap_count, "gap"),
                (std_thresh, std_count, "std")
            ]

            # filter out methods that give fewer than 20 papers (too strict)
            valid_candidates = [c for c in candidates if c[1] >= 20]

            if valid_candidates:
                # among valid methods, prefer one closest to target range (50-500)
                # use geometric mean of range (158) as target
                best = min(valid_candidates, key=lambda x: abs(x[1] - 158))
            else:
                # all methods too strict, use the one with most papers
                best = max(candidates, key=lambda x: x[1])

            threshold = best[0]

            # absolute minimum threshold to filter noise
            threshold = max(threshold, 0.01)

            logger.info(f"threshold selection: elbow={elbow_thresh:.3f}({elbow_count}), "
                       f"gap={gap_thresh:.3f}({gap_count}), std={std_thresh:.3f}({std_count}) -> using {best[2]}")
        else:
            threshold = 0.0

        # filter to papers above threshold (or seeds)
        topical_papers = [
            p for p in papers
            if p.topical_score >= threshold or p.is_seed
        ]

        # sort by topical score, then citations
        topical_papers.sort(key=lambda p: (p.topical_score, p.citation_count), reverse=True)

        # cluster by primary concept
        clusters = defaultdict(list)
        for p in topical_papers:
            if p.concepts:
                clusters[p.concepts[0]].append(p.id)

        logger.info(f"landscape: {len(topical_papers)} topical papers, {len(hubs)} methodology hubs, threshold={threshold:.3f}")

        return topical_papers, hubs, dict(clusters)

    def generate_leaders(self) -> Tuple[List[Leader], Dict[str, Any]]:
        """
        identify research leaders by topical contribution.
        uses natural breakpoints to determine significance.

        note: requires author data in database. candidate pools typically
        don't store authors - use Garden database or fetch from API.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # check if we have author data
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='authors'")
        has_authors_table = c.fetchone() is not None

        # also check for extra_json with author data
        c.execute("SELECT COUNT(*) FROM paper_candidates WHERE extra_json IS NOT NULL AND extra_json != '{}'")
        has_extra_json = (c.fetchone()[0] or 0) > 0

        if not has_authors_table and not has_extra_json:
            logger.warning("no author data available in database - leaders analysis requires "
                          "Garden database or API fetch to populate author information")
            conn.close()
            return [], {"method": "none", "threshold": 0, "reason": "no_author_data"}

        # first, get topical paper ids (recompute quickly)
        c.execute("""
            SELECT id, concepts_json, citation_count, year
            FROM paper_candidates
            WHERE concepts_json IS NOT NULL
        """)

        topical_paper_ids = set()
        paper_cites = {}
        paper_years = {}

        for row in c.fetchall():
            paper_id, concepts_json, cites, year = row
            score, is_hub = self.compute_topical_score(concepts_json)
            if score > 0.05 and not is_hub:  # minimal topical relevance
                topical_paper_ids.add(paper_id)
                paper_cites[paper_id] = cites or 0
                paper_years[paper_id] = year

        author_stats = defaultdict(lambda: {
            'papers': [], 'topical_papers': [], 'citations': 0,
            'topical_citations': 0, 'years': []
        })

        if has_extra_json:
            # try to extract from extra_json
            c.execute("""
                SELECT id, extra_json FROM paper_candidates
                WHERE extra_json IS NOT NULL AND extra_json != '{}'
            """)

            for row in c.fetchall():
                paper_id, extra_json = row
                if extra_json:
                    try:
                        extra = json.loads(extra_json)
                        authors = extra.get('authors', [])
                        for author in authors[:3]:  # first 3 authors
                            name = author if isinstance(author, str) else author.get('name', '')
                            if name:
                                author_stats[name]['papers'].append(paper_id)
                                if paper_id in topical_paper_ids:
                                    author_stats[name]['topical_papers'].append(paper_id)
                                    author_stats[name]['topical_citations'] += paper_cites.get(paper_id, 0)
                                author_stats[name]['citations'] += paper_cites.get(paper_id, 0)
                                if paper_years.get(paper_id):
                                    author_stats[name]['years'].append(paper_years[paper_id])
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass

        elif has_authors_table:
            # use authors table if available (Garden database)
            # query author-paper relationships
            c.execute("""
                SELECT a.name, p.id, p.citation_count, p.year
                FROM authors a
                JOIN edges e ON e.source_id = a.id OR e.target_id = a.id
                JOIN paper_candidates p ON p.id = e.source_id OR p.id = e.target_id
                WHERE e.edge_type IN ('authored', 'authored_by')
            """)

            for row in c.fetchall():
                name, paper_id, cites, year = row
                if name:
                    author_stats[name]['papers'].append(paper_id)
                    if paper_id in topical_paper_ids:
                        author_stats[name]['topical_papers'].append(paper_id)
                        author_stats[name]['topical_citations'] += cites or 0
                    author_stats[name]['citations'] += cites or 0
                    if year:
                        author_stats[name]['years'].append(year)

        conn.close()

        # compute leader scores
        leaders = []
        significance_scores = []

        for name, stats in author_stats.items():
            if len(stats['topical_papers']) < 2:  # need at least 2 on-topic papers
                continue

            years = stats['years']
            if not years:
                continue

            # significance combines:
            # - topical paper count (weight 2)
            # - topical citations (weight 1, log-scaled)
            # - active years (weight 0.5)
            topical_count = len(stats['topical_papers'])
            log_cites = math.log1p(stats['topical_citations'])
            active_years = max(years) - min(years) + 1

            significance = (
                topical_count * 2 +
                log_cites * 1 +
                active_years * 0.5
            )

            leaders.append(Leader(
                name=name,
                paper_count=len(stats['papers']),
                topical_papers=topical_count,
                total_citations=stats['citations'],
                topical_citations=stats['topical_citations'],
                first_year=min(years),
                last_year=max(years),
                active_years=active_years,
                significance_score=significance
            ))
            significance_scores.append(significance)

        # find natural breakpoint
        if significance_scores:
            threshold, count = self.find_natural_breakpoint(significance_scores, method="elbow")
            breakpoint_info = {
                "method": "elbow",
                "threshold": threshold,
                "leaders_above": count,
                "total_candidates": len(leaders)
            }
        else:
            threshold = 0
            breakpoint_info = {"method": "none", "threshold": 0}

        # filter and sort
        significant_leaders = [l for l in leaders if l.significance_score >= threshold]
        significant_leaders.sort(key=lambda l: l.significance_score, reverse=True)

        logger.info(f"leaders: {len(significant_leaders)} significant (threshold={threshold:.2f})")

        return significant_leaders, breakpoint_info

    def generate_topic_flow(self) -> Tuple[List[ConceptFlow], List[str], List[str]]:
        """
        analyze concept evolution over time.
        identifies emerging, stable, and declining concepts.
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT year, concepts_json FROM paper_candidates
            WHERE year IS NOT NULL AND concepts_json IS NOT NULL
            ORDER BY year
        """)

        # count concepts by year
        concept_years = defaultdict(lambda: defaultdict(int))

        for row in c.fetchall():
            year, concepts_json = row
            if not year:
                continue
            try:
                concepts = json.loads(concepts_json)
                for c_data in concepts[:5]:
                    name = c_data.get('name', '').lower()
                    # skip methodology concepts for topic flow
                    if name and name not in self.METHODOLOGY_CONCEPTS:
                        concept_years[name][year] += 1
            except:
                pass

        conn.close()

        # analyze each concept's trajectory
        flows = []

        for concept, year_counts in concept_years.items():
            if sum(year_counts.values()) < 5:  # skip rare concepts
                continue

            years = sorted(year_counts.keys())
            if len(years) < 2:
                continue

            # determine trend
            recent = sum(year_counts.get(y, 0) for y in years[-3:])
            early = sum(year_counts.get(y, 0) for y in years[:3])

            if len(years) >= 3:
                if recent > early * 1.5:
                    trend = "emerging"
                elif recent < early * 0.5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # find peak year
            peak_year = max(year_counts.items(), key=lambda x: x[1])[0]

            flows.append(ConceptFlow(
                concept=concept,
                appearances_by_year=dict(year_counts),
                peak_year=peak_year,
                trend=trend,
                total_papers=sum(year_counts.values())
            ))

        # sort by total papers
        flows.sort(key=lambda f: f.total_papers, reverse=True)

        # extract emerging and declining
        emerging = [f.concept for f in flows if f.trend == "emerging"][:10]
        declining = [f.concept for f in flows if f.trend == "declining"][:10]

        logger.info(f"topic flow: {len(flows)} concepts tracked, {len(emerging)} emerging, {len(declining)} declining")

        return flows, emerging, declining

    def generate_full_report(self) -> LiteratureReport:
        """generate complete literature review report."""
        logger.info(f"generating report from {self.db_path}")

        # generate each section
        topical_papers, hubs, clusters = self.generate_landscape()
        leaders, breakpoint_info = self.generate_leaders()
        flows, emerging, declining = self.generate_topic_flow()

        # compile natural breakpoints used
        topical_scores = [p.topical_score for p in topical_papers]
        topical_threshold, _ = self.find_natural_breakpoint(topical_scores, "elbow")

        natural_breakpoints = {
            "topical_score_threshold": topical_threshold,
            "leader_significance_threshold": breakpoint_info.get("threshold", 0),
            "method": "elbow (maximum curvature)"
        }

        # count seeds
        seed_count = sum(1 for p in topical_papers if p.is_seed)

        return LiteratureReport(
            topical_papers=topical_papers,
            clusters=clusters,
            methodology_hubs=hubs[:50],  # cap display of hubs
            leaders=leaders,
            breakpoint_info=breakpoint_info,
            concept_flow=flows[:30],  # top 30 concepts
            emerging_concepts=emerging,
            declining_concepts=declining,
            sparse_connections=[],  # TODO: compute from edges
            suggested_reading=[],  # TODO: gap analysis
            seed_count=seed_count,
            total_papers_analyzed=len(topical_papers) + len(hubs),
            natural_breakpoints=natural_breakpoints
        )

    def export_bibtex(self, papers: List[TopicalPaper], output_path: str) -> str:
        """export papers to BibTeX format."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # get DOIs for papers
        paper_ids = [p.id for p in papers]
        placeholders = ','.join('?' * len(paper_ids))

        c.execute(f"""
            SELECT id, doi, title, year, venue, extra_json
            FROM paper_candidates
            WHERE id IN ({placeholders})
        """, paper_ids)

        entries = []
        for row in c.fetchall():
            paper_id, doi, title, year, venue, extra_json = row

            # extract authors from extra_json
            authors = []
            if extra_json:
                try:
                    extra = json.loads(extra_json)
                    author_list = extra.get('authors', [])
                    for a in author_list:
                        if isinstance(a, str):
                            authors.append(a)
                        elif isinstance(a, dict):
                            authors.append(a.get('name', ''))
                except:
                    pass

            # generate cite key
            first_author = authors[0].split()[-1] if authors else "Unknown"
            cite_key = f"{first_author}{year or 'xxxx'}"

            # bibtex entry
            entry = f"@article{{{cite_key},\n"
            entry += f"  title = {{{title or 'Untitled'}}},\n"
            if authors:
                entry += f"  author = {{{' and '.join(authors[:5])}}},\n"
            if year:
                entry += f"  year = {{{year}}},\n"
            if venue:
                entry += f"  journal = {{{venue}}},\n"
            if doi:
                entry += f"  doi = {{{doi}}},\n"
            entry += f"  note = {{topical_score: {next((p.topical_score for p in papers if p.id == paper_id), 0):.3f}}}\n"
            entry += "}"

            entries.append(entry)

        conn.close()

        # write to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            f.write("% generated by refnet report generator\n")
            f.write(f"% {len(entries)} entries\n\n")
            f.write("\n\n".join(entries))

        logger.info(f"exported {len(entries)} entries to {output_path}")
        return str(output)


def print_report(report: LiteratureReport, verbose: bool = False):
    """print report to console."""
    print("\n" + "=" * 60)
    print("LITERATURE REVIEW REPORT")
    print("=" * 60)

    print(f"\n--- LANDSCAPE ---")
    print(f"Seeds: {report.seed_count}")
    print(f"Topical papers: {len(report.topical_papers)}")
    print(f"Methodology hubs (excluded): {len(report.methodology_hubs)}")
    print(f"Clusters: {len(report.clusters)}")

    print(f"\nTop clusters:")
    sorted_clusters = sorted(report.clusters.items(), key=lambda x: len(x[1]), reverse=True)
    for name, paper_ids in sorted_clusters[:7]:
        print(f"  • {name}: {len(paper_ids)} papers")

    if verbose:
        print(f"\nTop topical papers:")
        for p in report.topical_papers[:10]:
            seed_mark = "★" if p.is_seed else " "
            print(f"  {seed_mark} [{p.topical_score:.2f}] {(p.title or '?')[:50]}... ({p.year})")

    print(f"\n--- LEADERS ---")
    if report.breakpoint_info.get("reason") == "no_author_data":
        print(f"  ⚠ No author data in database")
        print(f"  Use Garden database or fetch author info from API")
    else:
        print(f"Significant researchers: {len(report.leaders)}")
        print(f"Breakpoint: {report.breakpoint_info}")

        print(f"\nTop leaders:")
        for l in report.leaders[:10]:
            print(f"  • {l.name}")
            print(f"    {l.topical_papers} topical papers, {l.topical_citations} cites, {l.first_year}-{l.last_year}")

    print(f"\n--- TOPIC FLOW ---")
    print(f"Emerging concepts: {', '.join(report.emerging_concepts[:5])}")
    print(f"Declining concepts: {', '.join(report.declining_concepts[:5])}")

    print(f"\n--- NATURAL BREAKPOINTS ---")
    for key, value in report.natural_breakpoints.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
