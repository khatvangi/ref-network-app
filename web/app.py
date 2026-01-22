"""
refnet web application - submit queries, view visualizations.

run:
    cd /storage/kiran-stuff/ref-network-app
    uvicorn web.app:app --host 0.0.0.0 --port 8765

endpoints:
    GET  /                  → landing page
    POST /api/analyze       → submit query, returns job_id
    GET  /api/status/{id}   → check job status
    GET  /results/{id}      → view results page
    GET  /api/results/{id}  → get results JSON
"""

import os
import sys
import json
import uuid
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# add parent to path for refnet imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from refnet.providers.openalex import OpenAlexProvider
from refnet.pipeline import Pipeline, PipelineConfig, VerifiedConfig, QuickConfig
from refnet.visualization import GraphBuilder, GraphExporter, HTMLRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("refnet.web")

# paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "jobs.db"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# app
app = FastAPI(title="RefNet", description="Literature network analysis")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# database setup
def init_db():
    """initialize sqlite database."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                query_type TEXT DEFAULT 'paper',
                config TEXT DEFAULT 'quick',
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                paper_count INTEGER,
                author_count INTEGER,
                error TEXT
            )
        """)

@contextmanager
def get_db():
    """get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# models
class AnalyzeRequest(BaseModel):
    query: str
    query_type: str = "paper"  # "paper" or "author"
    config: str = "quick"  # "quick", "standard", "verified"


class JobStatus(BaseModel):
    id: str
    status: str  # "pending", "running", "completed", "failed"
    query: str
    query_type: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    paper_count: Optional[int] = None
    author_count: Optional[int] = None
    error: Optional[str] = None


# background job processor
def run_analysis(job_id: str, query: str, query_type: str, config_name: str):
    """run pipeline analysis in background."""
    logger.info(f"starting job {job_id}: {query}")

    # update status to running
    with get_db() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
            (datetime.now().isoformat(), job_id)
        )

    try:
        # select config
        if config_name == "quick":
            config = QuickConfig()
        elif config_name == "verified":
            config = VerifiedConfig()
        else:
            config = PipelineConfig()

        # run pipeline
        provider = OpenAlexProvider()
        pipeline = Pipeline(provider, config=config)

        if query_type == "author":
            result = pipeline.analyze_author(query)
        else:
            result = pipeline.analyze_paper(query)

        # save results
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        # save summary JSON
        summary = {
            "seed_query": result.seed_query,
            "seed_type": result.seed_type,
            "paper_count": result.paper_count,
            "author_count": len(result.key_authors),
            "insights": result.insights,
            "errors": result.errors,
            "warnings": result.warnings,
            "duration_seconds": result.duration_seconds,
            "seed_paper": {
                "title": result.seed_paper.title if result.seed_paper else None,
                "year": result.seed_paper.year if result.seed_paper else None,
                "doi": result.seed_paper.doi if result.seed_paper else None,
                "citations": result.seed_paper.citation_count if result.seed_paper else None
            } if result.seed_paper else None,
            "key_authors": [
                {"name": a.name, "papers": a.paper_count, "citations": a.citation_count}
                for a in result.key_authors[:10]
            ],
            "reading_list": [
                {
                    "title": item.paper.title,
                    "year": item.paper.year,
                    "doi": item.paper.doi,
                    "category": item.category,
                    "priority": item.priority,
                    "score": item.relevance_score
                }
                for item in result.reading_list[:20]
            ],
            "landscape": {
                "core_topics": result.landscape.core_topics[:10] if result.landscape else [],
                "emerging_topics": result.landscape.emerging_topics[:5] if result.landscape else [],
                "year_range": list(result.landscape.year_range) if result.landscape else []
            } if result.landscape else None,
            "gaps": {
                "concept_gaps": [
                    {"a": g.concept_a, "b": g.concept_b, "score": g.gap_score}
                    for g in (result.gaps.concept_gaps[:5] if result.gaps else [])
                ]
            } if result.gaps else None
        }

        with open(job_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # generate visualizations
        builder = GraphBuilder(result)
        renderer = HTMLRenderer()

        citation_graph = builder.build_citation_graph()
        author_graph = builder.build_author_graph()
        topic_graph = builder.build_topic_graph()

        renderer.render_citation_graph(citation_graph, str(job_dir / "citations.html"))
        renderer.render_author_graph(author_graph, str(job_dir / "authors.html"))
        renderer.render_topic_graph(topic_graph, str(job_dir / "topics.html"))

        # update database
        with get_db() as conn:
            conn.execute("""
                UPDATE jobs SET
                    status = 'completed',
                    completed_at = ?,
                    paper_count = ?,
                    author_count = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), result.paper_count, len(result.key_authors), job_id))

        logger.info(f"job {job_id} completed: {result.paper_count} papers")

    except Exception as e:
        logger.error(f"job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()

        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, completed_at = ? WHERE id = ?",
                (str(e), datetime.now().isoformat(), job_id)
            )


# routes
@app.on_event("startup")
async def startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """submit analysis job."""
    job_id = str(uuid.uuid4())[:8]

    with get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, query, query_type, config, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
            (job_id, req.query, req.query_type, req.config, datetime.now().isoformat())
        )

    # run in background
    background_tasks.add_task(run_analysis, job_id, req.query, req.query_type, req.config)

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """check job status."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**dict(row))


@app.get("/api/results/{job_id}")
async def results_json(job_id: str):
    """get results JSON."""
    job_dir = RESULTS_DIR / job_id
    summary_file = job_dir / "summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")

    with open(summary_file) as f:
        return json.load(f)


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    """results page with visualizations."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dict(row)

    # load summary if completed
    summary = None
    if job["status"] == "completed":
        summary_file = RESULTS_DIR / job_id / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "job": job,
        "summary": summary
    })


@app.get("/viz/{job_id}/{viz_type}")
async def visualization(job_id: str, viz_type: str):
    """serve visualization HTML."""
    if viz_type not in ["citations", "authors", "topics"]:
        raise HTTPException(status_code=400, detail="Invalid visualization type")

    viz_file = RESULTS_DIR / job_id / f"{viz_type}.html"

    if not viz_file.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")

    return FileResponse(viz_file)


@app.get("/api/recent")
async def recent_jobs(limit: int = 10):
    """get recent jobs."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
