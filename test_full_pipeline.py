#!/usr/bin/env python
"""
test the complete pipeline with LLM extraction and visualization.

usage:
    python test_full_pipeline.py

tests:
1. basic pipeline run
2. LLM extraction (if ollama available)
3. graph visualization output
"""

import sys
import logging
from pathlib import Path

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("test")

def test_basic_pipeline():
    """test basic pipeline run without LLM."""
    from refnet.providers.openalex import OpenAlexProvider
    from refnet.pipeline import Pipeline, QuickConfig

    logger.info("=== testing basic pipeline ===")

    provider = OpenAlexProvider()
    config = QuickConfig()
    pipeline = Pipeline(provider, config=config)

    # test with a well-known paper (attention is all you need)
    result = pipeline.analyze_paper("attention is all you need")

    logger.info(f"  papers found: {result.paper_count}")
    logger.info(f"  key authors: {len(result.key_authors)}")
    logger.info(f"  reading list: {len(result.reading_list)}")

    assert result.paper_count > 0, "should find papers"
    assert result.seed_paper is not None, "should resolve seed"

    logger.info("  ✓ basic pipeline test passed")
    return result

def test_verified_pipeline():
    """test pipeline with verification enabled."""
    from refnet.providers.openalex import OpenAlexProvider
    from refnet.pipeline import Pipeline, VerifiedConfig

    logger.info("=== testing verified pipeline ===")

    provider = OpenAlexProvider()
    config = VerifiedConfig()
    pipeline = Pipeline(provider, config=config)

    result = pipeline.analyze_paper("BERT: Pre-training of Deep Bidirectional Transformers")

    logger.info(f"  papers found: {result.paper_count}")
    logger.info(f"  resolved field: {result.resolved_field.primary_field.name if result.resolved_field else 'none'}")

    if result.verification_summary:
        logger.info(f"  verification passed: {result.verification_summary['passed']}")
        logger.info(f"  confidence: {result.verification_summary['confidence']:.2f}")

    assert result.paper_count > 0, "should find papers"

    logger.info("  ✓ verified pipeline test passed")
    return result

def test_llm_extraction():
    """test LLM extraction (requires ollama)."""
    from refnet.llm import OllamaProvider, PaperExtractor
    from refnet.core.models import Paper

    logger.info("=== testing LLM extraction ===")

    provider = OllamaProvider(model="qwen3:32b")

    if not provider.is_available():
        logger.warning("  ⚠ ollama not available, skipping LLM test")
        return None

    extractor = PaperExtractor(provider)

    # create test paper
    paper = Paper(
        id="test123",
        title="Attention Is All You Need",
        year=2017,
        venue="NeurIPS",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        concepts=[
            {"name": "attention mechanism", "score": 0.9},
            {"name": "neural networks", "score": 0.8},
            {"name": "transformers", "score": 0.95}
        ]
    )

    # extract info
    info = extractor.extract(paper)

    logger.info(f"  extracted from: {info.paper_title}")
    logger.info(f"  key findings: {info.key_findings}")
    logger.info(f"  methods: {info.methods}")
    logger.info(f"  contribution type: {info.contribution_type}")
    logger.info(f"  confidence: {info.extraction_confidence}")

    assert info.extraction_confidence > 0.5, "should extract with confidence"

    logger.info("  ✓ LLM extraction test passed")
    return info

def test_visualization(result):
    """test visualization from analysis result."""
    from refnet.visualization import GraphBuilder, GraphExporter, HTMLRenderer

    logger.info("=== testing visualization ===")

    builder = GraphBuilder(result)
    exporter = GraphExporter()
    renderer = HTMLRenderer()

    # build graphs
    citation_graph = builder.build_citation_graph()
    author_graph = builder.build_author_graph()
    topic_graph = builder.build_topic_graph()

    logger.info(f"  citation graph: {citation_graph.paper_count} nodes, {citation_graph.edge_count} edges")
    logger.info(f"  author graph: {author_graph.author_count} nodes, {author_graph.edge_count} edges")
    logger.info(f"  topic graph: {topic_graph.topic_count} nodes, {topic_graph.edge_count} edges")

    # export to files
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # JSON exports
    exporter.to_json(citation_graph.graph, str(output_dir / "citations.json"))
    exporter.to_json(author_graph.graph, str(output_dir / "authors.json"))

    # HTML visualizations
    renderer.render_citation_graph(citation_graph, str(output_dir / "citations.html"))
    renderer.render_author_graph(author_graph, str(output_dir / "authors.html"))
    renderer.render_topic_graph(topic_graph, str(output_dir / "topics.html"))

    logger.info(f"  outputs in: {output_dir.absolute()}")
    logger.info("  ✓ visualization test passed")

def main():
    """run all tests."""
    logger.info("RefNet Full Pipeline Test")
    logger.info("=" * 50)

    try:
        # test 1: basic pipeline
        result = test_basic_pipeline()

        # test 2: verified pipeline
        verified_result = test_verified_pipeline()

        # test 3: LLM extraction
        test_llm_extraction()

        # test 4: visualization (using verified result which has more data)
        test_visualization(verified_result)

        logger.info("")
        logger.info("=" * 50)
        logger.info("All tests passed ✓")
        return 0

    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
