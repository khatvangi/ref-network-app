"""
relevance scoring for papers.
v0 implementation uses simple keyword/BM25-style matching.
"""

import re
import math
from typing import List, Set

from providers.base import PaperStub


def tokenize(text: str) -> List[str]:
    """simple tokenization: lowercase, split on non-alphanumeric."""
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_term_frequency(tokens: List[str]) -> dict:
    """compute term frequency."""
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return tf


def compute_relevance_score(paper: PaperStub, topic: str) -> float:
    """
    compute relevance of paper to topic.
    uses simple BM25-inspired scoring.

    returns score in [0, 1].
    """
    # tokenize topic
    topic_tokens = tokenize(topic)
    if not topic_tokens:
        return 0.0

    topic_set = set(topic_tokens)

    # tokenize paper content (title + abstract + concepts)
    paper_text = (paper.title or "") + " "
    if paper.abstract:
        paper_text += paper.abstract + " "
    for concept in paper.concepts:
        paper_text += concept.get('name', '') + " "

    paper_tokens = tokenize(paper_text)
    if not paper_tokens:
        return 0.0

    paper_tf = compute_term_frequency(paper_tokens)

    # BM25-style scoring
    k1 = 1.5
    b = 0.75
    avg_dl = 200  # assumed average document length

    doc_len = len(paper_tokens)
    score = 0.0

    for term in topic_set:
        if term not in paper_tf:
            continue

        tf = paper_tf[term]
        # simplified BM25: no IDF (would need corpus stats)
        # just use TF saturation
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
        score += numerator / denominator

    # normalize by number of query terms
    score = score / len(topic_set)

    # cap at 1.0
    score = min(score, 1.0)

    # boost for exact phrase match in title
    title_lower = (paper.title or "").lower()
    topic_lower = topic.lower()
    if topic_lower in title_lower:
        score = min(score + 0.3, 1.0)

    # boost for review papers on topic
    if paper.is_review and score > 0.1:
        score = min(score + 0.1, 1.0)

    return score


def fuzzy_title_match(title1: str, title2: str, threshold: float = 0.8) -> bool:
    """
    check if two titles are similar enough to be duplicates.
    uses jaccard similarity on tokens.
    """
    tokens1 = set(tokenize(title1))
    tokens2 = set(tokenize(title2))

    if not tokens1 or not tokens2:
        return False

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    similarity = len(intersection) / len(union)
    return similarity >= threshold


# simple test
if __name__ == "__main__":
    # test relevance scoring
    paper = PaperStub(
        title="Ancestral sequence reconstruction of ancient proteins",
        abstract="We review methods for reconstructing ancestral proteins using phylogenetic analysis.",
        concepts=[{'name': 'Phylogenetics', 'score': 0.9}]
    )

    score = compute_relevance_score(paper, "ancestral protein reconstruction")
    print(f"Relevance score: {score:.3f}")

    # test fuzzy match
    title1 = "Deep learning for protein structure prediction"
    title2 = "Protein structure prediction using deep learning"
    print(f"Fuzzy match: {fuzzy_title_match(title1, title2)}")
