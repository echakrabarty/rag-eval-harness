"""Offline metric scorers.

Heuristic implementations that run with no API keys. For production use,
swap faithfulness() for an LLM judge and answer_relevance() for embeddings —
the harness treats them as drop-in functions.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"[a-z]+")


def _tokens(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def context_precision(retrieved_ids, relevant_ids):
    """Fraction of retrieved chunks that are relevant."""
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(retrieved_ids)


def context_recall(retrieved_ids, relevant_ids):
    """Fraction of relevant chunks that were retrieved."""
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)


def answer_relevance(question, answer):
    """Cosine similarity between question and answer (TF-IDF)."""
    if not answer.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english").fit_transform([question, answer])
    return float(cosine_similarity(vec[0], vec[1])[0][0])


def faithfulness(answer, contexts):
    """Share of answer content-words grounded in retrieved context.

    Cheap offline proxy for "did the model hallucinate?". For a stronger
    signal, replace with an LLM judge over (answer, contexts).
    """
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens.update(_tokens(c["text"]))
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    return sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens)
