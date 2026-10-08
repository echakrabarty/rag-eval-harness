"""Retrievers.

Ships a TF-IDF retriever so the harness runs offline. To plug in your own
retriever (vector DB, hybrid search, ...), implement a class with:

    retrieve(query: str, top_k: int = 3) -> list[{"id": str, "text": str, "score": float}]
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c["text"] for c in corpus])

    def retrieve(self, query, top_k=3):
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {
                "id": self.corpus[i]["id"],
                "text": self.corpus[i]["text"],
                "score": float(scores[i]),
            }
            for i in ranked
        ]
