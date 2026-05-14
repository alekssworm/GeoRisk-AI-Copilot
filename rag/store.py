from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.config import RAG_INDEX_PATH
from rag.text import DocumentChunk


class TfidfRAGStore:
    """Small local vector store suitable for demos, tests, and offline use.

    The interface is intentionally simple so the implementation can be swapped
    for Chroma or FAISS without changing the API and frontend layers.
    """

    def __init__(self, chunks: list[DocumentChunk] | None = None):
        self.chunks = chunks or []
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=50000
        )
        self.matrix = None
        if self.chunks:
            self._fit()

    def _fit(self) -> None:
        try:
            self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        except ValueError:
            self.matrix = None

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        existing_keys = {(chunk.source, chunk.page, chunk.text.strip()) for chunk in self.chunks}
        new_chunks = []
        for chunk in chunks:
            chunk.text = chunk.text.strip()
            if not chunk.text:
                continue
            key = (chunk.source, chunk.page, chunk.text)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_chunks.append(chunk)

        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        self._fit()
        return len(new_chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not query.strip() or not self.chunks or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[int(index)]
            score = float(scores[int(index)])
            if score <= 0:
                continue
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk": chunk,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                }
            )
        return results

    def save(self, path: str | Path = RAG_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"chunks": self.chunks}, path)

    @classmethod
    def load(cls, path: str | Path = RAG_INDEX_PATH) -> "TfidfRAGStore":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            payload = joblib.load(path)
        except Exception:
            return cls()
        return cls(chunks=payload.get("chunks", []))
