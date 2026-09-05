import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
from filelock import FileLock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.config import RAG_INDEX_PATH
from rag.shelves import ShelfEntry, ShelfManager, ShelfPolicy
from rag.text import DocumentChunk


class TfidfRAGStore:
    """Small local vector store suitable for demos, tests, and offline use.

    The interface is intentionally simple so the implementation can be swapped
    for Chroma or FAISS without changing the API and frontend layers.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        *,
        policy: ShelfPolicy | None = None,
    ):
        self.chunks = list(chunks or [])
        self.shelves = ShelfManager([ShelfEntry() for _ in self.chunks], policy=policy)
        self.last_search: dict = {}
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

    def add_chunks(self, chunks: list[DocumentChunk], *, stable: bool = False) -> int:
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
        self.shelves.add(len(new_chunks), stable=stable)
        self._fit()
        return len(new_chunks)

    def search(self, query: str, top_k: int = 4, *, full_search: bool = False) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        self.last_search = {
            "mode": "all" if full_search else "shelves",
            "shelves_searched": [],
            "chunks_scored": 0,
            "total_chunks": len(self.chunks),
            "early_stopped": False,
            "shelf_counts": self.shelves.counts(),
        }
        if not query.strip() or not self.chunks or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        # Scores use one shared vocabulary/IDF, so candidates from all shelves
        # remain comparable. Only rows on shelves actually visited are scored.
        candidates: list[tuple[int, float, int]] = []
        analyzer = self.vectorizer.build_analyzer()
        query_terms = {term for term in analyzer(query) if " " not in term}
        for shelf, indices in self.shelves.indices.items():
            if not indices:
                continue
            self.last_search["shelves_searched"].append(shelf)
            self.last_search["chunks_scored"] += len(indices)
            scores = cosine_similarity(query_vector, self.matrix[indices]).ravel()
            candidates.extend(
                (index, float(score), shelf)
                for index, score in zip(indices, scores, strict=True)
                if score > 0
            )
            candidates.sort(key=lambda item: (-item[1], item[0]))
            candidates = candidates[:top_k]
            if not full_search and len(candidates) == top_k:
                enough_score = candidates[-1][1] >= self.shelves.policy.min_score
                covered_terms = {
                    term
                    for index, _, _ in candidates
                    for term in analyzer(self.chunks[index].text)
                    if " " not in term
                }
                if enough_score and query_terms <= covered_terms:
                    self.last_search["early_stopped"] = self.last_search["chunks_scored"] < len(
                        self.chunks
                    )
                    break

        results = []
        for rank, (index, score, shelf) in enumerate(candidates, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk": chunk,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "shelf": shelf,
                }
            )
        # Popularity measures passages delivered as context, not every match.
        # Weak matches may be returned for compatibility, but do not gain heat.
        self.shelves.record_hits(
            [index for index, score, _ in candidates if score >= self.shelves.policy.min_score]
        )
        return results

    def save(self, path: str | Path = RAG_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "chunks": self.chunks,
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "shelf_entries": self.shelves.entries,
            "query_count": self.shelves.query_count,
        }
        # Readers see either the previous complete index or the new one.
        with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as temp:
            temporary_path = Path(temp.name)
        try:
            joblib.dump(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls, path: str | Path = RAG_INDEX_PATH, *, policy: ShelfPolicy | None = None
    ) -> "TfidfRAGStore":
        path = Path(path)
        if not path.exists():
            return cls(policy=policy)
        try:
            payload = joblib.load(path)
        except Exception:  # noqa: BLE001 - corrupt indexes safely reset to an empty store
            return cls(policy=policy)
        if not isinstance(payload, dict):
            return cls(policy=policy)
        if payload.get("version") != 2:
            # Legacy chunk-only indexes are fitted once, then upgraded on save.
            return cls(chunks=payload.get("chunks", []), policy=policy)
        store = cls(policy=policy)
        store.chunks = payload["chunks"]
        store.vectorizer = payload["vectorizer"]
        store.matrix = payload["matrix"]
        store.shelves = ShelfManager(
            payload["shelf_entries"], query_count=payload["query_count"], policy=policy
        )
        return store

    @classmethod
    @contextmanager
    def transaction(cls, path: str | Path = RAG_INDEX_PATH):
        """Serialize read/modify/write across API threads and local workers.

        Keep LLM/network calls outside this context. Direct load/save users must
        use this transaction too when several writers share an index.
        """
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=30):
            store = cls.load(path)
            yield store
            if store.chunks:
                store.save(path)
