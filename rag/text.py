from dataclasses import dataclass
import re
from uuid import uuid4


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int

    def as_citation(self, rank: int, score: float) -> dict:
        page_text = f", page {self.page}" if self.page is not None else ""
        return {
            "citation_id": f"[{rank}]",
            "source": self.source,
            "page": self.page,
            "score": round(float(score), 4),
            "label": f"{self.source}{page_text}",
        }


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_size: int = 1200,
    overlap: int = 180,
) -> list[DocumentChunk]:
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = chunk_size - 1

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    chunks = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            next_period = cleaned.rfind(". ", start, end)
            if next_period > start + chunk_size * 0.55:
                end = next_period + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(
                DocumentChunk(
                    id=str(uuid4()),
                    text=chunk,
                    source=source,
                    page=page,
                    chunk_index=index,
                )
            )
            index += 1
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks
