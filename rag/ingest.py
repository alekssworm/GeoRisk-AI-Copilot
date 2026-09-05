from pathlib import Path

from pypdf import PdfReader

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, RAG_INDEX_PATH
from rag.store import TfidfRAGStore
from rag.text import chunk_text


def chunks_from_text(text: str, source: str, page: int | None = None):
    return chunk_text(
        text,
        source=source,
        page=page,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )


def extract_pdf_chunks(path: str | Path, source_name: str | None = None):
    path = Path(path)
    reader = PdfReader(str(path))
    chunks = []
    source = source_name or path.name
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        chunks.extend(chunks_from_text(page_text, source=source, page=page_number))
    return chunks


def ingest_pdf(
    path: str | Path,
    source_name: str | None = None,
    index_path: str | Path = RAG_INDEX_PATH,
    *,
    stable: bool = False,
) -> dict:
    chunks = extract_pdf_chunks(path, source_name=source_name)
    with TfidfRAGStore.transaction(index_path) as store:
        added = store.add_chunks(chunks, stable=stable)
        total_chunks = len(store.chunks)
        shelf_counts = store.shelves.counts()
    return {
        "message": "PDF ingested",
        "source": source_name or Path(path).name,
        "chunks_added": added,
        "total_chunks": total_chunks,
        "index_path": str(index_path),
        "stable": stable,
        "shelf_counts": shelf_counts,
    }
