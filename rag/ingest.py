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
) -> dict:
    chunks = extract_pdf_chunks(path, source_name=source_name)
    store = TfidfRAGStore.load(index_path)
    added = store.add_chunks(chunks)
    store.save(index_path)
    return {
        "message": "PDF ingested",
        "source": source_name or Path(path).name,
        "chunks_added": added,
        "total_chunks": len(store.chunks),
        "index_path": str(index_path),
    }
