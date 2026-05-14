from rag.ingest import chunks_from_text
from rag.llm import LLMClient
from rag.qa import RAGAssistant
from rag.store import TfidfRAGStore
from rag.text import chunk_text


class NoLLM(LLMClient):
    def generate(self, prompt: str):
        return None


def test_rag_retrieves_relevant_chunk():
    chunks = chunks_from_text(
        "Cesium-137 binds strongly to clay minerals in many soils. "
        "Heavy rainfall can increase runoff and change downstream monitoring needs.",
        source="guidance.txt",
    )
    store = TfidfRAGStore()
    store.add_chunks(chunks)

    results = store.search("How does clay affect cesium mobility?", top_k=1)
    assert results
    assert "clay" in results[0]["text"].lower()

    assistant = RAGAssistant(store=store, llm_client=NoLLM())
    answer = assistant.answer("How does clay affect cesium mobility?", top_k=1)
    assert answer["citations"][0]["citation_id"] == "[1]"
    assert "OPENAI_API_KEY" in answer["answer"]


def test_rag_store_deduplicates_chunks_and_handles_empty_vocabulary():
    chunks = chunk_text("A I", source="minimal.txt", chunk_size=10, overlap=2)
    store = TfidfRAGStore()

    assert store.add_chunks(chunks) == 1
    assert store.add_chunks(chunks) == 0
    assert store.search("anything", top_k=1) == []


def test_chunk_text_clamps_overlap_to_avoid_stalling():
    chunks = chunk_text("abcdef", source="small.txt", chunk_size=2, overlap=2)

    assert chunks
    assert "".join(chunk.text for chunk in chunks)
