from concurrent.futures import ThreadPoolExecutor

import joblib
import pytest
from sklearn.metrics.pairwise import cosine_similarity

import rag.ingest as ingest_module
import rag.store as store_module
from rag.qa import RAGAssistant
from rag.shelves import ShelfEntry, ShelfManager, ShelfPolicy
from rag.store import TfidfRAGStore
from rag.text import chunk_text


class NoLLM:
    def generate(self, prompt):
        return None


def passages(*texts):
    return [
        chunk_text(text, source=f"reference-{index}.pdf", page=index + 1)[0]
        for index, text in enumerate(texts)
    ]


def small_store(*texts, **policy):
    settings = {"top_capacity": 2, "middle_capacity": 2, "recent_slots": 1} | policy
    return TfidfRAGStore(passages(*texts), policy=ShelfPolicy(**settings))


def test_new_arrivals_and_popular_passages_share_bounded_top_shelf():
    store = small_store("cesium clay retention", "strontium water transport")
    for _ in range(3):
        store.search("cesium clay retention", top_k=1)

    arrivals = passages(*(f"rainfall runoff region{index}" for index in range(20)))
    assert store.add_chunks(arrivals) == 20

    assert store.shelves.indices[1] == [21, 0]
    assert store.shelves.counts() == {"1": 2, "2": 2, "3": 18}
    assert sum(store.shelves.counts().values()) == len(store.chunks)
    before = list(store.shelves.entries)
    assert store.add_chunks(arrivals) == 0
    assert store.shelves.entries == before


def test_top_shelf_hit_only_scores_its_rows(monkeypatch):
    store = small_store(
        "strontium groundwater transport",
        "radon basement ventilation",
        "potassium granite background",
        "rainfall runoff monitoring",
        "cesium clay retention",
    )
    scored_rows = []

    def spy(query, matrix):
        scored_rows.append(matrix.shape[0])
        return cosine_similarity(query, matrix)

    monkeypatch.setattr(store_module, "cosine_similarity", spy)
    results = store.search("cesium clay retention", top_k=1)

    assert results[0]["text"] == "cesium clay retention"
    assert results[0]["shelf"] == 1
    assert scored_rows == [2]
    assert store.last_search["chunks_scored"] == 2
    assert store.last_search["early_stopped"] is True


def test_bottom_shelf_hit_is_promoted_and_next_search_stays_small():
    store = small_store(
        "cesium clay retention",
        "strontium groundwater transport",
        "radon basement ventilation",
        "potassium granite background",
        "rainfall runoff monitoring",
    )
    original = store.chunks[0]
    first = store.search("cesium clay retention", top_k=1)
    assert first[0]["shelf"] == 3
    assert store.last_search["shelves_searched"] == [1, 2, 3]
    assert 0 in store.shelves.indices[1]

    second = store.search("cesium clay retention", top_k=1)
    assert second[0]["shelf"] == 1
    assert store.last_search["chunks_scored"] == 2
    assert store.shelves.entries[0].hits == 2
    assert second[0]["chunk"].as_citation(1, second[0]["score"])["page"] == 1
    assert second[0]["chunk"] is original


def test_stable_reference_remains_on_bottom_after_repeated_hits():
    store = small_store("rainfall runoff monitoring", "potassium granite background")
    store.add_chunks(passages("cesium clay retention"), stable=True)
    for _ in range(3):
        result = store.search("cesium clay retention", top_k=1)
        assert result[0]["shelf"] == 3
        assert store.shelves.indices[3] == [2]
    assert store.shelves.entries[2].hits == 3
    assert store.chunks[2].text == "cesium clay retention"


def test_frequency_decays_so_old_popularity_does_not_dominate():
    shelves = ShelfManager(
        [ShelfEntry() for _ in range(3)],
        policy=ShelfPolicy(top_capacity=2, middle_capacity=1, recent_slots=1, heat_half_life=2),
    )
    for _ in range(4):
        shelves.record_hits([0])
    assert 0 in shelves.indices[1]
    for _ in range(10):
        shelves.record_hits([])
    shelves.record_hits([1])
    assert shelves.indices[1] == [2, 1]
    assert shelves.entries[0].hits == 4


def test_missing_query_terms_force_deeper_search_despite_high_top_score():
    store = small_store(
        "soil cesium retention",
        "strontium groundwater transport",
        "radon basement ventilation",
        "soil",
        "soil soil",
        min_score=0.1,
    )
    results = store.search("soil cesium", top_k=1)
    assert results[0]["text"] == "soil cesium retention"
    assert store.last_search["shelves_searched"] == [1, 2, 3]


def test_too_few_good_results_expand_search_and_keep_global_ranking():
    store = small_store(
        "soil cesium retention",
        "soil strontium transport",
        "radon basement ventilation",
        "potassium granite background",
        "soil runoff",
    )
    results = store.search("soil", top_k=3)
    assert store.last_search["chunks_scored"] == 5
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert [result["score"] for result in results] == sorted(
        (result["score"] for result in results), reverse=True
    )


def test_full_search_matches_flat_tfidf_ranking():
    store = small_store(
        "soil cesium retention",
        "soil strontium transport",
        "soil rainfall runoff",
        "soil granite background",
        "soil",
        min_score=0.1,
    )
    vector = store.vectorizer.transform(["soil"])
    scores = cosine_similarity(vector, store.matrix).ravel()
    expected = sorted(range(5), key=lambda index: (-scores[index], index))[:3]
    results = store.search("soil", top_k=3, full_search=True)
    assert [result["chunk"].id for result in results] == [
        store.chunks[index].id for index in expected
    ]
    assert store.last_search["chunks_scored"] == 5
    assert store.last_search["mode"] == "all"
    assert store.last_search["early_stopped"] is False


@pytest.mark.parametrize("query", ["", "  ", "unseenword"])
def test_empty_and_unknown_queries_do_not_promote_or_scan(query):
    store = small_store("cesium clay retention")
    store.search("cesium", top_k=1)
    hits = store.shelves.entries[0].hits
    assert store.search(query) == []
    assert store.last_search["chunks_scored"] == 0
    assert store.shelves.entries[0].hits == hits


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        TfidfRAGStore().search("cesium", top_k=top_k)


def test_legacy_index_migrates_and_saved_vectors_are_reused(tmp_path, monkeypatch):
    path = tmp_path / "index.joblib"
    chunks = passages("cesium clay retention", "rainfall runoff monitoring")
    joblib.dump({"chunks": chunks}, path)
    store = TfidfRAGStore.load(path)
    store.add_chunks(passages("strontium groundwater transport"), stable=True)
    store.search("cesium clay retention", top_k=1)
    store.save(path)

    def fail_fit(self):
        pytest.fail("Loading an existing v2 index must not refit TF-IDF")

    monkeypatch.setattr(TfidfRAGStore, "_fit", fail_fit)
    reloaded = TfidfRAGStore.load(path)
    assert [chunk.id for chunk in reloaded.chunks[:2]] == [chunk.id for chunk in chunks]
    assert reloaded.shelves.entries[0].hits == 1
    assert reloaded.shelves.entries[2].stable is True
    assert reloaded.shelves.indices[3] == [2]
    assert reloaded.search("cesium clay retention", top_k=1)


def test_assistant_persists_hits_between_instances_and_releases_lock_before_llm(tmp_path):
    path = tmp_path / "index.joblib"
    store = small_store("cesium clay retention")
    store.save(path)

    class IngestDuringSynthesis:
        def generate(self, prompt):
            # This would time out if synthesis still held the index transaction.
            with ThreadPoolExecutor(max_workers=1) as pool:

                def add_document():
                    with TfidfRAGStore.transaction(path) as current:
                        current.add_chunks(passages("rainfall runoff monitoring"))

                pool.submit(add_document).result(timeout=5)

    first = RAGAssistant(index_path=path, llm_client=IngestDuringSynthesis())
    answer = first.answer("cesium clay retention", top_k=1)
    assert answer["citations"][0]["source"] == "reference-0.pdf"
    assert answer["retrieved_context"][0]["shelf"] == 1
    assert answer["retrieval"]["chunks_scored"] == 1
    RAGAssistant(index_path=path, llm_client=NoLLM()).answer("cesium clay retention", top_k=1)
    reloaded = TfidfRAGStore.load(path)
    assert reloaded.shelves.entries[0].hits == 2
    assert len(reloaded.chunks) == 2


def test_concurrent_answers_and_ingestion_do_not_lose_updates(tmp_path):
    path = tmp_path / "index.joblib"
    TfidfRAGStore(passages("cesium clay retention")).save(path)

    def answer(_):
        return RAGAssistant(index_path=path, llm_client=NoLLM()).answer(
            "cesium clay retention", top_k=1
        )

    def ingest():
        with TfidfRAGStore.transaction(path) as store:
            store.add_chunks(passages("rainfall runoff monitoring"), stable=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(answer, i) for i in range(8)] + [pool.submit(ingest)]
        for future in futures:
            future.result(timeout=10)
    reloaded = TfidfRAGStore.load(path)
    assert reloaded.shelves.entries[0].hits == 8
    assert len(reloaded.chunks) == 2
    assert reloaded.shelves.entries[1].stable is True


def test_failed_atomic_save_preserves_existing_index(tmp_path, monkeypatch):
    path = tmp_path / "index.joblib"
    store = small_store("cesium clay retention")
    store.save(path)
    before = path.read_bytes()

    def fail_dump(payload, destination):
        destination.write_bytes(b"incomplete")
        raise OSError("disk write failed")

    monkeypatch.setattr(store_module.joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="disk write failed"):
        store.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_pdf_ingestion_saves_stability_and_deduplicates(tmp_path, monkeypatch):
    path = tmp_path / "index.joblib"
    monkeypatch.setattr(
        ingest_module, "extract_pdf_chunks", lambda *a, **kw: passages("cesium clay retention")
    )
    first = ingest_module.ingest_pdf("reference.pdf", index_path=path, stable=True)
    second = ingest_module.ingest_pdf("reference.pdf", index_path=path, stable=True)
    assert first["chunks_added"] == 1
    assert first["shelf_counts"] == {"1": 0, "2": 0, "3": 1}
    assert second["chunks_added"] == 0
    assert second["total_chunks"] == 1
    assert TfidfRAGStore.load(path).shelves.entries[0].stable is True
