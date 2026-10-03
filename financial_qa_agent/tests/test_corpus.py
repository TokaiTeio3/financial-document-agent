from app.retrieval.corpus import CorpusRegistry


def test_corpus_keeps_page_and_source():
    from pathlib import Path

    fixture_root = Path(__file__).parent / "fixtures" / "corpus"
    registry = CorpusRegistry(fixture_root, chunk_size=200, overlap=20)
    hits = registry.get("insurance").search("等待期九十日", top_k=2)
    assert hits
    assert hits[0].page == 3
    assert hits[0].document_id == "policy"
    assert hits[0].source == "insurance/policy.md"
