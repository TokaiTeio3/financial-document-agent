from pathlib import Path

from app.retrieval.corpus import CorpusRegistry


def test_search_results_do_not_allow_one_document_to_crowd_out_others():
    fixture_root = Path(__file__).parent / "fixtures" / "corpus"
    registry = CorpusRegistry(fixture_root, chunk_size=35, overlap=0)
    hits = registry.get("financial_reports").search("营业收入 净利润", top_k=6)
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.document_id] = counts.get(hit.document_id, 0) + 1
    assert all(count <= 2 for count in counts.values())
