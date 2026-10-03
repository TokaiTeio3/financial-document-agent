from app.retrieval.bm25 import BM25Index, query_anchors, tokenize


def test_chinese_tokenizer_and_anchor_ranking():
    index = BM25Index([
        ("公司营业收入达到一百亿元，归母净利润稳步增长。", "比亚迪2024年年度报告"),
        ("本保险合同的等待期为九十日。", "保险条款"),
    ])
    scores = index.scores("比亚迪2024年营业收入是多少？")
    assert scores[0] > scores[1]
    assert tokenize("营业收入2024年")
    assert any("营业收入" in anchor for anchor in query_anchors("2024年营业收入是多少"))
