from pathlib import Path

from app.retrieval.corpus import chunk_file


def test_page_chunks_include_both_adjacent_page_contexts(tmp_path: Path):
    document = tmp_path / "sample.md"
    document.write_text(
        "<!-- page: 1; source: page_0001.md -->\n## 第一页\n上一页独有的偿付能力要求。\n"
        "<!-- page: 2; source: page_0002.md -->\n## 第二页\n当前页说明信息披露期限。\n"
        "<!-- page: 3; source: page_0003.md -->\n## 第三页\n下一页独有的处罚规定。\n",
        encoding="utf-8",
    )
    chunks = chunk_file(document, "regulatory", chunk_size=2000, overlap=100, page_overlap=80)
    page_two = next(chunk for chunk in chunks if chunk.page == 2)
    assert "偿付能力要求" in page_two.text
    assert "信息披露期限" in page_two.text
    assert "处罚规定" in page_two.text
