from pathlib import Path


def test_frontend_contains_streaming_trace_contract():
    static = Path(__file__).parents[1] / "app" / "web" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    javascript = (static / "app.js").read_text(encoding="utf-8")
    for element_id in ("questionForm", "traceList", "answerPanel", "citations", "tokenModels"):
        assert f'id="{element_id}"' in html
        assert f"#{element_id}" in javascript
    assert "getReader()" in javascript
    assert "查看输入、输出与公开理由" in javascript
