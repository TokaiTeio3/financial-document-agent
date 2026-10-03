from app.schemas import StructuredAnswer


def test_provider_json_variations_are_coerced():
    answer = StructuredAnswer.model_validate({
        "answer": "答案：AC",
        "selected_options": "AC",
        "final_value": 12.3,
        "key_points": [{"point": "结论", "evidence": "原文"}],
        "citations": None,
        "confidence": "高",
        "limitations": "无",
    })
    assert answer.selected_options == ["A", "C"]
    assert answer.final_value == "12.3"
    assert answer.key_points == ["结论：原文"]
    assert answer.citations == []
    assert answer.confidence == "high"
    assert answer.limitations == []
