from pathlib import Path

from app.evaluation.ground_truth import GroundTruthItem, load_ground_truth, repair_mojibake, score_answer


GROUND_TRUTH = Path(__file__).parent / "fixtures" / "ground_truth.json"


def test_synthetic_ground_truth_loads_and_repairs_encoding():
    items = load_ground_truth(GROUND_TRUTH)
    assert len(items) == 5
    assert {item.domain for item in items} == {
        "financial_contracts", "financial_reports", "insurance", "regulatory", "research"
    }
    assert items[0].question.startswith("根据《示例")
    assert items[0].expected_document_ids == ["sample"]
    text = "根据示例合同"
    assert repair_mojibake(text.encode("gbk").decode("latin-1")) == text


def test_ground_truth_choice_and_calculation_scoring():
    items = load_ground_truth(GROUND_TRUTH)
    choice = next(item for item in items if "选题" in item.type)
    assert score_answer(choice, {"selected_options": list(choice.answer)})
    calculation = next(item for item in items if item.type == "计算题")
    assert score_answer(calculation, {"final_value": calculation.answer})


def test_calculation_scoring_treats_trailing_zero_as_equivalent():
    item = GroundTruthItem(qid="synthetic_decimal", domain="insurance", question="合成小数比较", type="计算题", answer="42.00")
    assert score_answer(item, {"final_value": "42"})
