from langchain_core.messages import HumanMessage

from app.agent.graph import FinancialQAAgent, _route_domains
from app.config import Settings
from app.schemas import StructuredAnswer


class _FakeRunner:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return next(self.responses)


class _FakeModel:
    def __init__(self, responses):
        self.runner = _FakeRunner(responses)

    def with_structured_output(self, schema, include_raw=False):
        assert schema is StructuredAnswer
        assert include_raw is True
        return self.runner


def test_structured_answer_retries_after_validation_failure():
    model = _FakeModel([
        {"parsed": None, "parsing_error": "confidence must be low, medium or high"},
        {"parsed": StructuredAnswer(answer="修正后的答案", confidence="high"), "parsing_error": None},
    ])
    parsed, attempts, error = FinancialQAAgent._invoke_structured(
        model,
        StructuredAnswer,
        [HumanMessage(content="问题")],
        retries=2,
    )
    assert parsed is not None
    assert parsed.confidence == "high"
    assert attempts == 2
    assert error is None
    assert model.runner.calls == 2


def test_unknown_keyword_domain_requires_api_classification():
    assert _route_domains("请分析这份材料并回答问题") == []
    assert _route_domains("保险等待期是多少") == ["insurance"]


def test_finalize_returns_safe_structure_when_all_retries_fail(tmp_path):
    model = _FakeModel([
        {"parsed": None, "parsing_error": "confidence input_value=1.0"},
        {"parsed": None, "parsing_error": "confidence input_value=1.0"},
    ])
    agent = object.__new__(FinancialQAAgent)
    agent.settings = Settings(
        data_root=tmp_path,
        model="flash",
        answer_model="max",
        base_url=None,
        api_key="test",
        demo_mode=False,
        final_answer_retries=1,
    )
    agent.answer_model = model
    result = agent._finalize({"question": "问题", "messages": [], "trace": []})
    assert result["final_answer"]["confidence"] == "low"
    assert "多次返回" in result["final_answer"]["answer"]
    assert result["trace"][0]["details"]["attempts"] == 2
    assert result["trace"][0]["details"]["fallback_used"] is True


def test_insufficient_evidence_reuses_api_query_with_candidate_domain(tmp_path):
    agent = object.__new__(FinancialQAAgent)
    agent.settings = Settings(
        data_root=tmp_path,
        model="flash",
        answer_model="max",
        base_url=None,
        api_key="test",
        demo_mode=False,
        max_tool_rounds=3,
    )
    result = agent._agent({
        "question": "原问题",
        "messages": [],
        "trace": [],
        "tool_rounds": 1,
        "candidate_domains": ["financial_contracts"],
        "evidence_sufficient": False,
        "refined_query": "API 生成的新检索词",
    })
    call = result["messages"][0].tool_calls[0]
    assert call["name"] == "search_financial_contracts"
    assert call["args"]["query"] == "API 生成的新检索词"
