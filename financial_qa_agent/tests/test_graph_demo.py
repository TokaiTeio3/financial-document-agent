import pytest

pytest.importorskip("langgraph")

from app.agent.graph import FinancialQAAgent
from app.config import Settings
from app.retrieval.corpus import CorpusRegistry


@pytest.mark.asyncio
async def test_demo_graph_streams_trace_and_structured_answer():
    from pathlib import Path

    fixture_root = Path(__file__).parent / "fixtures" / "corpus"
    settings = Settings(
        data_root=fixture_root,
        model="unused",
        base_url=None,
        api_key=None,
        demo_mode=True,
        chunk_size=300,
        chunk_overlap=20,
        search_top_k=3,
        max_tool_rounds=2,
    )
    agent = FinancialQAAgent(settings, CorpusRegistry(fixture_root, 300, 20))
    events = [event async for event in agent.stream("示例文件的营业收入是多少？")]
    assert [event for event in events if event["type"] == "trace"]
    answer = next(event["data"] for event in events if event["type"] == "answer")
    assert answer["citations"][0]["document_id"] == "sample"
    assert answer["confidence"] in {"low", "medium", "high"}
