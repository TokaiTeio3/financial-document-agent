from langchain_core.messages import AIMessage

from app.agent.graph import _usage_from_message


def test_usage_metadata_is_normalized_for_frontend():
    message = AIMessage(content="ok", usage_metadata={
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    })
    usage = _usage_from_message(message, "flash", "evidence_compression")
    assert usage == {
        "model": "flash",
        "stage": "evidence_compression",
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }
