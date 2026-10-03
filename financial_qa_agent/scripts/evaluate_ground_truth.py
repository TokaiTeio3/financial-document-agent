from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.agent.graph import FinancialQAAgent
from app.config import Settings
from app.evaluation.ground_truth import GroundTruthItem, load_ground_truth, score_answer
from app.retrieval.corpus import CorpusRegistry


DEFAULT_GROUND_TRUTH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ground_truth.json"


def document_matches(actual: str, expected: list[str]) -> bool:
    actual = actual.lower()
    return any(actual == item or actual.startswith(item) or item.startswith(actual) for item in expected)


def retrieval_result(registry: CorpusRegistry, item: GroundTruthItem, top_k: int) -> dict[str, Any]:
    hits = registry.get(item.domain).search(item.prompt, top_k=top_k)
    expected_docs = item.expected_document_ids
    doc_rank = next(
        (rank for rank, hit in enumerate(hits, 1) if document_matches(hit.document_id, expected_docs)),
        None,
    )
    page_hit = any(
        document_matches(hit.document_id, expected_docs)
        and hit.page is not None
        and any(abs(hit.page - page) <= 1 for page in item.expected_pages)
        for hit in hits
    ) if item.expected_pages else None
    return {
        "qid": item.qid,
        "domain": item.domain,
        "expected_documents": expected_docs,
        "expected_pages": item.expected_pages,
        "document_rank": doc_rank,
        "document_hit": doc_rank is not None,
        "page_hit": page_hit,
        "hits": [
            {"document_id": hit.document_id, "page": hit.page, "score": hit.score}
            for hit in hits
        ],
    }


async def agent_result(agent: FinancialQAAgent, item: GroundTruthItem) -> dict[str, Any]:
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    token_usage: dict[str, Any] = {}
    answer: dict[str, Any] = {}
    error: str | None = None
    try:
        async for event in agent.stream(item.prompt):
            if event["type"] == "trace":
                traces.append(event["data"])
            elif event["type"] == "token_usage":
                token_usage = event["data"]
            elif event["type"] == "answer":
                answer = event["data"]
    except Exception as exc:
        error = str(exc)
    trace_nodes = {event.get("node") for event in traces}
    flow_completed = bool(answer) and error is None and {"prepare", "decompose", "worker", "finalize"}.issubset(trace_nodes)
    return {
        "qid": item.qid,
        "domain": item.domain,
        "question_type": item.type,
        "expected": item.answer,
        "answer": answer,
        "correct": bool(answer) and score_answer(item, answer),
        "flow_completed": flow_completed,
        "trace": traces,
        "token_usage": token_usage,
        "error": error,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agent flow on synthetic fixtures or an explicitly supplied, authorized dataset.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--mode", choices=("retrieval", "agent", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected questions")
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args()

    items = load_ground_truth(args.ground_truth)
    if args.qid:
        wanted = set(args.qid)
        items = [item for item in items if item.qid in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        parser.error("No questions selected; check the supplied dataset and --qid filter.")

    settings = Settings.from_env()
    registry = CorpusRegistry(
        settings.data_root,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.page_overlap,
        settings.index_root,
    )
    payload: dict[str, Any] = {
        "ground_truth": str(args.ground_truth.resolve()),
        "question_count": len(items),
        "mode": args.mode,
        "evaluation_scope": "flow_only" if settings.demo_mode else "model_run",
    }

    if args.mode in {"retrieval", "all"}:
        retrieval = [retrieval_result(registry, item, args.top_k) for item in items]
        payload["retrieval"] = retrieval
        page_rows = [row for row in retrieval if row["page_hit"] is not None]
        payload["retrieval_summary"] = {
            "document_recall": sum(row["document_hit"] for row in retrieval) / max(1, len(retrieval)),
            "page_recall": sum(row["page_hit"] for row in page_rows) / max(1, len(page_rows)),
            "document_hits": sum(row["document_hit"] for row in retrieval),
            "page_hits": sum(row["page_hit"] for row in page_rows),
        }
        print(json.dumps(payload["retrieval_summary"], ensure_ascii=False))

    if args.mode in {"agent", "all"}:
        agent = FinancialQAAgent(settings, registry)
        results = []
        for index, item in enumerate(items, 1):
            result = await agent_result(agent, item)
            results.append(result)
            status = "FLOW_OK" if result["flow_completed"] else "FLOW_FAIL"
            print(f"[{index}/{len(items)}] {item.qid}: {status} ({result['elapsed_seconds']}s)")
        payload["agent"] = results
        payload["agent_summary"] = {
            "flow_success_rate": sum(row["flow_completed"] for row in results) / max(1, len(results)),
            "flow_completed": sum(row["flow_completed"] for row in results),
            "errors": sum(bool(row["error"]) for row in results),
            "answer_accuracy_reference": None if settings.demo_mode else sum(row["correct"] for row in results) / max(1, len(results)),
        }
        print(json.dumps(payload["agent_summary"], ensure_ascii=False))

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    if args.mode in {"agent", "all"} and any(not row["flow_completed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
