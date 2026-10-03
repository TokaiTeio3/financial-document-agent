from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import FinancialQAAgent
from app.config import settings
from app.retrieval.corpus import CorpusRegistry
from app.schemas import ChatRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = CorpusRegistry(
        settings.data_root,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.page_overlap,
        settings.index_root,
    )
    app.state.agent = FinancialQAAgent(settings, registry)
    yield


app = FastAPI(title="金融文档 Agent", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "demo" if settings.demo_mode else "model", "data_root_exists": settings.data_root.is_dir()}


@app.get("/api/meta")
async def meta():
    return {
        "mode": "demo" if settings.demo_mode else "model",
        "model": None if settings.demo_mode else settings.model,
        "tool_model": None if settings.demo_mode else settings.model,
        "answer_model": None if settings.demo_mode else settings.model,
        "planner_model": None if settings.demo_mode else settings.planner_model,
        "domains": list(CorpusRegistry.DOMAINS),
        "retrieval": "BM25 + keyword anchors + page overlap",
    }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def events():
        try:
            yield json.dumps({"type": "status", "data": {"message": "Agent 已接收问题"}}, ensure_ascii=False) + "\n"
            async for event in app.state.agent.stream(request.question):
                yield json.dumps(event, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done", "data": {}}, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "data": {"message": str(exc)}}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


STATIC_DIR = Path(__file__).parent / "web" / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
