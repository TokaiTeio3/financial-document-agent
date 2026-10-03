from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "corpus"
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True, slots=True)
class Settings:
    data_root: Path
    model: str
    base_url: str | None
    api_key: str | None
    demo_mode: bool
    answer_model: str | None = None
    planner_model: str | None = None
    index_root: Path = PROJECT_ROOT / "runtime_index"
    chunk_size: int = 900
    chunk_overlap: int = 120
    page_overlap: int = 240
    search_top_k: int = 10
    max_tool_rounds: int = 3
    final_answer_retries: int = 2
    structured_call_retries: int = 1
    request_timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "Settings":
        data_root = Path(os.getenv("FINQA_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        data_root = data_root.resolve()
        index_root = Path(os.getenv("FINQA_INDEX_ROOT", str(PROJECT_ROOT / "runtime_index"))).expanduser()
        if not index_root.is_absolute():
            index_root = PROJECT_ROOT / index_root
        api_key = os.getenv("OPENAI_API_KEY")
        requested_demo = os.getenv("FINQA_DEMO_MODE", "").lower() in {"1", "true", "yes"}
        tool_model = os.getenv("FINQA_TOOL_MODEL", os.getenv("FINQA_MODEL", "qwen3.7-flash"))
        return cls(
            data_root=data_root,
            model=tool_model,
            answer_model=os.getenv("FINQA_ANSWER_MODEL", tool_model),
            planner_model=os.getenv("FINQA_PLANNER_MODEL", "qwen3.7-max"),
            index_root=index_root.resolve(),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=api_key,
            demo_mode=requested_demo or not api_key,
            chunk_size=int(os.getenv("FINQA_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("FINQA_CHUNK_OVERLAP", "120")),
            page_overlap=int(os.getenv("FINQA_PAGE_OVERLAP", "240")),
            search_top_k=int(os.getenv("FINQA_TOP_K", "10")),
            max_tool_rounds=int(os.getenv("FINQA_MAX_TOOL_ROUNDS", "3")),
            final_answer_retries=int(os.getenv("FINQA_FINAL_RETRIES", "2")),
            structured_call_retries=int(os.getenv("FINQA_STRUCTURED_RETRIES", "1")),
            request_timeout=float(os.getenv("FINQA_REQUEST_TIMEOUT", "120")),
        )


settings = Settings.from_env()
