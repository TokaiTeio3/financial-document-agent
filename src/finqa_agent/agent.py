"""金融长文档问答 Agent 的可组合外观入口。

具体行为分布在按职责拆分的混入类中。本模块特意保留运行器、测试和审计工具使用的
稳定 ``CleanFinancialQAAgent`` 导入路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .reasoning.agent_orchestration import AgentOrchestrationMixin
from .core.agent_types import AnswerResult
from .reasoning.answer_memory import PriorApiAnswerMemory
from .runtime.audit import RunAuditLogger
from .calculation.calculation_routes import CalculationRoutesMixin
from .calculation.calculator import RestrictedPythonExecutor
from .reasoning.compact_option_routes import CompactOptionRoutesMixin
from .reasoning.decision_rules import DecisionRulesMixin
from .core.entities import RuntimeEntityMemory
from .retrieval.index import DocumentIndex
from .reasoning.joint_reasoning_routes import JointReasoningRoutesMixin
from .runtime.llm import QwenClient
from .retrieval.planner import QueryPlanner
from .retrieval.retrieval_helpers import RetrievalHelpersMixin
from .retrieval.retrieval_strategies import RetrievalStrategiesMixin


class CleanFinancialQAAgent(
    AgentOrchestrationMixin,
    CompactOptionRoutesMixin,
    RetrievalStrategiesMixin,
    JointReasoningRoutesMixin,
    CalculationRoutesMixin,
    RetrievalHelpersMixin,
    DecisionRulesMixin,
):
    def __init__(
        self,
        config: Dict[str, object],
        profile_name: str = "long",
        audit_logger: RunAuditLogger | None = None,
    ) -> None:
        self.config = config
        profile = (config.get("profiles") or {}).get(profile_name, {})
        if not isinstance(profile, dict):
            profile = {}
        self.profile = profile
        self.profile_name = profile_name
        self.audit = audit_logger
        model_cfg = config.get("model") or {}
        data_cfg = config.get("data") or {}
        self.llm = QwenClient(
            model=str(model_cfg.get("name", "qwen3.7-max-2026-06-08")),
            fallback_model=str(model_cfg.get("fallback_name", model_cfg.get("name", "qwen3.7-max-2026-06-08"))),
            api_base=str(model_cfg.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            temperature=float(model_cfg.get("temperature", 0.0)),
            top_p=float(model_cfg.get("top_p", 1.0)),
            seed=(
                int(model_cfg["seed"])
                if model_cfg.get("seed") is not None
                else None
            ),
            enable_thinking=bool(model_cfg.get("enable_thinking", False)),
            token_budget=int(profile.get("token_budget", 0)),
            timeout_seconds=float(model_cfg.get("timeout_seconds", 180.0)),
            max_retries=int(model_cfg.get("max_retries", 2)),
            audit_logger=audit_logger,
        )
        runtime_dir = Path(str(data_cfg.get("runtime_index_dir", "runtime_index")))
        self.index = DocumentIndex(
            docs_dir=str(data_cfg.get("docs_dir", "data/processed_pymupdf4llm_merged_normalized")),
            cache_dir=runtime_dir,
        )
        self.index.load()
        self.planner = QueryPlanner(self.llm, int(profile.get("query_max_tokens", 1200)))
        self.executor = RestrictedPythonExecutor()
        agent_cfg = config.get("agent") or {}
        self.memory = RuntimeEntityMemory(
            runtime_dir / "question_entity_memory.jsonl",
            enabled=bool(agent_cfg.get("runtime_entity_memory", True)),
        )
        self.answer_memory = PriorApiAnswerMemory(
            str(data_cfg.get("prior_api_answer_memory", runtime_dir / "prior_api_answer_memory.json")),
            enabled=bool(profile.get("use_prior_api_answer_memory", False)),
        )
