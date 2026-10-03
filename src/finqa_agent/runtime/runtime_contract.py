"""冷启动竞赛运行的生产运行契约检查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class RuntimeContract:
    schema_version: int
    profile: str
    model: str
    cold_start: bool
    no_embedding: bool
    answer_memory: bool
    cross_round_memory: bool
    runtime_entity_memory: bool
    token_budget: int
    submission_token_limit: int
    workers: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def validate_runtime_contract(
    config: Mapping[str, object],
    profile_name: str,
    *,
    workers: int,
) -> RuntimeContract:
    model_cfg = config.get("model") or {}
    profiles = config.get("profiles") or {}
    profile = profiles.get(profile_name) or {}
    agent_cfg = config.get("agent") or {}
    model = str(model_cfg.get("name", ""))
    if not model.lower().startswith("qwen"):
        raise ValueError("production inference model must be a Qwen model")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    no_embedding = bool(agent_cfg.get("no_embedding", False))
    if not no_embedding:
        raise ValueError("production contract requires no_embedding=true")

    answer_memory = bool(profile.get("use_prior_api_answer_memory", False))
    cross_round = bool(agent_cfg.get("cross_round_memory", False))
    entity_memory = bool(agent_cfg.get("runtime_entity_memory", False))
    cold_start = not (answer_memory or cross_round or entity_memory)
    if profile_name == "reproduction" and not cold_start:
        raise ValueError("reproduction profile must be a cold-start run")

    token_budget = int(profile.get("token_budget", 0))
    submission_limit = int(profile.get("submission_token_limit", 0))
    if submission_limit and token_budget and submission_limit > token_budget:
        raise ValueError(
            "submission_token_limit cannot exceed the protected token_budget"
        )
    return RuntimeContract(
        schema_version=1,
        profile=profile_name,
        model=model,
        cold_start=cold_start,
        no_embedding=no_embedding,
        answer_memory=answer_memory,
        cross_round_memory=cross_round,
        runtime_entity_memory=entity_memory,
        token_budget=token_budget,
        submission_token_limit=submission_limit,
        workers=workers,
    )
