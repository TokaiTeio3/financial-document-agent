"""按真实运行轨迹组织的一次问答任务流水线。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..agent import CleanFinancialQAAgent
from .audit import RunAuditLogger
from ..retrieval.evidence_trace import reasoning_evidence_references
from .io import (
    answer_parts,
    load_config,
    load_questions,
    save_json,
    timestamped_run_dir,
    write_submission,
)
from .observability import export_observability_artifacts
from .run_identity import capture_run_identity, capture_system_identity
from .runtime_contract import validate_runtime_contract


@dataclass(frozen=True)
class RunOptions:
    """CLI 参数解析后的稳定运行选项。"""

    split: str
    profile: str
    config: str
    questions_dir: str | None
    docs_dir: str | None
    runtime_index_dir: str | None
    output: str | None
    submission_output: str | None
    run_dir: str | None
    limit: int | None
    workers: int
    qids: set[str]
    option: str | None


class FinancialQARunPipeline:
    """将一次运行拆成契约校验、数据准备、单题推理、落盘与审计。"""

    def __init__(self, options: RunOptions) -> None:
        self.options = options
        self.config = load_config(options.config)
        self.profile = self.config["profiles"][options.profile]
        self.runtime_contract = validate_runtime_contract(
            self.config,
            options.profile,
            workers=max(1, options.workers),
        )
        self.system_identity = capture_system_identity(
            options.config,
            options.profile,
            self.profile,
        )
        self.data_cfg = self.config.setdefault("data", {})

    def run(self) -> None:
        self._enforce_git_policy()
        self._apply_path_overrides()
        questions = self._load_selected_questions()
        if not questions:
            print("No questions found.")
            return

        run_identity = self._capture_run_identity(questions)
        run_dir = self._create_run_dir()
        audit = self._start_audit(run_dir, questions, run_identity)
        agent = CleanFinancialQAAgent(
            self.config,
            self.options.profile,
            audit_logger=audit,
        )
        results: Dict[str, Dict[str, object]] = {}

        try:
            self._answer_questions(results, agent, audit, questions)
        except Exception as exc:
            self._write_failure_manifest(
                run_dir,
                audit,
                results,
                questions,
                run_identity,
                exc,
            )
            print(f"Failed run directory: {run_dir}")
            raise

        self._write_success_outputs(run_dir, audit, results, run_identity)

    def _enforce_git_policy(self) -> None:
        if bool(self.profile.get("require_clean_git", False)) and bool(
            self.system_identity["git_dirty"]
        ):
            raise RuntimeError(
                "The selected profile requires a clean tracked Git worktree."
            )

    def _apply_path_overrides(self) -> None:
        if self.options.questions_dir:
            self.data_cfg["questions_dir"] = self.options.questions_dir
        if self.options.docs_dir:
            self.data_cfg["docs_dir"] = self.options.docs_dir
        if self.options.runtime_index_dir:
            runtime_index = Path(self.options.runtime_index_dir)
            self.data_cfg["runtime_index_dir"] = str(runtime_index)
            self.data_cfg["prior_api_answer_memory"] = str(
                runtime_index / "prior_api_answer_memory.json"
            )

    def _load_selected_questions(self) -> List[Dict[str, object]]:
        questions = load_questions(
            self.data_cfg.get("questions_dir", "data/raw_dataset/questions"),
            self.options.split,
            self.options.qids,
        )
        if self.options.option:
            for question in questions:
                options = question.get("options") or {}
                question["options"] = (
                    {self.options.option: options[self.options.option]}
                    if self.options.option in options
                    else {}
                )
        if self.options.limit is not None:
            questions = questions[: self.options.limit]
        return questions

    def _capture_run_identity(
        self,
        questions: List[Dict[str, object]],
    ) -> Dict[str, object]:
        selected_qids = sorted(str(question.get("qid", "")) for question in questions)
        return capture_run_identity(
            self.system_identity,
            {
                "split": self.options.split,
                "profile": self.options.profile,
                "option": self.options.option,
                "qids": selected_qids,
            },
        )

    def _create_run_dir(self) -> Path:
        output_dir = Path(self.data_cfg.get("output_dir", "output"))
        run_dir = (
            Path(self.options.run_dir)
            if self.options.run_dir
            else timestamped_run_dir(output_dir)
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _start_audit(
        self,
        run_dir: Path,
        questions: List[Dict[str, object]],
        run_identity: Dict[str, object],
    ) -> RunAuditLogger:
        audit = RunAuditLogger(
            run_dir / "run.log.jsonl",
            run_id=str(run_identity["run_fingerprint"]),
        )
        audit.event(
            "runtime_contract_validated",
            profile=self.options.profile,
            contract=self.runtime_contract.to_dict(),
        )
        audit.event(
            "run_started",
            split=self.options.split,
            profile=self.options.profile,
            workers=max(1, self.options.workers),
            question_count=len(questions),
            run_dir=str(run_dir),
            run_identity=run_identity,
        )
        return audit

    def _answer_questions(
        self,
        results: Dict[str, Dict[str, object]],
        agent: CleanFinancialQAAgent,
        audit: RunAuditLogger,
        questions: List[Dict[str, object]],
    ) -> None:
        with ThreadPoolExecutor(max_workers=max(1, self.options.workers)) as pool:
            future_to_qid = {
                pool.submit(
                    self._answer_one,
                    agent,
                    audit,
                    index,
                    len(questions),
                    question,
                ): str(question.get("qid", ""))
                for index, question in enumerate(questions)
            }
            for future in as_completed(future_to_qid):
                row = future.result()
                results[str(row["qid"])] = row

    def _answer_one(
        self,
        agent: CleanFinancialQAAgent,
        audit: RunAuditLogger,
        index: int,
        total_questions: int,
        question: Dict[str, object],
    ) -> Dict[str, object]:
        audit.event(
            "question_dispatched",
            qid=str(question.get("qid", "")),
            index=index,
            domain=str(question.get("domain", "")),
            answer_format=str(question.get("answer_format", "")),
        )
        try:
            result = agent.answer_question(question)
        except Exception as exc:
            audit.event(
                "question_failed",
                qid=str(question.get("qid", "")),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        row = {
            "qid": result.qid,
            "answer": result.answer,
            "answer_parts": answer_parts(
                result.answer,
                str(question.get("answer_format", "")),
            ),
            "reasoning": result.reasoning,
            "domain": question.get("domain", ""),
            "split": question.get("split", self.options.split),
            **result.usage.to_dict(),
            "system_fingerprint": self.system_identity["system_fingerprint"],
            **result.metadata,
        }
        print(
            f"[{index + 1}/{total_questions}] {result.qid} -> "
            f"{result.answer} | tokens={result.usage.total_tokens}"
        )
        audit.event(
            "question_completed",
            qid=result.qid,
            answer=result.answer,
            reasoning=result.reasoning,
            reasoning_evidence_refs=reasoning_evidence_references(
                result.reasoning
            ),
            usage=result.usage.to_dict(),
        )
        return row

    def _write_failure_manifest(
        self,
        run_dir: Path,
        audit: RunAuditLogger,
        results: Dict[str, Dict[str, object]],
        questions: List[Dict[str, object]],
        run_identity: Dict[str, object],
        exc: Exception,
    ) -> None:
        total = sum(int(row.get("total_tokens", 0)) for row in results.values())
        save_json(
            {
                "status": "failed",
                "split": self.options.split,
                "profile": self.options.profile,
                "question_count": len(results),
                "requested_question_count": len(questions),
                "total_tokens": total,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "audit_log": audit.path.name,
                "run_identity": run_identity,
            },
            run_dir / "manifest.json",
        )
        audit.event(
            "run_failed",
            completed_question_count=len(results),
            requested_question_count=len(questions),
            total_tokens=total,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def _write_success_outputs(
        self,
        run_dir: Path,
        audit: RunAuditLogger,
        results: Dict[str, Dict[str, object]],
        run_identity: Dict[str, object],
    ) -> None:
        json_name = (
            Path(self.options.output).name
            if self.options.output
            else f"results_{self.options.split.lower()}_{self.options.profile}.json"
        )
        csv_name = (
            Path(self.options.submission_output).name
            if self.options.submission_output
            else f"submission_{self.options.split.lower()}_{self.options.profile}.csv"
        )
        json_path = run_dir / json_name
        csv_path = run_dir / csv_name
        save_json(results, json_path)
        write_submission(results, csv_path)
        total = sum(int(r.get("total_tokens", 0)) for r in results.values())
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV: {csv_path}")
        print(f"Total tokens: {total}")
        audit.event(
            "run_completed",
            question_count=len(results),
            total_tokens=total,
        )
        observability_artifacts = export_observability_artifacts(
            audit.path,
            run_dir,
        )
        save_json(
            {
                "status": "completed",
                "split": self.options.split,
                "profile": self.options.profile,
                "question_count": len(results),
                "prompt_tokens": sum(
                    int(r.get("prompt_tokens", 0)) for r in results.values()
                ),
                "completion_tokens": sum(
                    int(r.get("completion_tokens", 0)) for r in results.values()
                ),
                "total_tokens": total,
                "results": json_path.name,
                "submission": csv_path.name,
                "audit_log": audit.path.name,
                **observability_artifacts,
                "runtime_contract": self.runtime_contract.to_dict(),
                "run_identity": run_identity,
            },
            run_dir / "manifest.json",
        )
        print(f"Run directory: {run_dir}")
