from __future__ import annotations

import json
import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from scripts.build_evidence import build_evidence
from scripts.config_dashboard import coerce_updates, materialize_config
from scripts.validate_submission_bundle import REQUIRED_COLUMNS, validate_answer
from src.finqa_agent.reasoning.answer_reconciliation import (
    explicit_segment_verdict,
    repair_numeric_self_consistency_verdict,
    repair_region_share_answer_from_reasoning,
)
from src.finqa_agent.runtime.audit import RunAuditLogger
from src.finqa_agent.calculation.calculation_reconciliation import (
    percent_rate_tail_from_reasoning,
)
from src.finqa_agent.retrieval.index import Page, SearchHit
from src.finqa_agent.retrieval.evidence_trace import (
    extract_evidence_table,
    reasoning_evidence_references,
)
from src.finqa_agent.runtime.llm import QwenClient
from src.finqa_agent.runtime.observability import export_observability_artifacts
from src.finqa_agent.reasoning.reasoning_policy import assign_reasoning_level
from src.finqa_agent.runtime.runtime_contract import validate_runtime_contract
from src.finqa_agent.core.entities import extract_constraint_terms
from src.finqa_agent.runtime.io import load_config
from src.finqa_agent.reasoning.decision_rules import DecisionRulesMixin
from src.finqa_agent.retrieval.retrieval_helpers import RetrievalHelpersMixin
from src.finqa_agent.retrieval.retrieval_strategies import RetrievalStrategiesMixin
from src.preprocess.normalize_tree import normalize_tree
from src.preprocess.prepare_data import unzip_dataset


class _FakeCompletions:
    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=17,
                completion_tokens=9,
                total_tokens=26,
            ),
            model=request["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"answer":"A","reasoning":"证据支持A"}'),
                    finish_reason="stop",
                )
            ],
        )


class ReproductionPipelineTest(unittest.TestCase):
    def test_preprocessor_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as value:
                value.writestr("dataset/../../escaped.txt", "unsafe")
            with self.assertRaises(ValueError):
                unzip_dataset(archive, root / "extract")
            self.assertFalse((root / "escaped.txt").exists())

    def test_dashboard_writes_a_valid_copy_without_mutating_frozen_config(self) -> None:
        source = Path("config/config.ultra.yaml")
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dashboard.yaml"
            config = materialize_config(
                source,
                destination,
                {"temperature": "0.15", "retrieval_top_k": "18"},
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(config["model"]["temperature"], 0.15)
            self.assertEqual(
                config["profiles"]["reproduction"]["retrieval_top_k"],
                18,
            )
        self.assertEqual(source.read_bytes(), before)

    def test_dashboard_rejects_out_of_range_submission_parameters(self) -> None:
        with self.assertRaises(ValueError):
            coerce_updates({"top_p": 1.5})

    def test_submission_answer_validator_checks_synthetic_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answer.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                for index in range(100):
                    writer.writerow({
                        "qid": f"synthetic_{index}", "answer_1": "A",
                        "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                        "reasoning": "合成证据支持示例选项。",
                    })
                writer.writerow({
                    "qid": "summary", "prompt_tokens": 1000,
                    "completion_tokens": 200, "total_tokens": 1200,
                })
            totals = validate_answer(path)
        self.assertEqual(totals["prompt_tokens"], 1000)
        self.assertEqual(totals["completion_tokens"], 200)
        self.assertEqual(totals["total_tokens"], 1200)

    def test_evidence_table_is_extracted_from_exact_prompt_blocks(self) -> None:
        rows = extract_evidence_table(
            [
                {
                    "role": "user",
                    "content": (
                        "本轮证据：\n"
                        "[E1] doc=text01 page=page_0002 score=8.50\n"
                        "期限为十年。\n"
                        "[E2] doc=text02 page=page_0003 score=7.25\n"
                        "存在明确例外。"
                    ),
                }
            ]
        )
        self.assertEqual([row["label"] for row in rows], ["E1", "E2"])
        self.assertEqual(rows[0]["doc_id"], "text01")
        self.assertEqual(rows[0]["page_id"], "page_0002")
        self.assertEqual(rows[0]["excerpt"], "期限为十年。")
        self.assertEqual(
            reasoning_evidence_references("E1支持期限，证据[2]说明例外。"),
            ["E1", "E2"],
        )

    def test_reasoning_level_exposes_structural_signals_without_routing(self) -> None:
        allocation = assign_reasoning_level(
            {
                "question": "比较2023、2024、2025年三项指标并计算差值",
                "answer_format": "calc",
                "options": {},
            },
            route="compact_calculation_program",
            document_count=2,
            profile={"compact_calc_reasoning_max_tokens": 210},
        )
        self.assertEqual(allocation.level, "L3")
        self.assertIn("multi_year_series", allocation.signals)
        self.assertEqual(allocation.configured_max_tokens, 1810)

    def test_reproduction_runtime_contract_is_cold_start(self) -> None:
        config = load_config("config/config.ultra.yaml")
        contract = validate_runtime_contract(
            config,
            "reproduction",
            workers=4,
        )
        self.assertTrue(contract.cold_start)
        self.assertTrue(contract.no_embedding)
        self.assertFalse(contract.answer_memory)

    def test_focused_excerpt_keeps_highest_scoring_table_row_first(self) -> None:
        text = "\n".join(
            [
                "|经营活动现金流入小计|100|20|",
                "|支付其他经营现金|80|30|",
                "|经营活动产生的现金流量净额|20|-10|",
                "|投资活动现金流入小计|500|600|",
                "|投资活动产生的现金流量净额|100|200|",
            ]
            * 8
        )
        excerpt = RetrievalHelpersMixin._focused_excerpt(
            text,
            "经营活动产生的现金流量净额",
            220,
        )
        self.assertIn("经营活动产生的现金流量净额|20|-10", excerpt)

    def test_reproduction_profile_has_no_answer_cardinality_prior(self) -> None:
        profile = load_config("config/config.ultra.yaml")["profiles"]["reproduction"]
        self.assertEqual(profile["multi_min_options"], 0)
        self.assertEqual(profile["submission_token_limit"], 575001)

    def test_constraint_terms_are_derived_from_current_clause(self) -> None:
        terms = extract_constraint_terms(
            "关于保险金请求权的诉讼时效期间为2年的说法",
            "insurance",
        )
        self.assertIn("诉讼时效期间", terms)
        self.assertIn("2年", terms)

    def test_directional_reasoning_overrides_stale_supported_verdict(self) -> None:
        self.assertEqual(
            repair_numeric_self_consistency_verdict(
                "利润减少额高于现金流减少额",
                "本期利润高于上期，实为增加而非减少。",
                "supported",
            ),
            "contradicted",
        )

    def test_negated_error_does_not_override_supported_verdict(self) -> None:
        self.assertEqual(
            explicit_segment_verdict(
                "省略另一并行门槛不构成错误，选项与证据完全一致。",
                "supported",
            ),
            "supported",
        )

    def test_invalid_dividend_sum_is_contradicted(self) -> None:
        self.assertEqual(
            repair_numeric_self_consistency_verdict(
                "年末每10股分红38元，加上每10股5元中期分红后，全年每10股分红为48元",
                "证据列出年末38元和中期5元。",
                "supported",
            ),
            "contradicted",
        )

    def test_compound_calculation_recovers_api_formula_results(self) -> None:
        reasoning = (
            "同比增幅=(310-221)/221≈40.05%；"
            "两期占比分别为38.65%和28.55%，提高10.10个百分点。"
        )
        self.assertEqual(
            DecisionRulesMixin._final_calculation_answer_from_reasoning(
                reasoning,
                "答案格式为“同比增幅；占比提高百分点”，均不带单位。",
            ),
            "40.05；10.10",
        )

    def test_negated_conflict_does_not_override_explicit_support(self):
        reasoning = (
            "证据明确禁止将结果用于广告，直接支持选项表述。"
            "义务主体和用途范围均一致，无冲突或信息缺失。"
        )
        self.assertEqual(
            explicit_segment_verdict(reasoning, "supported"),
            "supported",
        )

    def test_uncertain_whether_correct_is_not_parsed_as_supported(self):
        reasoning = (
            "证据缺少覆盖该年龄的给付比例，无法计算并比较，"
            "故无法判断选项是否正确。"
        )
        self.assertEqual(
            explicit_segment_verdict(reasoning, "unknown"),
            "unknown",
        )

    def test_english_verdict_next_to_chinese_text_has_a_boundary(self):
        reasoning = (
            "D项算式与选项一致，supported。"
            "综上A contradicted，其余supported。"
        )
        self.assertEqual(
            explicit_segment_verdict(reasoning, "contradicted"),
            "supported",
        )

    def test_document_cap_preserves_two_hits_per_compared_document(self):
        hits = [
            SearchHit(Page("financial_reports", doc, f"page_{index:04d}", "x", ""), 10 - index, "q")
            for doc in ("doc_a", "doc_b", "doc_c")
            for index in (1, 2)
        ]
        capped = RetrievalStrategiesMixin._cap_hits_preserving_documents(
            hits,
            cap=3,
            minimum_per_document=2,
        )
        counts = {
            doc: sum(hit.page.doc_id == doc for hit in capped)
            for doc in ("doc_a", "doc_b", "doc_c")
        }
        self.assertEqual(counts, {"doc_a": 2, "doc_b": 2, "doc_c": 2})

    def test_region_share_repair_distinguishes_points_from_relative_growth(self):
        reasoning = (
            "2025年境外收入310,740,988千元，中国493,223,970千元；"
            "2024年境外收入221,884,773千元，中国555,217,682千元。"
        )
        options = {
            "A": "境外收入占比提高约10.10个百分点",
            "B": "境外收入占比相对增幅约10.10%",
            "C": "另一项已核验结论",
        }
        self.assertEqual(
            repair_region_share_answer_from_reasoning(
                "BC",
                reasoning,
                options,
                "multi",
            ),
            "AC",
        )

    def test_decimal_rate_program_is_rendered_in_percentage_points(self):
        question = (
            "资产收益率ROA=净资产收益率ROE÷权益乘数，"
            "权益乘数=1÷(1-资产负债率)，收益率不带%。"
        )
        program = (
            "asset_liability_ratio = 61.17 / 100\n"
            "roe = 19.70 / 100"
        )
        self.assertEqual(
            percent_rate_tail_from_reasoning(program, question),
            "7.65",
        )

    def test_api_audit_contains_full_prompt_parameters_output_and_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log.jsonl"
            audit = RunAuditLogger(log_path)
            client = QwenClient(
                model="qwen-test",
                api_base="https://example.invalid/v1",
                api_key="test-key",
                temperature=0.0,
                top_p=0.9,
                seed=7,
                max_retries=0,
                audit_logger=audit,
            )
            completions = _FakeCompletions()
            client.client = SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )
            client.set_question_context("sample_qid")
            client.set_route_context("qwen_test_route")
            result = client.chat(
                [{"role": "user", "content": "完整测试 prompt"}],
                max_tokens=123,
            )

            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            request = next(row for row in records if row["event"] == "api_request")
            response = next(row for row in records if row["event"] == "api_response")
            self.assertEqual(request["qid"], "sample_qid")
            self.assertEqual(request["route"], "qwen_test_route")
            self.assertEqual(request["messages"][0]["content"], "完整测试 prompt")
            self.assertEqual(request["parameters"]["seed"], 7)
            self.assertEqual(request["parameters"]["top_p"], 0.9)
            self.assertEqual(response["raw_output"], result.content)
            self.assertEqual(response["usage"]["total_tokens"], 26)

    def test_api_audit_emits_call_scoped_evidence_table(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log.jsonl"
            audit = RunAuditLogger(log_path, run_id="run-test")
            client = QwenClient(
                model="qwen-test",
                api_base="https://example.invalid/v1",
                api_key="test-key",
                max_retries=0,
                audit_logger=audit,
            )
            client.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_FakeCompletions())
            )
            client.set_question_context("sample_qid")
            client.set_route_context("qwen_test_route")
            client.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "证据：\n"
                            "[1] domain=regulatory doc_id=text01 "
                            "page=page_0001 score=9.0\n期限为十年。"
                        ),
                    }
                ],
                max_tokens=50,
            )
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            table = next(
                row for row in records if row["event"] == "evidence_table"
            )
            self.assertEqual(table["run_id"], "run-test")
            self.assertEqual(table["rows"][0]["label"], "E1")
            self.assertEqual(table["rows"][0]["prompt_reference"], "[1]")
            self.assertEqual(table["rows"][0]["excerpt"], "期限为十年。")

    def test_observability_exports_evidence_tables_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "run.log.jsonl"
            records = [
                {
                    "event": "evidence_table",
                    "qid": "sample_qid",
                    "call_id": 1,
                    "route": "answer",
                    "table_id": "sample_qid:1:answer",
                    "rows": [{"label": "E1", "excerpt": "事实"}],
                },
                {
                    "event": "reasoning_level_assigned",
                    "qid": "sample_qid",
                    "allocation": {"level": "L2"},
                },
                {
                    "event": "api_response",
                    "qid": "sample_qid",
                    "route": "answer",
                },
            ]
            log_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in records),
                encoding="utf-8",
            )
            artifacts = export_observability_artifacts(log_path, root)
            tables = json.loads(
                (root / artifacts["evidence_tables"]).read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (root / artifacts["run_metrics"]).read_text(encoding="utf-8")
            )
            self.assertEqual(tables["sample_qid"][0]["rows"][0]["label"], "E1")
            self.assertEqual(metrics["reasoning_level_counts"]["L2"], 1)
            self.assertEqual(metrics["api_success_count"], 1)

    def test_normalized_pages_build_the_exact_merged_index_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pages = root / "pages" / "research" / "sample_document"
            pages.mkdir(parents=True)
            (pages / "page_0001.md").write_text(" 第一页  \n\n内容 ", encoding="utf-8")
            (pages / "page_0002.md").write_text("第二页\n表格", encoding="utf-8")

            results = normalize_tree(
                root / "pages",
                root / "normalized",
                root / "merged",
            )
            merged = root / "merged" / "research" / "sample_document.md"
            text = merged.read_text(encoding="utf-8")
            self.assertEqual(len(results), 1)
            self.assertIn("<!-- page: 1; source: page_0001.md -->", text)
            self.assertIn("## Page 2", text)
            self.assertIn("第二页", text)

    def test_evidence_builder_joins_requests_responses_and_final_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "run.log.jsonl"
            records = [
                {
                    "event": "evidence_table",
                    "qid": "sample_qid",
                    "call_id": 1,
                    "route": "answer",
                    "option": "A",
                    "table_id": "sample_qid:1:answer",
                    "rows": [
                        {
                            "label": "E1",
                            "doc_id": "text01",
                            "page_id": "page_0001",
                            "excerpt": "期限为十年。",
                        }
                    ],
                },
                {
                    "event": "api_request",
                    "qid": "sample_qid",
                    "call_id": 1,
                    "route": "answer",
                    "messages": [{"role": "user", "content": "evidence"}],
                },
                {
                    "event": "api_response",
                    "qid": "sample_qid",
                    "call_id": 1,
                    "raw_output": '{"answer":"A"}',
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
                {
                    "event": "question_completed",
                    "qid": "sample_qid",
                    "answer": "A",
                    "reasoning": "E1 supports A",
                    "reasoning_evidence_refs": ["E1"],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            ]
            log_path.write_text(
                "\n".join(json.dumps(row) for row in records),
                encoding="utf-8",
            )
            evidence = build_evidence(log_path)["sample_qid"]
            self.assertEqual(evidence["api_calls"][0]["raw_output"], '{"answer":"A"}')
            self.assertEqual(evidence["answer"], "A")
            self.assertEqual(evidence["token_usage"]["total_tokens"], 5)
            resolution = evidence["reasoning_evidence_resolution"][0]
            self.assertEqual(resolution["label"], "E1")
            self.assertEqual(
                resolution["candidates"][0]["evidence"]["page_id"],
                "page_0001",
            )


if __name__ == "__main__":
    unittest.main()
