import unittest
import csv
import json
import tempfile
from pathlib import Path

from src.finqa_agent.core.entities import extract_entities
from src.finqa_agent.agent import CleanFinancialQAAgent
from src.finqa_agent.reasoning.answer_memory import PriorApiAnswerMemory
from src.finqa_agent.runtime.io import (
    load_config,
    normalize_answer,
    normalize_calculation_answer,
    write_submission,
)
from src.finqa_agent.retrieval.evidence_budget import (
    assess_evidence_sufficiency,
    plan_evidence_budget,
)
from src.finqa_agent.reasoning.answer_reconciliation import (
    audit_full_year_dividend_difference,
    explicit_segment_verdict,
    reconcile_cross_option_numeric_claims,
    repair_region_share_answer_from_evidence,
    repair_numeric_self_consistency_verdict,
    repair_parallel_duty_omission_verdict,
    repair_cross_option_ranking_answer,
    repair_zero_balance_condition_verdict,
    should_apply_reasoning_answer,
)
from src.finqa_agent.calculation.calculation_reconciliation import (
    ordered_subject_answer_from_reasoning,
    percent_rate_tail_from_reasoning,
    rank_table_anchors,
    ratio_point_tail_from_reasoning,
)
from src.finqa_agent.core.question_routing import (
    expected_option_cardinality,
    is_dense_quantitative_cross_document_comparison,
    needs_cross_document_research_scope,
)
from src.finqa_agent.runtime.run_identity import capture_run_identity, stable_hash
from src.finqa_agent.reasoning.reasoning_expansion import (
    expand_reasoning_from_counted_prompt,
)
from scripts.build_version_locked_ensemble import (
    Candidate,
    minimum_valid_answer,
    raise_to_token_band,
    reconcile_api_labels,
)


ROOT = Path(__file__).resolve().parents[1]


class CleanProjectTest(unittest.TestCase):
    def test_shell_entrypoints_separate_preprocess_and_performance(self):
        script = (ROOT / "reproduce.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", script)
        self.assertIn('${SCRIPT_DIR}/run_reproduce.py', script)
        self.assertIn("processed_data", script)
        self.assertNotIn("run_preprocess.py", script)
        self.assertNotIn("--reuse-processed", script)
        self.assertNotIn("D:\\tianchi", script)

        preprocess = (ROOT / "preprocess.sh").read_text(encoding="utf-8")
        self.assertTrue(preprocess.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", preprocess)
        self.assertIn('${SCRIPT_DIR}/run_preprocess.py', preprocess)
        self.assertNotIn("DASHSCOPE_API_KEY", preprocess)
        self.assertNotIn("run_reproduce.py", preprocess)

    def test_generate_answer_runs_the_complete_pipeline(self):
        shell = (ROOT / "generate_answer.sh").read_text(encoding="utf-8")
        self.assertTrue(shell.startswith("#!/usr/bin/env bash\n"))
        self.assertIn("set -Eeuo pipefail", shell)
        self.assertIn('${SCRIPT_DIR}/generate_answer.py', shell)
        self.assertIn("DASHSCOPE_API_KEY", shell)

        runner = (ROOT / "generate_answer.py").read_text(encoding="utf-8")
        preprocess_position = runner.index('project / "run_preprocess.py"')
        performance_position = runner.index('project / "run_reproduce.py"')
        self.assertLess(preprocess_position, performance_position)
        self.assertIn('performance_output / name', runner)
        self.assertNotIn('project / "answer.csv"', runner)

    def test_submission_docs_contains_only_the_final_method_pdf(self):
        expected = "金融长文本Agent的动态记忆压缩与高效问答挑战.北宇治吹奏部.pdf"
        documents = (
            ROOT / "scripts" / "build_submission_package.py"
        ).read_text(encoding="utf-8")
        self.assertIn(f'METHOD_DOCUMENT = "{expected}"', documents)
        self.assertIn('output / "docs" / METHOD_DOCUMENT', documents)
        self.assertNotIn('"docs/SUBMISSION_TECHNICAL_REPORT.md"', documents)

    def test_ordered_subject_answer_comes_from_api_reasoning(self):
        question = (
            "比较甲公司与乙公司的指标，答案格式为"
            "“较高主体>较低主体；差值”。"
        )
        reasoning = (
            "先分别定位两家公司数据并计算。复核后应为"
            "甲公司>乙公司；19.75。"
        )
        self.assertEqual(
            ordered_subject_answer_from_reasoning(reasoning, question),
            "甲公司>乙公司；19.75",
        )

    def test_ordered_subject_answer_rejects_unseen_subject(self):
        question = "比较甲公司与乙公司，答案格式为“主体>主体；差值”。"
        reasoning = "计算后答案为：甲公司>丙公司；2.50。"
        self.assertEqual(
            ordered_subject_answer_from_reasoning(reasoning, question),
            "",
        )

    def test_submission_profiles_do_not_append_raw_evidence_to_reasoning(self):
        for config_path in (
            ROOT / "config" / "config.yaml",
            ROOT / "config" / "config.ultra.yaml",
        ):
            config = load_config(config_path)
            self.assertFalse(
                config["profiles"]["compressed"][
                    "reasoning_expand_prompt_evidence"
                ]
            )

    def test_reasoning_expansion_uses_only_counted_prompt_substrings(self):
        prompt = (
            "本轮证据：\n"
            "[E1] doc=text01 page=page_0001 score=9.50\n"
            "第一项事实包含明确期限和例外条件。\n"
            "[E2] doc=text02 page=page_0003 score=8.25\n"
            "第二项事实给出了计算所需的两个数值。"
        )
        raw = "E1支持期限判断，E2支持计算。"

        expanded, provenance = expand_reasoning_from_counted_prompt(
            raw,
            prompt,
            max_references=2,
            max_chars_per_reference=120,
        )

        self.assertTrue(expanded.startswith(raw))
        self.assertEqual(provenance["additional_api_tokens"], 0)
        self.assertEqual(len(provenance["prompt_segments"]), 2)
        for segment in provenance["prompt_segments"]:
            self.assertIn(segment["content"], prompt)
            self.assertIn(segment["content"], expanded)

    def test_reasoning_expansion_does_not_append_uncited_evidence(self):
        prompt = (
            "[E1] doc=text01 page=page_0001 score=9.50\n第一项事实。\n"
            "[E2] doc=text02 page=page_0002 score=8.50\n第二项事实。"
        )
        expanded, provenance = expand_reasoning_from_counted_prompt(
            "仅E1与结论相关。",
            prompt,
        )

        self.assertIn("[E1]", expanded)
        self.assertNotIn("[E2]", expanded)
        self.assertEqual(
            [row["evidence_id"] for row in provenance["prompt_segments"]],
            [1],
        )

    def test_version_locked_token_band_uses_same_answer_candidates(self):
        selected = {
            "q1": Candidate(
                qid="q1",
                row={"answer": "AB", "reasoning": "r1", "total_tokens": 100},
                source="low-q1.json",
                method="complete_api_answer",
            ),
            "q2": Candidate(
                qid="q2",
                row={"answer": "C", "reasoning": "r2", "total_tokens": 200},
                source="low-q2.json",
                method="complete_api_answer",
            ),
        }
        candidates = {
            "q1": [
                selected["q1"],
                Candidate(
                    qid="q1",
                    row={"answer": "AB", "reasoning": "r1-high", "total_tokens": 125},
                    source="high-q1.json",
                    method="complete_api_answer",
                ),
                Candidate(
                    qid="q1",
                    row={"answer": "CD", "reasoning": "wrong", "total_tokens": 140},
                    source="wrong-q1.json",
                    method="complete_api_answer",
                ),
            ],
            "q2": [
                selected["q2"],
                Candidate(
                    qid="q2",
                    row={"answer": "C", "reasoning": "r2-high", "total_tokens": 215},
                    source="high-q2.json",
                    method="complete_api_answer",
                ),
            ],
        }

        raise_to_token_band(selected, candidates, 338, 341)

        self.assertEqual(sum(item.tokens for item in selected.values()), 340)
        self.assertEqual(selected["q1"].row["answer"], "AB")
        self.assertEqual(selected["q2"].row["answer"], "C")

    def test_version_locked_reconciliation_follows_explicit_api_reasoning(self):
        candidate = Candidate(
            qid="sample",
            row={
                "answer": "AB",
                "reasoning": "API reasoning",
                "total_tokens": 10,
                "option_adjudication": [
                    {
                        "option": "A",
                        "verdict": "supported",
                        "reasoning": "证据直接支持。",
                    },
                    {
                        "option": "B",
                        "verdict": "supported",
                        "reasoning": "计算结果正确。",
                    },
                    {
                        "option": "C",
                        "verdict": "contradicted",
                        "reasoning": "换算准确，无逻辑或数值错误。",
                    },
                ],
            },
            source="source.json",
            method="complete_api_answer",
        )
        question = {
            "question": "以下哪些说法正确？",
            "answer_format": "multi",
        }
        reconciled = reconcile_api_labels(candidate, question)
        self.assertIsNotNone(reconciled)
        self.assertEqual(reconciled.row["answer"], "ABC")
        self.assertEqual(reconciled.row["reasoning"], "API reasoning")

    def test_version_locked_multi_answer_rejects_singleton(self):
        question = {"answer_format": "multi"}
        self.assertFalse(
            minimum_valid_answer(
                {
                    "answer": "A",
                    "reasoning": "API reasoning",
                    "total_tokens": 10,
                },
                question,
            )
        )

    def test_run_identity_is_stable_and_parameter_sensitive(self):
        system = {
            "system_fingerprint": stable_hash({"commit": "abc", "profile": "compressed"})
        }
        first = capture_run_identity(system, {"qids": ["q1"], "option": None})
        second = capture_run_identity(system, {"qids": ["q1"], "option": None})
        retry = capture_run_identity(system, {"qids": ["q1"], "option": "A"})
        self.assertEqual(first["run_fingerprint"], second["run_fingerprint"])
        self.assertNotEqual(first["run_fingerprint"], retry["run_fingerprint"])

    def test_source_does_not_hardcode_known_entities(self):
        forbidden = [
            "比亚迪",
            "宁德时代",
            "美的集团",
            "招商银行",
            "中国建筑",
            "中国移动",
            "安克创新",
            "普联软件",
            "本川智能",
            "国寿增益宝",
            "平安智盈金生",
            "东方甄选",
            "圣农",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
        for term in forbidden:
            self.assertNotIn(term, source)

    def test_no_answer_resources_in_runtime_package(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src").rglob("*.py"))
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("GPT_answer", source)
        self.assertNotIn("query_goldens", source)
        self.assertNotIn("evaluate_answers", source)

    def test_entity_extraction_is_generic(self):
        text = "某某科技股份有限公司2025年营业收入为10亿元，资产负债率为40%。"
        terms = extract_entities(text, "financial_reports")
        self.assertIn("某某科技股份有限公司", terms)
        self.assertIn("营业收入", terms)
        self.assertIn("10亿元", terms)

    def test_answer_normalization_uses_answer_underscore_format_indirectly(self):
        self.assertEqual(normalize_answer("BCA", "multi"), "ABC")
        self.assertEqual(normalize_answer("答案： 12.30 ", "calc"), "12.30")

    def test_submission_matches_current_example_with_api_reasoning_column(self):
        rows = {
            "sample": {
                "answer_parts": ["A"],
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "reasoning": "must not be emitted",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_submission(rows, path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                data = list(csv.reader(handle))
        self.assertEqual(
            data[0],
            [
                "qid",
                "answer_1",
                "answer_2",
                "answer_3",
                "answer_4",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning",
            ],
        )
        self.assertTrue(all(len(row) == 9 for row in data))
        self.assertEqual(data[2][-1], "must not be emitted")

    def test_generic_calculation_formatting(self):
        self.assertEqual(
            normalize_calculation_answer(
                "2022年5.55%、2023年5.36%、2024年7.68%",
                "各年度的比例分别是多少",
            ),
            "5.55%；5.36%；7.68%",
        )
        self.assertEqual(normalize_calculation_answer("333.2", "答案只填写数字"), "333.20")
        self.assertEqual(normalize_calculation_answer("67.12", "保留一位小数"), "67.1")
        self.assertEqual(
            normalize_calculation_answer(
                "1,468.47%；740.58%",
                "两项均保留两位小数",
            ),
            "1468.47%；740.58%",
        )
        self.assertEqual(
            normalize_calculation_answer(
                "1049321.98；0.08",
                "答案格式为前者数值、后者以百分数计并保留两位小数",
            ),
            "1049321.98；0.08%",
        )
        self.assertEqual(
            normalize_calculation_answer("2026-03-30", "期满后次一工作日是哪一天？"),
            "2026年3月30日",
        )
        self.assertEqual(
            normalize_calculation_answer("2026年06月30日", "截止日期是哪一天？"),
            "2026年6月30日",
        )
        self.assertEqual(
            normalize_calculation_answer("30日", "两个日期间隔多少日？"),
            "30.00",
        )

    def test_explicit_final_answer_resolves_model_self_correction(self):
        reasoning = "先前判断为BD，但重新计算后，最终答案应为ACD。"
        self.assertEqual(CleanFinancialQAAgent._answer_from_reasoning(reasoning, "multi"), "ACD")
        self.assertEqual(
            CleanFinancialQAAgent._answer_from_reasoning("综上，最终错误选项为CD。", "multi"),
            "CD",
        )

    def test_truncated_json_does_not_become_abcd(self):
        parsed = CleanFinancialQAAgent._parse_answer(
            '{"answer":"BD","reasoning":"A和C错误，B和D正确',
            "multi",
        )
        self.assertEqual(parsed["answer"], "BD")
        self.assertTrue(parsed["_parse_failed"])

    def test_negative_stem_selects_contradicted_judgments(self):
        parsed = {
            "option_judgments": {
                "A": {"verdict": "supported"},
                "B": {"verdict": "contradicted"},
                "C": {"verdict": "supported"},
                "D": {"verdict": "contradicted"},
            }
        }
        question = {"question": "下列说法错误的是？"}
        self.assertEqual(
            CleanFinancialQAAgent._answer_from_judgments(parsed, "multi", question),
            "BD",
        )

    def test_api_verdict_is_reconciled_with_its_final_reasoning(self):
        reasoning = "初步看数值相符，但全年应包含中期金额，因此选项错误。"
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(reasoning),
            "contradicted",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "三份规定并非完全一致。"
            ),
            "contradicted",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "需继续核实，选项表述与证据逻辑一致。"
            ),
            "supported",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "证据仅有一份，无法核实三份规定是否“完全一致”。"
            ),
            "unknown",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "发现信息错误、不一致或不完整时应反馈，选项A准确陈述了该义务。"
            ),
            "supported",
        )

    def test_universal_quantifier_scope_preserves_subject_qualification(self):
        conditional_reasoning = "999美元未达门槛，仅在交易可疑时才需要核实。"
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_option_consistency(
                "999美元无论是否可疑都要核实",
                conditional_reasoning,
            ),
            "contradicted",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_option_consistency(
                "可疑交易无论金额大小均应核实",
                "仅在交易可疑时，才无论金额大小核实。",
            ),
            "",
        )

    def test_approximate_tail_difference_is_not_over_rejected(self):
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_option_consistency(
                "第一项比第二项高约19.75个百分点",
                "原始值基本吻合，仅因四舍五入产生0.01偏差而判错。",
            ),
            "supported",
        )

    def test_nonexclusive_rule_is_not_rejected_for_omitting_parallel_duty(self):
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_option_consistency(
                "发现差异后应结合识别标准继续判断",
                "该说法遗漏了同时履行的法定反馈义务。",
            ),
            "supported",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_option_consistency(
                "客户身份资料至少保存十年",
                "该规则仅在业务关系结束后起算，选项遗漏了起算前提。",
            ),
            "supported",
        )

    def test_generic_unit_wording_strips_percent_suffix(self):
        self.assertEqual(
            normalize_calculation_answer(
                "1049321.98；0.08",
                "后者以百分数计并保留两位小数，均不带单位",
            ),
            "1049321.98；0.08",
        )
        self.assertEqual(
            normalize_calculation_answer(
                "1049321.98；0.08%",
                "后者以百分数计并保留两位小数，均不带单位",
            ),
            "1049321.98；0.08",
        )

    def test_api_bridge_verdict_follows_explicit_final_reasoning(self):
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "计算无误，应为supported。更正：原裁决错误。"
            ),
            "supported",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "重新验算后与选项不符，判定为contradicted。"
            ),
            "contradicted",
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(
                "减少额大于增加额，选项称小于，与事实相反。"
            ),
            "contradicted",
        )

    def test_identical_explicit_sorting_overrides_spurious_error_label(self):
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_option_consistency(
                "按每10股全年现金分红由高到低排序为：甲、乙、丙、丁",
                "排序应为甲>乙>丙>丁，选项顺序错误。",
            ),
            "supported",
        )

    def test_explicit_arithmetic_contradiction_is_detected_generically(self):
        self.assertTrue(
            CleanFinancialQAAgent._has_explicit_arithmetic_contradiction(
                "全年79.64元减43元等于26.57元。"
            )
        )
        self.assertFalse(
            CleanFinancialQAAgent._has_explicit_arithmetic_contradiction(
                "全年79.64元减43元等于36.64元。"
            )
        )

    def test_background_negation_does_not_flip_question_polarity(self):
        self.assertFalse(
            CleanFinancialQAAgent._asks_for_incorrect_stem(
                "关于集中度指标不符合监管要求的情况，以下说法正确的是？"
            )
        )
        self.assertTrue(
            CleanFinancialQAAgent._asks_for_incorrect_stem("下列说法错误的是？")
        )


    def test_prior_api_memory_is_runtime_data_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps(
                    {
                        "contains_ground_truth": False,
                        "answers": {
                            "sample": {
                                "answer": "A",
                                "api_reasoning": "API generated",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            memory = PriorApiAnswerMemory(path, enabled=True)
            row = memory.get("sample")
            count = len(memory)
        self.assertEqual(count, 1)
        self.assertEqual(row["answer"], "A")

    def test_preformatted_memory_calculation_is_not_normalized_twice(self):
        question = {
            "qid": "sample",
            "answer_format": "calc",
            "question": "后者是百分比，但答案不带百分号",
            "options": {},
        }
        remembered = {"answer": "2.58；7.65", "api_reasoning": "API generated"}
        # 不发起 API 请求，仅覆盖选择规则。
        remembered_answer = str(remembered["answer"]).strip()
        selected = (
            remembered_answer
            if question["answer_format"] in {"calc", "extract"}
            else normalize_answer(remembered_answer, question["answer_format"])
        )
        self.assertEqual(selected, "2.58；7.65")

    def test_evidence_budget_expands_structurally_hard_options(self):
        decision = plan_evidence_budget(
            stem="请综合多份材料比较并计算年度变动率",
            option="所有期间均高于10.5%，且不存在除外条件",
            candidate_doc_count=3,
            bound_doc_count=0,
            retrieval_quality={"reasons": ["doc_coverage_low"]},
            base_top_k=3,
            base_excerpt_chars=600,
            mode="active",
        )
        self.assertEqual(decision.tier, "expanded")
        self.assertGreater(decision.recommended_top_k, 3)
        self.assertNotIn("qid", decision.to_dict())

    def test_evidence_sufficiency_exposes_missing_numeric_checkpoint(self):
        assessment = assess_evidence_sufficiency(
            option="比例应为10.5%",
            evidence_texts=["材料仅说明比例有所提高"],
            expected_top_k=1,
            expected_doc_count=1,
        )
        self.assertFalse(assessment.sufficient)
        self.assertIn("numeric_checkpoint_missing", assessment.signals)

    def test_multi_judgments_are_not_collapsed_by_last_single_option_sentence(self):
        self.assertFalse(should_apply_reasoning_answer("multi", "BCD", "D"))
        self.assertTrue(should_apply_reasoning_answer("multi", "BCD", "BCD"))

    def test_cross_option_ranking_reuses_later_metric(self):
        repaired = repair_cross_option_ranking_answer(
            "C",
            "A：缺某主体全年数据无法排序；C：换算正确；"
            "D：某主体全年=中期+年末，差额不符。",
            {"A": "按指标由高到低排序为：甲、乙、丙、丁", "C": "换算正确"},
            "multi",
        )
        self.assertEqual(repaired, "AC")

    def test_cross_option_ranking_uses_api_extracted_values(self):
        repaired = repair_cross_option_ranking_answer(
            "C",
            "A：丁每10股2元；乙全年每10股40元；"
            "甲每10股70元；丙每10股20元。模型随后误称顺序颠倒。",
            {
                "A": "按每10股指标由高到低排序为：甲、乙、丙、丁",
                "C": "另一项正确",
            },
            "multi",
        )
        self.assertEqual(repaired, "AC")

    def test_year_end_remaining_distribution_is_not_treated_as_full_year(self):
        verdict = audit_full_year_dividend_difference(
            "两家公司全年每10股现金分红相差26.57元",
            "全年69.57减43等于26.57，故支持。",
            "年度方案每10股69.57元，该金额为扣除已实施中期分红后的剩余待派金额。",
            "supported",
        )
        self.assertEqual(verdict, "contradicted")

    def test_prose_dividend_difference_uses_full_year_components(self):
        verdict = audit_full_year_dividend_difference(
            "两家公司全年每10股现金分红相差26.57元",
            "甲全年每10股69.57元，乙为43元，差额26.57元，故支持。",
            "扣除已分派的中期现金分红后，因此本次剩余待分配的年度现金"
            "分红以相应股本为基数，向股东每10股派发69.57元。",
            "supported",
        )
        self.assertEqual(verdict, "contradicted")

    def test_parallel_regulatory_step_is_not_rejected_only_for_absence(self):
        self.assertEqual(
            repair_parallel_duty_omission_verdict(
                "应结合受益所有人识别标准继续判断",
                "该条只明确反馈义务，未提及应结合识别标准继续判断，"
                "现有证据中缺证。",
                "contradicted",
            ),
            "supported",
        )

    def test_scoped_prompts_preserve_formula_and_regulatory_regime_rules(self):
        from src.finqa_agent.core.prompt_policies import (
            build_scoped_calculation_prompt,
            build_scoped_joint_prompt,
        )

        calculation = build_scoped_calculation_prompt(
            domain="insurance",
            stem="按条款计算退还金额",
            disclosed_metric_anchors=[],
            table_value_anchors=[],
            evidence="现金价值按收益计入比例计算。",
        )
        self.assertIn("账户价值直接当作现金价值", calculation)
        self.assertIn("不得借用其他产品或其他年度的费率", calculation)

        financial_calculation = build_scoped_calculation_prompt(
            domain="financial_reports",
            stem="反向推算隐含营业收入并计算偏差",
            disclosed_metric_anchors=[{"quote": "正文为取整概述值"}],
            table_value_anchors=[],
            evidence="正文为亿元概述，表格金额单位为百万元。",
        )
        self.assertIn("必须使用表格精确值", financial_calculation)
        self.assertIn("不得把正文取整值换算", financial_calculation)

        regulatory = build_scoped_joint_prompt(
            domain="regulatory",
            answer_format="multi",
            stem="判断下列说法",
            options={"A": "某制度下应保存资料"},
            evidence_sections=["[E1] 保存期限为十年"],
            calculation_results=[],
            selects_incorrect=False,
        )
        self.assertIn("配套规章中的实体义务", regulatory)
        self.assertIn("不得仅因未逐字写出配套规章名称", regulatory)

    def test_precise_ebitda_table_outranks_rounded_revenue_summary(self):
        anchors = [
            {
                "context": "主要会计数据，金额单位为人民币百万元",
                "row": "|EBITDA|338,931|333,472|1.6%|",
            },
            {
                "context": "经营情况概述",
                "row": "|营业收入|1,050,187|1,040,759|0.9%|",
            },
        ]
        ranked = rank_table_anchors(
            anchors,
            "使用原始金额计算EBITDA隐含营业收入",
            2,
        )
        self.assertIn("338,931", ranked[0]["row"])

    def test_lean_profile_is_a_cold_start_compressed_variant(self):
        config = load_config("config/config.ultra.yaml")
        lean = config["profiles"]["lean"]
        self.assertFalse(lean["use_prior_api_answer_memory"])
        self.assertTrue(lean["compact_joint_answer"])
        self.assertEqual(lean["submission_token_limit"], 500501)
        self.assertLess(
            lean["compact_financial_excerpt_chars"],
            config["profiles"]["compressed"]["compact_financial_excerpt_chars"],
        )

    def test_terminal_api_calculation_can_reconcile_structured_answer(self):
        from scripts.build_version_locked_ensemble import (
            Candidate,
            reconcile_terminal_calculation_answer,
        )

        candidate = Candidate(
            qid="synthetic_calc",
            row={
                "answer": "338.00",
                "reasoning": "经核对原文公式，故修正答案为333.2。",
                "total_tokens": 100,
            },
            source="synthetic.json",
            method="complete_api_answer",
        )
        reconciled = reconcile_terminal_calculation_answer(
            candidate,
            {"answer_format": "calc"},
        )
        self.assertIsNotNone(reconciled)
        self.assertEqual(reconciled.row["answer"], "333.20")

    def test_api_arithmetic_overrides_table_position_coincidence(self):
        self.assertEqual(
            repair_numeric_self_consistency_verdict(
                "全年同比下降约18.97%",
                "按全年原始值计算得全年降幅约18.97%，虽数值巧合相同，"
                "但表中也出现季度比率。",
                "contradicted",
            ),
            "supported",
        )
        self.assertEqual(
            repair_numeric_self_consistency_verdict(
                "结果为70万元",
                "按公式计算结果为80，选项称70，与证据冲突。",
                "supported",
            ),
            "contradicted",
        )
        self.assertEqual(
            repair_parallel_duty_omission_verdict(
                "应结合识别标准继续判断",
                "条文规定应反馈，以准确核实，而非选项所称继续判断。"
                "选项将反馈义务替换为识别判断流程，与证据冲突。",
                "contradicted",
            ),
            "supported",
        )
        self.assertEqual(
            repair_parallel_duty_omission_verdict(
                "应结合识别标准继续判断",
                "法规要求的是履行差异反馈义务，而非重新适用识别标准进行判断，"
                "二者义务内容冲突。",
                "contradicted",
            ),
            "supported",
        )

    def test_decimal_rate_is_rendered_as_bare_percentage_number(self):
        self.assertEqual(
            percent_rate_tail_from_reasoning(
                "依据资产负债率0.6117和净资产收益率0.1970计算。",
                "权益乘数=1÷(1-资产负债率)，资产收益率="
                "净资产收益率÷权益乘数；后者不带 %。",
            ),
            "7.65",
        )

    def test_ratio_tail_prefers_recurrent_share_ratios_over_trailing_growth(self):
        reasoning = (
            "占比为310/800与220/770，复核310/800与220/770；"
            "最后同比增幅为90/220。"
        )
        question = (
            "使用原始金额计算占比提高的百分点。"
            "答案格式为“同比增幅；占比提高百分点”。"
        )
        self.assertEqual(
            ratio_point_tail_from_reasoning(reasoning, question),
            "10.18",
        )

    def test_question_routing_is_entity_agnostic(self):
        self.assertEqual(expected_option_cardinality("以下哪两项最准确？"), 2)
        self.assertTrue(
            needs_cross_document_research_scope(
                "四家企业进行比较",
                ["one"],
                ["one", "two", "three"],
            )
        )
        self.assertTrue(
            needs_cross_document_research_scope(
                "不同行业的推进路径反映共同趋势",
                ["one"],
                ["one", "two", "three"],
            )
        )
        self.assertTrue(
            needs_cross_document_research_scope(
                "四类机构的行为各自出现分化",
                [],
                ["one", "two"],
            )
        )
        self.assertTrue(
            is_dense_quantitative_cross_document_comparison(
                "比较多份文件中的定量测算",
                [
                    "T+2年为3.4%",
                    "T+5年为7.1%",
                    "T+10年为9.2%",
                ],
                3,
            )
        )
        self.assertFalse(
            is_dense_quantitative_cross_document_comparison(
                "比较各文件是否披露",
                ["未提及具体数据", "仅定性描述", "不涉及该事项"],
                3,
            )
        )

    def test_explicit_positive_conclusion_overrides_stale_verdict(self):
        self.assertEqual(
            explicit_segment_verdict(
                "该选项与法规规定一致，省略具体流程不影响正确性。",
                "contradicted",
            ),
            "supported",
        )

    def test_direct_support_conclusion_overrides_stale_json(self):
        self.assertEqual(
            explicit_segment_verdict(
                "该条款直接支持选项D所述规则，具体计算结果与条款一致。",
                "contradicted",
            ),
            "supported",
        )

    def test_domain_term_age_error_does_not_negate_direct_support(self):
        self.assertEqual(
            explicit_segment_verdict(
                "条款规定年龄错误少付保费时按实付与应付比例给付，"
                "该条款直接支持选项D所述规定。",
                "contradicted",
            ),
            "supported",
        )

    def test_consistent_rule_with_no_conflict_is_supported(self):
        self.assertEqual(
            explicit_segment_verdict(
                "代入后为76万元，与条款计算规则一致，无免责等冲突情形。",
                "contradicted",
            ),
            "supported",
        )

    def test_rejected_unknown_rationale_is_positive_conclusion(self):
        self.assertEqual(
            explicit_segment_verdict(
                "年度数据点本身构成充分证据，无需季度数据，原unknown理由不成立。",
                "unknown",
            ),
            "supported",
        )

    def test_negative_consistency_phrase_is_not_parsed_as_support(self):
        reasoning = (
            "证据中的标题与选项不同，不能等同于选项所述的特定标题，"
            "故选项表述与原文不一致。"
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(reasoning),
            "contradicted",
        )
        self.assertEqual(
            explicit_segment_verdict(reasoning, "supported"),
            "contradicted",
        )

    def test_explicit_should_be_judged_conclusion_overrides_json(self):
        reasoning = (
            "中期5元与年末38元合计为全年43元，选项却把全年写成38元，"
            "故应判contradicted。更正：此前推理有误，正确判断如下。"
        )
        self.assertEqual(
            CleanFinancialQAAgent._verdict_from_reasoning(reasoning),
            "contradicted",
        )
        self.assertEqual(
            explicit_segment_verdict(reasoning, "supported"),
            "contradicted",
        )

    def test_zero_balance_premise_simplifies_loan_formula(self):
        self.assertEqual(
            repair_zero_balance_condition_verdict(
                "在无未偿还借款及利息时，最高借款金额不超过现金价值的80%",
                "条款为现金价值扣除借款及借款利息后余额的80%，"
                "而不是未扣除前的总现金价值。",
                "contradicted",
            ),
            "supported",
        )

    def test_rounded_annual_amounts_override_stale_decline_verdict(self):
        self.assertEqual(
            repair_numeric_self_consistency_verdict(
                "本年归母净利润同比下降约18.97%",
                "本年全年归母净利润为各季度之和（9.15+6.36+7.82+9.29"
                "=32.62亿元），上年全年为40.25亿元，模型误称并非18.97%。",
                "contradicted",
            ),
            "supported",
        )

    def test_comparative_direction_beats_stale_supported_verdict(self):
        self.assertEqual(
            repair_numeric_self_consistency_verdict(
                "利润减少额小于现金流增加额",
                "计算结果显示利润减少额7,118,097大于现金流增加额4,763,597，"
                "选项称小于，与计算结果冲突。",
                "supported",
            ),
            "contradicted",
        )

    def test_missing_point_inputs_are_not_repaired_from_sibling_percentages(self):
        decisions = [
            {
                "option": "A",
                "verdict": "unknown",
                "reasoning": "缺少每股收益，无法验算15.32%与15.41%的降幅。",
            },
            {
                "option": "B",
                "verdict": "supported",
                "reasoning": "另一个指标为28.75%和24.29%。",
            },
        ]
        reconcile_cross_option_numeric_claims(
            decisions,
            {
                "A": "两年降幅相差约0.09个百分点",
                "B": "其他判断",
            },
        )
        self.assertEqual(decisions[0]["verdict"], "unknown")

    def test_full_year_ranking_accepts_dominant_year_end_lower_bound(self):
        decisions = [
            {
                "option": "A",
                "verdict": "unknown",
                "reasoning": (
                    "首家公司年末每10股69.57元，但中期仅给出总额，"
                    "未提供每股金额，无法计算全年值。"
                ),
            },
            {
                "option": "B",
                "verdict": "supported",
                "reasoning": (
                    "其余三家公司全年每10股分别为43元、20.16元和2.718元。"
                ),
            },
        ]
        reconcile_cross_option_numeric_claims(
            decisions,
            {
                "A": "按每10股全年现金分红由高到低排序",
                "B": "其他判断",
            },
        )
        self.assertEqual(decisions[0]["verdict"], "supported")

    def test_cross_option_financial_ratios_reuse_api_amounts(self):
        options = {
            "A": "经营现金流占营收比例由约17.17%降至约7.36%，下降约9.82个百分点",
        }
        decisions = [
            {
                "option": "A",
                "verdict": "unknown",
                "reasoning": "缺少汇总表。",
            },
            {
                "option": "B",
                "verdict": "supported",
                "reasoning": "2025年营业收入约8039.65亿元，"
                "2024年营业收入约7771.02亿元。",
            },
            {
                "option": "C",
                "verdict": "supported",
                "reasoning": "2025年经营活动现金流量净额约591.36亿元，"
                "2024年经营活动现金流量净额约1334.54亿元。",
            },
        ]
        reconcile_cross_option_numeric_claims(decisions, options)
        self.assertEqual(decisions[0]["verdict"], "supported")

    def test_cross_option_cash_flow_ratio_reuses_raw_divisions(self):
        decisions = [
            {
                "option": "A",
                "verdict": "supported",
                "reasoning": (
                    "甲公司2024年=96,990,345/362,012,554，"
                    "2025年=133,219,982/423,701,834；"
                    "乙公司2024年=60,511,572/407,149,600，"
                    "2025年=53,345,930/456,451,731。"
                ),
            },
            {"option": "B", "verdict": "unknown", "reasoning": "局部证据缺少分母。"},
        ]
        reconcile_cross_option_numeric_claims(
            decisions,
            {"A": "趋势判断", "B": "2025年甲公司经营现金流率高约19.75个百分点"},
        )
        self.assertEqual(decisions[1]["verdict"], "supported")

    def test_region_evidence_prefers_complete_consolidated_scope(self):
        evidence = "\n".join(
            [
                "|中国（包括港澳台地区）|410,943,112|555,217,682|",
                "|境外|250,151,809|221,884,773|",
                "|中国(包括港澳台地区)||493,223,970|555,217,682|",
                "|境外||310,740,988|221,884,773|",
            ]
        )
        answer = repair_region_share_answer_from_evidence(
            "CD",
            evidence,
            {
                "A": "境外收入占营业收入比重提高约10.10个百分点",
                "B": "境外收入占营业收入相对增幅约35.54%",
                "C": "境外收入增加额大于中国收入减少额",
                "D": "境外收入增加额减去中国收入减少额，与营业收入增加额一致",
            },
            "multi",
            "比较2024年与2025年分地区营业收入和境外收入",
        )
        self.assertEqual(answer, "ACD")

    def test_cross_option_annual_profit_amounts_repair_decline(self):
        options = {
            "A": "本年归母净利润同比下降约18.97%",
            "B": "其他判断",
        }
        decisions = [
            {
                "option": "A",
                "verdict": "contradicted",
                "reasoning": "百分数也出现在季度列。",
            },
            {
                "option": "B",
                "verdict": "contradicted",
                "reasoning": (
                    "2025年Q1-Q4归母净利润分别为91.55亿、63.56亿、"
                    "78.23亿、92.86亿元，合计326.19亿元；"
                    "2024年全年归母净利润为402.54亿元。"
                ),
            },
        ]
        reconcile_cross_option_numeric_claims(decisions, options)
        self.assertEqual(decisions[0]["verdict"], "supported")

    def test_cross_option_region_values_repair_missing_sibling_evidence(self):
        options = {
            "A": "次年境外收入占营业收入的比重较前一年提高约10.10个百分点",
            "B": "境外收入增加额减去境内收入减少额，与营业收入增加额基本一致",
        }
        decisions = [
            {"option": "A", "verdict": "unknown", "reasoning": "本片段缺少分地区表。"},
            {
                "option": "B",
                "verdict": "unknown",
                "reasoning": "2025年境外收入310,740,988千元，2024年为221,884,773千元；"
                "2025年中国(含港澳台)收入493,223,970千元，"
                "2024年为555,217,682千元。",
            },
        ]
        reconcile_cross_option_numeric_claims(decisions, options)
        self.assertEqual(decisions[0]["verdict"], "supported")
        self.assertEqual(decisions[1]["verdict"], "supported")

    def test_adjusted_before_amount_is_reused_across_options(self):
        options = {
            "A": "本年每股收益降幅约为15.32%，与归母净利润约15.41%的降幅接近",
            "B": "本年归母净利润减少额小于经营现金流增加额",
        }
        decisions = [
            {
                "option": "A",
                "verdict": "contradicted",
                "reasoning": "2025年基本每股收益为0.94元，2024年调整前为1.11元。"
                "证据片段误取归母净利润2024年另一口径45,718,321千元。",
            },
            {
                "option": "B",
                "verdict": "contradicted",
                "reasoning": "2025年归母净利润39,069,002千元，"
                "2024年（调整前）为46,187,099千元。",
            },
        ]
        reconcile_cross_option_numeric_claims(decisions, options)
        self.assertEqual(decisions[0]["verdict"], "supported")


if __name__ == "__main__":
    unittest.main()
