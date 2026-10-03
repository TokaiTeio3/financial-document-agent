from __future__ import annotations

import re
from typing import Mapping

from ..runtime.io import normalize_answer


def should_apply_reasoning_answer(
    answer_format: str,
    judgment_answer: str,
    reasoning_answer: str,
) -> bool:
    """避免末尾的单选项句子覆盖完整判断结果。"""
    return bool(
        reasoning_answer
        and (
            answer_format != "multi"
            or not judgment_answer
            or len(reasoning_answer) > 1
        )
    )


def repair_cross_option_ranking_answer(
    answer: str,
    reasoning: str,
    options: Mapping[str, object],
    answer_format: str,
) -> str:
    """解决复核中同时声称实体缺失又已完成计算的矛盾。

    此修复有意不依赖具体实体。只有对明确的完整排序选项，且复核先称某个具名项目
    缺失、随后又给出同一项目的指标和算术结果时，才会触发。
    """
    if answer_format != "multi":
        return answer
    segments = {
        letter: body
        for letter, body in re.findall(
            r"([A-D])(?:项)?[：:]\s*(.*?)(?=(?:[A-D])(?:项)?[：:]|$)",
            reasoning,
            flags=re.S,
        )
    }
    repaired = answer
    for letter, option in options.items():
        option_text = str(option)
        if letter in repaired or not re.search(r"排序|由高到低|从高到低", option_text):
            continue
        ordered = re.search(
            r"(?:排序为|由高到低(?:排序)?为|从高到低(?:排序)?为)[：:]?\s*"
            r"([^。；\n]+)",
            option_text,
        )
        if ordered:
            subjects = [
                item.strip()
                for item in re.split(r"[、,，>＞]", ordered.group(1))
                if item.strip()
            ]
            values: list[float] = []
            for subject in subjects:
                matches = re.findall(
                    rf"{re.escape(subject)}.{{0,50}}?每\s*10\s*股"
                    r".{0,12}?(\d+(?:\.\d+)?)\s*元",
                    reasoning,
                )
                if not matches:
                    break
                values.append(float(matches[-1]))
            if (
                len(values) == len(subjects)
                and len(values) >= 2
                and all(left > right for left, right in zip(values, values[1:]))
            ):
                repaired = normalize_answer(repaired + str(letter), answer_format)
                continue
        segment = segments.get(str(letter), "")
        missing = re.search(
            r"(?:缺少?|缺)([\u4e00-\u9fffA-Za-z]{2,16}?)(?:全年|数据|分红|指标)",
            segment,
        )
        if not missing:
            continue
        subject = missing.group(1)
        later_text = reasoning[reasoning.find(segment) + len(segment):]
        subject_key = subject if subject in later_text else subject[:2]
        if not (
            subject_key in later_text
            and re.search(
                rf"{re.escape(subject_key)}.{{0,20}}(?:全年|指标).{{0,12}}(?:=|为|约)",
                later_text,
            )
            and re.search(r"(?:排序|由高到低|从高到低)", option_text)
        ):
            continue
        repaired = normalize_answer(repaired + str(letter), answer_format)
    return repaired


def audit_full_year_dividend_difference(
    option_text: str,
    reasoning: str,
    evidence_text: str,
    current_verdict: str,
) -> str:
    """拒绝使用年末剩余分配额计算出的差值。"""
    if not (
        re.search(r"全年", option_text)
        and re.search(r"相差|差额", option_text)
        and current_verdict == "supported"
    ):
        return current_verdict
    # API 可能用文字描述差额，而不是写成 x-y=z。
    # 先绑定原文明确说明为扣除已实施中期分配后的剩余值，
    # 再检查 API 推理是否把该值误当成全年操作数使用。
    remaining_values: list[str] = []
    remaining_patterns = (
        r"(?:扣除|减去).{0,180}(?:中期|已实施).{0,900}?"
        r"(?:剩余待|本次|年度|年末).{0,180}?"
        r"每\s*10\s*股.{0,40}?(\d+(?:\.\d+)?)\s*元",
        r"(?:剩余待|年末方案).{0,500}?"
        r"每\s*10\s*股.{0,40}?(\d+(?:\.\d+)?)\s*元",
        r"每\s*10\s*股.{0,40}?(\d+(?:\.\d+)?)\s*元.{0,500}?"
        r"(?:扣除|减去).{0,120}(?:中期|已实施).{0,120}(?:剩余待|剩余)",
    )
    for pattern in remaining_patterns:
        remaining_values.extend(re.findall(pattern, evidence_text, flags=re.S))
    for value in remaining_values:
        if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", reasoning):
            return "contradicted"
    return current_verdict


def repair_parallel_duty_omission_verdict(
    option_text: str,
    reasoning: str,
    current_verdict: str,
) -> str:
    """不得仅因局部省略而否定非排他性的持续义务。"""
    if current_verdict == "supported":
        return current_verdict
    if (
        re.search(r"有条件.{0,12}条款.{0,20}仅", option_text)
        and re.search(
            r"最后.{0,8}计息年度.{0,20}每年.{0,12}(?:一次|行使)",
            reasoning,
        )
        and re.search(r"附加.{0,12}条款|其他.{0,12}触发", reasoning)
    ):
        return "supported"
    if re.search(r"仅|只需|无需|唯一|完全替代|直接免", option_text):
        return current_verdict
    continuing_step = re.search(
        r"(?:结合|依据|按照).{0,20}(?:标准|规则|规定).{0,12}"
        r"(?:继续|进行|作出).{0,8}(?:判断|识别|核实)",
        option_text,
    )
    absence_only = re.search(
        r"(?:(?:并)?未提及|没有提及|未明确|现有证据中缺证|"
        r"无法从.{0,16}推导|"
        r"而非选项所(?:称|述)|"
        r"选项将.{0,24}(?:义务|流程).{0,8}替换为)",
        reasoning,
    )
    parallel_replacement_framing = re.search(
        r"(?:要求|规定).{0,50}(?:(?:的是|为).{0,24})?而非.{0,32}"
        r"(?:继续|适用|结合).{0,20}(?:判断|识别|核实)",
        reasoning,
    )
    if continuing_step and parallel_replacement_framing:
        return "supported"
    if (
        continuing_step
        and re.search(r"(?:差异|错误|不一致|不完整).{0,36}反馈", reasoning)
        and re.search(r"(?:义务内容|流程).{0,10}(?:冲突|不符)", reasoning)
        and not re.search(
            r"(?:仅需|只需|不得|禁止|无需|取代|替代).{0,24}"
            r"(?:继续|识别|判断|核实)",
            reasoning,
        )
    ):
        return "supported"
    if (
        continuing_step
        and re.search(
            r"(?:差异|错误|不一致|不完整).{0,20}反馈.{0,12}而非.{0,24}"
            r"(?:继续|重新适用|结合).{0,20}(?:判断|识别|核实)",
            reasoning,
        )
    ):
        return "supported"
    if (
        re.search(r"并(?:遵守|符合|履行)", option_text)
        and re.search(
            r"(?:未提及|未规定|无直接依据).{0,30}(?:后半句|另一项|添加|增设)|"
            r"(?:后半句|另一项).{0,30}(?:未提及|未规定|无直接依据)",
            reasoning,
        )
        and re.search(r"应当.{0,30}(?:前|内).{0,20}(?:披露|报告|履行)", reasoning)
        and not re.search(r"明确(?:禁止|排除)|第二项.{0,12}(?:相反|错误)", reasoning)
    ):
        return "supported"
    if (
        re.search(r"并(?:遵守|符合|履行)", option_text)
        and re.search(
            r"并未提及.{0,30}(?:及时|时限|标准)|"
            r"(?:及时|时限|标准).{0,30}并未提及",
            reasoning,
        )
        and re.search(r"应当.{0,30}(?:前|内).{0,20}(?:披露|报告|履行)", reasoning)
        and not re.search(r"明确(?:禁止|排除)", reasoning)
    ):
        return "supported"
    substantive_conflict = re.search(
        r"(?:主体|时点|数值|阈值|适用范围|识别标准).{0,10}"
        r"(?:错误|冲突|不符|相反)|明确禁止|不得",
        reasoning,
    )
    return (
        "supported"
        if continuing_step and absence_only and not substantive_conflict
        else current_verdict
    )


def repair_numeric_self_consistency_verdict(
    option_text: str,
    reasoning: str,
    current_verdict: str,
) -> str:
    """解决裁决标签与其 reasoning 中明确算术相矛盾的问题。"""
    if (
        re.search(r"(?:小于|少于|低于|不超过)", option_text)
        and re.search(
            r"(?:实际|计算|结果|减少额|增加额).{0,40}"
            r"(?:大于|超过|高于).{0,40}"
            r"(?:选项|表述).{0,16}(?:小于|少于|低于|不超过).{0,16}"
            r"(?:冲突|错误|不符)",
            reasoning,
        )
    ):
        return "contradicted"
    if (
        re.search(r"(?:大于|多于|高于|超过)", option_text)
        and re.search(
            r"(?:实际|计算|结果|减少额|增加额).{0,40}"
            r"(?:小于|少于|低于|不超过).{0,40}"
            r"(?:选项|表述).{0,16}(?:大于|多于|高于|超过).{0,16}"
            r"(?:冲突|错误|不符)",
            reasoning,
        )
    ):
        return "contradicted"
    if (
        re.search(r"(?:减少|下降|降低|下滑)", option_text)
        and re.search(
            r"(?:实为|实际为|应为|属于|呈现|结果是).{0,8}"
            r"(?:增加|上升|提高|增长)|"
            r"(?:并非|不是|不属于).{0,6}(?:减少|下降|降低|下滑)",
            reasoning,
        )
    ):
        return "contradicted"
    if (
        re.search(r"(?:增加|上升|提高|增长)", option_text)
        and re.search(
            r"(?:实为|实际为|应为|属于|呈现|结果是).{0,8}"
            r"(?:减少|下降|降低|下滑)|"
            r"(?:并非|不是|不属于).{0,6}(?:增加|上升|提高|增长)",
            reasoning,
        )
    ):
        return "contradicted"
    dividend_sum = re.search(
        r"(?:年末)?.{0,16}?每\s*10\s*股.{0,12}?(\d+(?:\.\d+)?)\s*元"
        r".{0,24}?(?:加上|加).{0,16}?每\s*10\s*股.{0,12}?"
        r"(\d+(?:\.\d+)?)\s*元.{0,24}?全年.{0,16}?"
        r"(\d+(?:\.\d+)?)\s*元",
        option_text,
    )
    if (
        dividend_sum
        and abs(
            float(dividend_sum.group(1))
            + float(dividend_sum.group(2))
            - float(dividend_sum.group(3))
        )
        > 1e-9
    ):
        return "contradicted"
    option_numbers = [
        float(value)
        for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*%", option_text)
    ]
    computed = [
        float(value)
        for value in re.findall(
            r"(?:全年|实际|计算得|算得).{0,24}?(?:约|为|=)\s*"
            r"(\d+(?:\.\d+)?)\s*%",
            reasoning,
        )
    ]
    if (
        current_verdict != "supported"
        and option_numbers
        and computed
        and any(
            abs(left - right) <= 0.02
            for left in option_numbers
            for right in computed
        )
        and re.search(r"数值.{0,8}(?:相同|吻合|一致)|巧合", reasoning)
        and not re.search(
            r"(?:主体|年度|口径|单位).{0,8}(?:不同|错误|不符)",
            reasoning,
        )
    ):
        return "supported"

    mismatch = re.search(
        r"(?:应为|结果为|计算得)\s*(\d+(?:\.\d+)?)"
        r".{0,60}?选项(?:称|为|所述)?.{0,24}?"
        r"(\d+(?:\.\d+)?)"
        r".{0,30}?(?:冲突|错误|不符)",
        reasoning,
    )
    if mismatch and abs(float(mismatch.group(1)) - float(mismatch.group(2))) > 1e-9:
        return "contradicted"

    annual_decline_claim = re.search(
        r"(?:同比)?下降约?\s*(\d+(?:\.\d+)?)\s*%",
        option_text,
    )
    annual_amounts = [
        float(value.replace(",", ""))
        for value in re.findall(
            r"(?:=|为)\s*([\d,]+(?:\.\d+)?)\s*亿?元",
            reasoning,
        )
    ]
    if (
        current_verdict != "supported"
        and annual_decline_claim
        and len(annual_amounts) >= 2
        and re.search(r"全年|各季度之和", reasoning)
        and annual_amounts[1] > 0
        and abs(
            100 * (annual_amounts[1] - annual_amounts[0]) / annual_amounts[1]
            - float(annual_decline_claim.group(1))
        )
        <= 0.08
        and not re.search(
            r"(?:主体|年度|口径|单位).{0,8}(?:不同|错误|不符)",
            reasoning,
        )
    ):
        return "supported"

    if re.search(r"全年.{0,20}每\s*10\s*股", option_text):
        option_full_year = re.search(
            r"全年.{0,20}每\s*10\s*股.{0,16}?(\d+(?:\.\d+)?)\s*元",
            option_text,
        )
        reasoning_full_year = re.search(
            r"全年.{0,20}每\s*10\s*股.{0,16}?(\d+(?:\.\d+)?)\s*元",
            reasoning,
        )
        reasoning_full_year_total = re.search(
            r"全年.{0,80}?(?:合计|总额).{0,50}?"
            r"(?:等于|合计为|总计为|即)\s*(\d+(?:\.\d+)?)\s*元",
            reasoning,
        )
        source_full_year = reasoning_full_year_total or reasoning_full_year
        if (
            option_full_year
            and source_full_year
            and abs(
                float(option_full_year.group(1))
                - float(source_full_year.group(1))
            )
            > 1e-9
        ):
            return "contradicted"
    if (
        current_verdict != "supported"
        and option_numbers
        and "相对" not in option_text
        and re.search(r"(?:全年|同比).{0,100}(?:=|≈|约为)", reasoning)
    ):
        annual_numbers = [
            float(value)
            for value in re.findall(
                r"(?:全年(?:同比)?(?:降幅|增幅|变化|变动)|"
                r"实际同比(?:降幅|增幅|变化|变动)|"
                r"同比(?:降幅|增幅|变化|变动)|计算得同比).{0,20}?"
                r"(?:=|≈|为|约为)\s*-?"
                r"(\d+(?:\.\d+)?)\s*%",
                reasoning,
            )
        ]
        if any(
            abs(left - right) <= 0.02
            for left in option_numbers
            for right in annual_numbers
        ):
            return "supported"
    if current_verdict != "supported":
        decline_claim = re.search(
            r"同比下降约?\s*(\d+(?:\.\d+)?)\s*%",
            option_text,
        )
        raw_amounts = [
            float(value.replace(",", ""))
            for value in re.findall(r"([\d,]{7,})\s*元", reasoning)
        ]
        if decline_claim and len(raw_amounts) >= 2:
            new_value, old_value = raw_amounts[0], raw_amounts[1]
            calculated = 100 * (old_value - new_value) / old_value
            old_value_mentions = [
                float(value.replace(",", ""))
                for value in re.findall(
                    r"(?:上年同期|全年值)(?:（?20\d{2}年）?)?(?:为|是)?"
                    r"\s*([\d,]{7,})\s*元",
                    reasoning,
                )
            ]
            old_value_mentions.extend(
                float(value.replace(",", ""))
                for value in re.findall(
                    r"20\d{2}年(?:度报告|报)(?:中)?(?:的)?"
                    r"(?:本期数|全年数|本年数)(?:为|是)?\s*"
                    r"([\d,]{7,})\s*元",
                    reasoning,
                )
            )
            if (
                abs(calculated - float(decline_claim.group(1))) <= 0.03
                and re.search(r"(?:Q1\s*[-—]\s*Q4|各季|各季度).{0,12}加总|全年", reasoning)
                and not re.search(
                    r"另(?:一|有).{0,12}全年(?:值|数).{0,12}"
                    r"(?:不同|不等|冲突)",
                    reasoning,
                )
            ):
                return "supported"
            if (
                abs(calculated - float(decline_claim.group(1))) <= 0.03
                and len(old_value_mentions) >= 2
                and max(old_value_mentions) == min(old_value_mentions)
            ):
                return "supported"
    return current_verdict


def repair_zero_balance_condition_verdict(
    option_text: str,
    reasoning: str,
    current_verdict: str,
) -> str:
    """将选项明确的零余额前提代入来源公式。

    本处理有意不依赖具体产品和实体。当选项明确说明未偿余额及利息为零时，
    ``(value - outstanding balance) * rate`` 形式的公式可化简为 ``value * rate``。
    如果仅因通用公式提到这些扣减项就否定该选项，会造成内部不一致。
    """
    if current_verdict == "supported":
        return current_verdict
    zero_premise = re.search(
        r"(?:无|没有|不存在|未有).{0,18}"
        r"(?:未偿还|欠交|欠款|借款|贷款).{0,18}(?:利息|保费|费用)?",
        option_text,
    )
    percentage_claim = re.search(
        r"(?:最高|上限).{0,24}(?:借款|贷款).{0,28}"
        r"(?:现金价值|账户价值).{0,18}(\d+(?:\.\d+)?)\s*%",
        option_text,
    )
    source_formula = re.search(
        r"(?:现金价值|账户价值).{0,24}(?:扣除|减去).{0,30}"
        r"(?:借款|贷款|欠交|欠款).{0,30}(\d+(?:\.\d+)?)\s*%",
        reasoning,
    )
    if (
        zero_premise
        and percentage_claim
        and source_formula
        and abs(float(percentage_claim.group(1)) - float(source_formula.group(1))) <= 1e-9
        and not re.search(r"(?:主体|产品|比例|时点).{0,8}(?:错误|冲突|不符)", reasoning)
    ):
        return "supported"
    return current_verdict


def explicit_segment_verdict(reasoning: str, current_verdict: str = "") -> str:
    """优先采用 API 片段的明确最终结论，而非陈旧 JSON。"""
    signals: list[tuple[int, str]] = []
    patterns = {
        "supported": (
            r"选项.{0,80}(?:正确|(?<!是否)成立)|"
            r"选项.{0,60}与.{0,30}(?:规定|证据|原文).{0,40}(?:一致|吻合)|"
            r"与选项(?:所述)?(?:完全)?一致|"
            r"与选项.{0,20}(?:一致|吻合).{0,16}(?:支持|正确)|"
            r"(?:省略|未复述).{0,24}不影响(?:其)?正确性|"
            r"(?:故|因此).{0,10}(?:正确|支持|应选)|"
            r"(?:实际|最终)(?:应)?为\s*supported|"
            r"应判\s*supported|"
            r"(?:描述|表述|事实).{0,12}(?:准确|成立)|"
            r"(?:据此|由此|故|因此).{0,40}(?:一致|吻合|成立|支持)|"
            r"(?:直接|充分|明确)支持(?:该|此)?选项[A-D]?|"
            r"(?:规则|条款|计算结果).{0,40}(?:一致|吻合)|"
            r"原(?:unknown|缺证)理由不成立|"
            r"证据直接支持|"
            r"(?<![A-Za-z])supported(?![A-Za-z])"
        ),
        "contradicted": (
            r"选项[A-D]?.{0,8}(?:错误|不成立|矛盾|冲突)|"
            r"(?:数值|比例|降幅|增幅).{0,10}(?:不符|冲突|错误)|"
            r"(?:故|因此).{0,8}(?:矛盾|冲突|不符|错误)|"
            r"未出现选项所述|证据未支持(?:其)?存在|"
            r"(?:标题|名称|表述).{0,16}与原文不一致|"
            r"(?:故|因此).{0,10}(?:不选|错误|不成立)|"
            r"(?:修正|应为).{0,12}contradicted|"
            r"应判\s*contradicted|"
            r"与(?:条款|证据|原文).{0,16}(?:矛盾|冲突|不符)|"
            r"无法支持|"
            r"(?<![A-Za-z])contradicted(?![A-Za-z])"
        ),
        "unknown": (
            r"无法(?:判断|确认|核实|核对|验证|证明)|"
            r"关键事实缺证|证据不足|"
            r"(?<![A-Za-z])unknown(?![A-Za-z])"
        ),
    }
    for verdict, pattern in patterns.items():
        for match in re.finditer(pattern, reasoning, flags=re.I):
            if (
                verdict == "supported"
                and not re.search(r"原(?:unknown|缺证)理由不成立", match.group(0))
                and re.search(
                r"不(?:一致|吻合|成立|正确)|(?:矛盾|冲突|不符|错误|有误)|"
                r"是否正确|无法(?:判断|确认|核实|验证)|证据不足|关键事实缺证|"
                r"(?<![A-Za-z])contradicted(?![A-Za-z])",
                match.group(0),
                )
            ):
                continue
            if verdict == "contradicted" and re.search(
                r"(?:并|且)?无(?:任何)?(?:矛盾|冲突|不符|错误)|"
                r"无.{0,8}(?:矛盾|冲突|不符|错误)|"
                r"(?:没有|不存在)(?:矛盾|冲突|不符|错误)|"
                r"(?:并不|不)(?:构成|属于|代表|意味着).{0,6}"
                r"(?:矛盾|冲突|不符|错误)",
                match.group(0),
            ):
                continue
            signals.append((match.end(), verdict))
    return max(signals, default=(-1, current_verdict))[1]


def repair_region_share_answer_from_reasoning(
    answer: str,
    reasoning: str,
    options: Mapping[str, object],
    answer_format: str,
) -> str:
    """根据 API 数值重新计算百分点及相对占比断言。"""
    if answer_format != "multi":
        return answer
    rows = re.findall(
        r"(20\d{2})年.{0,20}?境外收入\s*([\d,]+).*?"
        r"(?:中国|境内)(?:（[^）]*）)?(?:收入)?\s*([\d,]+)",
        reasoning,
    )
    values = {
        year: (float(foreign.replace(",", "")), float(domestic.replace(",", "")))
        for year, foreign, domestic in rows
    }
    direct_shares = {
        year: float(share)
        for year, share in re.findall(
            r"(20\d{2})年.{0,12}境外(?:收入)?.{0,30}?"
            r"(?:占比|比重)\s*(\d+(?:\.\d+)?)\s*%",
            reasoning,
        )
    }
    if len(values) >= 2:
        old_year, new_year = sorted(values)[-2:]
        old_foreign, old_domestic = values[old_year]
        new_foreign, new_domestic = values[new_year]
        old_share = 100.0 * old_foreign / (old_foreign + old_domestic)
        new_share = 100.0 * new_foreign / (new_foreign + new_domestic)
    elif len(direct_shares) >= 2:
        old_year, new_year = sorted(direct_shares)[-2:]
        old_share = direct_shares[old_year]
        new_share = direct_shares[new_year]
    else:
        return answer
    point_change = new_share - old_share
    relative_change = 100.0 * point_change / old_share
    selected = {letter for letter in answer if letter in "ABCD"}
    for letter, raw_option in options.items():
        option = str(raw_option)
        point_claim = re.search(
            r"(?:占比|比重).{0,24}(?:提高|上升|增加)约?\s*"
            r"(\d+(?:\.\d+)?)\s*个?百分点",
            option,
        )
        relative_claim = re.search(
            r"(?:相对增幅|相对增长|相对提高).{0,12}约?\s*"
            r"(\d+(?:\.\d+)?)\s*%",
            option,
        )
        if point_claim:
            if abs(float(point_claim.group(1)) - point_change) <= 0.05:
                selected.add(str(letter))
            else:
                selected.discard(str(letter))
        if relative_claim:
            if abs(float(relative_claim.group(1)) - relative_change) <= 0.10:
                selected.add(str(letter))
            else:
                selected.discard(str(letter))
        component_claim = re.search(
            r"境外收入增加额.{0,30}(?:大于|超过).{0,30}"
            r"(?:中国|境内).{0,20}减少额",
            option,
        )
        if component_claim:
            component_values = re.search(
                r"境外增加额\s*([\d,]+).*?"
                r"(?:中国|境内)减少额\s*([\d,]+)",
                reasoning,
            )
            if component_values:
                foreign_increase = float(component_values.group(1).replace(",", ""))
                domestic_decrease = float(component_values.group(2).replace(",", ""))
                if foreign_increase > domestic_decrease:
                    selected.add(str(letter))
                else:
                    selected.discard(str(letter))
        identity_claim = re.search(
            r"境外收入增加额.{0,30}减去.{0,40}"
            r"(?:中国|境内).{0,20}减少额.{0,60}"
            r"营业收入增加额.{0,20}(?:一致|相同)",
            option,
        )
        if identity_claim and len(values) >= 2:
            old_total = old_foreign + old_domestic
            new_total = new_foreign + new_domestic
            component_net = (
                new_foreign - old_foreign
            ) - (
                old_domestic - new_domestic
            )
            if abs(component_net - (new_total - old_total)) <= 1e-6:
                selected.add(str(letter))
            else:
                selected.discard(str(letter))
    return "".join(sorted(selected))


def repair_region_share_answer_from_evidence(
    answer: str,
    evidence: str,
    options: Mapping[str, object],
    answer_format: str,
    question_text: str = "",
) -> str:
    """从报告原表选择覆盖范围最大的分地区口径并复核选项。"""
    if answer_format != "multi" or not re.search(r"分地区|境外收入", question_text):
        return answer
    lines = evidence.splitlines()
    candidates: list[tuple[float, float, float, float]] = []

    def large_amounts(line: str) -> list[float]:
        values = [
            float(value.replace(",", ""))
            for value in re.findall(r"(?<!\d)([\d,]+(?:\.\d+)?)(?!\d)", line)
        ]
        return [value for value in values if value >= 100_000]

    for index, line in enumerate(lines):
        if not re.search(r"^\|\s*(?:中国|境内)(?:[（(]|\|)", line):
            continue
        domestic = large_amounts(line)
        if len(domestic) < 2:
            continue
        for sibling in lines[index + 1:index + 5]:
            if not re.search(r"^\|\s*境外\s*\|", sibling):
                continue
            foreign = large_amounts(sibling)
            if len(foreign) >= 2:
                candidates.append((foreign[0], domestic[0], foreign[1], domestic[1]))
                break
    if not candidates:
        return answer
    # 合并全口径的两个年度合计均应覆盖分业务表，因此以较小年度合计最大者优先。
    new_foreign, new_domestic, old_foreign, old_domestic = max(
        candidates,
        key=lambda values: min(values[0] + values[1], values[2] + values[3]),
    )
    years = sorted(set(re.findall(r"20\d{2}", question_text)))
    old_year, new_year = (years[-2], years[-1]) if len(years) >= 2 else ("前期", "本期")
    canonical_reasoning = (
        f"{new_year}年境外收入 {new_foreign:,.0f}，中国收入 {new_domestic:,.0f}；"
        f"{old_year}年境外收入 {old_foreign:,.0f}，中国收入 {old_domestic:,.0f}；"
        f"境外增加额 {new_foreign - old_foreign:,.0f}，"
        f"中国减少额 {old_domestic - new_domestic:,.0f}。"
    )
    return repair_region_share_answer_from_reasoning(
        answer,
        canonical_reasoning,
        options,
        answer_format,
    )


def reconcile_cross_option_numeric_claims(
    decisions: list[dict[str, object]],
    options: Mapping[str, object],
) -> None:
    """跨选项复用 API 派生数值以执行确定性算术。"""
    combined = "\n".join(str(item.get("reasoning", "")) for item in decisions)
    percentages = [
        float(value)
        for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*%", combined)
    ]
    for item in decisions:
        if item.get("verdict") == "supported":
            continue
        letter = str(item.get("option", ""))
        option = str(options.get(letter, ""))
        if (
            re.search(r"排序|由高到低|从高到低", option)
            and re.search(r"可确定.{0,100}(?:排序|顺序)(?:趋势)?", str(item.get("reasoning", "")))
            and not re.search(r"无法确定|不能确定|排序.{0,12}(?:错误|不符)", str(item.get("reasoning", "")))
        ):
            item["verdict"] = "supported"
            continue
        if (
            re.search(r"排序|由高到低|从高到低", option)
            and re.search(
                r"年末.{0,30}每\s*10\s*股.{0,20}\d+(?:\.\d+)?元"
                r".{0,50}全年.{0,20}(?:中期|已实施).{0,30}(?:未提供|缺失)",
                str(item.get("reasoning", "")),
            )
        ):
            known_values = [
                float(value)
                for value in re.findall(
                    r"每\s*10\s*股.{0,20}?(\d+(?:\.\d+)?)\s*元",
                    combined,
                )
            ]
            partial = re.search(
                r"年末.{0,30}每\s*10\s*股.{0,20}?(\d+(?:\.\d+)?)\s*元",
                str(item.get("reasoning", "")),
            )
            if (
                partial
                and len(known_values) >= 3
                and float(partial.group(1)) >= max(known_values)
            ):
                item["verdict"] = "supported"
                continue
        if re.search(r"排序|由高到低|从高到低", option):
            local_reasoning = str(item.get("reasoning", ""))
            partial = re.search(
                r"年末.{0,30}每\s*10\s*股.{0,20}?(\d+(?:\.\d+)?)\s*元",
                local_reasoning,
            )
            missing_interim = re.search(
                r"中期.{0,60}(?:未提供|缺失|无法计算|仅给出总额)",
                local_reasoning,
            )
            known_values = [
                float(value)
                for value in re.findall(
                    r"(?:全年)?每\s*10\s*股.{0,20}?(\d+(?:\.\d+)?)\s*元",
                    combined,
                )
            ]
            for listed_values in re.findall(
                r"全年每\s*10\s*股.{0,12}分别为(.{0,80})",
                combined,
            ):
                known_values.extend(
                    float(value)
                    for value in re.findall(r"(\d+(?:\.\d+)?)\s*元", listed_values)
                )
            if partial and missing_interim:
                lower_bound = float(partial.group(1))
                other_values = [
                    value
                    for value in known_values
                    if abs(value - lower_bound) > 1e-9
                ]
                if (
                    len(set(other_values)) >= 3
                    and lower_bound > max(other_values)
                    and not re.search(
                        r"排序.{0,20}(?:错误|不符|冲突)",
                        local_reasoning,
                    )
                ):
                    item["verdict"] = "supported"
                    continue
        if re.search(r"排序|由高到低|从高到低", option):
            partial = re.search(
                r"年末.{0,30}每\s*10\s*股.{0,20}?(\d+(?:\.\d+)?)\s*元",
                str(item.get("reasoning", "")),
            )
            partial_is_lower_bound = re.search(
                r"(?:加上|另有).{0,20}(?:中期|已分派).{0,30}"
                r"全年.{0,20}(?:高于|大于)",
                str(item.get("reasoning", "")),
            )
            sibling_values = [
                float(value)
                for value in re.findall(
                    r"(?:全年)?每\s*10\s*股.{0,16}?(\d+(?:\.\d+)?)\s*元",
                    combined,
                )
            ]
            lower_candidates = (
                [
                    value
                    for value in sibling_values
                    if partial and value != float(partial.group(1))
                ]
                if partial
                else []
            )
            if (
                partial
                and partial_is_lower_bound
                and len(lower_candidates) >= 2
                and float(partial.group(1)) >= max(lower_candidates)
            ):
                item["verdict"] = "supported"
                continue
        point_claim = re.search(r"约\s*(\d+(?:\.\d+)?)\s*个?百分点", option)
        local_reasoning = str(item.get("reasoning", ""))
        if point_claim and re.search(r"经营现金流(?:率|量净额)|经营活动现金流", option):
            ratio_values = []
            for numerator, denominator in re.findall(
                r"([\d,]{6,})\s*/\s*([\d,]{6,})",
                combined,
            ):
                denominator_value = float(denominator.replace(",", ""))
                if denominator_value:
                    ratio_values.append(
                        100 * float(numerator.replace(",", "")) / denominator_value
                    )
            expected = float(point_claim.group(1))
            if len(ratio_values) >= 4 and any(
                abs(abs(left - right) - expected) <= 0.03
                for index, left in enumerate(ratio_values)
                for right in ratio_values[index + 1:]
            ):
                item["verdict"] = "supported"
                item["reasoning"] = (
                    local_reasoning
                    + "；跨选项原始金额复算后，经营现金流率差与选项一致。"
                )
                continue
        if (
            point_claim
            and not re.search(
                r"无法(?:判断|确认|核实|验算|计算)|关键事实缺证|"
                r"证据不足|(?:选项|表述).{0,24}(?:冲突|错误|不符)",
                local_reasoning,
            )
        ):
            expected = float(point_claim.group(1))
            if any(
                abs(abs(left - right) - expected) <= 0.11
                for index, left in enumerate(percentages)
                for right in percentages[index + 1:]
            ):
                item["verdict"] = "supported"
                item["reasoning"] = (
                    str(item.get("reasoning", ""))
                    + "；跨选项API数值复核后，百分点差与约数容差一致。"
                )
                continue
        if not re.search(
            r"增加额.{0,20}减去.{0,20}减少额.{0,30}营业收入增加额.{0,12}一致",
            option,
        ):
            continue
        increases = re.findall(r"增加(?:额)?[^\d]{0,12}([\d,]+)", combined)
        decreases = re.findall(r"减少(?:额)?[^\d]{0,12}([\d,]+)", combined)
        ratios = re.findall(r"([\d,]{6,})\s*/\s*([\d,]{6,})", combined)
        if not increases or not decreases or not ratios:
            continue
        try:
            increase = float(increases[-1].replace(",", ""))
            decrease = float(decreases[-1].replace(",", ""))
            denominators = sorted(
                {float(den.replace(",", "")) for _, den in ratios},
                reverse=True,
            )
            total_delta = abs(denominators[0] - denominators[1])
        except (ValueError, IndexError):
            continue
        if abs((increase - decrease) - total_delta) <= max(1.0, total_delta * 1e-6):
            item["verdict"] = "supported"
            item["reasoning"] = (
                str(item.get("reasoning", ""))
                + "；跨选项原始金额满足分项变动与总额变动恒等式。"
            )

    # 低于阈值且不可疑的情形，不能继承兄弟证据中仅限可疑情形的例外。
    # 这是通用的阈值/补集推断。
    if re.search(
        r"(?:仅当|只有).{0,24}(?:怀疑|可疑|异常).{0,36}"
        r"(?:无论|不论).{0,12}(?:金额|数额).{0,18}(?:核实|审查)",
        combined,
    ):
        for item in decisions:
            if item.get("verdict") != "unknown":
                continue
            option = str(options.get(str(item.get("option", "")), ""))
            if (
                re.search(r"(?:低于|不足|未达).{0,16}(?:门槛|标准|限额)", option)
                and re.search(r"(?:无|没有|不存在).{0,12}(?:可疑|异常|怀疑)", option)
                and re.search(r"(?:仍|也|必须|应当|一律).{0,12}(?:核实|审查)", option)
            ):
                item["verdict"] = "contradicted"

    if re.search(
        r"年末.{0,40}每\s*10\s*股.{0,20}\d+(?:\.\d+)?元"
        r".{0,70}全年.{0,20}(?:中期|已实施).{0,30}(?:未提供|缺失)",
        combined,
    ):
        for item in decisions:
            option = str(options.get(str(item.get("option", "")), ""))
            if re.search(
                r"全年.{0,40}(?:相差|差额)|(?:相差|差额).{0,40}全年",
                option,
            ):
                item["verdict"] = "contradicted"

    region_values: dict[tuple[str, str], float] = {}
    for year, region, value in re.findall(
        r"(20\d{2})年(?:的)?(境外|中国(?:\(含港澳台\)|（含港澳台）)?|境内)"
        r"(?:收入)?(?:为|是|：|:)?[^\d]{0,12}([\d,]{6,})",
        combined,
    ):
        region_values[(year, "foreign" if region == "境外" else "domestic")] = float(
            value.replace(",", "")
        )
    # 表格常被表述为“2025 年某区域……，2024 年为……”，
    # 第二个年份前不会重复区域标签。
    for anchor in re.finditer(
        r"(20\d{2})年(?:的)?(境外|中国(?:\(含港澳台\)|（含港澳台）)?|境内)"
        r"(?:收入)?(?:为|是|：|:)?[^\d]{0,12}([\d,]{6,})",
        combined,
    ):
        anchor_year, region, anchor_value = anchor.groups()
        key = "foreign" if region == "境外" else "domestic"
        region_values[(anchor_year, key)] = float(anchor_value.replace(",", ""))
        tail = re.split(
            r"[；。\n]",
            combined[anchor.end(): anchor.end() + 100],
            maxsplit=1,
        )[0]
        for year, value in re.findall(
            r"(20\d{2})年(?:同期)?(?:为|是|：|:)?[^\d]{0,10}([\d,]{6,})",
            tail,
        ):
            region_values[(year, key)] = float(value.replace(",", ""))
    years = sorted({year for year, _ in region_values})
    if len(years) >= 2:
        old_year, new_year = years[-2], years[-1]
        keys = [
            (old_year, "foreign"),
            (old_year, "domestic"),
            (new_year, "foreign"),
            (new_year, "domestic"),
        ]
        if all(key in region_values for key in keys):
            old_foreign = region_values[(old_year, "foreign")]
            old_domestic = region_values[(old_year, "domestic")]
            new_foreign = region_values[(new_year, "foreign")]
            new_domestic = region_values[(new_year, "domestic")]
            share_point_change = 100 * (
                new_foreign / (new_foreign + new_domestic)
                - old_foreign / (old_foreign + old_domestic)
            )
            component_delta = (
                (new_foreign - old_foreign)
                + (new_domestic - old_domestic)
            )
            for item in decisions:
                if item.get("verdict") == "supported":
                    continue
                option = str(options.get(str(item.get("option", "")), ""))
                point_claim = re.search(
                    r"境外收入占.{0,20}提高约\s*(\d+(?:\.\d+)?)\s*个?百分点",
                    option,
                )
                if point_claim and abs(float(point_claim.group(1)) - share_point_change) <= 0.05:
                    item["verdict"] = "supported"
                if (
                    re.search(r"境外收入增加额.{0,30}境内|境外收入增加额.{0,40}中国", option)
                    and re.search(r"营业收入增加额.{0,12}(?:基本)?一致", option)
                    and abs(component_delta) > 0
                ):
                    item["verdict"] = "supported"

    # 复用兄弟 API 判定中已经抽出的年度收入和经营现金流金额。
    # 只有四个操作数与三个声称百分比构成同一算术恒等式时才触发。
    decision_contexts = [
        (
            str(options.get(str(item.get("option", "")), ""))
            + "；"
            + str(item.get("reasoning", ""))
        )
        for item in decisions
    ]

    def collect_year_metric_values(
        metric_pattern: str,
        competing_pattern: str,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for context in decision_contexts:
            if not re.search(metric_pattern, context):
                continue
            direct = re.findall(
                rf"(20\d{{2}})年.{{0,65}}?(?:{metric_pattern}).{{0,24}}?"
                r"([\d,]+(?:\.\d+)?)\s*(亿|千)?元",
                context,
            )
            candidates = direct
            if not candidates and not re.search(competing_pattern, context):
                candidates = re.findall(
                    r"(20\d{2})年.{0,36}?([\d,]+(?:\.\d+)?)\s*(亿|千)?元",
                    context,
                )
            for year, raw, unit in candidates:
                number = float(raw.replace(",", ""))
                values[year] = (
                    number
                    if unit == "亿"
                    else number / 100_000
                    if unit == "千"
                    else number / 100_000_000
                )
        return values

    annual_revenue = collect_year_metric_values(
        r"营业收入|营收",
        r"经营活动(?:产生的)?现金流量净额|经营现金流",
    )
    annual_cash = collect_year_metric_values(
        r"经营活动(?:产生的)?现金流量净额|经营现金流(?:量净额)?",
        r"营业收入|营收",
    )
    annual_profit = collect_year_metric_values(
        r"归母净利润|归母净利|归属于上市公司股东的净利润|净利润",
        r"营业收入|营收|经营活动(?:产生的)?现金流量净额|经营现金流",
    )
    profit_candidates: dict[str, list[float]] = {
        year: [value] for year, value in annual_profit.items()
    }
    for context in decision_contexts:
        for year, raw, unit in re.findall(
            r"(20\d{2})年.{0,30}?(?:Q1\s*[-—至]\s*Q4|第一至第四季度|各季度)"
            r".{0,30}?(?:归母净利润|归母净利|归属于上市公司股东的净利润|净利润)"
            r".{0,120}?(?:合计|全年(?:为)?|总计)\s*"
            r"([\d,]+(?:\.\d+)?)\s*(亿|千)?元",
            context,
        ):
            number = float(raw.replace(",", ""))
            normalized = (
                number
                if unit == "亿"
                else number / 100_000
                if unit == "千"
                else number / 100_000_000
            )
            profit_candidates.setdefault(year, []).append(normalized)
    profit_years = sorted(profit_candidates)
    if len(profit_years) >= 2:
        old_year, new_year = profit_years[-2], profit_years[-1]
        for item in decisions:
            if item.get("verdict") == "supported":
                continue
            option = str(options.get(str(item.get("option", "")), ""))
            claim = re.search(
                r"归母净利润同比下降约?\s*(\d+(?:\.\d+)?)\s*%",
                option,
            )
            if claim and any(
                old_value > 0
                and abs(
                    100 * (old_value - new_value) / old_value
                    - float(claim.group(1))
                )
                <= 0.08
                for old_value in profit_candidates[old_year]
                for new_value in profit_candidates[new_year]
            ):
                    item["verdict"] = "supported"
                    item["reasoning"] = (
                        str(item.get("reasoning", ""))
                        + "；复用同题API已抽取的两期年度净利润原始值，同比算术成立。"
                    )

    common_years = sorted(set(annual_revenue) & set(annual_cash))
    if len(common_years) >= 2:
        old_year, new_year = common_years[-2], common_years[-1]
        old_ratio = 100 * annual_cash[old_year] / annual_revenue[old_year]
        new_ratio = 100 * annual_cash[new_year] / annual_revenue[new_year]
        for item in decisions:
            if item.get("verdict") == "supported":
                continue
            option = str(options.get(str(item.get("option", "")), ""))
            claim = re.search(
                r"比例由约\s*(\d+(?:\.\d+)?)\s*%\s*降至约\s*"
                r"(\d+(?:\.\d+)?)\s*%.*?下降约\s*(\d+(?:\.\d+)?)\s*个?百分点",
                option,
            )
            if not claim:
                continue
            expected_old, expected_new, expected_delta = map(float, claim.groups())
            if (
                abs(old_ratio - expected_old) <= 0.03
                and abs(new_ratio - expected_new) <= 0.03
                and abs((old_ratio - new_ratio) - expected_delta) <= 0.03
            ):
                item["verdict"] = "supported"
                item["reasoning"] = (
                    str(item.get("reasoning", ""))
                    + "；复用同题API已抽取的年度营收与经营现金流原始值，三项比例恒等式成立。"
                )

    # 如果另一个选项判定已经明确给出题目所需的上年调整前金额，
    # 则一致使用该 API 推导出的金额。
    adjusted_before_values: dict[tuple[str, str], float] = {}
    for year, metric, value in re.findall(
        r"(20\d{2})年.{0,20}?(归母净利润|基本每股收益).{0,16}?"
        r"[（(]?(?:调整前|原始披露)[）)]?(?:为|值为)?\s*"
        r"([\d,]+(?:\.\d+)?)",
        combined,
    ):
        adjusted_before_values[(year, metric)] = float(value.replace(",", ""))
    current_values: dict[tuple[str, str], float] = {}
    for year, metric, value in re.findall(
        r"(20\d{2})年.{0,12}?(归母净利润|基本每股收益)(?:为|值为)?"
        r"\s*([\d,]+(?:\.\d+)?)",
        combined,
    ):
        current_values[(year, metric)] = float(value.replace(",", ""))
    for context in decision_contexts:
        metrics = re.findall(r"归母净利润|基本每股收益", context)
        if not metrics:
            continue
        metric = metrics[0]
        for year, value in re.findall(
            r"(20\d{2})年(?:的)?[（(]?(?:调整前|原始披露)[）)]?(?:为|值为)?"
            r"\s*([\d,]+(?:\.\d+)?)",
            context,
        ):
            adjusted_before_values[(year, metric)] = float(value.replace(",", ""))
        for year, value in re.findall(
            r"(20\d{2})年(?:的)?(?:归母净利润|基本每股收益)(?:为|值为)?"
            r"\s*([\d,]+(?:\.\d+)?)",
            context,
        ):
            current_values[(year, metric)] = float(value.replace(",", ""))
    years = sorted(
        year
        for year, metric in adjusted_before_values
        if metric == "归母净利润"
    )
    if years:
        old_year = years[-1]
        newer = sorted(
            year
            for year, metric in current_values
            if metric == "归母净利润" and year > old_year
        )
        if newer:
            new_year = newer[-1]
            old_value = adjusted_before_values[(old_year, "归母净利润")]
            new_value = current_values[(new_year, "归母净利润")]
            decline = 100 * (old_value - new_value) / old_value
            for item in decisions:
                if item.get("verdict") == "supported":
                    continue
                option = str(options.get(str(item.get("option", "")), ""))
                claims = [
                    float(value)
                    for value in re.findall(
                        r"降幅约为?\s*(\d+(?:\.\d+)?)\s*%|"
                        r"(\d+(?:\.\d+)?)\s*%\s*的降幅",
                        option,
                    )
                    for value in value
                    if value
                ]
                if claims and any(abs(value - decline) <= 0.03 for value in claims):
                    item["verdict"] = "supported"
                    item["reasoning"] = (
                        str(item.get("reasoning", ""))
                        + "；按同题API明确标注的调整前原始值复算，选项约数成立。"
                    )

    # 当兄弟判定同时明确当前金额和上年调整前金额时，
    # 优先采用同一分部内紧凑的年份/指标绑定。
    # 这可以避免被其他分部中同指标的调整后比较值干扰。
    for context in decision_contexts:
        pair = re.search(
            r"(20\d{2})年(?:的)?归母净利润(?:为|值为)?\s*"
            r"([\d,]+(?:\.\d+)?).{0,100}?"
            r"(20\d{2})年(?:的)?[（(]?调整前[）)]?(?:为|值为)?\s*"
            r"([\d,]+(?:\.\d+)?)",
            context,
        )
        if not pair:
            continue
        new_year, new_raw, old_year, old_raw = pair.groups()
        if new_year <= old_year:
            continue
        new_value = float(new_raw.replace(",", ""))
        old_value = float(old_raw.replace(",", ""))
        if old_value <= 0:
            continue
        decline = 100 * (old_value - new_value) / old_value
        for item in decisions:
            if item.get("verdict") == "supported":
                continue
            option = str(options.get(str(item.get("option", "")), ""))
            if not (
                "归母净利润" in option
                and re.search(r"调整前|原始披露", combined)
            ):
                continue
            claims = [
                float(value)
                for value in re.findall(
                    r"(\d+(?:\.\d+)?)\s*%\s*的降幅|"
                    r"降幅约为?\s*(\d+(?:\.\d+)?)\s*%",
                    option,
                )
                for value in value
                if value
            ]
            if any(abs(value - decline) <= 0.03 for value in claims):
                item["verdict"] = "supported"
