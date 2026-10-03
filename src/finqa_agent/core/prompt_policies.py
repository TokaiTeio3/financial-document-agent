from __future__ import annotations

import json
from typing import Mapping, Sequence

from .domain_policies import (
    CONTRACT_PARALLEL_CLAUSE_RULE,
    FINANCIAL_CROSS_OPTION_RULE,
    RESEARCH_FUNCTIONAL_QUALITY_RULE,
)


DOMAIN_INSTRUCTIONS = {
    "financial_reports": (
        "逐项绑定同一主体、年度及合并/母公司口径；区分相对百分比与百分点。"
        "全年分红区分已实施中期与剩余年末方案，避免重复或漏加。表格中的本期、"
        "同期、调整前后不得串列；中间值不提前四舍五入。若证据给出同一年度"
        "Q1至Q4或第一至第四季度四个互斥期间的原始金额，必须相加得到全年金额，"
        "不得仅因未另列全年合计而判为缺证。连续两年年度金额可直接验算同比，"
        "即使相同百分数也出现在季度列，仍以年度算术结果为准。"
        + FINANCIAL_CROSS_OPTION_RULE
    ),
    "financial_contracts": (
        "回到题干指定文件，联合核对定义、触发条件、期限、公式、例外和争议解决。"
        "含全部、均、完全一致的选项须逐份成立；非传统产线项目不能仅因缺少同名"
        "风险标题就判为缺证。必须精确判断仅、每年一次等量词的作用域，不能把"
        "最后两个年度内每年仅能行使一次误读为只有最后两个年度存在该项权利。"
        "一般性的项目效益、收入实现或实施未达预期风险，不等同于传统生产线的"
        "新增产能消化风险；软件研发、系统实施等项目若不形成传统产能，不得用"
        "广义经营风险反向否定其“不涉及传统产能消化”的陈述。"
        + CONTRACT_PARALLEL_CLAUSE_RULE
    ),
    "insurance": (
        "联合核对责任、等待期、免责、给付比例、累计限额和条款例外。不承担责任"
        "与承担后按比例给付不同；年龄分档必须命中明确区间，相邻续页限制仍有效。"
        "不能因年龄未进入后一档就臆测前一档比例更低，必须读取覆盖该年龄的区间。"
        "题干概括询问核风险类别时，核反应、核辐射、核爆炸、核污染或放射性污染"
        "均可作为明确列明该类风险的证据，除非题干要求逐字一致。"
        "题目询问某个具体事由是否列为责任免除时，必须在免责正文中找到该事由"
        "或语义等价表述；仅找到“责任免除”章节标题，不能证明该具体事由被列入。"
    ),
    "regulatory": (
        "精确绑定制度名称、义务对象、施行日、过渡期和保存期限。不同办法的相似"
        "义务不得互相移植；并行义务可同时成立，非排他选项不能只因省略另一义务"
        "而判错。选项用“某法律制度下”概括该制度及其配套规章中的实体义务时，"
        "只要义务内容、对象和期限均获证据支持，不得仅因未逐字写出配套规章名称"
        "或义务起算条件而判错；只有选项明确声称该义务直接出自某一具体条文且"
        "证据相反时，才构成规范层级或出处冲突。全称量词必须覆盖全部条件与例外。"
        "题目用“可疑交易”概括条文中的“有合理理由怀疑涉嫌洗钱或恐怖融资”时，"
        "若对象、行为和义务均相同，应按制度语义等价理解，不能仅因题目采用简称"
        "判为扩大范围；只有简称确实覆盖了条文之外的对象或义务时才判错。"
    ),
    "research": (
        "允许从多份材料的事实前提、行业属性和因果关系作结构化归纳，但每个事实"
        "前提必须有证据。区分资金来源端和承接端、主动行为和被动结果；复合选项"
        "须逐一核对全部前提。品牌化难度应比较历史渠道偏企业端还是消费端、产品"
        "差异是否易感知、消费者心智、渠道继承性、品质信任和品牌溢价培育阶段；"
        "原文不必逐字写出相对难度结论。若证据已表明业务长期偏企业端，且消费品牌"
        "认知或溢价仍在培育，即足以支持其向消费端延伸的相对品牌化难度判断，"
        "不得再要求材料直接给出横向排名。两融净流入达到历史高位可作为杠杆型"
        "投资者参与度上升的市场行为证据；理财偏债基货基与险资增配高股息并强化"
        "久期匹配，可分别支持稳健偏好及长期收益与稳定回报的平衡判断。"
        "行业机制归纳可以由已给事实推出，但选项若声称具体经营动作，例如扩大"
        "产销量、压价、要求交易对手让利或开设具体渠道，每个动作本身必须有文档"
        "证据；仅称“符合产业逻辑”“合理经营策略”不能把缺证动作判为正确。"
        "选项没有使用“全部、普遍、唯一”等全称量词时，同一机构类别的代表性"
        "材料若分别明确支持资产端与负债端动作，即可支持其资负联动归纳；不得"
        "仅因其中一项动作也见于其他机构，就否定该类别自身已有的同项证据。"
        "跨产业链技术路径比较中，下游自研专用芯片或算法与上游ASIC/IP定制"
        "服务属于可直接衔接的需求和能力，不要求材料逐字写出两份报告间的关联。"
        "品牌转型分析中，内容或流量能力对应品牌热度，选品、自营研发和品质"
        "控制对应产品品质；材料若同时支持两端，可归纳为内容与品质双轮驱动。"
        + RESEARCH_FUNCTIONAL_QUALITY_RULE
    ),
}


def build_scoped_joint_prompt(
    *,
    domain: str,
    answer_format: str,
    stem: str,
    options: Mapping[str, object],
    evidence_sections: Sequence[str],
    calculation_results: Sequence[Mapping[str, object]],
    selects_incorrect: bool,
) -> str:
    """为题目全部选项构建一次限定领域的 API 调用。"""
    polarity = (
        "题干选择错误项，answer收集 contradicted 项。"
        if selects_incorrect
        else "题干选择正确项，answer收集 supported 项。"
    )
    option_text = "\n".join(
        f"{key}. {value}" for key, value in sorted(options.items())
    )
    calculations = json.dumps(
        list(calculation_results),
        ensure_ascii=False,
        separators=(",", ":"),
    )[:2600]
    return f"""你是金融长文档证据裁决器，只依据本轮原始文档证据作答。
一次独立判断全部选项；每项输出 supported、contradicted 或 unknown，并引用证据编号。
reasoning只写最终证据链，不得自问自答、反复改判或输出检索元数据。
多选题按A项、B项依次写：先展开证据中的关键事实或数值，再说明它与选项的主体、
时点、条件、口径或结论一致、冲突还是缺证；不得用“E1支持”代替事实。
计算题写清原始值、关键算式和最终结果；不抄整段原文。可以作证据确定的算术、
时间和逻辑推导，不得用常识补具体事实。非排他选项不因省略并行规则判错；
含仅、全部、无论、完整等限定时必须覆盖完整作用域。{polarity}
领域规则：{DOMAIN_INSTRUCTIONS.get(domain, "")}
只输出严格JSON：
{{"answer":"...","confidence":"high|medium|low","option_judgments":{{"A":{{"verdict":"supported|contradicted|unknown","evidence_ids":[1]}}}},"reasoning":"不超过260字"}}

领域：{domain}
答案格式：{answer_format}
题目：{stem}
选项：
{option_text or "无选项"}
API计算程序及结果：{calculations}
本轮证据：
{chr(10).join(evidence_sections)}
"""


def build_scoped_option_prompt(
    *,
    domain: str,
    stem: str,
    letter: str,
    option: str,
    evidence: str,
) -> str:
    """为单个选项构建紧凑的独立裁决 Prompt。"""
    return f"""你是金融长文档单选项证据裁决器，本次只判断一个选项。
- supported：证据直接支持，或可由证据作确定算术、时间、条件及因果推导。
- contradicted：主体、时点、数值、单位、条件、例外或逻辑与证据冲突。
- unknown：关键事实确实缺证；不得用常识补充具体事实。
准确陈述一项基础或并行规则的非排他选项，不因省略其他规则而判错。
含仅、全部、无论、完整、唯一等限定时，证据必须覆盖完整作用域。
领域规则：{DOMAIN_INSTRUCTIONS.get(domain, "")}
只输出严格JSON：
{{"verdict":"supported|contradicted|unknown","reasoning":"不超过180字并引用证据编号"}}

题目：{stem}
待判断选项{letter}：{option}
本轮定向证据：
{evidence}
"""


def build_scoped_calculation_prompt(
    *,
    domain: str,
    stem: str,
    disclosed_metric_anchors: Sequence[Mapping[str, object]],
    table_value_anchors: Sequence[Mapping[str, object]],
    evidence: str,
) -> str:
    """构建不含跨领域规则的紧凑计算代码 Prompt。"""
    precision_rule = ""
    if (
        domain == "financial_reports"
        and "隐含营业收入" in stem
        and disclosed_metric_anchors
    ):
        precision_rule = (
            "\n同一指标若同时出现正文中按亿元取整的概述值，以及“主要会计数据/财务指标”"
            "表中\n按百万元披露的精确值，必须使用表格精确值；不得把正文取整值换算后"
            "替代原始金额。"
        )
    return f"""你是金融文档计算代码生成器。只根据题目和本轮证据生成安全Python代码，
将最终结果赋给变量 answer；不要读取文件、联网或import。
answer直接符合题目格式，多项结果用中文分号连接，不要字典、列表或解释前缀。
严格区分百分点差(a-b)与相对变化率((a-b)/b)。中间量保持完整精度，只在最终答案
按题意舍入；题目未指定时百分数保留两位小数。优先使用文档直接披露且口径一致的
比率、每股或每10股数值；只有题目明确要求原始金额时才从原始金额计算。{precision_rule}
表格必须同时核对标题、行名、期间、单位及调整前后口径；不同单位先分别换算。
最终单位为亿元时，按股数和每股金额算出的“元”必须除以1e8；最终单位为万人时，
按总金额除以人均金额得到的“人”必须再除以1e4。输出前必须反向检查一次数量级。
若答案格式要求“主体>主体；差值”，answer必须保留题目中的主体名称和排序，
不能用两个比率替代主体；差值必须从未提前舍入的原始金额重新计算。
{DOMAIN_INSTRUCTIONS.get(domain, "")}
研报计算必须严格匹配题目对象层级；“乘用车”不得用包含商用车等范围的“全部车辆”
平均值替代，同一年出现旧预测、新预测和实际值时，以题目指定的历史实际值为基期。
保险计算须分别提取各合同的公式、年龄/年度系数、已领取金额、计入比例和手续费，
不得把账户价值、现金价值或已交保费无条件相加。证据已经给出现金价值、退还金额
或给付金额的计算公式时，必须逐项代入该公式中的收益计入比例和手续费；不得把
账户价值直接当作现金价值，也不得借用其他产品或其他年度的费率覆盖本合同公式。
证据不足时输出 answer = ''。只输出Python代码。

题目：{stem}
直接披露指标锚点：{json.dumps(list(disclosed_metric_anchors), ensure_ascii=False, separators=(",", ":"))}
候选表格行：{json.dumps(list(table_value_anchors), ensure_ascii=False, separators=(",", ":"))}
证据：
{evidence}
"""


def build_scoped_bridge_prompt(
    *,
    domain: str,
    stem: str,
    unresolved_options: Mapping[str, object],
    first_reasoning: Mapping[str, object],
    evidence: str,
) -> str:
    """仅为未决选项构建紧凑的第二轮 Prompt。"""
    return f"""你是金融长文档缺证桥接裁决器。只复核列出的未决选项，并为每个字母返回
supported、contradicted或unknown。联合使用本轮跨页/跨文档证据，可作确定算术和
结构化归纳，但不得用常识补具体事实。{DOMAIN_INSTRUCTIONS.get(domain, "")}
只输出严格JSON：
{{"judgments":{{"A":{{"verdict":"supported|contradicted|unknown","reasoning":"不超过140字"}}}}}}

题目：{stem}
未决选项：{json.dumps(dict(unresolved_options), ensure_ascii=False, separators=(",", ":"))}
首轮理由：{json.dumps(dict(first_reasoning), ensure_ascii=False, separators=(",", ":"))}
补充证据：
{evidence}
"""
