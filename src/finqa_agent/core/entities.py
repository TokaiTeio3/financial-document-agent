from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


def repair_mojibake(text: str) -> str:
    """恢复被单字节编码错误解码的 UTF-8/GBK 文本。"""
    value = str(text or "")
    if not value:
        return value
    suspicious = re.compile(r"[ÃÂåæçèéäöüÊÎÐÑÒÓÆ½°²µÄÖÐ¹ú]")

    def score(candidate: str) -> float:
        cjk = len(re.findall(r"[\u4e00-\u9fff]", candidate))
        bad = len(suspicious.findall(candidate))
        replacement = candidate.count("\ufffd")
        controls = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", candidate))
        return cjk * 3.0 - bad * 1.5 - replacement * 8.0 - controls * 5.0

    def decoded_candidates(fragment: str) -> List[str]:
        candidates = [fragment]
        for source_encoding in ("latin1", "cp1252"):
            try:
                raw = fragment.encode(source_encoding)
            except (UnicodeEncodeError, LookupError):
                continue
            for target_encoding in ("utf-8", "gb18030"):
                try:
                    candidates.append(raw.decode(target_encoding))
                except (UnicodeDecodeError, LookupError):
                    continue
        return candidates

    def repair_line(line: str) -> str:
        candidates = decoded_candidates(line)
        best = max(candidates, key=score)
        if score(best) > score(line) + 2.0:
            return best
        # 部分 PDF 转换结果会把私有区项目符号和可恢复的单字节乱码混在一起。
        # 仅修复字节范围片段，并保留私有区字形作为结构分隔符。
        segmented = re.sub(
            r"[\x00-\xff]{4,}",
            lambda match: max(decoded_candidates(match.group(0)), key=score),
            line,
        )
        return segmented if score(segmented) > score(line) + 2.0 else line

    return "\n".join(repair_line(line) for line in value.split("\n"))


def dedupe(values: Iterable[object], limit: int | None = None) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


GENERIC_TERMS = {
    "公司",
    "产品",
    "文档",
    "报告",
    "内容",
    "规定",
    "条款",
    "下列",
    "正确",
    "错误",
    "关于",
    "根据",
    "判断",
    "计算",
}

DOMAIN_TERMS: Dict[str, List[str]] = {
    "financial_reports": [
        "营业收入",
        "营业总收入",
        "归母净利润",
        "净利润",
        "经营现金流",
        "现金流量净额",
        "资产负债率",
        "研发费用",
        "研发投入",
        "现金分红",
        "每股收益",
        "同比",
        "增长率",
    ],
    "insurance": [
        "保险责任",
        "责任免除",
        "等待期",
        "免赔额",
        "赔付比例",
        "现金价值",
        "退保",
        "保单贷款",
        "犹豫期",
        "宽限期",
        "身故保险金",
        "特定药品",
    ],
    "financial_contracts": [
        "发行人",
        "发行金额",
        "注册金额",
        "主体评级",
        "债项评级",
        "主承销商",
        "募集资金用途",
        "转股价格",
        "赎回",
        "回售",
        "向下修正",
        "违约利息",
    ],
    "regulatory": [
        "应当",
        "不得",
        "可以",
        "无需",
        "报送",
        "披露",
        "公示",
        "备案",
        "客户尽职调查",
        "受益所有人",
        "反洗钱",
        "定期报告",
    ],
    "research": [
        "市场规模",
        "市场空间",
        "CAGR",
        "复合增速",
        "同比",
        "渗透率",
        "出货量",
        "销量",
        "净利润",
        "PE",
        "市场份额",
        "预计",
        "预测",
    ],
}

NEGATIVE_TERMS = ["不", "未", "不得", "不承担", "除外", "不适用", "无需", "不能", "禁止"]
CALC_TERMS = ["合计", "差额", "比例", "增长率", "同比", "排序", "最高", "最低", "多少", "计算"]


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def extract_entities(text: str, domain: str = "") -> List[str]:
    raw = str(text or "")
    terms: List[str] = []
    suffixes = (
        "股份有限公司",
        "有限责任公司",
        "集团有限公司",
        "控股有限公司",
        "集团",
        "银行",
        "证券",
        "保险",
        "基金",
        "科技",
        "股份",
        "办法",
        "规定",
        "准则",
        "指引",
        "报告",
        "债券",
        "保险金",
        "医疗保险",
        "责任保险",
    )
    suffix_pattern = "|".join(map(re.escape, suffixes))
    terms.extend(re.findall(rf"[\u4e00-\u9fffA-Za-z0-9（）()·\-]{{2,36}}(?:{suffix_pattern})", raw))
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9._-]{1,18}", raw))
    terms.extend(re.findall(r"20\d{2}(?:年|Q[1-4])?", raw))
    terms.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|个工作日|个自然日|日|个月|年|倍)", raw))
    for term in DOMAIN_TERMS.get(domain, []):
        if term in raw:
            terms.append(term)
    for match in re.findall(r"[“\"《](.*?)[”\"》]", raw):
        if 2 <= len(match) <= 50:
            terms.append(match)
    return [t for t in dedupe(terms, 80) if t not in GENERIC_TERMS]


def extract_constraint_terms(text: str, domain: str = "") -> List[str]:
    """仅根据当前题目文本派生字面审计锚点。"""
    raw = str(text or "")
    terms: List[str] = []
    quoted = re.findall(r"[“\"《](.*?)[”\"》]", raw)
    topics = re.findall(
        r"(?:关于|明确约定|规定)(.{2,36}?)(?:的说法|的表述|为|是|，|。|？|\?)",
        raw,
    )
    clauses = [*quoted, *topics]
    clauses.extend(
        item
        for item in re.split(r"[。；;：:\n]", raw)
        if 3 <= len(compact(item)) <= 60
    )
    split_pattern = (
        r"[,，、/／]|(?:但是|但|以及|并且|同时|其中|除外|除非|"
        r"之后|以前|以内|届满|导致|按照|按|不得|不应|禁止|"
        r"可覆盖|覆盖|用于|因|则|且|与|及|将|纳入|属于|包括|通过|的|为|后)"
    )
    for clause in clauses:
        cleaned = re.sub(
            r"^(?:下列|以下|关于|根据|哪(?:一|些)项|选项[A-Z]?)",
            "",
            clause.strip(),
        )
        if 3 <= len(compact(cleaned)) <= 36:
            terms.append(cleaned)
        for atom in re.split(split_pattern, cleaned):
            atom = atom.strip(" ：:，,。；;（）()")
            if 2 <= len(compact(atom)) <= 18 and _is_constraint_term(atom):
                terms.append(atom)
    terms.extend(
        re.findall(
            r"\d+(?:\.\d+)?\s*(?:%|年|个月|日|个工作日|个交易日|倍|元|万元|亿元)",
            raw,
        )
    )
    terms.extend(term for term in DOMAIN_TERMS.get(domain, []) if term in raw)
    for match in re.finditer(
        r"\d+(?:\.\d+)?\s*(?:%|年|个月|日|个工作日|个交易日|倍)",
        raw,
    ):
        left = max(0, match.start() - 10)
        right = min(len(raw), match.end() + 10)
        window = raw[left:right].strip(" ：:，,。；;（）()")
        if 3 <= len(compact(window)) <= 28:
            terms.append(window)
    return [term for term in dedupe(terms, 20) if _is_constraint_term(term)]


def _is_constraint_term(term: str) -> bool:
    """排除题目脚手架文本，同时保留可由来源核验的词语。"""
    value = compact(term)
    if len(value) < 2 or value in GENERIC_TERMS:
        return False
    if (
        re.fullmatch(r"(?:下列|以下|哪项|哪些|说法|表述|正确|错误|是)+[？?]?", value)
        or re.search(r"(?:下列|以下).{0,8}(?:说法|表述).{0,4}[？?]?$", value)
    ):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9%]", value))


@dataclass
class RuntimeEntityMemory:
    path: Path
    enabled: bool = True
    _items: List[Dict[str, object]] = field(default_factory=list)

    def remember(self, qid: str, domain: str, entities: List[str], source: str, token_usage: Dict[str, int] | None = None) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        item = {
            "qid": qid,
            "domain": domain,
            "source": source,
            "entities": dedupe(entities, 80),
            "token_usage": token_usage or {},
        }
        self._items.append(item)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
