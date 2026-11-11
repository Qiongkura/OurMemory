# -*- coding: utf-8 -*-
"""Bidirectional Civilization Membrane (文明交互膜).

Implements Row ④ of the architecture ("文明交互膜：双向净化"):
- Inbound: filters rhetorical questions, jokes, commands, and LLM placeholder
  garbage before they can enter memory.
- Outbound:
  1. ``check_reply_consistency``: catches replies that contradict active
     identity/scene slots (e.g. claiming to be in another city or school).
  2. ``ungrounded_event_terms`` / ``strip_ungrounded_events``: catches
     ungrounded first-person claims of major life events (interviews,
     internships, hospitalization, surgery, bereavement, breakups, etc.)
     when neither user context nor stored memory contains those terms.
"""

from __future__ import annotations

import dataclasses
import re

from .soul import DEFAULT_SOUL, SoulSpec
from .store import MemoryStore
from .text import normalize

SENSITIVE_EVENT_TERMS: tuple[str, ...] = (
    "面试",
    "笔试",
    "实习",
    "找工作",
    "投简历",
    "改简历",
    "简历",
    "拿offer",
    "拿到offer",
    "offer",
    "入职",
    "辞职",
    "离职",
    "跳槽",
    "转正",
    "裁员",
    "招聘",
    "岗位",
    "运营岗",
    "住院",
    "手术",
    "急诊",
    "骨折",
    "怀孕",
    "流产",
    "车祸",
    "去世",
    "葬礼",
    "搬家",
    "租房",
    "退租",
    "出差",
    "出国",
    "签证",
    "相亲",
    "分手",
    "复合",
    "订婚",
    "结婚",
    "领证",
    "见家长",
    # English equivalents, so the guard is not silently Chinese-only.
    "interview",
    "job interview",
    "internship",
    "intern",
    "résumé",
    "resume",
    "job offer",
    "offer letter",
    "hired",
    "fired",
    "laid off",
    "layoff",
    "promotion",
    "hospital",
    "hospitalized",
    "surgery",
    "operation",
    "emergency room",
    "pregnant",
    "pregnancy",
    "miscarriage",
    "car accident",
    "passed away",
    "funeral",
    "moving out",
    "moving house",
    "relocating",
    "business trip",
    "visa",
    "blind date",
    "broke up",
    "breakup",
    "got back together",
    "engaged",
    "engagement",
    "getting married",
    "married",
    "met the parents",
    "offered a job",
    "got the job",
    "quit my job",
    "lost my job",
)


def _term_pattern(term: str) -> str:
    """Word-boundary ASCII terms, plain terms for CJK.

    Without the boundaries ``intern`` matches inside ``internet`` and
    ``operation`` inside ``operations`` — the guard would then fire on
    ordinary technical talk.
    """
    escaped = re.escape(term)
    if term.isascii():
        return rf"(?<![A-Za-z]){escaped}(?![A-Za-z])"
    return escaped


_SENSITIVE_RE = re.compile(
    "|".join(
        _term_pattern(term)
        for term in sorted(SENSITIVE_EVENT_TERMS, key=len, reverse=True)
    ),
    re.IGNORECASE,
)

# English words that are also ordinary non-event vocabulary.  They only count
# as a life-event claim when the surrounding words make the event reading the
# plausible one ("fixing my resume", not "resume the video").
_AMBIGUOUS_EN = {
    "resume": re.compile(
        r"(?:\bmy\b|\bthe\b|\ba\b|\bupdate\b|\bfix\w*\b|\bsend\b|\bsent\b|"
        r"\bsubmit\w*\b|\bpolish\w*\b|\bprint\w*\b|\battach\w*\b)"
        r"[^.!?\n]{0,20}?\bresume\b"
        r"|\bresume\b[^.!?\n]{0,12}?\b(?:is|was|got|needs|ready)\b",
        re.IGNORECASE,
    ),
    "operation": re.compile(
        r"\b(?:had|have|having|needs?|needed|underwent|undergo|surgery|"
        r"recover(?:ing|ed)? from)\b[^.!?\n]{0,20}?\boperation\b",
        re.IGNORECASE,
    ),
    "promotion": re.compile(
        r"\b(?:got|get|getting|earned|offered|denied|up for)\b"
        r"[^.!?\n]{0,12}?\bpromotion\b",
        re.IGNORECASE,
    ),
    "engaged": re.compile(
        r"\b(?:got|get|getting|just|newly|recently)\s+engaged\b"
        r"|\bengaged\s+to\b"
        r"|\bi(?:'m| am)\s+(?:now\s+)?engaged\b",
        re.IGNORECASE,
    ),
    "married": re.compile(
        r"\b(?:got|get|getting|just|already|newly|recently)\s+married\b"
        r"|\bmarried\s+(?:to|next|in)\b"
        r"|\bi(?:'m| am)\s+(?:already\s+)?married\b",
        re.IGNORECASE,
    ),
}

_CLAUSE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+|\.\s+")
_SUBCLAUSE_SPLIT_RE = re.compile(r"[，,]+")

_THIRD_PARTY_PREFIX_RE = re.compile(
    r"(?:我(?:有个|一个|那个)?(?:室友|舍友|同学|朋友|闺蜜|学姐|学长|表姐|表哥|堂姐|堂哥|姐姐|哥哥|弟弟|妹妹|爸|妈|父母|老师|同事|亲戚)|"
    r"别人|他们|她们|她|他|大家|网上|电视剧|小说|电影|新闻|刷到|看到有人|听说|"
    # English third parties.  ``my roommate`` must be excluded before the
    # first-person check, or every "my X" reads as a self claim.
    r"\bmy\s+(?:roommate|roomie|flatmate|housemate|friend|best\s+friend|classmate|"
    r"coworker|co-?worker|colleague|boss|manager|teacher|professor|advisor|"
    r"sister|brother|mom|mum|mother|dad|father|parents|cousin|aunt|uncle|"
    r"girlfriend|boyfriend|partner|husband|wife|neighbou?r|teammate|labmate|"
    r"senior|junior|intern)\b|"
    r"\b(?:he|she|they|someone|somebody|people|everyone|others)\b|"
    r"\b(?:a|one\s+of\s+my)\s+(?:friend|classmate|coworker|colleague)\b|"
    r"\bi\s+(?:heard|saw|read)\b|"
    r"\b(?:online|on\s+the\s+news|in\s+the\s+news|in\s+a\s+movie|in\s+the\s+novel)\b)"
)

_NEGATED_OR_META_RE = re.compile(
    r"(?:没有|没去|不去|不用|又没|才没|哪有|怎么可能|不可能|乱说|编造|瞎编|假装|梦到|做梦|"
    r"\b(?:didn'?t|did\s+not|never|not\s+really|no\s+way|i\s+wish|if\s+only|"
    r"dreamt|dreamed|in\s+(?:my|a)\s+dream|hypothetically|"
    r"imagine|suppose|pretend|role-?play|joking|kidding|make[- ]believe|as\s+if|"
    r"don'?t\s+(?:have|got|get)|do\s+not\s+have)\b)"
)

_USER_DIRECTED_RE = re.compile(
    r"你(?:怎么|干嘛|为什么|要去|去不去|是不是|有没有|在不在|准备|打算)?[^，。！？\n]{0,8}"
    r"(?:面试|笔试|实习|找工作|投简历|改简历|简历|住院|手术|出差|搬家|相亲|分手|订婚|结婚)|"
    r"\byou(?:'re|\s+are|\s+were)?\b[^.!?\n]{0,20}?"
    r"\b(?:interview|offer|internship|hospital|surgery|operation|visa|"
    r"promotion|wedding|exam)\b|"
    r"\byour\b[^.!?\n]{0,12}?"
    r"\b(?:interview|offer|internship|surgery|operation|visa|promotion|wedding)\b"
)

_SELF_EVENT_VERB_RE = re.compile(
    r"(?:^|[，,。！？\s]|去|在|要|刚|准备|打算|赶着|忙着|投了|改了|写了|收到|拿到|参加|过了|挂了|约了|排了)"
    r"(?:面试|笔试|实习|找工作|投简历|改简历|简历|拿offer|拿到offer|offer|入职|辞职|离职|招聘|岗位|运营岗|住院|手术|急诊|搬家|租房|出差|出国|相亲|订婚|结婚|领证|见家长)"
)

_ASSERTIVE_CUE_RE = re.compile(
    r"(?:我|刚|今天|明天|昨天|后天|下周|上午|下午|晚上|这几天|最近|正在|准备|打算|要去|去了|在弄|在忙|赶着|心里没底|好紧张|好累|压力好大)"
)

# English first person.  ``your`` is deliberately absent: a reply that asks
# "how was your interview?" is about the user, not a self claim.
_EN_SELF_RE = re.compile(
    r"\b(?:i|i'm|i've|i'd|i'll|my|me|myself|we|we're|we've|our)\b",
    re.IGNORECASE,
)

_EN_SELF_EVENT_VERB_RE = re.compile(
    r"\b(?:have|has|had|having|got|get|getting|gotten|"
    r"going\s+to|gonna|scheduled|booked|applied|interviewing|"
    r"recovering|preparing\s+for|studying\s+for|"
    r"started|starting|quit|quitting|left|leaving|moving|moved|"
    r"just|still|already|currently|"
    r"went|happened|finished|done|over)\b",
    re.IGNORECASE,
)

_EN_ASSERTIVE_CUE_RE = re.compile(
    r"\b(?:i|i'm|i've|i'd|i'll|my|me|we|we're|our|"
    r"today|tonight|tomorrow|yesterday|"
    r"this\s+(?:morning|afternoon|evening|week|weekend)|"
    r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"last\s+(?:night|week|month)|right\s+now|at\s+the\s+moment|"
    r"these\s+days|lately|recently)\b",
    re.IGNORECASE,
)

_CITIES = (
    "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆",
    "武汉", "西安", "长沙", "天津", "苏州", "郑州", "青岛", "厦门",
    "福州", "合肥", "南昌", "昆明", "贵阳", "南宁", "海口", "太原",
    "石家庄", "沈阳", "大连", "长春", "哈尔滨", "济南", "珠海", "佛山",
    "东莞",
)

_LOC_CLAIM = re.compile(
    r"我(?:现在|这会儿|今天|刚刚|正)?(?:在|回到了?|去了|搬到)([\u4e00-\u9fa5]{2,10})"
)
_SCHOOL_CLAIM = re.compile(
    r"我(?:是|在)([\u4e00-\u9fa5]{2,12}?(?:大学|学院))(?:的|读书|上学|念书)?"
)
_NAME_CLAIM = re.compile(
    r"我(?:叫|的名字是|真名叫)([\u4e00-\u9fa5]{2,5})"
)
_HOME_LOC = {"家", "家里", "宿舍", "寝室", "房间", "床上", "屋里", "被窝"}
_QUESTION_END = re.compile(r"[？?吗呢吧呀]\s*$")


@dataclasses.dataclass(slots=True)
class ConsistencyIssue:
    slot: str
    expected: str
    claimed: str
    reason: str


@dataclasses.dataclass(slots=True)
class GuardResult:
    cleaned_reply: str
    ungrounded_events: list[str]
    consistency_issues: list[ConsistencyIssue]
    needs_regeneration: bool
    repair_prompt: str


def _canonical_term(raw: str) -> str:
    low = raw.lower()
    if "offer" in low:
        return "offer"
    return raw


def _clauses(text: str) -> list[str]:
    result: list[str] = []
    for line in _CLAUSE_SPLIT_RE.split(str(text or "")):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in _SUBCLAUSE_SPLIT_RE.split(line) if p.strip()]
        result.extend(parts or [line])
    return result


def _is_mostly_ascii(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isascii()) / len(letters) > 0.8


def _term_is_event(term: str, clause: str) -> bool:
    """Reject ambiguous English terms used in their everyday sense."""
    gate = _AMBIGUOUS_EN.get(term.lower())
    if gate is None:
        return True
    return bool(gate.search(clause))


def _is_first_person_event_clause(clause: str, full_text: str = "") -> bool:
    if not _SENSITIVE_RE.search(clause):
        return False
    if _THIRD_PARTY_PREFIX_RE.search(clause):
        return False
    if full_text and _THIRD_PARTY_PREFIX_RE.search(full_text) and not (
        "我" in clause or _EN_SELF_RE.search(clause)
    ):
        return False
    if _NEGATED_OR_META_RE.search(clause):
        return False
    if _USER_DIRECTED_RE.search(clause) and not (
        "我" in clause or _EN_SELF_RE.search(clause)
    ):
        return False
    if "我" in clause or "咱" in clause or "自己" in clause:
        return True
    if _EN_SELF_RE.search(clause) and _EN_SELF_EVENT_VERB_RE.search(clause):
        return True
    if _SELF_EVENT_VERB_RE.search(clause):
        return True
    if full_text and (
        _ASSERTIVE_CUE_RE.search(full_text)
        or _EN_ASSERTIVE_CUE_RE.search(full_text)
    ):
        return True
    return False


def sensitive_claims(reply: str) -> list[str]:
    """Return sensitive event terms that ``reply`` claims in first person."""
    text = str(reply or "").strip()
    if not text or not _SENSITIVE_RE.search(text):
        return []
    found: list[str] = []
    clauses = _clauses(text)
    for clause in clauses:
        if not _is_first_person_event_clause(clause, full_text=text):
            continue
        for match in _SENSITIVE_RE.finditer(clause):
            if not _term_is_event(match.group(0), clause):
                continue
            term = _canonical_term(match.group(0))
            if term not in found:
                found.append(term)
    if found:
        return found
    for match in _SENSITIVE_RE.finditer(text):
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 16)
        window = text[start:end]
        if not _term_is_event(match.group(0), window):
            continue
        if _THIRD_PARTY_PREFIX_RE.search(window) or _NEGATED_OR_META_RE.search(window):
            continue
        if _USER_DIRECTED_RE.search(window) and not (
            "我" in window or _EN_SELF_RE.search(window)
        ):
            continue
        if (
            _ASSERTIVE_CUE_RE.search(window)
            or _SELF_EVENT_VERB_RE.search(window)
            or (_EN_SELF_RE.search(window) and _EN_SELF_EVENT_VERB_RE.search(window))
        ):
            term = _canonical_term(match.group(0))
            if term not in found:
                found.append(term)
    return found


def ungrounded_event_terms(reply: str, grounding_corpus: str) -> list[str]:
    """Return sensitive event terms claimed in ``reply`` missing from ``grounding_corpus``."""
    claimed = sensitive_claims(reply)
    if not claimed:
        return []
    corpus = str(grounding_corpus or "").lower()
    ungrounded: list[str] = []
    for term in claimed:
        probe = term.lower()
        if probe == "offer":
            if "offer" in corpus or "录用" in corpus:
                continue
        elif probe in corpus:
            continue
        ungrounded.append(term)
    return ungrounded


def strip_ungrounded_events(reply: str, ungrounded: list[str] | tuple[str, ...]) -> str:
    """Remove clauses containing ungrounded event terms from ``reply``."""
    text = str(reply or "").strip()
    if not text or not ungrounded:
        return text
    lowered_terms = [t.lower() for t in ungrounded if t]
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        subclauses = [p.strip() for p in re.split(r"[，,。！？!?；;]+", line) if p.strip()]
        kept_subs: list[str] = []
        for sub in subclauses:
            low = sub.lower()
            if any(term in low for term in lowered_terms):
                continue
            kept_subs.append(sub)
        if kept_subs:
            joiner = ", " if _is_mostly_ascii(line) else "，"
            kept_lines.append(joiner.join(kept_subs))
    return "\n".join(kept_lines).strip()


def filter_ungrounded_history(
    history: list[dict],
    grounding_corpus: str = "",
) -> list[dict]:
    """Filter or sanitize assistant turns in conversation history that contain
    ungrounded sensitive event claims so prior hallucinations do not self-reinforce.
    """
    if not history:
        return []
    user_blob = "\n".join(
        str(m.get("content") or "")
        for m in history
        if isinstance(m, dict) and m.get("role") == "user"
    )
    full_grounding = f"{grounding_corpus}\n{user_blob}"
    cleaned: list[dict] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            cleaned.append(msg)
            continue
        content = str(msg.get("content") or "")
        bad = ungrounded_event_terms(content, full_grounding)
        if not bad:
            cleaned.append(msg)
            continue
        stripped = strip_ungrounded_events(content, bad)
        if stripped:
            copy = dict(msg)
            copy["content"] = stripped
            cleaned.append(copy)
    return cleaned


def _slot_map(store: MemoryStore, conv_key: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in store.active_slots(conv_key):
        if str(row["orbit"] or "confirmed") == "voided":
            continue
        subj = str(row["subject"] or "user")
        slot = str(row["slot"] or "")
        val = normalize(str(row["value"] or ""))
        if slot and val:
            out[(subj, slot)] = val
    return out


def check_reply_consistency(
    store: MemoryStore,
    conv_key: str,
    reply: str,
    *,
    soul: SoulSpec = DEFAULT_SOUL,
) -> list[ConsistencyIssue]:
    text = normalize(reply)
    if not text or _QUESTION_END.search(text):
        return []
    slots = _slot_map(store, conv_key)
    issues: list[ConsistencyIssue] = []

    bot_city = slots.get(("bot", "city")) or ""
    bot_loc = (
        slots.get(("both", "location"))
        or slots.get(("bot", "bot.location"))
        or slots.get(("bot", "location"))
        or ""
    )
    for match in _LOC_CLAIM.finditer(text):
        claimed = match.group(1)
        for city in _CITIES:
            if city in claimed and bot_city and city not in bot_city and city not in bot_loc:
                issues.append(ConsistencyIssue(
                    slot="bot.city",
                    expected=bot_city,
                    claimed=city,
                    reason=f"回复自称在{city}，但当前所在城市/地点是{bot_city or bot_loc}",
                ))
        if (
            claimed in _HOME_LOC
            and bot_loc
            and bot_loc not in _HOME_LOC
            and not any(h in bot_loc for h in _HOME_LOC)
        ):
            if any(k in bot_loc for k in ("商场", "公园", "餐厅", "饭店", "电影院", "超市", "外面", "路上", "地铁", "高铁")):
                issues.append(ConsistencyIssue(
                    slot="bot.location",
                    expected=bot_loc,
                    claimed=claimed,
                    reason=f"回复自称在{claimed}，但当前场景地点是{bot_loc}",
                ))

    bot_school = slots.get(("bot", "school")) or ""
    if bot_school:
        for match in _SCHOOL_CLAIM.finditer(text):
            claimed = match.group(1)
            if claimed and claimed not in bot_school and bot_school not in claimed:
                issues.append(ConsistencyIssue(
                    slot="bot.school",
                    expected=bot_school,
                    claimed=claimed,
                    reason=f"回复自称在{claimed}，但设定学校是{bot_school}",
                ))

    bot_name = slots.get(("bot", "name")) or soul.bot_name
    for match in _NAME_CLAIM.finditer(text):
        claimed = match.group(1)
        if claimed and claimed not in soul.all_bot_names() and claimed != bot_name:
            issues.append(ConsistencyIssue(
                slot="bot.name",
                expected=bot_name,
                claimed=claimed,
                reason=f"回复自称叫{claimed}，与设定名字{bot_name}矛盾",
            ))

    return issues


def guard_outbound_reply(
    store: MemoryStore,
    conv_key: str,
    reply: str,
    *,
    user_text: str = "",
    workspace_text: str = "",
    soul: SoulSpec = DEFAULT_SOUL,
) -> GuardResult:
    """Run full outbound membrane verification on a candidate reply."""
    grounding = f"{user_text}\n{workspace_text}"
    ungrounded = ungrounded_event_terms(reply, grounding)
    consistency = check_reply_consistency(store, conv_key, reply, soul=soul)
    cleaned = strip_ungrounded_events(reply, ungrounded) if ungrounded else str(reply or "").strip()

    reasons: list[str] = []
    if ungrounded:
        joined = "、".join(ungrounded)
        reasons.append(
            f"不要编造「{joined}」等没有在对话或记忆里出现过的人生大事或行程"
        )
    for issue in consistency[:2]:
        reasons.append(f"{issue.reason}（应为：{issue.expected}）")

    repair_prompt = ""
    if reasons:
        repair_prompt = (
            "【系统修正提示：刚才的草稿有以下失真："
            + "；".join(reasons)
            + "。请严格依据当前场景和真实记忆自然重答，只接对方当下的话。】"
        )

    return GuardResult(
        cleaned_reply=cleaned,
        ungrounded_events=ungrounded,
        consistency_issues=consistency,
        needs_regeneration=bool(ungrounded or consistency),
        repair_prompt=repair_prompt,
    )
