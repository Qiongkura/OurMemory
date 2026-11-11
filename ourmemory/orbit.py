# -*- coding: utf-8 -*-
"""Three-State Orbit (三态引力轨道: confirmed / pending / voided) & Verbatim Gate.

Implements Row ④ of the architecture:
- ``confirmed`` (确认): durable identity, verified preferences, explicit
  commitments, and episodes.  Never auto-downgraded; only explicit user
  correction can supersede.
- ``pending`` (待定/挂账): short-term plans ("计划带她去公园散步"), one-off
  orders ("喝奶茶要冰的"), temporary sensations ("走久了脚疼"), and single-turn
  conversational remarks ("提出背对方").  Tagged with a deadline (``valid_to``);
  automatically promoted to ``confirmed`` if re-stated across turns, or retired
  to ``voided`` once completed or expired.
- ``voided`` (作废): retired, superseded, or unverified claims.  Excluded from
  injection and recall while retaining an audit trail.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .soul import DEFAULT_SOUL, SoulSpec
from .store import MemoryStore
from .text import normalize

SUBJECT_USER = "user"
SUBJECT_BOT = "bot"
SUBJECT_BOTH = "both"

ORBIT_CONFIRMED = "confirmed"
ORBIT_PENDING = "pending"
ORBIT_VOIDED = "voided"

ALLOWED_SLOTS = frozenset({
    "name", "nickname", "hometown", "city", "home", "school", "major",
    "grade", "work", "job", "age", "birthday", "height", "weight",
    "location", "activity", "mood", "with", "next",
    "bot.location", "bot.activity", "bot.mood", "bot.with", "bot.next",
    "user.location", "user.activity", "user.mood",
    "relationship", "boundary", "preference", "pet",
})

TRANSIENT_SLOTS = frozenset({
    "location", "activity", "mood", "with", "next",
    "bot.location", "bot.activity", "bot.mood", "bot.with", "bot.next",
    "user.location", "user.activity", "user.mood",
})

_CONFLICT_SLOTS = frozenset({
    "name", "hometown", "city", "home", "school", "major", "grade",
    "work", "job", "age", "birthday", "height", "weight", "pet",
    "relationship",
})

PREFERENCE_MARKER = re.compile(
    r"喜欢|爱|讨厌|不喜欢|偏爱|最爱|受不了|怕|接受|拒绝|偏好|口味|忌口|上瘾"
)

SPEECH_ACTS = frozenset({
    "statement", "correction", "promise", "plan", "preference", "boundary",
    "milestone",
})
NON_FACT_ACTS = frozenset({
    "question", "rhetorical", "joke", "hypothetical", "quote",
})

_QUESTION_END = re.compile(r"[?？]\s*$|(?:吗|呢|么|嘛|来着)\s*[?？]?\s*$")
_INTERROGATIVE = re.compile(
    r"什么|啥|哪(?:里|儿|个|些|所)?|谁|多少|几岁|几点|几天|怎么|为什么|"
    r"多久|是不是|有没有|要不要|能不能|会不会|可不可以|好不好|行不行|如何|干嘛"
)
_ASSERTIVE_NEGATION = re.compile(
    r"不是[^，,。！？!?]{1,12}是|并不是|从来不|从不|不再|没有在|不在|"
    r"记错了|不对[，,]|不是的"
)
_PLACEHOLDER_GARBAGE = re.compile(
    r"^(?:无|暂无|没有|none|null)(?:\s*待跟进[:：]\s*无)?$|^待跟进[:：]\s*无$",
    re.IGNORECASE,
)

# Short-term plans or ephemeral turn remarks that belong in ``pending`` (挂账)
# rather than polluting permanent semantic facts forever.
_SHORT_TERM_PLAN_RE = re.compile(
    r"(?:计划|准备|打算|提议|要去|带(?:她|他|bot|对方)?去)"
    r"[^，。；;]{0,10}(?:散步|逛逛|逛街|商场|公园|吃饭|看电影|电玩|超市|买)"
)
_EPHEMERAL_REMARK_RE = re.compile(
    r"走久了脚疼|脚疼|肚子疼|有点困|喝奶茶要|想喝她的|提出背|表示都听|"
    r"间接接吻|手挺暖|手上还湿"
)
_DURABLE_FACT_RE = re.compile(
    r"叫|名字|大学|学院|学生|身高|体重|老家|住在|在(?:广州|深圳|北京|上海|杭州|"
    r"成都|重庆|武汉|西安|南京)|喜欢|讨厌|约定|一直喜欢|不会做饭|会做饭|作息|"
    r"自称|接受|生日|纪念日|养了|宠物"
)


@dataclass
class MemoryItem:
    text: str
    subject: str = SUBJECT_USER
    slot: str = ""
    value: str = ""
    speech_act: str = "statement"
    source: str = "user_explicit"
    evidence_quote: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.85
    importance: float = 0.6
    valence: float = 0.0
    arousal: float = 0.0
    layer: str = "semantic"
    time_scope: str = "long"
    orbit: str = ""
    valid_to: float | None = None


@dataclass
class Rejected:
    item: MemoryItem
    reason: str


def _is_question(text: str) -> bool:
    s = normalize(text)
    if not s:
        return False
    if _ASSERTIVE_NEGATION.search(s) and not s.endswith(("?", "？", "吗")):
        return False
    if _QUESTION_END.search(s):
        return True
    return bool(_INTERROGATIVE.search(s))


def user_owns_value(user_text: str, value: str) -> bool:
    """True when ``value`` appears inside the user's own first-person clause."""
    value = normalize(value)
    if not value:
        return False
    for clause in re.split(r"[，,。；;！!？?\n]", normalize(user_text or "")):
        if re.search(r"我(?!们)[^，。；;！!？?\n]{0,8}" + re.escape(value), clause):
            return True
    return False


def classify_orbit(item: MemoryItem, *, now: float | None = None) -> tuple[str, float | None]:
    """Assign an item to ``confirmed`` or ``pending`` (with expiration timestamp)."""
    now = time.time() if now is None else float(now)
    if item.orbit in (ORBIT_CONFIRMED, ORBIT_PENDING, ORBIT_VOIDED):
        return item.orbit, item.valid_to
    text = normalize(item.text)
    if _PLACEHOLDER_GARBAGE.match(text.replace("\n", " ").strip()):
        return ORBIT_VOIDED, now
    if item.slot in TRANSIENT_SLOTS:
        ttl = 6 * 3600 if "location" in item.slot else 2 * 3600
        return ORBIT_PENDING, now + ttl
    if item.layer == "episodic":
        return ORBIT_CONFIRMED, None
    if item.time_scope in ("instant", "today"):
        return ORBIT_PENDING, now + 6 * 3600
    if item.slot and item.slot in ALLOWED_SLOTS:
        return ORBIT_CONFIRMED, None
    if item.layer == "prospective" and _SHORT_TERM_PLAN_RE.search(text):
        return ORBIT_PENDING, now + 6 * 3600
    if _EPHEMERAL_REMARK_RE.search(text) and not _DURABLE_FACT_RE.search(text):
        return ORBIT_PENDING, now + 6 * 3600
    if float(item.confidence or 0.0) < 0.72 and item.speech_act not in ("correction", "milestone"):
        return ORBIT_PENDING, now + 12 * 3600
    return ORBIT_CONFIRMED, None


def validate(
    item: MemoryItem,
    *,
    user_text: str = "",
    bot_text: str = "",
    is_master_private: bool = True,
    soul: SoulSpec = DEFAULT_SOUL,
) -> str:
    """Return ``""`` when valid, or a deterministic rejection reason code."""
    text = normalize(item.text)
    if not text or _PLACEHOLDER_GARBAGE.match(text.replace("\n", " ").strip()):
        return "empty_or_placeholder"
    if item.speech_act in NON_FACT_ACTS:
        return "not_a_fact"
    if item.speech_act not in SPEECH_ACTS:
        return "unknown_speech_act"
    if item.slot and item.slot not in ALLOWED_SLOTS:
        return "unknown_slot"
    if item.subject not in (SUBJECT_USER, SUBJECT_BOT, SUBJECT_BOTH) \
            and not item.subject.startswith("person:"):
        return "unknown_subject"

    if bot_text and item.evidence_quote and item.evidence_quote in bot_text \
            and item.evidence_quote not in (user_text or ""):
        if item.subject == SUBJECT_USER:
            return "bot_words_cannot_state_user_fact"

    if (item.subject == SUBJECT_BOT and item.source == "inferred"
            and item.slot in ("bot.location", "bot.activity", "bot.mood")
            and user_owns_value(user_text, item.value)
            and normalize(item.value) not in normalize(bot_text)):
        return "bot_state_from_user_words"

    if item.evidence_quote:
        haystack = user_text if item.subject != SUBJECT_BOT else (bot_text or user_text)
        if normalize(item.evidence_quote) not in normalize(haystack or ""):
            return "evidence_not_verbatim"

    probe = item.evidence_quote or (
        (bot_text or "") if item.subject == SUBJECT_BOT else (user_text or "")
    )
    if (probe and _is_question(probe)
            and not (item.source == "inferred" and item.slot in TRANSIENT_SLOTS)):
        return "question_is_not_a_fact"

    if item.slot == "nickname" and not re.search(
        r"叫我|称呼我|希望被称|以后叫|别叫|喜欢被叫|喜欢叫",
        user_text or item.text,
    ):
        return "nickname_needs_explicit_preference"

    return ""
