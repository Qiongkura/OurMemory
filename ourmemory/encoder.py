# -*- coding: utf-8 -*-
"""Fast-Path Rules & Slow-Path Structured Extraction (快慢双通道记忆编码)."""

from __future__ import annotations

import json
import re
import time

from .orbit import (
    PREFERENCE_MARKER, MemoryItem, SUBJECT_BOT, SUBJECT_BOTH, SUBJECT_USER,
    TRANSIENT_SLOTS, user_owns_value,
)
from .scene import write_items
from .soul import DEFAULT_SOUL, SoulSpec
from .text import normalize

EXTRACT_SYSTEM = """你是记忆抽取器。读下面几轮对话，只输出 JSON，不要解释。

判断规则：
1. 只抽取"陈述、纠正、承诺、计划、偏好、边界、里程碑"，问句、玩笑、假设、引用一律不抽。
2. subject 只能是 user / bot / both / person:<名字>。说话人是用户时，他的事写 user；她自己的事写 bot。
3. slot 只能从这些里选（不适用就留空字符串）：
   name nickname hometown city home school major grade work job age birthday
   height weight location activity mood with next
   bot.location bot.activity bot.with bot.next
   user.location user.activity user.mood
   relationship boundary preference pet
4. evidence_quote 必须逐字复制原话里的片段，不能改写。找不到逐字片段就不要输出这一条。
5. 用户明确纠正旧说法时 speech_act 写 correction。
6. 不确定的不要写。宁可少写，不要编。
7. bot_state_change / user_state_change / shared_state_change 只写"本轮结束时仍然成立"的状态：
   - 已经做完的动作不要写：洗完澡、收拾完、吃完饭、饭做好之后，当前状态是做完之后的状态，不是"洗澡/收拾/吃饭/做饭"。
   - 抱一下、亲一下、喂一口、递东西、摸头、擦手这类瞬间动作不要写进 activity。
   - 双方在一起做同一件事、或明确落在同一个地点时，写 shared_state_change，bot/user 的同名字段留空。
   - activity 写持续动作（吃饭、做饭、收拾、洗澡、休息、散步、逛街、上课、做实验），不要写"准备/打算/刚起床"。
   - bot_state_change 只能来自她自己的话。用户说"我去做饭"不能写进她的状态。

输出格式：
{"items":[{"text":"用户老家在湖南","subject":"user","slot":"hometown","value":"湖南",
"speech_act":"statement","evidence_quote":"我老家湖南的","confidence":0.95,
"importance":4,"time_scope":"long"}],
"bot_state_change":{"location":"","activity":"","mood":""},
"user_state_change":{"location":"","activity":"","mood":""},
"shared_state_change":{"location":"","activity":"","mood":""}}"""


def _canonical_boundary(value: str) -> str:
    name = re.sub(r"^(?:不要|不想被|别|不准|不许)(?:叫|喊)?", "", normalize(value))
    name = name.strip(" ，,。；;！!？?")
    return f"不想被叫{name}" if name else normalize(value)


_STATE_KEYS = ("location", "activity", "mood")
_FUTURE_FACT_SLOTS = frozenset({"work", "job", "school", "major", "grade"})
_FUTURE_TIME = re.compile(
    r"明年|后年|毕业后|毕业以后|以后|将来|未来|下个?月|下下个?月|"
    r"过年|暑假|寒假|下周|下一年"
)
_NEXT_FUTURE = re.compile(
    r"明天|后天|下周|下个?月|下下个?月|明年|后年|毕业后|以后|将来|未来|"
    r"下一年|过年|暑假|寒假"
)
_NEXT_SLOTS = frozenset({"next", "bot.next"})
_PAST_TIME = re.compile(
    r"去年|前年|小时候|上次|那次|那天|以前|曾经|之前|昨天|上周|上个月|那年"
)
_BARE_STATE_SLOTS = ("location", "activity", "mood", "with")
_PRESENT_TIME = re.compile(r"现在|目前|已经|正在|当前|如今")
_PLAN_ACTS = frozenset({"plan", "promise"})

_MOMENTARY_ACTIVITY = re.compile(
    r"亲|吻|抱|搂|牵|摸头|摸|递|喂|擦|拍拍|拍一|凑近|贴"
)
_MEAL_START = re.compile(r"(?<!我)(?:吃饭了|开饭了|来吃饭|饭好了|可以吃了)")
_MEAL_REPLY = re.compile(r"来了|吃|尝|饿")
_MEAL_DONE = re.compile(r"(?:饭|菜).{0,3}(?:做好|好了|熟了)|吃完饭|吃完了")
_FINAL_LOCATION = re.compile(
    r"(?:坐到|坐在|躺在|躺到|趴在|靠在|回到|到了|来到|走到|抱到|抱回|"
    r"放到|搬回|进到|走进|冲进|躲在|到)"
    r"(?P<loc>床上|沙发上|沙发|家里|家|宿舍|客厅|厨房|浴室|卫生间|餐桌|"
    r"椅子|商场|公园|学校|教室|实验室|食堂|图书馆|公司|医院|车站|机场|"
    r"广州|深圳|北京|上海|杭州|成都|重庆|武汉)"
)

_FUTURE_FACT_WORDS = {
    "work": ("工作", "上班", "就业"),
    "job": ("工作", "上班", "职业"),
    "school": ("大学", "学校", "读", "上学"),
    "major": ("专业",),
    "grade": ("大一", "大二", "大三", "大四", "研一", "研二", "研三", "年级"),
}


def _final_location_match(text: str):
    found = list(_FINAL_LOCATION.finditer(text or ""))
    return found[-1] if found else None


def _completed_activity(value: str, text: str) -> bool:
    value = normalize(value)
    if not value:
        return False
    if re.search(re.escape(value) + r"(?:完|完了|好了|好|结束)", text or ""):
        return True
    return value in ("做饭", "做菜") and bool(_MEAL_DONE.search(text or ""))


def _shared_value(value: str, soul: SoulSpec = DEFAULT_SOUL) -> str:
    cleaned = soul.shared_strip_pattern().sub("", normalize(value))
    return cleaned or normalize(value)


def _future_fact_slot(slot: str, text: str) -> bool:
    for word in _FUTURE_FACT_WORDS.get(slot, ()):
        pos = text.find(word)
        while pos >= 0:
            before = text[max(0, pos - 20):pos]
            if _FUTURE_TIME.search(before) and not _PRESENT_TIME.search(before):
                return True
            pos = text.find(word, pos + 1)
    return False


def build_extract_messages(history, *, core_card: str = "", limit: int = 6):
    lines = []
    for message in (history or [])[-limit:]:
        if not isinstance(message, dict):
            continue
        role = "他" if message.get("role") == "user" else "她"
        content = normalize(message.get("content") or "")
        if content:
            lines.append(f"{role}：{content[:400]}")
    if not lines:
        return []
    user_block = ""
    if core_card:
        user_block += "【已知的稳定记忆】\n" + core_card + "\n\n"
    user_block += "【最近对话】\n" + "\n".join(lines)
    user_block += "\n\n只输出 JSON。"
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user_block},
    ]


def _strip_fence(text: str) -> str:
    body = str(text or "").strip()
    body = re.sub(r"^```(?:json)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    return body.strip()


def parse_extraction(raw: str) -> dict:
    body = _strip_fence(raw)
    empty = {
        "items": [],
        "bot_state_change": {},
        "user_state_change": {},
        "shared_state_change": {},
    }
    if not body:
        return empty
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return empty
    try:
        data = json.loads(body[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    return {
        "items": [item for item in items if isinstance(item, dict)],
        "bot_state_change": data.get("bot_state_change")
        if isinstance(data.get("bot_state_change"), dict) else {},
        "user_state_change": data.get("user_state_change")
        if isinstance(data.get("user_state_change"), dict) else {},
        "shared_state_change": data.get("shared_state_change")
        if isinstance(data.get("shared_state_change"), dict) else {},
    }


def _to_item(raw: dict) -> MemoryItem | None:
    text = normalize(raw.get("text") or "")
    if not text:
        return None
    subject = str(raw.get("subject") or SUBJECT_USER)
    if subject not in (SUBJECT_USER, SUBJECT_BOT, SUBJECT_BOTH) \
            and not subject.startswith("person:"):
        subject = SUBJECT_USER
    try:
        confidence = float(raw.get("confidence") or 0.9)
    except (TypeError, ValueError):
        confidence = 0.9
    try:
        importance = float(raw.get("importance") or 3) / 5.0
    except (TypeError, ValueError):
        importance = 0.6
    slot = str(raw.get("slot") or "")
    if subject == SUBJECT_USER and slot in _BARE_STATE_SLOTS:
        slot = "user." + slot
    elif subject == SUBJECT_BOT and (slot in _BARE_STATE_SLOTS or slot == "next"):
        slot = "bot." + slot
    value = (
        _canonical_boundary(raw.get("value") or "")
        if slot == "boundary" else normalize(raw.get("value") or "")
    )
    speech_act = str(raw.get("speech_act") or "statement")
    layer = "semantic"
    if slot in TRANSIENT_SLOTS and _PAST_TIME.search(text):
        slot = ""
        layer = "episodic"
    elif slot in _FUTURE_FACT_SLOTS and (
        speech_act in _PLAN_ACTS or _future_fact_slot(slot, text)
    ):
        slot = ""
        layer = "prospective"
    elif slot in _NEXT_SLOTS and (
        speech_act in _PLAN_ACTS or _NEXT_FUTURE.search(text)
    ):
        slot = ""
        layer = "prospective"
    if slot == "preference" and not PREFERENCE_MARKER.search(text):
        slot = ""
    return MemoryItem(
        text=text,
        subject=subject,
        slot=slot,
        value=value,
        speech_act=speech_act,
        source="user_correction" if speech_act == "correction" else "user_explicit",
        evidence_quote=str(raw.get("evidence_quote") or ""),
        confidence=max(0.0, min(1.0, confidence)),
        importance=max(0.0, min(1.0, importance)),
        layer=layer,
        time_scope=str(raw.get("time_scope") or "long"),
    )


def state_items(
    extraction: dict,
    *,
    now: float | None = None,
    skip: set[tuple[str, str]] | None = None,
    user_text: str = "",
    bot_text: str = "",
    soul: SoulSpec = DEFAULT_SOUL,
) -> list[MemoryItem]:
    """Convert state-change blocks into validated end-of-turn state items."""
    now = time.time() if now is None else float(now)
    skip = skip or set()
    text_all = normalize(f"{user_text} {bot_text}")
    user_loc = _final_location_match(user_text)
    bot_loc = _final_location_match(bot_text)
    loc_match = user_loc or bot_loc
    final_loc = loc_match.group("loc") if loc_match else ""
    near = ""
    if loc_match is not None:
        source = user_text if user_loc is not None else bot_text
        near = normalize(source[max(0, loc_match.start() - 8):loc_match.start()])
    shared_loc_re = soul.shared_location_pattern()
    her_object_re = soul.her_object_pattern()
    joint_re = soul.joint_pattern()

    joint_final = bool(final_loc and shared_loc_re.search(near))
    her_final = bool(final_loc and her_object_re.search(user_text or ""))
    shared_block = extraction.get("shared_state_change") or {}
    shared_done: set[str] = set()
    loc_done: set[str] = set()
    out: list[MemoryItem] = []

    def add(subject: str, field: str, value: str, *, label: str) -> None:
        if subject == SUBJECT_BOTH:
            slot = field
            layer = "relation"
            who = soul.shared_label
        else:
            prefix = "bot." if subject == SUBJECT_BOT else "user."
            slot = prefix + field
            layer = "self" if subject == SUBJECT_BOT else "semantic"
            who = "她" if subject == SUBJECT_BOT else "他"
        out.append(MemoryItem(
            text=f"{who}{label}：{value}",
            subject=subject,
            slot=slot,
            value=value,
            speech_act="statement",
            source="inferred",
            confidence=0.8,
            importance=0.6,
            layer=layer,
            time_scope="instant",
        ))

    labels = {"location": "地点", "activity": "在做", "mood": "心情"}
    for field in _STATE_KEYS:
        if ("both", field) in skip:
            continue
        value = normalize(shared_block.get(field) or "")
        if field == "location" and joint_final and final_loc:
            value = final_loc
        if field == "activity":
            if _MOMENTARY_ACTIVITY.search(value):
                value = ""
            elif value and _completed_activity(value, text_all):
                value = ""
        if not value:
            continue
        add(SUBJECT_BOTH, field, value, label=labels[field])
        shared_done.add(field)
        if field == "location":
            loc_done.add("both")

    if "location" not in shared_done and joint_final:
        add(SUBJECT_BOTH, "location", final_loc, label="地点")
        shared_done.add("location")
        loc_done.add("both")
    if ("activity" not in shared_done
            and _MEAL_START.search(normalize(user_text))
            and _MEAL_REPLY.search(normalize(bot_text))):
        add(SUBJECT_BOTH, "activity", "吃饭", label="在做")
        shared_done.add("activity")

    if final_loc and not joint_final:
        if her_final and "location" not in shared_done:
            add(SUBJECT_BOT, "location", final_loc, label="地点")
            loc_done.add(SUBJECT_BOT)
        elif final_loc in normalize(bot_text):
            add(SUBJECT_BOT, "location", final_loc, label="地点")
            loc_done.add(SUBJECT_BOT)
        elif final_loc in normalize(user_text) and not her_object_re.search(normalize(user_text)):
            add(SUBJECT_USER, "location", final_loc, label="地点")
            loc_done.add(SUBJECT_USER)

    for key, subject in (
        ("bot_state_change", SUBJECT_BOT),
        ("user_state_change", SUBJECT_USER),
    ):
        block = extraction.get(key) or {}
        own_text = bot_text if subject == SUBJECT_BOT else user_text
        for field in _STATE_KEYS:
            if (subject, ("bot." if subject == SUBJECT_BOT else "user.") + field) in skip:
                continue
            if field in shared_done:
                continue
            if field == "location" and subject in loc_done:
                continue
            value = normalize(block.get(field) or "")
            if field == "location" and final_loc:
                if joint_final:
                    continue
                if her_final and subject == SUBJECT_USER:
                    pass
                elif final_loc in normalize(own_text):
                    value = final_loc
            if field == "activity":
                if _MOMENTARY_ACTIVITY.search(value):
                    continue
                if value and _completed_activity(value, text_all):
                    continue
                if subject == SUBJECT_USER and joint_re.search(value):
                    shared_val = _shared_value(value, soul=soul)
                    if shared_val and _completed_activity(shared_val, text_all):
                        continue
                    add(SUBJECT_BOTH, "activity", shared_val, label="在做")
                    shared_done.add("activity")
                    continue
                if (subject == SUBJECT_BOT and value
                        and user_owns_value(user_text, value)
                        and value not in normalize(bot_text)):
                    continue
            if not value:
                continue
            add(subject, field, value, label=labels[field])
    return out


# ---------------- Fast Path ----------------
_FAST_RULES = (
    ("boundary", re.compile(r"(?:别|不要|不准|不许|不想被)(?:再)?(?:叫|喊)我([^\s，,。；;！!？?]{1,12})")),
    ("nickname", re.compile(r"(?:以后|就)叫我([^\s，,。；;！!？?]{1,12})")),
    ("nickname", re.compile(r"我(?:很|也|特别|挺)?喜欢被叫(?:做)?[“\"']?([^\s，,。；;！!？?\"”']{1,6})")),
    ("hometown", re.compile(r"我老家(?:是|在)([^\s，,。；;！!？?]{1,12})")),
    ("height", re.compile(r"我身高\s*(?:是|有)?\s*(\d{3})")),
    ("weight", re.compile(r"我体重\s*(?:是|有)?\s*(\d{2,3})")),
    ("grade", re.compile(r"我(?:已经|现在)?(?:是)?(大一|大二|大三|大四|研一|研二|研三|博一|博二|博三)")),
)

_NOT_NICKNAME = frozenset({"醒", "起来", "起床", "走", "回去", "停"})

_PET_RULE = re.compile(
    r"我(?:养|有)了?(?:一|1)?[只条]?"
    r"(猫|狗|仓鼠|兔子|鸟|鱼|乌龟)"
    r"(?:[，,、\s]*(?:名字)?叫([^\s，,。；;！!？?]{1,6}))"
)

_INTERROGATIVE_VALUE = re.compile(
    r"^(?:哪(?:里|儿|个|些|所)?|什么|啥|谁|多少|几(?:岁|点|天|个|号|年级)?|"
    r"怎么|为什么|如何|多久|干嘛|干啥|什么样|怎么样)"
)

_CORRECTION_RULES = (
    ("height", re.compile(
        r"我(?:身高)?(?:不是|没有)\s*\d{3}[^\d]{0,6}?(?:我)?(?:身高)?(?:是|有)?\s*(\d{3})"
    )),
    ("weight", re.compile(
        r"我(?:体重)?(?:不是|没有)\s*\d{2,3}[^\d]{0,6}?(?:我)?(?:体重)?(?:是|有)?\s*(\d{2,3})"
    )),
    ("hometown", re.compile(
        r"我(?:老家|家乡)(?:不是|不在)\s*[^\s，,。；;！!？?]{1,8}"
        r"[^\u4e00-\u9fff]{0,4}(?:是|在)\s*([^\s，,。；;！!？?]{1,12})"
    )),
    ("grade", re.compile(
        r"我(?:不是|不再是)\s*(?:大一|大二|大三|大四|研一|研二|研三|博一|博二|博三)"
        r"[^\u4e00-\u9fff]{0,6}(?:我)?(?:已经)?(?:是)?"
        r"(大一|大二|大三|大四|研一|研二|研三|博一|博二|博三)"
    )),
)

_LOCATION_CORRECTION = re.compile(
    r"我(?:现在)?(?:不在|没在)\s*[^\s，,。；;！!？?]{1,8}"
    r"[^\u4e00-\u9fff]{0,4}(?:我)?(?:现在)?在\s*([^\s，,。；;！!？?]{1,8})"
)
_NICKNAME_CORRECTION = re.compile(
    r"(?:别|不要|不准|不许)(?:再)?(?:叫|喊)我[^\s，,。；;！!？?]{1,12}"
    r"[^\u4e00-\u9fff]{0,6}(?:叫我|喊我)([^\s，,。；;！!？?]{1,12})"
)

_LABELS = {
    "hometown": "老家在",
    "height": "身高",
    "weight": "体重",
    "grade": "是",
    "location": "现在在",
}


def _is_fact_value(value: str) -> bool:
    if not value or _INTERROGATIVE_VALUE.search(value):
        return False
    if value.endswith(("吗", "呢", "么", "嘛")):
        return False
    return True


def _followed_by_question(text: str, match) -> bool:
    tail = text[match.end():match.end() + 3]
    tail = tail[1:] if tail.startswith(("了", "的")) else tail
    return tail.startswith(("吗", "呢", "么", "嘛", "没", "不", "？", "?"))


def _slot_item(slot: str, value: str, quote: str, *,
               confidence: float = 0.97) -> MemoryItem:
    label = _LABELS[slot]
    return MemoryItem(
        text=f"用户{label}{value}",
        subject=SUBJECT_USER,
        slot=slot,
        value=value,
        speech_act="statement",
        source="user_explicit",
        evidence_quote=quote,
        confidence=confidence,
        importance=0.85,
    )


def _plausible(slot: str, value: str, text: str) -> bool:
    if slot not in ("height", "weight"):
        return True
    try:
        number = int(re.sub(r"\D", "", value) or 0)
    except ValueError:
        return False
    if slot == "height":
        return 120 <= number <= 230
    if "体重" in text:
        return 30 <= number <= 300
    return 30 <= number <= 119


def fast_path(user_text: str, *, now: float | None = None) -> list[MemoryItem]:
    text = normalize(user_text)
    if not text:
        return []
    out: list[MemoryItem] = []
    for slot, pattern in _FAST_RULES:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            continue
        if slot != "boundary" and (
            not _is_fact_value(value) or _followed_by_question(text, match)
        ):
            continue
        if slot == "boundary":
            canonical = _canonical_boundary(f"别叫{value}")
            out.append(MemoryItem(
                text=f"用户{canonical}",
                subject=SUBJECT_USER,
                slot="boundary",
                value=canonical,
                speech_act="boundary",
                source="user_explicit",
                evidence_quote=match.group(0),
                confidence=0.98,
                importance=1.0,
            ))
            continue
        if slot == "nickname":
            if value in _NOT_NICKNAME:
                continue
            out.append(MemoryItem(
                text=f"用户希望被叫{value}",
                subject=SUBJECT_USER,
                slot="nickname",
                value=value,
                speech_act="preference",
                source="user_explicit",
                evidence_quote=match.group(0),
                confidence=0.98,
                importance=0.9,
            ))
            continue
        out.append(_slot_item(slot, value, match.group(0)))

    match = _PET_RULE.search(text)
    if match and not _followed_by_question(text, match):
        kind, name = match.group(1), match.group(2).strip()
        if _is_fact_value(name):
            out.append(MemoryItem(
                text=f"用户养了{kind}叫{name}",
                subject=SUBJECT_USER,
                slot="pet",
                value=f"{kind}叫{name}",
                speech_act="statement",
                source="user_explicit",
                evidence_quote=match.group(0),
                confidence=0.95,
                importance=0.8,
            ))

    corrected = {slot for slot, pattern in _CORRECTION_RULES if pattern.search(text)}
    out = [item for item in out if item.slot not in corrected]
    for slot, pattern in _CORRECTION_RULES:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        if not _is_fact_value(value) or not _plausible(slot, value, text):
            continue
        item = _slot_item(slot, value, match.group(0), confidence=0.98)
        item.speech_act = "correction"
        item.source = "user_correction"
        out.append(item)

    match = _LOCATION_CORRECTION.search(text)
    if match:
        value = match.group(1).strip()
        if _is_fact_value(value):
            out.append(MemoryItem(
                text=f"用户现在在{value}",
                subject=SUBJECT_USER,
                slot="user.location",
                value=value,
                speech_act="correction",
                source="user_correction",
                evidence_quote=match.group(0),
                confidence=0.96,
                importance=0.8,
            ))

    match = _NICKNAME_CORRECTION.search(text)
    if match:
        value = match.group(1).strip()
        if _is_fact_value(value) and value not in _NOT_NICKNAME:
            out.append(MemoryItem(
                text=f"用户希望被叫{value}",
                subject=SUBJECT_USER,
                slot="nickname",
                value=value,
                speech_act="correction",
                source="user_correction",
                evidence_quote=match.group(0),
                confidence=0.98,
                importance=0.95,
            ))
    return out


def apply_fast_path(
    store,
    conv_key: str,
    user_text: str,
    *,
    is_master_private: bool = True,
    soul: SoulSpec = DEFAULT_SOUL,
    now: float | None = None,
) -> list[str]:
    items = fast_path(user_text, now=now)
    if not items:
        return []
    written, _ = write_items(
        store, conv_key, items, user_text=user_text,
        is_master_private=is_master_private, soul=soul, now=now,
    )
    return written


def encode_turn(
    store,
    conv_key: str,
    history,
    *,
    user_text: str = "",
    bot_text: str = "",
    core_card: str = "",
    is_master_private: bool = True,
    soul: SoulSpec = DEFAULT_SOUL,
    now: float | None = None,
    extractor=None,
) -> dict:
    now = time.time() if now is None else float(now)
    messages = build_extract_messages(history, core_card=core_card)
    if not messages or extractor is None:
        return {"status": "skipped", "written": 0, "rejected": 0, "usage": {}}
    try:
        raw = extractor(messages)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error": type(exc).__name__,
            "written": 0,
            "rejected": 0,
            "usage": {},
        }
    usage = dict(getattr(raw, "usage", None) or {})
    extraction = parse_extraction(raw)
    items = [
        item for item in (_to_item(raw_item) for raw_item in extraction["items"])
        if item is not None
    ]
    covered = {(item.subject, item.slot) for item in items if item.slot}
    items.extend(
        state_items(
            extraction, now=now, skip=covered,
            user_text=user_text, bot_text=bot_text, soul=soul,
        )
    )
    if not items:
        return {"status": "empty", "written": 0, "rejected": 0, "usage": usage}
    written, rejected = write_items(
        store, conv_key, items, user_text=user_text, bot_text=bot_text,
        is_master_private=is_master_private, soul=soul, now=now,
    )
    return {
        "status": "ok",
        "written": len(written),
        "rejected": len(rejected),
        "reasons": sorted({item.reason for item in rejected}),
        "usage": usage,
    }
