# -*- coding: utf-8 -*-
"""Dual-Subject & Shared Scene State Machine (双人同场状态机).

Tracks ``user``, ``bot``, and ``both`` (shared) states (`location`, `activity`,
`mood`, `next`), mirrors shared scene transitions onto both participants,
retires completed actions and short-term plans (`prospective`/`pending`), and
maintains the Three-State Orbit (`confirmed`, `pending`, `voided`).
"""

from __future__ import annotations

import json
import re
import time

from .orbit import (
     _CONFLICT_SLOTS, _DURABLE_FACT_RE, _EPHEMERAL_REMARK_RE,
    _PLACEHOLDER_GARBAGE, _SHORT_TERM_PLAN_RE, MemoryItem, ORBIT_CONFIRMED,
    ORBIT_PENDING, ORBIT_VOIDED, PREFERENCE_MARKER, Rejected, SUBJECT_BOT,
    SUBJECT_BOTH, SUBJECT_USER, TRANSIENT_SLOTS, classify_orbit, validate,
)
from .soul import DEFAULT_SOUL, SoulSpec
from .store import MemoryStore
from .text import normalize

_SHARED_STATE_SLOTS = frozenset({"location", "activity", "mood", "next"})
_STATE_CONCEPT_SLOTS = {
    ("bot", "location"): "bot.location",
    ("bot", "activity"): "bot.activity",
    ("bot", "mood"): "bot.mood",
    ("bot", "next"): "bot.next",
    ("user", "location"): "user.location",
    ("user", "activity"): "user.activity",
    ("user", "mood"): "user.mood",
    ("user", "next"): "next",
    ("both", "location"): "location",
    ("both", "activity"): "activity",
    ("both", "mood"): "mood",
    ("both", "next"): "next",
}

_SUB_LOCATIONS = frozenset({
    "床上", "沙发上", "沙发", "客厅", "厨房", "浴室", "卫生间",
    "餐桌", "椅子上",
})
_COOK_ACTS = ("做饭", "做菜", "煮饭", "下厨")
_CLEAN_ACTS = ("收拾", "洗碗", "收拾碗筷", "打扫")
_BATH_ACTS = ("洗澡", "洗漱")
_MEAL_ACTS = ("吃饭", "吃早餐", "吃午饭", "吃晚饭", "吃宵夜")
_REST_ACTS = ("休息", "睡觉", "眯", "睡")

_OUTING_KEYWORDS = ("商场", "公园", "散步", "逛街", "逛逛", "电玩", "面馆", "超市", "电影")
_HOME_KEYWORDS = ("到家", "回家", "回到家", "在家", "沙发", "床上", "厨房", "做饭", "洗碗")


def _slot_for(subject: str, concept: str) -> str:
    return _STATE_CONCEPT_SLOTS.get((subject, concept), concept)


def _clear_slot(store: MemoryStore, conv_key: str, subject: str, slot: str,
                now: float) -> None:
    current = store.get_slot(conv_key, subject, slot)
    if current is None:
        return
    with store.db.tx():
        store.db.conn.execute(
            "UPDATE memories SET status='superseded', valid_to=? WHERE id=?",
            (now, str(current["id"])),
        )
        store.db.record_audit(
            conv_key,
            "clear_state_slot",
            str(current["id"]),
            subject=subject,
            slot=slot,
            value=str(current["value"] or "")[:60],
            orbit="voided",
            reason="scene_transition",
            ts=now,
        )


def _activity_kind(value: str) -> str:
    value = normalize(value)
    if any(word in value for word in _COOK_ACTS):
        return "cook"
    if any(word in value for word in _CLEAN_ACTS):
        return "clean"
    if any(word in value for word in _BATH_ACTS):
        return "bath"
    if any(word in value for word in _MEAL_ACTS):
        return "meal"
    if any(word in value for word in _REST_ACTS):
        return "rest"
    return "other"


def _plan_started(next_value: str, activity: str, text: str) -> bool:
    planned = _activity_kind(next_value)
    current = _activity_kind(activity)
    text = normalize(text)
    if planned == "cook":
        return current in ("meal", "clean") or bool(
            re.search(r"吃饭了|开饭了|饭好了|做好了.*吃", text)
        )
    if planned == "bath":
        return current == "bath" or bool(re.search(r"洗完澡|洗好了", text))
    if planned == "clean":
        return current == "clean"
    if planned == "meal":
        return current == "meal"
    return False


def _activity_finished(value: str, text: str) -> bool:
    value = normalize(value)
    text = normalize(text)
    if not value:
        return False
    if re.search(re.escape(value) + r"(?:完|完了|好了|好|结束)", text):
        return True
    kind = _activity_kind(value)
    if kind == "bath":
        return bool(re.search(r"洗完澡|洗好了|洗完了", text))
    if kind == "meal":
        return bool(re.search(r"吃完了|吃好了|吃完饭|收拾|洗碗", text))
    if kind == "cook":
        return bool(
            re.search(r"饭做好了|菜做好了|做好了|熟了|吃饭了|开饭了|收拾|洗碗", text)
        )
    return False


def _retire_completed_plans(store: MemoryStore, conv_key: str, text_all: str,
                            now: float) -> int:
    """Void short-term prospective plans once their outing/action completes or returns home."""
    returned_home = any(word in text_all for word in _HOME_KEYWORDS)
    arrived_or_done = bool(re.search(r"到了|到[\u4e00-\u9fa5]{1,6}啦|逛完|散完|买完|吃完|看完", text_all))
    retired = 0

    # Clear active `next`/`bot.next` slots pointing to an outing once reached or back home
    for subj, nslot in ((SUBJECT_BOTH, "next"), (SUBJECT_USER, "next"), (SUBJECT_BOT, "bot.next")):
        nrow = store.get_slot(conv_key, subj, nslot)
        if nrow is None:
            continue
        nval = normalize(str(nrow["value"] or nrow["text"] or ""))
        matched = [w for w in _OUTING_KEYWORDS if w in nval]
        if matched and (returned_home or (arrived_or_done and any(w in text_all for w in matched))):
            _clear_slot(store, conv_key, subj, nslot, now)
            retired += 1

    rows = store.db.query(
        "SELECT id, text, ts FROM memories WHERE conv_key=? AND status='active' "
        "AND layer='prospective' AND (slot IS NULL OR slot='')",
        (conv_key,),
    )
    if not rows:
        return retired
    for row in rows:
        plan_text = normalize(str(row["text"] or ""))
        if not _SHORT_TERM_PLAN_RE.search(plan_text):
            continue
        matched_outing = [w for w in _OUTING_KEYWORDS if w in plan_text]
        completed = False
        if matched_outing and any(w in text_all for w in matched_outing) and arrived_or_done:
            completed = True
        if returned_home and float(row["ts"] or 0) <= now:
            completed = True
        if completed:
            store.void_memory(str(row["id"]), now=now, reason="short_term_plan_completed")
            retired += 1
    return retired


def apply_state_transitions(
    store: MemoryStore,
    conv_key: str,
    items: list[MemoryItem],
    *,
    user_text: str,
    bot_text: str,
    now: float,
) -> None:
    """Event-driven state transition engine."""
    text_all = normalize(f"{user_text} {bot_text}")
    activities: dict[str, str] = {}
    locations: set[str] = set()
    for item in items:
        if not item.slot or item.slot not in TRANSIENT_SLOTS:
            continue
        if item.subject == SUBJECT_BOTH:
            if item.slot == "location":
                locations.add("both")
            continue
        concept = item.slot.split(".", 1)[-1]
        if concept == "activity":
            activities[item.subject] = normalize(item.value or item.text)
        elif concept == "location":
            locations.add(item.subject)

    for subject in (SUBJECT_BOT, SUBJECT_USER):
        slot = _slot_for(subject, "activity")
        current = store.get_slot(conv_key, subject, slot)
        new_activity = activities.get(subject)
        if current is not None and not new_activity:
            if _activity_finished(str(current["value"] or ""), text_all):
                _clear_slot(store, conv_key, subject, slot, now)
                current = None
        if new_activity:
            if current is not None:
                if _activity_finished(str(current["value"] or ""), text_all):
                    _clear_slot(store, conv_key, subject, slot, now)
                current = None
            next_slot = _slot_for(subject, "next")
            next_row = store.get_slot(conv_key, subject, next_slot)
            if (next_row is not None
                    and _plan_started(str(next_row["value"] or ""), new_activity, text_all)):
                _clear_slot(store, conv_key, subject, next_slot, now)
            kind = _activity_kind(new_activity)
            loc_slot = _slot_for(subject, "location")
            loc_row = store.get_slot(conv_key, subject, loc_slot)
            if (loc_row is not None
                    and normalize(str(loc_row["value"] or "")) in _SUB_LOCATIONS
                    and kind in ("cook", "clean", "bath", "meal")):
                store.set_slot(
                    conv_key, subject, loc_slot, "家", text="状态联动：家",
                    source="inferred", orbit=ORBIT_PENDING,
                    valid_to=now + 12 * 3600, ts=now, confidence=0.6, importance=0.5,
                )

    shared = next(
        (item for item in items if item.subject == SUBJECT_BOTH and item.slot == "activity"),
        None,
    )
    shared_current = store.get_slot(conv_key, SUBJECT_BOTH, "activity")
    if (shared_current is not None and shared is None
            and (_activity_finished(str(shared_current["value"] or ""), text_all)
                 or any(value != normalize(str(shared_current["value"] or ""))
                        for value in activities.values()))):
        _clear_slot(store, conv_key, SUBJECT_BOTH, "activity", now)
    if shared is not None:
        value = normalize(shared.value or shared.text)
        kind = _activity_kind(value)
        for subject in (SUBJECT_BOT, SUBJECT_USER):
            next_slot = _slot_for(subject, "next")
            next_row = store.get_slot(conv_key, subject, next_slot)
            if (next_row is not None
                    and _plan_started(str(next_row["value"] or ""), value, text_all)):
                _clear_slot(store, conv_key, subject, next_slot, now)
            loc_slot = _slot_for(subject, "location")
            loc_row = store.get_slot(conv_key, subject, loc_slot)
            if (loc_row is not None
                    and normalize(str(loc_row["value"] or "")) in _SUB_LOCATIONS
                    and subject not in locations
                    and kind in ("cook", "clean", "bath", "meal")):
                store.set_slot(
                    conv_key, subject, loc_slot, "家", text="状态联动：家",
                    source="inferred", orbit=ORBIT_PENDING,
                    valid_to=now + 12 * 3600, ts=now, confidence=0.6, importance=0.5,
                )
            shared_loc = store.get_slot(conv_key, SUBJECT_BOTH, "location")
            if (shared_loc is not None
                    and normalize(str(shared_loc["value"] or "")) in _SUB_LOCATIONS
                    and "both" not in locations
                    and kind in ("cook", "clean", "bath", "meal")):
                store.set_slot(
                    conv_key, SUBJECT_BOTH, "location", "家", text="状态联动：家",
                    source="inferred", orbit=ORBIT_PENDING,
                    valid_to=now + 12 * 3600, ts=now, confidence=0.6, importance=0.5,
                )

    _retire_completed_plans(store, conv_key, text_all, now)


def _mirror_shared_state(store: MemoryStore, conv_key: str, item: MemoryItem,
                         now: float) -> None:
    if (item.subject != SUBJECT_BOTH or item.slot not in _SHARED_STATE_SLOTS
            or not normalize(item.value or "")):
        return
    concept = item.slot
    orbit, valid_to = classify_orbit(item, now=now)
    for subject in (SUBJECT_BOT, SUBJECT_USER):
        store.set_slot(
            conv_key, subject, _slot_for(subject, concept),
            item.value or item.text, text=item.text, layer=item.layer,
            speech_act=item.speech_act, source="inferred", orbit=orbit,
            valid_to=valid_to, confidence=item.confidence,
            importance=item.importance, ts=now,
        )


def _keep_previous_preference(store: MemoryStore, conv_key: str, current,
                              now: float) -> None:
    if not current:
        return
    text = normalize(str(current["text"] or ""))
    if not text:
        return
    exists = store.db.query_one(
        "SELECT id FROM memories WHERE conv_key=? AND status='active' "
        "AND (slot IS NULL OR slot='') AND text=? LIMIT 1",
        (conv_key, text),
    )
    if exists is not None:
        return
    memory_id, _ = store.add(
        "semantic", text, summary=text[:80], ts=now,
        source=str(current["source"] or "user_explicit"), conv_key=conv_key,
        importance=float(current["importance"] or 0.6),
        confidence=float(current["confidence"] or 0.6),
        orbit=ORBIT_CONFIRMED,
    )
    store.db.conn.execute(
        "UPDATE memories SET subject=?, layer=?, status='active', "
        "orbit='confirmed', valid_from=? WHERE id=?",
        (str(current["subject"] or SUBJECT_USER),
         str(current["layer"] or "semantic"), now, memory_id),
    )


def write_items(
    store: MemoryStore,
    conv_key: str,
    items: list[MemoryItem],
    *,
    user_text: str = "",
    bot_text: str = "",
    is_master_private: bool = True,
    soul: SoulSpec = DEFAULT_SOUL,
    now: float | None = None,
) -> tuple[list[str], list[Rejected]]:
    """Validate, classify Three-State Orbit, and write memory items."""
    now = time.time() if now is None else float(now)
    store.expire_pending(conv_key, now=now)
    written: list[str] = []
    rejected: list[Rejected] = []
    accepted: list[MemoryItem] = []

    for item in items:
        reason = validate(
            item, user_text=user_text, bot_text=bot_text,
            is_master_private=is_master_private, soul=soul,
        )
        if reason:
            rejected.append(Rejected(item, reason))
            continue

        orbit, valid_to = classify_orbit(item, now=now)
        if orbit == ORBIT_VOIDED:
            rejected.append(Rejected(item, "voided_orbit"))
            continue

        if item.slot:
            corrected = (
                item.speech_act == "correction" or item.source == "user_correction"
            )
            current = store.get_slot(conv_key, item.subject, item.slot)
            value = normalize(item.value or item.text)
            conflicts = (
                current is not None
                and item.subject == SUBJECT_BOT
                and item.slot in _CONFLICT_SLOTS
                and not corrected
                and item.speech_act not in ("boundary",)
                and normalize(str(current["value"] or "")) != value
            )
            if corrected:
                with store.db.tx():
                    store.db.conn.execute(
                        "UPDATE memories SET status='superseded', valid_to=? "
                        "WHERE conv_key=? AND subject=? AND slot=? AND status='conflicted'",
                        (now, conv_key, item.subject, item.slot),
                    )
            if conflicts:
                duplicate = store.db.query_one(
                    "SELECT id FROM memories WHERE conv_key=? AND subject=? "
                    "AND slot=? AND status='conflicted' AND value=? LIMIT 1",
                    (conv_key, item.subject, item.slot, value),
                )
                if duplicate is not None:
                    continue
                memory_id, _ = store.add(
                    "semantic", item.text, summary=item.text[:80], ts=now,
                    source=item.source, conv_key=conv_key,
                    importance=item.importance, confidence=item.confidence,
                    orbit=ORBIT_PENDING,
                    meta={"speech_act": item.speech_act,
                          "evidence_ids": item.evidence_ids,
                          "slot": item.slot, "value": value},
                )
                store.db.conn.execute(
                    "UPDATE memories SET subject=?, slot=?, value=?, speech_act=?, "
                    "evidence_ids=?, layer=?, status='conflicted', orbit='pending', "
                    "valid_from=? WHERE id=?",
                    (item.subject, item.slot, value, item.speech_act,
                     json.dumps(item.evidence_ids, ensure_ascii=False),
                     item.layer, now, memory_id),
                )
            else:
                if (item.slot == "preference" and current is not None
                        and normalize(str(current["value"] or "")) != value):
                    _keep_previous_preference(store, conv_key, current, now)
                memory_id, _ = store.set_slot(
                    conv_key, item.subject, item.slot,
                    item.value or item.text,
                    text=item.text, layer=item.layer,
                    speech_act=item.speech_act,
                    source=item.source, orbit=orbit, valid_to=valid_to,
                    evidence_ids=item.evidence_ids,
                    confidence=item.confidence, importance=item.importance,
                    ts=now,
                )
                _mirror_shared_state(store, conv_key, item, now)
                accepted.append(item)
        else:
            existing = store.db.query_one(
                "SELECT id, orbit, ts FROM memories WHERE conv_key=? AND status='active' "
                "AND (slot IS NULL OR slot='') AND text=? LIMIT 1",
                (conv_key, normalize(item.text)),
            )
            if existing is not None:
                # Three-State Orbit rule: a ``pending`` memory re-stated in a
                # later turn is promoted to ``confirmed``!
                if (str(existing["orbit"] or "") == ORBIT_PENDING
                        and now - float(existing["ts"] or 0) >= 60):
                    store.promote_orbit(
                        str(existing["id"]), now=now, reason="repeated_across_turns"
                    )
                continue
            memory_id, _ = store.add(
                "semantic", item.text, summary=item.text[:80], ts=now,
                valid_to=valid_to, source=item.source, conv_key=conv_key,
                importance=item.importance, confidence=item.confidence,
                valence=item.valence, arousal=item.arousal, orbit=orbit,
                meta={"speech_act": item.speech_act, "evidence_ids": item.evidence_ids},
            )
            store.db.conn.execute(
                "UPDATE memories SET subject=?, speech_act=?, evidence_ids=?, "
                "layer=?, status='active', orbit=?, valid_from=? WHERE id=?",
                (item.subject, item.speech_act,
                 json.dumps(item.evidence_ids, ensure_ascii=False),
                 item.layer, orbit, now, memory_id),
            )
            store.db.record_audit(
                conv_key,
                "add_memory",
                memory_id,
                subject=item.subject,
                slot="",
                value=normalize(item.text)[:60],
                orbit=orbit,
                reason=item.layer,
                ts=now,
            )
            accepted.append(item)
        written.append(memory_id)

    apply_state_transitions(
        store, conv_key, accepted, user_text=user_text, bot_text=bot_text, now=now
    )
    return written, rejected


def repair_store(
    store: MemoryStore,
    conv_key: str,
    *,
    soul: SoulSpec = DEFAULT_SOUL,
    now: float | None = None,
) -> dict[str, int]:
    """Sanitize a store by applying Three-State Orbit and subject rules to existing rows."""
    now = time.time() if now is None else float(now)
    stats = {
        "voided_garbage": 0,
        "retired_plans": 0,
        "demoted_ephemeral": 0,
        "repaired_subjects": 0,
        "deduped_facts": 0,
    }
    rows = store.db.query(
        "SELECT * FROM memories WHERE conv_key=? AND status='active'",
        (conv_key,),
    )
    bot_names = soul.all_bot_names()
    for row in rows:
        mid = str(row["id"])
        text = normalize(str(row["text"] or ""))
        slot = str(row["slot"] or "")
        layer = str(row["layer"] or "")

        # 1. Void empty/placeholder garbage ("无\n待跟进：无")
        if not text or _PLACEHOLDER_GARBAGE.match(text.replace("\n", " ").strip()):
            store.void_memory(mid, now=now, reason="placeholder_garbage")
            stats["voided_garbage"] += 1
            continue

        # 2. Retire completed or stale short-term prospective plans
        if not slot and layer == "prospective" and _SHORT_TERM_PLAN_RE.search(text):
            age = now - float(row["ts"] or now)
            if age > 2 * 3600:
                store.void_memory(mid, now=now, reason="stale_short_term_plan")
                stats["retired_plans"] += 1
                continue

        # 3. Move ephemeral single-turn remarks from permanent confirmed to pending/voided
        if not slot and layer == "semantic" and _EPHEMERAL_REMARK_RE.search(text) \
                and not _DURABLE_FACT_RE.search(text):
            age = now - float(row["ts"] or now)
            if age > 6 * 3600:
                store.void_memory(mid, now=now, reason="expired_ephemeral_remark")
            else:
                with store.db.tx():
                    store.db.conn.execute(
                        "UPDATE memories SET orbit='pending', valid_to=? WHERE id=?",
                        (float(row["ts"] or now) + 6 * 3600, mid),
                    )
            stats["demoted_ephemeral"] += 1

        # 4. Fix legacy subject misattribution: a first-person bot utterance
        # ("我坐G6550从深圳北去广州东找用户") belongs to ``bot``, not ``user``.
        if not slot and str(row["source"] or "") == "legacy":
            if text.startswith("我") and ("找用户" in text or layer == "self"):
                target_subject = SUBJECT_BOT
            else:
                has_user = any(w in text for w in ("用户", "主人"))
                has_bot = any(w in text for w in bot_names)
                if has_user and has_bot:
                    target_subject = SUBJECT_BOTH
                elif has_user and not text.startswith("我"):
                    target_subject = SUBJECT_USER
                else:
                    target_subject = SUBJECT_BOT
            if target_subject != str(row["subject"] or ""):
                with store.db.tx():
                    store.db.conn.execute(
                        "UPDATE memories SET subject=? WHERE id=?",
                        (target_subject, mid),
                    )
                stats["repaired_subjects"] += 1

    # 5. Deduplicate identical slotless active facts
    active_slotless = store.db.query(
        "SELECT id, text, ts FROM memories WHERE conv_key=? AND status='active' "
        "AND (slot IS NULL OR slot='') ORDER BY ts DESC, id DESC",
        (conv_key,),
    )
    seen: dict[str, str] = {}
    for row in active_slotless:
        key = normalize(str(row["text"] or ""))
        mid = str(row["id"])
        if key in seen:
            store.void_memory(mid, now=now, reason=f"duplicate_of:{seen[key]}")
            stats["deduped_facts"] += 1
        else:
            seen[key] = mid

    return stats
