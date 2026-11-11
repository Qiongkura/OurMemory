# -*- coding: utf-8 -*-
"""Comprehensive test suite for ourmemory (100% fictional personas, zero PII)."""

from __future__ import annotations

import time

from ourmemory import (
    MemoryItem,
    MemoryKernel,
    ORBIT_CONFIRMED,
    ORBIT_PENDING,
    SoulSpec,
    classify_orbit,
    validate,
    write_items,
)


def _demo_soul() -> SoulSpec:
    return SoulSpec(
        bot_name="小星",
        bot_aliases=("星宝",),
        user_label="阿泽",
        invariants={
            ("bot", "name"): "小星",
            ("bot", "city"): "广州",
            ("bot", "school"): "星海大学",
            ("both", "relationship"): "恋人",
        },
    )


def test_soul_invariants_and_anchor_verification():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:anchors"
    seeded = kernel.seed_invariants(conv)
    assert seeded == 4

    report = kernel.verify_anchors(conv)
    assert report.ok
    assert not report.missing
    assert not report.drifted

    # Slow-path inference must NOT overwrite protected bot identity slots
    # (instead it enters conflicted state without replacing active slot)
    write_items(
        kernel.store,
        conv,
        [
            MemoryItem(
                layer="semantic",
                subject="bot",
                slot="school",
                value="北方学院",
                text="小星在北方学院上学",
                confidence=0.95,
                source="user_explicit",
            )
        ],
        soul=soul,
    )
    assert kernel.verify_anchors(conv).ok
    row = kernel.store.get_slot(conv, "bot", "school")
    assert row is not None and row["value"] == "星海大学"


def test_three_state_orbit_classification_and_promotion():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:orbit"
    now = 1_760_000_000.0

    # 1. One-off drink order / temporary sensation / single-turn remark -> pending
    items = [
        MemoryItem(
            layer="semantic",
            subject="bot",
            slot="",
            value="喝奶茶要冰的",
            text="小星喝奶茶要冰的",
            confidence=0.85,
        ),
        MemoryItem(
            layer="semantic",
            subject="bot",
            slot="",
            value="走久了脚疼",
            text="小星走久了脚疼",
            confidence=0.85,
        ),
        MemoryItem(
            layer="semantic",
            subject="user",
            slot="",
            value="提出背对方",
            text="阿泽提出背对方",
            confidence=0.85,
        ),
    ]
    for it in items:
        orbit, ttl = classify_orbit(it, now=now)
        assert orbit == ORBIT_PENDING
        assert ttl is not None and ttl > now

    ids, rejected = write_items(kernel.store, conv, items, now=now, soul=soul)
    assert not rejected
    assert len(ids) == 3
    for mid in ids:
        row = kernel.store.get(mid)
        assert row is not None
        assert row["orbit"] == ORBIT_PENDING

    # 2. Restating the same fact in a later turn promotes it from pending -> confirmed
    write_items(
        kernel.store,
        conv,
        [
            MemoryItem(
                layer="semantic",
                subject="bot",
                slot="",
                value="喝奶茶要冰的",
                text="小星喝奶茶要冰的",
                confidence=0.90,
            )
        ],
        now=now + 600,
        soul=soul,
    )
    row = kernel.store.get(ids[0])
    assert row is not None
    assert row["orbit"] == ORBIT_CONFIRMED


def test_short_term_plan_auto_retires_when_outing_completes():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:plan_retire"
    now = 1_760_000_000.0

    # User proposes going to the park
    write_items(
        kernel.store,
        conv,
        [
            MemoryItem(
                layer="prospective",
                subject="user",
                slot="",
                value="计划带她去公园散步",
                text="计划带她去公园散步",
                confidence=0.88,
            ),
            MemoryItem(
                layer="prospective",
                subject="both",
                slot="next",
                value="去公园散步",
                text="接下来：去公园散步",
                confidence=0.88,
            ),
            # Long-term promise should NOT be retired when returning home today
            MemoryItem(
                layer="prospective",
                subject="user",
                slot="",
                value="约定以后经常带她出去玩",
                text="约定以后经常带她出去玩",
                confidence=0.92,
            ),
        ],
        now=now,
        soul=soul,
    )

    # Now they return home and cook dinner
    kernel.observe(conv, "回到家啦，我们做晚饭吧", "好呀我来洗菜", now=now + 3600)

    active_prospective = kernel.db.query(
        "SELECT text, orbit, status FROM memories "
        "WHERE conv_key=? AND layer='prospective' AND status='active' AND orbit<>'voided'",
        (conv,),
    )
    texts = [str(r["text"]) for r in active_prospective]
    assert not any("去公园散步" in t for t in texts)
    assert any("约定以后经常带她出去玩" in t for t in texts)


def test_time_window_recall_handles_what_did_we_do_today():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:today_recall"
    now = time.time()

    write_items(
        kernel.store,
        conv,
        [
            MemoryItem(
                layer="episodic",
                subject="both",
                slot="",
                value="一起去万达买衣服、逛公园，晚上回家做饭",
                text="今天阿泽和小星一起去万达买衣服、去公园散步，晚上回家做饭",
                importance=0.88,
                confidence=0.92,
            )
        ],
        now=now - 1800,
        soul=soul,
    )

    # Previously failed because "干了" and "了啥" were treated as required content words
    for query in ("我们今天干了啥", "今天我们做了什么呀", "今天干嘛了"):
        hits = kernel.search(conv, query, now=now)
        assert hits, f"Expected time-window recall hits for query={query!r}"
        assert any("万达" in h.text or "公园" in h.text for h in hits)

    workspace = kernel.awaken(conv, "我们今天干了啥", now=now, memory_question=True)
    assert "【今日现场画】" in workspace
    assert "【相关记忆】" in workspace
    assert "万达" in workspace


def test_prospective_recall_for_tomorrow_inquiry():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:tomorrow_recall"
    now = time.time()

    write_items(
        kernel.store,
        conv,
        [
            MemoryItem(
                layer="prospective",
                subject="both",
                slot="",
                value="明天一起去海边看日出",
                text="约定明天一起去海边看日出",
                importance=0.9,
                confidence=0.9,
            )
        ],
        now=now - 60,
        soul=soul,
    )
    hits = kernel.search(conv, "我们明天干嘛呀", now=now)
    assert hits
    assert any("海边看日出" in h.text for h in hits)


def test_placeholder_garbage_rejected_and_repaired():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:garbage"
    now = time.time()

    bad = MemoryItem(
        layer="episodic",
        subject="both",
        slot="",
        value="无\n待跟进：无",
        text="无\n待跟进：无",
        confidence=0.8,
    )
    reason = validate(bad, soul=soul)
    assert reason == "empty_or_placeholder"

    # Even if legacy DB had a row like "无\n待跟进：无", repair() voids it
    kernel.store.add("episode", "无\n待跟进：无", conv_key=conv, ts=now)
    stats = kernel.repair(conv, now=now)
    assert stats["voided_garbage"] >= 1
    rows = kernel.db.query(
        "SELECT * FROM memories WHERE conv_key=? AND status='active' AND orbit<>'voided'",
        (conv,),
    )
    assert not rows


def test_outbound_membrane_blocks_ungrounded_interview_claims():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:membrane"
    kernel.seed_invariants(conv)

    # 1. Multi-line ungrounded operational internship claim is caught
    reply = "就一个运营岗\n我也没底\n简历都是赶着改的"
    guard = kernel.guard_reply(conv, reply, user_text="你在忙什么呢", workspace_text="")
    assert guard.needs_regeneration
    assert "运营岗" in guard.ungrounded_events
    assert "简历" in guard.ungrounded_events
    assert guard.cleaned_reply == "我也没底"

    # 2. Grounded claim (user explicitly asked about her interview) passes
    grounded_guard = kernel.guard_reply(
        conv,
        "今天下午刚去面试了一个运营岗，有点紧张",
        user_text="你今天下午的运营岗面试怎么样啦？简历过了吗",
        workspace_text="",
    )
    assert not grounded_guard.ungrounded_events

    # 3. Third-party mention passes
    third_party = kernel.guard_reply(
        conv,
        "我室友今天去面试实习了，看她改简历改到半夜",
        user_text="你们宿舍今天热闹吗",
        workspace_text="",
    )
    assert not third_party.ungrounded_events

    # 4. History sanitization strips prior ungrounded assistant hallucinations
    history = [
        {"role": "user", "content": "干嘛呢"},
        {"role": "assistant", "content": "刚改完简历，明天要去面试实习"},
        {"role": "user", "content": "什么实习？"},
        {"role": "assistant", "content": "就一个运营岗\n我也没底\n简历都是赶着改的"},
    ]
    cleaned_hist = kernel.sanitize_history(history)
    assistant_texts = [m["content"] for m in cleaned_hist if m["role"] == "assistant"]
    assert not any("面试" in t or "运营岗" in t or "简历" in t for t in assistant_texts)


def test_audit_hash_chain_and_tamper_detection():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:audit"
    kernel.seed_invariants(conv)
    kernel.observe(conv, "我们去公园散步吧", "好呀")
    kernel.observe(conv, "回到家啦，我去做饭", "我来洗碗")

    ok, count = kernel.verify_audit(conv)
    assert ok
    assert count >= 4

    pack = kernel.export_pack(conv)
    assert pack["audit_verified"] is True
    assert pack["schema_version"] == 3

    # Tamper with an audit row and verify detection
    first_row = kernel.db.query_one(
        "SELECT id FROM audit_log WHERE conv_key=? ORDER BY id ASC LIMIT 1",
        (conv,),
    )
    assert first_row is not None
    kernel.db.conn.execute(
        "UPDATE audit_log SET value='TAMPERED' WHERE id=?",
        (first_row["id"],),
    )
    ok_after, _ = kernel.verify_audit(conv)
    assert not ok_after


def test_english_outbound_membrane_blocks_ungrounded_life_events():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "test:membrane-en"
    kernel.seed_invariants(conv)

    # 1. First-person English life event with no grounding is caught and stripped
    guard = kernel.guard_reply(
        conv,
        "I have a job interview tomorrow, I am still fixing my resume",
        user_text="what are you up to",
        workspace_text="",
    )
    assert guard.needs_regeneration
    assert "job interview" in guard.ungrounded_events
    assert "resume" in guard.ungrounded_events
    assert guard.cleaned_reply == ""

    # 2. Partially grounded reply keeps the grounded clause only
    partial = kernel.guard_reply(
        conv,
        "I had surgery this morning, but I still want milk tea",
        user_text="do you want milk tea",
        workspace_text="",
    )
    assert "surgery" in partial.ungrounded_events
    assert "milk tea" in partial.cleaned_reply
    assert "surgery" not in partial.cleaned_reply

    # 3. Third-party English mention passes
    third_party = kernel.guard_reply(
        conv,
        "my roommate has a job interview tomorrow, she is nervous",
        user_text="how is your dorm",
        workspace_text="",
    )
    assert not third_party.ungrounded_events

    # 4. Asking the user about their own interview is not a self claim
    asking = kernel.guard_reply(
        conv,
        "did you have your interview today?",
        user_text="hey",
        workspace_text="",
    )
    assert not asking.ungrounded_events

    # 5. Meta talk (dreams, negation) is not a claim
    for meta in (
        "I dreamt I had a job interview",
        "I never had an interview, I am just chilling",
    ):
        result = kernel.guard_reply(conv, meta, user_text="hey", workspace_text="")
        assert not result.ungrounded_events, meta

    # 6. Ordinary non-event uses of ambiguous words are ignored
    for benign in ("I resume the video at 3pm", "the operation runs in O(n)"):
        result = kernel.guard_reply(conv, benign, user_text="hey", workspace_text="")
        assert not result.ungrounded_events, benign


def test_english_history_sanitization_strips_prior_hallucinations():
    kernel = MemoryKernel(":memory:", soul=_demo_soul())
    history = [
        {"role": "user", "content": "what's up"},
        {"role": "assistant", "content": "I got engaged last week, planning the wedding"},
        {"role": "user", "content": "really?"},
        {"role": "assistant", "content": "just kidding, I am home watching a movie"},
    ]
    cleaned = kernel.sanitize_history(history)
    assistant_texts = [m["content"] for m in cleaned if m["role"] == "assistant"]
    assert not any("engaged" in t for t in assistant_texts)
    assert any("movie" in t for t in assistant_texts)


def test_custom_embedder_is_used_when_writing_not_just_querying():
    calls: list[int] = []

    def embedder(texts):
        calls.append(len(texts))
        return [[0.02] * 512 for _ in texts]

    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul, embedder=embedder)
    conv = "test:embedder"
    kernel.observe(conv, "我们下午去公园散步吧", "好呀")

    # The injected model must be used at write time; otherwise stored rows stay
    # in the 384-dim hashing space and the vector route silently skips them.
    assert calls, "embedder was never called while storing"
    _ids, matrix = kernel.store.vectors()
    assert matrix.shape[1] == 512


def test_english_activity_and_time_queries_are_understood():
    soul = _demo_soul()
    kernel = MemoryKernel(":memory:", soul=soul, tz="Asia/Shanghai")
    conv = "test:english"
    kernel.observe(conv, "we went to the park and had milk tea", "sounds fun")

    hits = kernel.search(conv, "did we go to the park?")
    assert hits and any("park" in hit.text for hit in hits)

    workspace = kernel.awaken(conv, "what did we do today?", memory_question=True)
    assert "park" in workspace
    # Small talk must not drag memories in.
    assert kernel.awaken(conv, "what is the weather") == ""


def test_day_boundary_follows_the_configured_timezone():
    from ourmemory.clock import day_bounds, stamp

    # 2026-11-12 00:00 UTC is 08:00 in Shanghai, so Shanghai's "today" starts
    # 8 hours earlier in epoch terms than UTC's.
    instant = 1762905600.0
    utc_start, _ = day_bounds(instant, "UTC")
    sh_start, _ = day_bounds(instant, "Asia/Shanghai")
    assert sh_start < utc_start
    assert stamp(utc_start, "UTC").endswith("00:00")
    assert stamp(sh_start, "Asia/Shanghai").endswith("00:00")

    kernel = MemoryKernel(":memory:", soul=_demo_soul(), tz="Asia/Shanghai")
    conv = "test:tz"
    kernel.observe(conv, "我们去公园散步吧", "好呀")
    assert "【今日现场画】" in kernel.awaken(conv, "我们今天干了啥", now=instant + 3600)
