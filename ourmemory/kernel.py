# -*- coding: utf-8 -*-
"""MemoryKernel: Stateless Three-State Memory Engine for Relational AI.

Provides the four core lifecycle operations inspired by the Digital Life
Architecture:
1. ``awaken(conv_key, query)`` -> builds the stateless prompt workspace
   (Core Card + Current Scene + Resident Today's Scene Canvas + 4-Route Recall).
2. ``observe(conv_key, user_text, bot_text)`` -> passes dialogue through the
   inbound membrane, updates the dual-subject scene state machine, classifies
   facts into the Three-State Orbit (`confirmed` / `pending` / `voided`),
   links entities into the NeuroGraph, and appends to the hash-chained audit log.
3. ``guard_reply(conv_key, reply, user_text, workspace_text)`` -> runs the
   outbound civilization membrane (blocks ungrounded major life event claims
   and slot contradictions).
4. ``verify_anchors(conv_key)`` & ``verify_audit(conv_key)`` -> verifies
   origin invariants and cryptographic audit chain integrity.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from .canvas import build_workspace, core_card, recall_block, scene_canvas, state_block
from .db import Database
from .encoder import apply_fast_path, encode_turn, state_items
from .graph import NeuroGraph
from .membrane import (
    GuardResult,
    filter_ungrounded_history,
    guard_outbound_reply,
)
from .orbit import MemoryItem, ORBIT_CONFIRMED, SUBJECT_BOTH
from .retrieve import HybridRetriever
from .scene import apply_state_transitions, repair_store, write_items
from .soul import DEFAULT_SOUL, AnchorReport, SoulSpec, verify_anchors
from .store import MemoryStore
from .text import extract_entities, normalize
import re

_OFFLINE_SHARED_PLAN_RE = re.compile(
    r"(?:我们|咱们|一起)[^，。！？!?]{0,6}?去([\u4e00-\u9fa5]{2,10})(?:吧|呀|好不好|好吗)?"
)
_OFFLINE_SHARED_ACT_RE = re.compile(
    r"(?:我们|咱们|一起)(?:来|去|在)?(做晚饭|做饭|吃饭|洗碗|散步|看电影|打游戏|休息)"
)


class MemoryKernel:
    """Stateless, Three-State Memory Kernel (`confirmed` / `pending` / `voided`)."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        soul: SoulSpec = DEFAULT_SOUL,
        llm_call: Callable[[list[dict]], str] | None = None,
        embedder=None,
    ):
        self.path = str(db_path)
        self.soul = soul
        self.llm_call = llm_call
        self.embedder = embedder
        self.db = Database(self.path)
        self.store = MemoryStore(self.db)
        self.graph = NeuroGraph(self.db)
        if self.soul.invariants:
            self.seed_invariants("global")

    def close(self) -> None:
        self.db.close()

    def seed_invariants(self, conv_key: str, *, now: float | None = None) -> int:
        """Seed SoulSpec origin invariants into `conv_key` without overwriting user corrections."""
        now = time.time() if now is None else float(now)
        count = 0
        for (subject, slot), value in self.soul.invariants.items():
            existing = self.store.get_slot(conv_key, subject, slot)
            if existing is not None:
                continue
            mid, status = self.store.set_slot(
                conv_key,
                subject,
                slot,
                value,
                text=f"{subject}.{slot}：{value}",
                layer="identity",
                speech_act="statement",
                source="seed",
                orbit=ORBIT_CONFIRMED,
                confidence=0.95,
                importance=0.95,
                ts=now,
            )
            if mid and status in ("created", "superseded"):
                count += 1
        return count

    def awaken(
        self,
        conv_key: str,
        query: str = "",
        *,
        now: float | None = None,
        include_card: bool = True,
        include_canvas: bool = True,
        memory_question: bool = False,
    ) -> str:
        """Build the stateless memory workspace prompt for a single turn."""
        return build_workspace(
            self.store,
            conv_key,
            query,
            now=now,
            embedder=self.embedder,
            soul=self.soul,
            include_card=include_card,
            include_canvas=include_canvas,
            memory_question=memory_question,
        )

    def observe(
        self,
        conv_key: str,
        user_text: str,
        bot_text: str = "",
        *,
        turn_id: str = "",
        now: float | None = None,
        use_slow_path: bool = True,
    ) -> list[str]:
        """Observe a dialogue turn, update scene & Three-State Orbit, and link entities."""
        now = time.time() if now is None else float(now)
        u_clean = normalize(user_text)
        b_clean = normalize(bot_text)
        if not u_clean and not b_clean:
            return []

        ids = list(
            apply_fast_path(
                self.store,
                conv_key,
                u_clean,
                soul=self.soul,
                now=now,
            )
        )

        # Deterministic end-of-turn state transitions (shared locations, returning home, plan retirement)
        shared_hint: dict[str, str] = {}
        act_m = _OFFLINE_SHARED_ACT_RE.search(u_clean)
        if act_m:
            shared_hint["activity"] = act_m.group(1)
        if ("到" in u_clean or "回" in u_clean) and any(
            w in b_clean for w in ("好", "嗯", "我来", "一起", "甜", "香")
        ):
            for loc_word in ("公园", "商场", "家", "学校", "超市", "电影院"):
                if loc_word in u_clean:
                    shared_hint["location"] = loc_word
                    break

        inferred_states = state_items(
            {
                "shared_state_change": shared_hint,
                "bot_state_change": {},
                "user_state_change": {},
            },
            now=now,
            user_text=u_clean,
            bot_text=b_clean,
            soul=self.soul,
        )
        plan_m = _OFFLINE_SHARED_PLAN_RE.search(u_clean)
        if plan_m and "location" not in shared_hint:
            plan_val = f"去{plan_m.group(1)}"
            inferred_states.append(
                MemoryItem(
                    text=f"接下来：{plan_val}",
                    subject=SUBJECT_BOTH,
                    slot="next",
                    value=plan_val,
                    speech_act="plan",
                    source="inferred",
                    confidence=0.85,
                    importance=0.65,
                    layer="prospective",
                    time_scope="today",
                )
            )
            inferred_states.append(
                MemoryItem(
                    text=f"计划一起{plan_val}",
                    subject=SUBJECT_BOTH,
                    slot="",
                    value=f"计划一起{plan_val}",
                    speech_act="plan",
                    source="inferred",
                    confidence=0.85,
                    importance=0.65,
                    layer="prospective",
                    time_scope="today",
                )
            )
        if shared_hint.get("location") or shared_hint.get("activity"):
            ep_parts = []
            if shared_hint.get("location"):
                ep_parts.append(f"到了{shared_hint['location']}")
            if shared_hint.get("activity"):
                ep_parts.append(f"一起{shared_hint['activity']}")
            ep_text = f"{self.soul.user_label}和{self.soul.bot_name}" + "，".join(ep_parts)
            inferred_states.append(
                MemoryItem(
                    text=ep_text,
                    subject=SUBJECT_BOTH,
                    slot="",
                    value=ep_text,
                    speech_act="statement",
                    source="inferred",
                    confidence=0.85,
                    importance=0.72,
                    layer="episodic",
                    time_scope="today",
                )
            )

        if inferred_states:
            w_state, _ = write_items(
                self.store,
                conv_key,
                inferred_states,
                user_text=u_clean,
                bot_text=b_clean,
                soul=self.soul,
                now=now,
            )
            ids.extend(w_state)
        else:
            apply_state_transitions(
                self.store,
                conv_key,
                [],
                user_text=u_clean,
                bot_text=b_clean,
                now=now,
            )

        if use_slow_path and self.llm_call is not None:
            history = []
            if u_clean:
                history.append({"role": "user", "content": u_clean})
            if b_clean:
                history.append({"role": "assistant", "content": b_clean})
            encode_turn(
                self.store,
                conv_key,
                history,
                user_text=u_clean,
                bot_text=b_clean,
                core_card=self.core_card(conv_key, now=now),
                soul=self.soul,
                now=now,
                extractor=self.llm_call,
            )

        # Link entities into NeuroGraph for every newly written memory
        for mid in ids:
            row = self.store.get(mid)
            if row is None:
                continue
            mentions = extract_entities(str(row["text"] or ""))
            if mentions:
                eids = self.graph.link_memory(mid, mentions, now=now)
                self.graph.reinforce_cooccurrence(
                    eids, importance=float(row["importance"] or 0.5), now=now
                )

        return ids

    def search(self, conv_key: str, query: str, *, limit: int = 6, now: float | None = None):
        """Run four-route hybrid retrieval directly and return `RetrievalHit` objects."""
        return HybridRetriever(
            self.store, conv_key, embedder=self.embedder, soul=self.soul
        ).search(query, limit=limit, now=now)

    def guard_reply(
        self,
        conv_key: str,
        reply: str,
        *,
        user_text: str = "",
        workspace_text: str = "",
    ) -> GuardResult:
        """Verify an outbound candidate reply against active slots and grounding."""
        return guard_outbound_reply(
            self.store,
            conv_key,
            reply,
            user_text=user_text,
            workspace_text=workspace_text,
            soul=self.soul,
        )

    def sanitize_history(self, history: list[dict], *, workspace_text: str = "") -> list[dict]:
        """Filter ungrounded sensitive claims from prior assistant messages in history."""
        return filter_ungrounded_history(history, grounding_corpus=workspace_text)

    def repair(self, conv_key: str, *, now: float | None = None) -> dict[str, int]:
        """Clean placeholder garbage, retire stale short-term plans, and dedupe slots."""
        return repair_store(self.store, conv_key, now=now, soul=self.soul)

    def verify_anchors(self, conv_key: str) -> AnchorReport:
        """Check whether SoulSpec invariant anchors are intact in `conv_key`."""
        return verify_anchors(self.store, conv_key, self.soul)

    def verify_audit(self, conv_key: str) -> tuple[bool, int]:
        """Verify cryptographic SHA-256 hash chain in `audit_log` for `conv_key`."""
        ok = self.db.verify_audit_chain(conv_key)
        count = int(
            self.db.scalar(
                "SELECT COUNT(*) FROM audit_log WHERE conv_key=?",
                (conv_key,),
                default=0,
            )
            or 0
        )
        return ok, count

    def export_pack(self, conv_key: str) -> dict:
        """Export a portable, zero-lock-in JSON-serializable snapshot of a conversation's memory."""
        memories = [
            dict(row)
            for row in self.db.query(
                "SELECT id, layer, subject, slot, value, text, importance, confidence, "
                "status, orbit, ts, valid_from, valid_to, source "
                "FROM memories WHERE conv_key=? AND status='active' AND orbit<>'voided' "
                "ORDER BY ts ASC",
                (conv_key,),
            )
        ]
        audit_ok, audit_count = self.verify_audit(conv_key)
        return {
            "schema_version": 3,
            "conv_key": conv_key,
            "soul": {
                "bot_name": self.soul.bot_name,
                "bot_aliases": list(self.soul.bot_aliases),
                "user_label": self.soul.user_label,
            },
            "audit_verified": audit_ok,
            "audit_records": audit_count,
            "memories": memories,
        }

    def stats(self, conv_key: str | None = None) -> dict:
        base = self.store.stats()
        if conv_key:
            orbits = {}
            for orbit in ("confirmed", "pending", "voided"):
                row = self.db.query_one(
                    "SELECT COUNT(*) AS n FROM memories WHERE conv_key=? AND orbit=?",
                    (conv_key, orbit),
                )
                orbits[orbit] = int(row["n"]) if row else 0
            base["conv_orbits"] = orbits
        return base

    # Convenience accessors for individual canvas sections
    def core_card(self, conv_key: str, *, now: float | None = None) -> str:
        return core_card(self.store, conv_key, now=now)

    def state_block(self, conv_key: str, *, now: float | None = None) -> str:
        return state_block(self.store, conv_key, now=now)

    def scene_canvas(self, conv_key: str, *, now: float | None = None) -> str:
        return scene_canvas(self.store, conv_key, now=now)

    def recall_block(self, conv_key: str, query: str, *, now: float | None = None) -> str:
        return recall_block(self.store, conv_key, query, now=now, soul=self.soul)


def default_db_path() -> str:
    env = os.environ.get("OURMEMORY_DB", "").strip()
    if env:
        return env
    return str(Path.cwd() / "ourmemory.sqlite3")
