# -*- coding: utf-8 -*-
"""SoulCore (灵核) & Origin Invariants (原点不变量契约).

Decouples the memory kernel from any hardcoded persona.  A ``SoulSpec``
declares who the companion is, how the two speakers refer to each other, and
which origin invariants must hold across every migration, consolidation, and
prompt assembly pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SoulSpec:
    """Declarative identity anchor for a relational digital-life agent."""

    bot_name: str = "星野"
    bot_aliases: tuple[str, ...] = ("她", "宝贝", "宝宝")
    user_label: str = "他"
    user_aliases: tuple[str, ...] = ("用户", "对方", "主人")
    shared_label: str = "你们"
    invariants: Mapping[str, str] = field(default_factory=dict)
    protected_slots: tuple[str, ...] = (
        "name", "relationship", "boundary", "hometown", "school", "city",
    )

    def all_bot_names(self) -> tuple[str, ...]:
        names = [self.bot_name, *self.bot_aliases]
        out: list[str] = []
        for item in names:
            clean = str(item or "").strip()
            if clean and clean not in out:
                out.append(clean)
        return tuple(out)

    def all_generic_pronouns(self) -> frozenset[str]:
        return frozenset({
            self.bot_name, *self.bot_aliases,
            self.user_label, *self.user_aliases,
            "你", "我", "他", "她", "它", "我们", "你们", "咱们",
        })

    def joint_pattern(self) -> re.Pattern[str]:
        tokens = [re.escape(name) for name in self.all_bot_names()]
        tokens.extend(["两人", "双方", "一起", "一块儿"])
        return re.compile("|".join(tokens))

    def shared_location_pattern(self) -> re.Pattern[str]:
        tokens = [
            "两人", "双方", "一起", "一块儿", "我们", "咱们",
        ]
        for name in self.all_bot_names():
            if name not in ("她",):
                tokens.extend([
                    f"和{re.escape(name)}",
                    f"跟{re.escape(name)}",
                    f"带{re.escape(name)}",
                ])
        return re.compile("|".join(tokens))

    def her_object_pattern(self) -> re.Pattern[str]:
        names = "|".join(re.escape(name) for name in self.all_bot_names())
        return re.compile(rf"(?:把|抱|放|带|拉|牵)(?:着|起|住)?(?:{names})")

    def shared_strip_pattern(self) -> re.Pattern[str]:
        names = "|".join(re.escape(name) for name in self.all_bot_names())
        return re.compile(rf"^(?:和|跟|与)?(?:{names})?(?:一起|一块儿)?")

    def self_claim_pattern(self) -> re.Pattern[str]:
        names = [re.escape(self.bot_name), "我", "俺", "人家"]
        return re.compile("|".join(dict.fromkeys(names)))

    def manifest_digest(self) -> str:
        norm_inv = {
            (f"{k[0]}.{k[1]}" if isinstance(k, tuple) else str(k)): str(v)
            for k, v in (self.invariants or {}).items()
        }
        payload = {
            "bot_name": self.bot_name,
            "bot_aliases": list(self.bot_aliases),
            "user_label": self.user_label,
            "invariants": dict(sorted(norm_inv.items())),
            "protected_slots": list(self.protected_slots),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


DEFAULT_SOUL = SoulSpec()


@dataclass(frozen=True)
class AnchorReport:
    """Result of pre-render origin invariant verification."""

    ok: bool
    manifest: str
    violations: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    drifted: tuple[str, ...] = ()


def verify_anchors(store, conv_key: str,
                   soul: SoulSpec = DEFAULT_SOUL) -> AnchorReport:
    """Verify that core invariants hold before rendering a stateless pack.

    Checks:
    1. Every declared ``soul.invariants`` key (`slot` or `(subject, slot)`) matches
       if present in the store, or is seeded/protected from silent overwrite.
    2. No active `bot.*` state slot contains a user-only prefix (`用户...`).
    3. Database boundary matches ``conv_key``.
    """
    violations: list[str] = []
    missing: list[str] = []
    drifted: list[str] = []
    boundary = store.db.meta_get("conv_key")
    if boundary is not None and str(boundary) != conv_key:
        violations.append(f"boundary_mismatch:{boundary}!={conv_key}")

    for key, expected in (soul.invariants or {}).items():
        if isinstance(key, tuple) and len(key) == 2:
            subject, slot = str(key[0]), str(key[1])
        else:
            subject, slot = "bot", str(key)
        row = store.get_slot(conv_key, subject, slot)
        if row is None:
            missing.append(f"{subject}.{slot}")
            violations.append(f"invariant_missing:{subject}.{slot}")
        else:
            actual = str(row["value"] or "").strip()
            if actual and expected and actual != expected:
                drifted.append(f"{subject}.{slot}:expected={expected},got={actual}")
                violations.append(
                    f"invariant_drift:{subject}.{slot}:expected={expected},got={actual}"
                )

    for row in store.active_slots(conv_key, subject="bot"):
        slot = str(row["slot"] or "")
        text = str(row["text"] or "")
        if slot.startswith("bot.") and text.startswith("用户"):
            violations.append(f"subject_bleed:{slot}:{text[:24]}")

    return AnchorReport(
        ok=not violations,
        manifest=soul.manifest_digest(),
        violations=tuple(violations),
        missing=tuple(missing),
        drifted=tuple(drifted),
    )
