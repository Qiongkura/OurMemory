# -*- coding: utf-8 -*-
"""ourmemory: Stateless, Three-State Memory Kernel for Relational AI & Digital Life."""

from .canvas import (
    build_workspace,
    certainty_block,
    conflicted_block,
    core_card,
    recall_block,
    scene_canvas,
    state_block,
)
from .db import Database
from .encoder import apply_fast_path, encode_turn, fast_path, state_items
from .graph import REGIONS, NeuroGraph
from .kernel import MemoryKernel, default_db_path
from .membrane import (
    SENSITIVE_EVENT_TERMS,
    ConsistencyIssue,
    GuardResult,
    check_reply_consistency,
    filter_ungrounded_history,
    guard_outbound_reply,
    sensitive_claims,
    strip_ungrounded_events,
    ungrounded_event_terms,
)
from .orbit import (
    ORBIT_CONFIRMED,
    ORBIT_PENDING,
    ORBIT_VOIDED,
    MemoryItem,
    Rejected,
    classify_orbit,
    validate,
)
from .retrieve import HybridRetriever, RetrievalHit, render_hits
from .scene import apply_state_transitions, repair_store, write_items
from .soul import DEFAULT_SOUL, AnchorReport, SoulSpec, verify_anchors
from .store import MemoryStore

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AnchorReport",
    "ConsistencyIssue",
    "DEFAULT_SOUL",
    "Database",
    "GuardResult",
    "HybridRetriever",
    "MemoryItem",
    "MemoryKernel",
    "MemoryStore",
    "NeuroGraph",
    "ORBIT_CONFIRMED",
    "ORBIT_PENDING",
    "ORBIT_VOIDED",
    "REGIONS",
    "Rejected",
    "RetrievalHit",
    "SENSITIVE_EVENT_TERMS",
    "SoulSpec",
    "apply_fast_path",
    "apply_state_transitions",
    "build_workspace",
    "certainty_block",
    "check_reply_consistency",
    "classify_orbit",
    "conflicted_block",
    "core_card",
    "default_db_path",
    "encode_turn",
    "fast_path",
    "filter_ungrounded_history",
    "guard_outbound_reply",
    "recall_block",
    "render_hits",
    "repair_store",
    "scene_canvas",
    "sensitive_claims",
    "state_block",
    "state_items",
    "strip_ungrounded_events",
    "ungrounded_event_terms",
    "validate",
    "verify_anchors",
    "write_items",
]
