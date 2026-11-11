# -*- coding: utf-8 -*-
"""MemoryStore: CRUD, slot management, Three-State Orbit (confirmed/pending/voided), and vectors."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from array import array
from typing import Iterable

try:  # optional accelerator; the kernel runs on the standard library alone
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in no-numpy environments
    np = None

from .db import Database
from .text import (
    VECTOR_DIM, clamp, fts_query, fts_query_terms, fts_text, hash_vector,
    normalize, stable_hash, vector_tokens, vectors_matrix,
)

KINDS = ("episode", "semantic", "preference", "procedure", "plan", "reflection", "utterance")
ORBITS = ("confirmed", "pending", "voided")

DECAY_BASE_DAYS = 14.0
DECAY_FLOOR = 0.08
ARCHIVE_STRENGTH = 0.18
ARCHIVE_IMPORTANCE = 0.45
ARCHIVE_AGE_DAYS = 30.0


def _pack_vector(vec) -> bytes:
    """Serialize a float vector to bytes without requiring NumPy."""
    if np is not None:
        return np.asarray(vec, dtype=np.float32).tobytes()
    return array("f", (float(value) for value in vec)).tobytes()


def _unpack_vector(blob: bytes):
    """Deserialize bytes back into a NumPy array or a plain list of floats."""
    if np is not None:
        return np.frombuffer(blob, dtype=np.float32)
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


class MemoryStore:
    def __init__(self, db: Database, embedder=None):
        self.db = db
        self.embedder = embedder
        self._vec_cache = None

    def _embedding_for(self, text: str):
        """Embedding for a new row: the injected model, else built-in hashing.

        When an embedder is configured but fails, the row is stored without a
        vector instead of silently falling back to a different vector space —
        mixed dimensions would make the vector route skip rows unpredictably.
        """
        if self.embedder is None:
            return hash_vector(text)
        try:
            vectors = self.embedder([text])
            if vectors and len(vectors[0]) > 0:
                return vectors[0]
        except Exception:  # noqa: BLE001
            return None
        return None

    def add(
        self,
        kind: str,
        text: str,
        *,
        summary: str = "",
        ts: float | None = None,
        valid_to: float | None = None,
        source: str = "",
        conv_key: str = "",
        uid: int | None = None,
        gid: int | None = None,
        importance: float = 0.5,
        strength: float = 1.0,
        confidence: float = 0.7,
        valence: float = 0.0,
        arousal: float = 0.0,
        emotion: str = "平静",
        orbit: str = "confirmed",
        embedding: np.ndarray | list[float] | None = None,
        ref: str | None = None,
        meta: dict | None = None,
    ) -> tuple[str, bool]:
        text = normalize(text)
        if not text:
            raise ValueError("empty text")
        kind = kind if kind in KINDS else "episode"
        orbit = orbit if orbit in ORBITS else "confirmed"
        ts = time.time() if ts is None else float(ts)
        if ref:
            row = self.db.query_one("SELECT id FROM memories WHERE ref=?", (ref,))
            if row is not None:
                return str(row["id"]), False
        content_hash = stable_hash(kind + "\x1f" + text, 20)
        memory_id = f"m_{int(ts * 1000)}_{uuid.uuid4().hex}"
        tokens = fts_text(text + "\n" + summary)
        if embedding is None:
            embedding = self._embedding_for(text + " " + summary)
        blob = _pack_vector(embedding) if embedding is not None else None
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        status = "voided" if orbit == "voided" else "active"
        with self.db.tx():
            try:
                self.db.conn.execute(
                    "INSERT INTO memories(id,kind,text,summary,tokens,ts,valid_from,valid_to,"
                    "source,conv_key,uid,gid,importance,strength,confidence,valence,arousal,"
                    "emotion,access_count,last_access,archived,embedding,ref,hash,meta,status,orbit) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        memory_id, kind, text, summary, tokens, ts, ts, valid_to,
                        source, conv_key, uid, gid, clamp(importance), clamp(strength),
                        clamp(confidence), float(valence), clamp(arousal), emotion,
                        0, None, 0 if status == "active" else 1, blob, ref, content_hash,
                        meta_json, status, orbit,
                    ),
                )
            except sqlite3.IntegrityError:
                row = self.db.query_one("SELECT id FROM memories WHERE ref=?", (ref,)) if ref else None
                if row is not None:
                    return str(row["id"]), False
                raise
            if status == "active":
                self.db.conn.execute(
                    "INSERT INTO memory_fts(memory_id,tokens) VALUES(?,?)",
                    (memory_id, tokens),
                )
                self._bump_df(vector_tokens(text + " " + summary))
            self.db.conn.execute(
                "INSERT INTO meta(k,v) VALUES('vec_epoch',?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (str(time.time_ns()),),
            )
        return memory_id, True

    def _bump_df(self, toks: Iterable[str]) -> None:
        unique = sorted(set(t for t in toks if t))
        if not unique:
            return
        self.db.conn.executemany(
            "INSERT INTO token_df(token,df) VALUES(?,1) "
            "ON CONFLICT(token) DO UPDATE SET df=df+1",
            [(t,) for t in unique],
        )

    def promote_orbit(self, memory_id: str, *, now: float | None = None,
                      reason: str = "reconfirmed") -> bool:
        """Promote a ``pending`` memory to ``confirmed`` (clears its pending TTL)."""
        now = time.time() if now is None else float(now)
        row = self.get(memory_id)
        if row is None or str(row["orbit"] or "") == "confirmed":
            return False
        with self.db.tx():
            self.db.conn.execute(
                "UPDATE memories SET orbit='confirmed', status='active', "
                "valid_to=NULL, archived=0, confidence=MAX(confidence, 0.9) "
                "WHERE id=?",
                (memory_id,),
            )
            self.db.record_audit(
                str(row["conv_key"] or ""),
                "promote",
                memory_id,
                subject=str(row["subject"] or ""),
                slot=str(row["slot"] or ""),
                value=str(row["value"] or row["text"] or "")[:60],
                orbit="confirmed",
                reason=reason,
                ts=now,
            )
        return True

    def void_memory(self, memory_id: str, *, now: float | None = None,
                    reason: str = "voided") -> bool:
        """Move a memory to the ``voided`` orbit: never recalled, kept as tombstone."""
        now = time.time() if now is None else float(now)
        row = self.get(memory_id)
        if row is None:
            return False
        with self.db.tx():
            self.db.fts_delete(memory_id)
            self.db.conn.execute(
                "UPDATE memories SET status='voided', orbit='voided', "
                "archived=1, valid_to=? WHERE id=?",
                (now, memory_id),
            )
            self.db.record_audit(
                str(row["conv_key"] or ""),
                "void",
                memory_id,
                subject=str(row["subject"] or ""),
                slot=str(row["slot"] or ""),
                value=str(row["value"] or row["text"] or "")[:60],
                orbit="voided",
                reason=reason,
                ts=now,
            )
        self._vec_cache = None
        return True

    def expire_pending(self, conv_key: str, *, now: float | None = None) -> int:
        """Retire ``pending`` items whose ``valid_to`` deadline has passed."""
        now = time.time() if now is None else float(now)
        rows = self.db.query(
            "SELECT id, subject, slot, value, text FROM memories "
            "WHERE conv_key=? AND status='active' AND orbit='pending' "
            "AND valid_to IS NOT NULL AND valid_to <= ?",
            (conv_key, now),
        )
        if not rows:
            return 0
        with self.db.tx():
            for row in rows:
                mid = str(row["id"])
                self.db.fts_delete(mid)
                self.db.conn.execute(
                    "UPDATE memories SET status='voided', orbit='voided', archived=1 "
                    "WHERE id=?",
                    (mid,),
                )
                self.db.record_audit(
                    conv_key,
                    "expire_pending",
                    mid,
                    subject=str(row["subject"] or ""),
                    slot=str(row["slot"] or ""),
                    value=str(row["value"] or row["text"] or "")[:60],
                    orbit="voided",
                    reason="pending_ttl_elapsed",
                    ts=now,
                )
        self._vec_cache = None
        return len(rows)

    def set_slot(
        self,
        conv_key: str,
        subject: str,
        slot: str,
        value: str,
        *,
        text: str = "",
        layer: str = "semantic",
        speech_act: str = "statement",
        source: str = "user_explicit",
        orbit: str = "confirmed",
        valid_to: float | None = None,
        evidence_ids: Iterable[str] = (),
        confidence: float = 0.9,
        importance: float = 0.7,
        ts: float | None = None,
        ref: str | None = None,
        meta: dict | None = None,
    ) -> tuple[str, str]:
        if not conv_key or not subject or not slot:
            raise ValueError("conv_key, subject and slot are required")
        value = normalize(value)
        if not value:
            raise ValueError("empty value")
        ts = time.time() if ts is None else float(ts)
        evidence = [str(item) for item in evidence_ids if item]
        current = self.db.query_one(
            "SELECT * FROM memories WHERE conv_key=? AND subject=? AND slot=? "
            "AND status='active' LIMIT 1",
            (conv_key, subject, slot),
        )
        if current is not None and normalize(str(current["value"] or "")) == value:
            if str(current["orbit"] or "") == "pending" and orbit == "confirmed":
                self.promote_orbit(str(current["id"]), now=ts, reason="slot_reconfirmed")
            return str(current["id"]), "unchanged"
        if current is not None:
            current_from = float(current["valid_from"] or current["ts"] or 0)
            if ts < current_from:
                return str(current["id"]), "stale_ignored"

        body = normalize(text) or f"{subject}.{slot}={value}"
        meta_payload = dict(meta or {})
        meta_payload.setdefault("slot", slot)
        meta_payload.setdefault("value", value)
        with self.db.tx():
            if current is not None:
                self.db.conn.execute(
                    "UPDATE memories SET status='superseded', valid_to=?, "
                    "superseded_by=NULL WHERE id=?",
                    (ts, str(current["id"])),
                )
            memory_id, created = self.add(
                "semantic", body, summary=body[:80], ts=ts, valid_to=valid_to,
                source=source, conv_key=conv_key, importance=importance,
                confidence=confidence, orbit=orbit, ref=ref, meta=meta_payload,
            )
            if created:
                self.db.conn.execute(
                    "UPDATE memories SET subject=?, slot=?, value=?, speech_act=?, "
                    "evidence_ids=?, status='active', orbit=?, layer=?, valid_from=? "
                    "WHERE id=?",
                    (subject, slot, value, speech_act,
                     json.dumps(evidence, ensure_ascii=False), orbit, layer, ts, memory_id),
                )
            if current is not None:
                self.db.conn.execute(
                    "UPDATE memories SET superseded_by=? WHERE id=?",
                    (memory_id, str(current["id"])),
                )
            self.db.record_audit(
                conv_key,
                "supersede_slot" if current is not None else "set_slot",
                memory_id,
                subject=subject,
                slot=slot,
                value=value[:60],
                orbit=orbit,
                reason=speech_act,
                ts=ts,
            )
        return memory_id, ("superseded" if current is not None else "created")

    def get_slot(self, conv_key: str, subject: str, slot: str):
        return self.db.query_one(
            "SELECT * FROM memories WHERE conv_key=? AND subject=? AND slot=? "
            "AND status='active' LIMIT 1",
            (conv_key, subject, slot),
        )

    def active_slots(self, conv_key: str, subject: str | None = None) -> list:
        if subject is None:
            return list(self.db.query(
                "SELECT * FROM memories WHERE conv_key=? AND status='active' "
                "AND slot IS NOT NULL AND slot<>'' ORDER BY importance DESC, ts DESC",
                (conv_key,),
            ))
        return list(self.db.query(
            "SELECT * FROM memories WHERE conv_key=? AND subject=? AND status='active' "
            "AND slot IS NOT NULL AND slot<>'' ORDER BY importance DESC, ts DESC",
            (conv_key, subject),
        ))

    def get(self, memory_id: str):
        return self.db.query_one("SELECT * FROM memories WHERE id=?", (memory_id,))

    def get_many(self, memory_ids: Iterable[str]) -> dict:
        ids = [i for i in memory_ids if i]
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self.db.query(f"SELECT * FROM memories WHERE id IN ({marks})", tuple(ids))
        return {str(r["id"]): r for r in rows}

    def search_fts_terms(self, terms, limit: int = 60) -> list[tuple[str, float]]:
        q = fts_query_terms(terms)
        if not q:
            return []
        try:
            rows = self.db.query(
                "SELECT f.memory_id, bm25(memory_fts) AS rank "
                "FROM memory_fts f JOIN memories m ON m.id = f.memory_id "
                "WHERE memory_fts MATCH ? AND m.archived=0 "
                "AND m.status='active' AND m.orbit<>'voided' "
                "ORDER BY rank LIMIT ?",
                (q, limit),
            )
        except sqlite3.OperationalError:
            return []
        return [(str(r["memory_id"]), float(r["rank"])) for r in rows]

    def touch(self, memory_ids: Iterable[str], now: float | None = None) -> None:
        now = time.time() if now is None else now
        ids = [(now, mid) for mid in memory_ids if mid]
        if not ids:
            return
        with self.db.tx():
            for _, mid in ids:
                self.db.conn.execute(
                    "UPDATE memories SET access_count=access_count+1, last_access=?, "
                    "strength=MIN(1.0, strength + 0.08*(1.0-strength)) WHERE id=?",
                    (now, mid),
                )

    def count(self, where: str = "", params=()) -> int:
        sql = "SELECT COUNT(*) FROM memories"
        if where:
            sql += " WHERE " + where
        return int(self.db.scalar(sql, params, 0))

    def idf(self, toks: Iterable[str] | None = None) -> dict[str, float]:
        n = max(1, self.count("archived=0 AND status='active'"))
        if toks is None:
            rows = self.db.query("SELECT token, df FROM token_df")
        else:
            unique = sorted(set(t for t in toks if t))
            if not unique:
                return {}
            marks = ",".join("?" for _ in unique)
            rows = self.db.query(
                f"SELECT token, df FROM token_df WHERE token IN ({marks})", tuple(unique)
            )
        return {
            str(r["token"]): math.log((n + 1.0) / (float(r["df"]) + 1.0)) + 1.0
            for r in rows
        }

    def vectors(self):
        epoch = str(self.db.meta_get("vec_epoch", "0"))
        if self._vec_cache and self._vec_cache[0] == epoch:
            return self._vec_cache[1], self._vec_cache[2]
        rows = self.db.query(
            "SELECT id, embedding FROM memories "
            "WHERE archived=0 AND status='active' AND orbit<>'voided' "
            "AND embedding IS NOT NULL"
        )
        ids: list[str] = []
        vecs: list = []
        width = 0
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            vec = _unpack_vector(blob)
            size = int(vec.size) if np is not None else len(vec)
            if size == 0:
                continue
            if width == 0:
                width = size
            if size != width:
                continue
            if np is not None:
                norm = float(np.linalg.norm(vec))
                if norm <= 0:
                    continue
                vecs.append(vec / norm)
            else:
                norm = math.sqrt(sum(value * value for value in vec))
                if norm <= 0:
                    continue
                vecs.append([value / norm for value in vec])
            ids.append(str(row["id"]))
        matrix = vectors_matrix(vecs, width or VECTOR_DIM)
        self._vec_cache = (epoch, ids, matrix)
        return ids, matrix
