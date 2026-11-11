# -*- coding: utf-8 -*-
"""SQLite persistence layer: WAL, FTS5, transactions, and hash-chained audit log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 1,
    conv_key TEXT DEFAULT '',
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_msg_ts ON messages(ts);
CREATE INDEX IF NOT EXISTS idx_msg_key ON messages(conv_key);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    summary TEXT DEFAULT '',
    tokens TEXT DEFAULT '',
    ts REAL NOT NULL,
    valid_from REAL,
    valid_to REAL,
    source TEXT DEFAULT '',
    conv_key TEXT DEFAULT '',
    uid INTEGER,
    gid INTEGER,
    importance REAL DEFAULT 0.5,
    strength REAL DEFAULT 1.0,
    confidence REAL DEFAULT 0.7,
    valence REAL DEFAULT 0.0,
    arousal REAL DEFAULT 0.0,
    emotion TEXT DEFAULT '平静',
    access_count INTEGER DEFAULT 0,
    last_access REAL,
    archived INTEGER DEFAULT 0,
    embedding BLOB,
    ref TEXT,
    hash TEXT DEFAULT '',
    meta TEXT DEFAULT '{}',
    subject TEXT DEFAULT '',
    slot TEXT,
    value TEXT,
    speech_act TEXT DEFAULT '',
    evidence_ids TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    orbit TEXT DEFAULT 'confirmed',
    superseded_by TEXT,
    layer TEXT DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mem_ref ON memories(ref) WHERE ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories(ts);
CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(conv_key);
CREATE INDEX IF NOT EXISTS idx_mem_arch ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(hash);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mem_slot_active
    ON memories(conv_key, subject, slot)
    WHERE status='active' AND slot IS NOT NULL AND slot<>'';
CREATE INDEX IF NOT EXISTS idx_mem_slot ON memories(conv_key, subject, slot);
CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_mem_orbit ON memories(orbit);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_id UNINDEXED,
    tokens,
    tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    norm TEXT NOT NULL UNIQUE,
    type TEXT DEFAULT 'concept',
    aliases TEXT DEFAULT '',
    salience REAL DEFAULT 0.5,
    mentions INTEGER DEFAULT 0,
    created REAL,
    last_seen REAL
);
CREATE INDEX IF NOT EXISTS idx_ent_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_ent_sal ON entities(salience);

CREATE TABLE IF NOT EXISTS mentions (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_men_ent ON mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_men_mem ON mentions(memory_id);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    src_type TEXT DEFAULT 'entity',
    dst_type TEXT DEFAULT 'entity',
    rel TEXT DEFAULT 'co',
    weight REAL DEFAULT 0.1,
    ts REAL,
    last_fire REAL,
    valid_from REAL,
    valid_to REAL,
    meta TEXT DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_edge ON edges(src, dst, rel);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edge_rel ON edges(rel);

CREATE TABLE IF NOT EXISTS token_df (
    token TEXT PRIMARY KEY,
    df INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    conv_key TEXT NOT NULL,
    action TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    subject TEXT DEFAULT '',
    slot TEXT DEFAULT '',
    value TEXT DEFAULT '',
    orbit TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_key ON audit_log(conv_key, id);
"""


class Database:
    """Thread-safe SQLite connection with WAL, FTS5, and schema auto-migration."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0
        self.conn = sqlite3.connect(
            str(self.path), timeout=15.0, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.isolation_level = None
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=15000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        current = int(self.meta_get("schema_version", "0") or 0)
        if current < SCHEMA_VERSION:
            self._ensure_columns()
            self.meta_set("schema_version", str(SCHEMA_VERSION))

    def _ensure_columns(self) -> None:
        existing = {
            str(row["name"])
            for row in self.query("PRAGMA table_info(memories)")
        }
        additions = (
            ("subject", "TEXT DEFAULT ''"),
            ("slot", "TEXT"),
            ("value", "TEXT"),
            ("speech_act", "TEXT DEFAULT ''"),
            ("evidence_ids", "TEXT DEFAULT '[]'"),
            ("status", "TEXT DEFAULT 'active'"),
            ("orbit", "TEXT DEFAULT 'confirmed'"),
            ("superseded_by", "TEXT"),
            ("layer", "TEXT DEFAULT ''"),
        )
        with self.tx():
            for name, decl in additions:
                if name not in existing:
                    self.conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {name} {decl}"
                    )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_orbit ON memories(orbit)"
            )

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows) -> None:
        with self._lock:
            self.conn.executemany(sql, rows)

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params=()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params=(), default=None):
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    @contextmanager
    def tx(self):
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._depth = 0
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            else:
                self._depth = 0
                self.conn.execute("COMMIT")

    def meta_get(self, key: str, default=None):
        row = self.query_one("SELECT v FROM meta WHERE k=?", (key,))
        return default if row is None else row["v"]

    def meta_set(self, key: str, value: str) -> None:
        with self.tx():
            self.conn.execute(
                "INSERT INTO meta(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, str(value)),
            )

    def meta_json(self, key: str, default=None):
        raw = self.meta_get(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default

    def meta_set_json(self, key: str, value) -> None:
        self.meta_set(key, json.dumps(value, ensure_ascii=False))

    def record_audit(
        self,
        conv_key: str,
        action: str,
        memory_id: str,
        *,
        subject: str = "",
        slot: str = "",
        value: str = "",
        orbit: str = "",
        reason: str = "",
        ts: float | None = None,
    ) -> str:
        """Append a hash-chained audit record for any memory state change."""
        ts = time.time() if ts is None else float(ts)
        with self.tx():
            last = self.query_one(
                "SELECT hash FROM audit_log WHERE conv_key=? ORDER BY id DESC LIMIT 1",
                (conv_key,),
            )
            prev_hash = str(last["hash"]) if last else "0" * 16
            payload = f"{prev_hash}|{ts:.3f}|{conv_key}|{action}|{memory_id}|{subject}|{slot}|{value}|{orbit}|{reason}"
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            self.conn.execute(
                "INSERT INTO audit_log(ts,conv_key,action,memory_id,subject,slot,value,orbit,reason,prev_hash,hash) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (ts, conv_key, action, memory_id, subject, slot, value, orbit, reason, prev_hash, digest),
            )
        return digest

    def verify_audit_chain(self, conv_key: str) -> bool:
        """Verify that no audit row in ``conv_key`` was tampered with."""
        rows = self.query(
            "SELECT * FROM audit_log WHERE conv_key=? ORDER BY id ASC",
            (conv_key,),
        )
        expected_prev = "0" * 16
        for row in rows:
            if str(row["prev_hash"]) != expected_prev:
                return False
            payload = (
                f"{expected_prev}|{float(row['ts']):.3f}|{row['conv_key']}|"
                f"{row['action']}|{row['memory_id']}|{row['subject']}|"
                f"{row['slot']}|{row['value']}|{row['orbit']}|{row['reason']}"
            )
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            if str(row["hash"]) != digest:
                return False
            expected_prev = digest
        return True

    def fts_upsert(self, memory_id: str, tokens: str) -> None:
        with self.tx():
            self.conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))
            self.conn.execute(
                "INSERT INTO memory_fts(memory_id, tokens) VALUES(?,?)",
                (memory_id, tokens),
            )

    def fts_delete(self, memory_id: str) -> None:
        self.conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))

    def fts_rebuild(self, rows) -> None:
        with self.tx():
            self.conn.execute("DELETE FROM memory_fts")
            self.conn.executemany(
                "INSERT INTO memory_fts(memory_id, tokens) VALUES(?,?)", list(rows)
            )

    def close(self) -> None:
        with self._lock:
            self.conn.close()
