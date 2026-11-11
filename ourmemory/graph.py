# -*- coding: utf-8 -*-
"""NeuroGraph: entity nodes, Hebbian co-activation edges, and PageRank spreading."""

from __future__ import annotations

import time
from collections import defaultdict

from .db import Database
from .text import EntityMention, norm_entity, stable_hash

REGIONS = [
    {"id": "episodes", "name": "情景记忆", "x": 0.50, "y": 0.82, "color": "#fbbf24"},
    {"id": "knowledge", "name": "确认事实", "x": 0.75, "y": 0.66, "color": "#34d399"},
    {"id": "pending", "name": "待定挂账", "x": 0.62, "y": 0.48, "color": "#38bdf8"},
    {"id": "plans", "name": "前瞻计划", "x": 0.78, "y": 0.90, "color": "#fb923c"},
    {"id": "graph", "name": "实体关联", "x": 0.82, "y": 0.32, "color": "#60a5fa"},
    {"id": "audit", "name": "审计链", "x": 0.22, "y": 0.72, "color": "#a78bfa"},
]


def entity_id(norm: str) -> str:
    return "e_" + stable_hash(norm, 12)


class NeuroGraph:
    def __init__(self, db: Database):
        self.db = db

    def ensure_entity(self, mention: EntityMention | str, typ: str = "concept",
                      now: float | None = None) -> str:
        if isinstance(mention, EntityMention):
            name, norm, typ, conf = mention.name, mention.norm, mention.type, mention.confidence
        else:
            name = str(mention)
            norm = norm_entity(name)
            conf = 0.5
        if not norm:
            return ""
        eid = entity_id(norm)
        now = time.time() if now is None else now
        with self.db.tx():
            row = self.db.query_one("SELECT id FROM entities WHERE norm=?", (norm,))
            if row is not None:
                self.db.conn.execute(
                    "UPDATE entities SET last_seen=?, salience=MIN(1.0, salience+?) WHERE id=?",
                    (now, 0.002 * conf, row["id"]),
                )
                return str(row["id"])
            self.db.conn.execute(
                "INSERT INTO entities(id,name,norm,type,aliases,salience,mentions,created,last_seen) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (eid, name[:40], norm, typ, "", min(0.9, 0.35 + conf * 0.4), 0, now, now),
            )
        return eid

    def link_memory(self, memory_id: str, mentions: list[EntityMention],
                    now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        ids: list[str] = []
        with self.db.tx():
            for mention in mentions:
                eid = self.ensure_entity(mention, now=now)
                if not eid:
                    continue
                self.db.conn.execute(
                    "INSERT INTO mentions(memory_id,entity_id,weight) VALUES(?,?,?) "
                    "ON CONFLICT(memory_id,entity_id) DO UPDATE SET weight=MIN(2.0, weight+0.2)",
                    (memory_id, eid, 0.6 + mention.confidence * 0.4),
                )
                self.db.conn.execute(
                    "UPDATE entities SET mentions=mentions+1, last_seen=?, "
                    "salience=MIN(1.0, salience+0.01) WHERE id=?",
                    (now, eid),
                )
                ids.append(eid)
        return ids

    def add_edge(self, src: str, dst: str, rel: str = "co", delta: float = 0.1,
                 src_type: str = "entity", dst_type: str = "entity",
                 now: float | None = None) -> bool:
        if not src or not dst or src == dst:
            return False
        now = time.time() if now is None else now
        delta = max(0.001, float(delta))
        eid = "x_" + stable_hash(f"{src}|{dst}|{rel}", 12)
        with self.db.tx():
            cur = self.db.conn.execute(
                "UPDATE edges SET weight=MIN(3.0, weight*0.97+?), last_fire=?, valid_to=NULL "
                "WHERE src=? AND dst=? AND rel=?",
                (delta, now, src, dst, rel),
            )
            if cur.rowcount:
                return True
            self.db.conn.execute(
                "INSERT INTO edges(id,src,dst,src_type,dst_type,rel,weight,ts,last_fire,"
                "valid_from,valid_to,meta) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (eid, src, dst, src_type, dst_type, rel, delta, now, now, now, None, "{}"),
            )
        return True

    def reinforce_cooccurrence(self, entity_ids: list[str], importance: float = 0.5,
                               now: float | None = None) -> int:
        ids = sorted(set(i for i in entity_ids if i))
        if len(ids) < 2:
            return 0
        delta = 0.06 + 0.10 * max(0.0, min(1.0, importance))
        count = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self.add_edge(ids[i], ids[j], "co", delta, now=now):
                    count += 1
        return count

    def _adjacency(self):
        rows = self.db.query(
            "SELECT src,dst,weight FROM edges "
            "WHERE src_type='entity' AND dst_type='entity' "
            "AND valid_to IS NULL AND weight>0"
        )
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in rows:
            src, dst, w = str(row["src"]), str(row["dst"]), float(row["weight"])
            adj[src].append((dst, w))
            adj[dst].append((src, w))
        return adj

    def spread(self, seeds: dict[str, float], iterations: int = 16,
               damping: float = 0.85) -> dict[str, float]:
        seeds = {k: float(v) for k, v in seeds.items() if k and v > 0}
        if not seeds:
            return {}
        total = sum(seeds.values())
        seeds = {k: v / total for k, v in seeds.items()}
        adj = self._adjacency()
        scores = dict(seeds)
        for _ in range(iterations):
            nxt = {node: (1.0 - damping) * weight for node, weight in seeds.items()}
            for node, value in scores.items():
                neighbors = adj.get(node)
                if not neighbors:
                    continue
                wsum = sum(w for _, w in neighbors) or 1.0
                for target, w in neighbors:
                    nxt[target] = nxt.get(target, 0.0) + damping * value * (w / wsum)
            scores = nxt
        return scores
