# -*- coding: utf-8 -*-
"""Local HTTP dashboard for inspecting ourmemory's Three-State Orbit & NeuroGraph."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..graph import REGIONS
from ..kernel import MemoryKernel, default_db_path
from ..soul import DEFAULT_SOUL, SoulSpec

_WEB_DIR = Path(__file__).with_name("web")


def _snapshot(kernel: MemoryKernel, conv_key: str) -> dict:
    now = time.time()
    rows = kernel.db.query(
        "SELECT id, conv_key, layer, subject, slot, value, text, importance, "
        "confidence, status, orbit, ts, valid_from, valid_to, source "
        "FROM memories WHERE conv_key=? ORDER BY ts DESC LIMIT 200",
        (conv_key,),
    )
    memories = [dict(r) for r in rows]
    entities = [
        dict(r)
        for r in kernel.db.query(
            "SELECT id, name, norm, type, salience, mentions, last_seen "
            "FROM entities ORDER BY salience DESC, mentions DESC LIMIT 80"
        )
    ]
    edges = [
        dict(r)
        for r in kernel.db.query(
            "SELECT id, src, dst, rel, weight FROM edges "
            "WHERE valid_to IS NULL AND weight > 0 ORDER BY weight DESC LIMIT 160"
        )
    ]
    audits = [
        dict(r)
        for r in kernel.db.query(
            "SELECT id AS seq, ts, conv_key, action, memory_id, subject, slot, "
            "value AS new_value, orbit, reason, prev_hash, hash AS row_hash "
            "FROM audit_log WHERE conv_key=? ORDER BY id DESC LIMIT 60",
            (conv_key,),
        )
    ]
    audit_ok, audit_count = kernel.verify_audit(conv_key)
    convs = [
        str(r["conv_key"])
        for r in kernel.db.query(
            "SELECT DISTINCT conv_key FROM memories ORDER BY conv_key"
        )
    ]
    if conv_key not in convs:
        convs.insert(0, conv_key)

    return {
        "conv_key": conv_key,
        "conversations": convs,
        "stats": kernel.stats(conv_key),
        "audit": {"verified": audit_ok, "count": audit_count, "records": audits},
        "regions": REGIONS,
        "core_card": kernel.core_card(conv_key, now=now),
        "state_block": kernel.state_block(conv_key, now=now),
        "scene_canvas": kernel.scene_canvas(conv_key, now=now),
        "memories": memories,
        "entities": entities,
        "edges": edges,
    }


def make_handler(kernel: MemoryKernel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                html_path = _WEB_DIR / "index.html"
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            qs = parse_qs(parsed.query)
            conv_key = (qs.get("conv_key") or ["demo:companion"])[0]

            if parsed.path == "/api/snapshot":
                self._send_json(_snapshot(kernel, conv_key))
                return

            if parsed.path == "/api/search":
                q = (qs.get("q") or [""])[0]
                hits = kernel.search(conv_key, q, limit=8)
                self._send_json({
                    "query": q,
                    "workspace": kernel.awaken(conv_key, q, memory_question=True),
                    "hits": [
                        {
                            "memory_id": h.memory_id,
                            "text": h.text,
                            "layer": h.layer,
                            "orbit": h.orbit,
                            "score": round(h.score, 4),
                            "certainty": round(h.certainty, 3),
                            "routes": h.routes,
                        }
                        for h in hits
                    ],
                })
                return

            self._send_json({"error": "not_found"}, status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                data = json.loads(raw)
            except Exception:
                data = {}
            conv_key = str(data.get("conv_key") or "demo:companion")

            if parsed.path == "/api/observe":
                u = str(data.get("user_text") or "")
                b = str(data.get("bot_text") or "")
                ids = kernel.observe(conv_key, u, b)
                self._send_json({"ok": True, "written_ids": ids, "snapshot": _snapshot(kernel, conv_key)})
                return

            if parsed.path == "/api/promote":
                mid = str(data.get("memory_id") or "")
                ok = kernel.store.promote_orbit(mid, reason="dashboard_promote")
                self._send_json({"ok": ok, "snapshot": _snapshot(kernel, conv_key)})
                return

            if parsed.path == "/api/void":
                mid = str(data.get("memory_id") or "")
                ok = kernel.store.void_memory(mid, reason="dashboard_void")
                self._send_json({"ok": ok, "snapshot": _snapshot(kernel, conv_key)})
                return

            self._send_json({"error": "not_found"}, status=404)

    return Handler


def run_dashboard(
    db_path: str | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    soul: SoulSpec = DEFAULT_SOUL,
) -> None:
    path = db_path or default_db_path()
    kernel = MemoryKernel(path, soul=soul)
    server = ThreadingHTTPServer((host, port), make_handler(kernel))
    print(f"ourmemory dashboard listening on http://{host}:{port} (db={path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        kernel.close()
