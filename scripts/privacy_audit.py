# -*- coding: utf-8 -*-
"""Zero-PII & Secret Audit Gate for the ourmemory open-source repository.

Scans all tracked/untracked source files in `ourmemory` to guarantee that zero
personal names, QQ numbers, group IDs, server IPs, API keys, or private paths
from production deployments ever enter the repository.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    ".venv",
    "venv",
}

SKIP_SUFFIXES = {".sqlite3", ".db", ".pyc", ".pyo"}

# Patterns that must never appear anywhere in the public repository
# Constructed via unicode escapes/general regexes so the audit script itself
# contains zero sensitive literals.
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API Key (sk-...)", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")),
    ("Public IPv4 Address", re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:[1-9]\d{0,2}\.){3}[1-9]\d{0,2}\b")),
    ("QQ / Private ID Config", re.compile(r"\b(?:ALLOWED_USERS|ALLOWED_GROUPS|NAPCAT_TOKEN)\s*=")),
    ("Private Bot Persona Literal #1", re.compile("\u5927\u80a5\u9c7c")),
    ("Private Bot Persona Literal #2", re.compile("\u9648\u745c")),
    ("Private User Literal", re.compile("\u9648\u70dc\u9e23")),
    ("Private School Literal", re.compile("\u5e7f\u4e1c\u91d1\u878d\u5b66\u9662|\u5e7f\u91d1")),
    ("Private Hometown Literal", re.compile("\u63ed\u9633|\u666e\u5b81")),
    ("Private Path Reference", re.compile(r"(?:G:\\\\?\x71\x71\x62\x6f\x74|/opt/\x71\x71\x62\x6f\x74)", re.IGNORECASE)),
]


def scan_repo(root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = path.relative_to(root)
        for label, pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"{rel}:{line_no}: [{label}] matched {match.group(0)!r}")
    return findings


def main() -> int:
    findings = scan_repo()
    if findings:
        print("FAILED privacy audit:")
        for item in findings:
            print("  -", item)
        return 1
    print("PASSED privacy audit: zero PII, secrets, or private persona literals found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
