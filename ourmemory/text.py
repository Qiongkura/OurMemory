# -*- coding: utf-8 -*-
"""Chinese-friendly text processing: normalization, bigrams, entities, hash vectors."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

try:  # optional accelerator; the kernel runs on the standard library alone
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in no-numpy environments
    np = None

VECTOR_DIM = 384

CJK = r"\u4e00-\u9fff"
_TOKEN_RE = re.compile(rf"[{CJK}]+|[A-Za-z][A-Za-z0-9_+#.\-]*|\d+")
_ASCII_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]{1,24}")
_QUOTE_RE = re.compile(r"[“\"「『《]([^”\"」』》\n]{1,20})[”\"」』》]")
_CALL_RE = re.compile(
    r"(?:我叫|名字叫|昵称叫|叫他|叫她|叫它|名叫|叫做)\s*"
    r"([\u4e00-\u9fffA-Za-z0-9_]{1,12})"
)
PERSON_SUFFIXES = (
    "老师", "同学", "哥哥", "姐姐", "弟弟", "妹妹", "老板", "医生",
    "阿姨", "叔叔", "师傅", "学长", "学姐", "主任", "队长", "太太", "先生",
)
_PERSON_RE = re.compile(rf"([{CJK}]{{1,2}})({'|'.join(PERSON_SUFFIXES)})")

STOP_ENTITIES = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这个", "那个",
    "什么", "怎么", "为什么", "现在", "今天", "明天", "昨天", "刚才", "然后",
    "但是", "因为", "所以", "如果", "可以", "一个", "一下", "真的", "不是",
    "没有", "就是", "还是", "已经", "东西", "事情", "时候", "地方", "感觉",
    "用户", "对方", "主人",
}
STOP_PREFIX = {
    "的", "了", "是", "和", "跟", "给", "在", "我", "你", "他", "她", "它",
    "找", "问", "对", "把", "被", "快向", "向", "让", "叫", "请", "替", "帮",
    "多", "少", "再", "又", "就", "才", "都", "也",
}

ASCII_STOP = {
    "the", "and", "for", "you", "are", "this", "that", "with", "from", "have",
    "not", "but", "all", "can", "just", "now", "was", "were", "will", "would",
    "http", "https", "www", "com", "cn", "jpg", "jpeg", "png", "gif", "webp",
    "mp3", "wav", "flac", "m4a", "ogg", "txt", "json", "html", "exe", "zip",
}

KNOWN_ENTITIES = {
    "猫": "animal", "狗": "animal", "兔子": "animal", "仓鼠": "animal",
    "火锅": "food", "重庆火锅": "food", "串串": "food", "奶茶": "food",
    "咖啡": "food", "米饭": "food", "烧烤": "food", "冰淇淋": "food",
    "外卖": "food", "食堂": "place",
    "学校": "place", "宿舍": "place", "图书馆": "place", "实验室": "place",
    "公园": "place", "商场": "place",
    "考试": "event", "作业": "event", "上课": "event", "生日": "event",
    "纪念日": "event", "电影": "activity", "散步": "activity",
}

QUESTION_HINTS = (
    "什么", "怎么", "为什么", "为啥", "哪", "谁", "几点", "多少", "是不是",
    "能不能", "可不可以", "要不要", "有没有", "好不好", "行不行", "如何",
    "多久", "几个", "干嘛", "在吗",
)
COMMAND_HINTS = (
    "记住", "忘记", "唱", "发", "说", "来", "帮我", "给我", "查", "搜",
    "打开", "关", "叫", "提醒", "别忘", "教我", "讲",
)
GREETING_HINTS = ("在吗", "在干嘛", "早安", "晚安", "你好", "hello", "hi", "嗨")

EXPLICIT_MARKERS = (
    "记住", "记一下", "别忘了", "提醒我", "约定", "答应我", "说好",
    "重要", "第一次", "生日", "纪念日",
)


@dataclass(frozen=True)
class EntityMention:
    name: str
    norm: str
    type: str = "concept"
    confidence: float = 0.6


_CJK_PUNCTUATION = {
    ",": "，", ":": "：", ";": "；", "?": "？", "!": "！",
    "(": "（", ")": "）", "[": "【", "]": "】",
}
_HAS_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _restore_cjk_punctuation(text: str) -> str:
    if not _HAS_CJK_RE.search(text):
        return text
    out = []
    for index, char in enumerate(text):
        if char in _CJK_PUNCTUATION:
            before = text[index - 1] if index else ""
            after = text[index + 1] if index + 1 < len(text) else ""
            if not (_HAS_CJK_RE.match(before or "") or _HAS_CJK_RE.match(after or "")):
                out.append(char)
                continue
            if char == ":" and before.isdigit() and after.isdigit():
                out.append(char)
                continue
            if char == "," and before.isdigit() and after.isdigit():
                out.append(char)
                continue
            out.append(_CJK_PUNCTUATION[char])
        else:
            out.append(char)
    return "".join(out)


def normalize(text) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _restore_cjk_punctuation(text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def norm_entity(name: str) -> str:
    s = normalize(name)
    s = s.strip(" \t\r\n，。！？!?、；;：:（）()[]【】<>《》\"'“”‘’")
    if re.match(r"^[A-Za-z]", s):
        s = s.lower()
    return s[:24]


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(str(text).encode("utf-8", "ignore")).hexdigest()[:length]


def _iter_tokens(text: str):
    for m in _TOKEN_RE.finditer(normalize(text)):
        piece = m.group(0)
        if re.match(rf"^[{CJK}]+$", piece):
            if len(piece) == 1:
                yield piece
            else:
                for i in range(len(piece) - 1):
                    yield piece[i:i + 2]
        elif piece.isdigit():
            if len(piece) >= 2:
                yield piece
        else:
            word = piece.lower().strip(".-")
            if len(word) >= 2 and word not in ASCII_STOP:
                yield word


def tokens(text: str) -> list[str]:
    return list(_iter_tokens(text))


def token_set(text: str) -> set[str]:
    return set(_iter_tokens(text))


def fts_text(text: str) -> str:
    seen, out = set(), []
    for t in _iter_tokens(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)


def fts_query(text: str, max_terms: int = 32) -> str:
    terms = []
    seen = set()
    for t in _iter_tokens(text):
        if t in seen:
            continue
        seen.add(t)
        if '"' in t or "'" in t:
            continue
        terms.append(t)
        if len(terms) >= max_terms:
            break
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


def vector_tokens(text: str) -> list[str]:
    out = []
    for m in _TOKEN_RE.finditer(normalize(text)):
        piece = m.group(0)
        if re.match(rf"^[{CJK}]+$", piece):
            if len(piece) == 1:
                out.append(piece)
                continue
            for i in range(len(piece) - 1):
                out.append(piece[i:i + 2])
            for i in range(len(piece) - 2):
                out.append(piece[i:i + 3])
        elif piece.isdigit():
            if len(piece) >= 2:
                out.append(piece)
        else:
            word = piece.lower().strip(".-")
            if len(word) >= 2 and word not in ASCII_STOP:
                out.append(word)
    return out


def fts_query_terms(terms) -> str:
    seen, out = set(), []
    for t in terms:
        if not t or t in seen:
            continue
        if '"' in t or "'" in t:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 32:
            break
    return " OR ".join(f'"{t}"' for t in out)


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8", "ignore"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    idx = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return idx, sign


def hash_vector(text: str, idf: dict[str, float] | None = None,
                dim: int = VECTOR_DIM):
    """Signed feature hashing: a deterministic embedding with no model or API.

    Returns a NumPy array when NumPy is installed, otherwise a plain list of
    floats.  Both paths produce the same values, so a store written on one
    machine can be read on another.
    """
    counts = Counter(vector_tokens(text))
    if np is not None:
        vec = np.zeros(dim, dtype=np.float32)
        if not counts:
            return vec
        for token, count in counts.items():
            idx, sign = _hash_token(token, dim)
            weight = 1.0 + math.log(count)
            if idf:
                weight *= float(idf.get(token, 1.0))
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec
    vec_list = [0.0] * dim
    if not counts:
        return vec_list
    for token, count in counts.items():
        idx, sign = _hash_token(token, dim)
        weight = 1.0 + math.log(count)
        if idf:
            weight *= float(idf.get(token, 1.0))
        vec_list[idx] += sign * weight
    norm = math.sqrt(sum(value * value for value in vec_list))
    if norm > 0:
        vec_list = [value / norm for value in vec_list]
    return vec_list


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    if np is not None:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    left = [float(value) for value in a]
    right = [float(value) for value in b]
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(x * y for x, y in zip(left, right))
    na = math.sqrt(sum(x * x for x in left))
    nb = math.sqrt(sum(y * y for y in right))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def vector_width(matrix) -> int:
    """Number of columns in a matrix-like object, NumPy or plain lists."""
    if matrix is None:
        return 0
    if np is not None and hasattr(matrix, "shape"):
        shape = getattr(matrix, "shape", ())
        return int(shape[1]) if len(shape) == 2 else 0
    try:
        if len(matrix) == 0:
            return 0
        return len(matrix[0])
    except TypeError:
        return 0


def vectors_matrix(vecs: list, width: int):
    """Stack row vectors into a matrix, with or without NumPy."""
    if np is not None:
        if not vecs:
            return np.zeros((0, width or VECTOR_DIM), dtype=np.float32)
        return np.vstack([np.asarray(vec, dtype=np.float32) for vec in vecs])
    return [list(vec) for vec in vecs]


def rank_vectors(query_vec, ids: list[str], matrix, limit: int, floor: float):
    """Return ``[(memory_id, cosine)]`` sorted by similarity, above ``floor``."""
    if not ids or matrix is None:
        return []
    if np is not None:
        query = np.asarray(query_vec, dtype=np.float32)
        if matrix.shape[0] == 0 or matrix.shape[1] != query.size:
            return []
        scores = matrix @ query
        order = scores.argsort()[::-1]
        out: list[tuple[str, float]] = []
        for index in order:
            score = float(scores[int(index)])
            if score < floor:
                break
            out.append((ids[int(index)], score))
            if len(out) >= limit:
                break
        return out
    query = [float(value) for value in query_vec]
    query_norm = math.sqrt(sum(value * value for value in query)) or 1.0
    scored: list[tuple[float, str]] = []
    for memory_id, row in zip(ids, matrix):
        if len(row) != len(query):
            continue
        dot = sum(x * y for x, y in zip(row, query))
        row_norm = math.sqrt(sum(value * value for value in row)) or 1.0
        scored.append((dot / (row_norm * query_norm), memory_id))
    scored.sort(reverse=True)
    out = []
    for score, memory_id in scored:
        if score < floor:
            break
        out.append((memory_id, score))
        if len(out) >= limit:
            break
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _known_matches(text: str) -> list[EntityMention]:
    lower = normalize(text).lower()
    spans: list[tuple[int, int]] = []
    found: list[EntityMention] = []
    for name in sorted(KNOWN_ENTITIES, key=len, reverse=True):
        needle = name.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            end = idx + len(needle)
            if not any(s < end and idx < e for s, e in spans):
                spans.append((idx, end))
                found.append(
                    EntityMention(name=name, norm=norm_entity(name),
                                  type=KNOWN_ENTITIES[name], confidence=0.9)
                )
            start = idx + 1
    return found


def extract_entities(text: str, known: list[str] | None = None,
                     limit: int = 10) -> list[EntityMention]:
    raw = normalize(text)
    if not raw:
        return []
    out: list[EntityMention] = []
    out.extend(_known_matches(raw))

    named = [norm_entity(x) for x in (known or []) if norm_entity(x)]
    if named:
        lower = raw.lower()
        for name in sorted(set(named), key=len, reverse=True)[:400]:
            if len(name) >= 2 and name.lower() in lower:
                out.append(EntityMention(name=name, norm=name, type="concept",
                                         confidence=0.5))

    for m in _QUOTE_RE.finditer(raw):
        val = norm_entity(m.group(1))
        if val and val not in STOP_ENTITIES and not val.isdigit():
            out.append(EntityMention(val, val, "concept", 0.85))

    for m in _PERSON_RE.finditer(raw):
        prefix, suffix = m.group(1), m.group(2)
        if prefix in STOP_PREFIX or (prefix in ("大", "小") and len(prefix) == 1):
            continue
        if any(prefix.startswith(x) for x in STOP_PREFIX):
            continue
        if any(x in prefix for x in (
            "向", "让", "请", "帮", "替", "给", "跟", "找", "问", "对", "把",
            "被", "的", "是", "了", "人", "叫", "去", "抓", "记", "得", "和",
            "与", "在", "又", "就", "才", "都", "也",
        )):
            continue
        name = norm_entity(prefix + suffix)
        out.append(EntityMention(name, name, "person", 0.7))

    for m in _CALL_RE.finditer(raw):
        val = norm_entity(m.group(1))
        if val and val not in ("什么", "啥", "哪", "谁") and val not in STOP_ENTITIES:
            out.append(EntityMention(val, val, "person", 0.65))

    for m in _ASCII_RE.finditer(raw):
        word = m.group(0).strip(".-")
        low = word.lower()
        if len(low) < 2 or low in ASCII_STOP or low in STOP_ENTITIES:
            continue
        typ = "tech" if any(ch.isupper() for ch in word) else "concept"
        out.append(EntityMention(word, low, typ, 0.5))

    best: dict[str, EntityMention] = {}
    for item in out:
        key = item.norm
        if not key or key in STOP_ENTITIES or len(key) < 1:
            continue
        if key.isdigit():
            continue
        old = best.get(key)
        if old is None or (item.confidence, len(item.name)) > (old.confidence, len(old.name)):
            best[key] = item
    ranked = sorted(best.values(), key=lambda x: (x.confidence, len(x.name)), reverse=True)
    return ranked[:limit]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def short(text: str, limit: int = 80) -> str:
    s = re.sub(r"\s+", " ", normalize(text))
    return s if len(s) <= limit else s[: max(1, limit - 1)] + "…"
