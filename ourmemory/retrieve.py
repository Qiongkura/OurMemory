# -*- coding: utf-8 -*-
"""Four-Way Hybrid Retrieval (FTS5 + Vector + NeuroGraph + Time Window RRF)."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from .soul import DEFAULT_SOUL, SoulSpec
from .text import (
    hash_vector, normalize, rank_vectors, token_set, vector_width,
)

_WEIGHTS = {
    "fts": 1.0,
    "vector": 1.2,
    "graph": 0.5,
    "time": 0.85,
}
_RRF_K = 60.0
_VECTOR_FLOOR = 0.42
_VECTOR_STRONG = 0.72

_TIME_HINT_RE = re.compile(
    r"昨天|昨晚|前天|上次|之前|那天|上周|上个月|以前|当时|那时候|"
    r"今天|今晚|刚才|刚刚|最近|这几天|"
    r"\b(?:today|tonight|yesterday|last night|this morning|this afternoon|"
    r"this evening|earlier|just now|recently|last week|last month)\b",
    re.IGNORECASE,
)
_DAYS_RE = re.compile(r"(\d+)\s*(?:天前|days?\s+ago)", re.IGNORECASE)
_HOURS_RE = re.compile(r"(\d+)\s*(?:个)?(?:小时前|hours?\s+ago)", re.IGNORECASE)

_TIME_WORDS = frozenset({
    "昨天", "昨晚", "前天", "上次", "之前", "那天", "上周", "上个月",
    "以前", "当时", "那时候", "今天", "今晚", "刚才", "刚刚", "最近",
    "这几天", "天前", "小时", "时候",
    "today", "tonight", "yesterday", "earlier", "recently", "ago",
    "morning", "afternoon", "evening", "night", "week", "month",
})

# Activity inquiry words in questions like "我们今天干了啥" / "昨天做了什么"
# ask *what happened* in the time window; they are not literal content tokens
# that must appear verbatim inside the episodic memory text.
_ACTIVITY_INQUIRY_BIGRAMS = frozenset({
    "干了", "了啥", "做了", "了什", "做什", "干什", "干嘛", "嘛了",
    "发生", "生了", "些什", "么事", "哪些", "做过", "去哪", "去了",
    "吃啥", "吃了", "玩了", "聊了",
})
_ACTIVITY_INQUIRY_RE = re.compile(
    r"我们|你们|咱们|都|到底|究竟|"
    r"干了(?:什么|啥)|做了(?:什么|啥)|干嘛了|做什么了|干什么了|"
    r"发生了(?:什么|啥)|发生(?:什么|啥)|做过(?:什么|啥)|"
    r"去了(?:哪里|哪儿|哪)|去哪(?:里|儿)?了|玩了(?:什么|啥)|聊了(?:什么|啥)|"
    r"\bwhat\s+(?:did|have)\s+(?:we|you|i)\s+(?:do|done|been\s+up\s+to)\b|"
    r"\bwhat\s+happened\b",
    re.IGNORECASE,
)

_PROSPECTIVE_QUERY_RE = re.compile(
    r"(?:明天|后天|待会|待会儿|等下|等会儿|接下来|下周|周末|以后)"
    r".{0,8}(?:干嘛|干啥|做什么|干什么|去哪|安排|计划|打算)|"
    r"有什么(?:计划|安排|打算)|"
    r"\bwhat\s+(?:are|will)\s+(?:you|we)\s+(?:doing|do)\b|"
    r"\b(?:any\s+)?plans?\s+for\b",
    re.IGNORECASE,
)

_SLOT_SYNONYMS = {
    "多高": "身高", "身高": "身高", "几米": "身高",
    "多重": "体重", "体重": "体重", "多少斤": "体重", "多少公斤": "体重",
    "名字": "名字", "叫什么": "名字", "叫啥": "名字", "谁": "名字",
    "我叫": "名字", "他叫": "名字", "叫什": "名字", "么名": "名字",
    "学校": "学校", "哪个学校": "学校", "哪所学校": "学校", "读哪": "学校",
    "专业": "专业", "学什么": "专业",
    "住": "所在城市", "哪里人": "所在城市", "哪个城市": "所在城市",
    "城市": "所在城市", "在哪": "位置", "位置": "位置",
    "生日": "生日", "几岁": "年龄", "多大": "年龄", "年龄": "年龄",
    "喜欢": "喜欢", "喜好": "喜欢", "偏好": "喜欢", "爱好": "喜欢",
    "他喜": "喜欢", "欢什": "喜欢",
    "讨厌": "讨厌", "关系": "关系", "什么关系": "关系",
}

_SLOT_EVIDENCE = {
    "身高": ("身高", "cm", "厘米", "米"),
    "体重": ("体重", "kg", "公斤", "斤"),
    "名字": ("叫", "名字", "自称", "昵称"),
    "学校": ("学校", "大学", "学院", "读"),
    "专业": ("专业", "系"),
    "所在城市": ("城市", "住在", "在"),
    "年龄": ("岁", "年龄", "生日"),
    "喜欢": ("喜欢", "爱", "爱好"),
    "讨厌": ("讨厌", "不喜欢", "受不了"),
    "关系": ("关系", "约定", "在一起"),
}

_JUNK_BIGRAMS = frozenset({
    "天怎", "么样", "怎么", "什么", "天说", "说什", "过什",
    "前说", "天前", "这怎", "那怎", "有没", "是不是", "要不要",
    "天干", "在干", "干嘛", "干啥", "天做", "在做", "天在", "天有",
    "天吃", "天去", "天想", "天过", "过得", "得怎",
})

_GENERIC = frozenset({
    "今天", "明天", "昨天", "现在", "以前", "最近", "什么", "怎么", "怎么样",
    "喜欢", "知道", "觉得", "可能", "是不是", "那个", "这个", "可以", "我们",
    "你说", "我说", "时候", "事情", "一个", "一下", "记得", "告诉", "认为",
    "还是", "就是", "真的", "自己", "已经", "没有", "不是", "然后", "因为",
    "主人", "对方", "用户", "你", "我", "他", "她",
    "在哪", "哪里", "哪儿", "那里", "这里", "刚才", "刚刚", "到底",
    # English function words: without these, "what did we do today" treats
    # every word as a content term and matches nothing meaningful.
    "the", "and", "for", "you", "your", "are", "was", "were", "this", "that",
    "with", "from", "have", "has", "had", "not", "but", "all", "can", "just",
    "now", "will", "would", "what", "did", "does", "doing", "done", "we", "us",
    "our", "me", "my", "i", "do", "today", "tonight", "yesterday", "about",
    "there", "here", "when", "where", "who", "how", "why", "any", "some",
})

_ATTRIBUTE_TRIGGERS = {
    "food": re.compile(
        r"吃|喝|菜|饭|水果|零食|口味|忌口|奶茶|咖啡|酒|辣|甜|食堂|外卖|火锅"
    ),
    "music": re.compile(r"歌|音乐|专辑|歌手|乐队|唱"),
    "color": re.compile(r"颜色|色系|红|蓝|绿|黄|紫|粉|黑白"),
    "intimacy": re.compile(
        r"亲密|摸|抱|亲|胸|私密|身体|接触|"
        r"性(?:相关|话题|方面|生活|欲|爱|行为|需求)"
    ),
    "nickname": re.compile(r"叫啥|称呼|昵称|喊我|叫我"),
    "play": re.compile(r"逗|欺负|闹着玩"),
}

_ATTRIBUTE_EVIDENCE = {
    "food": ("吃", "喝", "菜", "饭", "水果", "零食", "口味", "奶茶", "咖啡",
             "酒", "辣", "甜", "火锅", "外卖", "食堂", "忌口", "过敏"),
    "music": ("歌", "音乐", "专辑", "歌手", "乐队", "唱"),
    "color": ("颜色", "色", "红", "蓝", "绿", "黄", "紫", "粉", "黑", "白"),
    "intimacy": ("亲密", "摸", "抱", "亲", "胸", "性", "私密", "身体", "接触"),
    "nickname": ("叫", "称呼", "昵称", "宝宝"),
    "play": ("逗", "欺负", "玩", "闹"),
}

_SLOT_WORDS = frozenset({
    "喜欢", "喜好", "偏好", "爱好", "讨厌", "名字", "学校", "专业", "身高",
    "体重", "年龄", "生日", "关系", "城市", "工作", "家乡", "习惯", "作息",
})

_SLOT_CUES = {
    "name": ("名字", "叫什么", "叫啥"),
    "nickname": ("叫我", "称呼", "昵称"),
    "hometown": ("老家", "家乡", "哪里人"),
    "city": ("城市", "哪个城市"),
    "home": ("住哪", "住处", "住在"),
    "school": ("学校", "大学", "上学", "哪个学校"),
    "major": ("专业",),
    "grade": ("年级", "大几", "几年级"),
    "work": ("工作", "上班"),
    "job": ("职业", "做什么工作"),
    "age": ("年龄", "多大", "几岁"),
    "birthday": ("生日",),
    "height": ("身高", "多高"),
    "weight": ("体重", "多重"),
    "pet": ("宠物", "养了什么"),
    "preference": ("喜欢", "喜好", "偏好", "爱好", "讨厌"),
    "relationship": ("关系", "在一起"),
    "boundary": ("边界", "底线"),
    "location": ("在哪", "哪里", "位置"),
    "activity": ("在干嘛", "做什么", "忙什么"),
    "mood": ("心情", "开心", "难过"),
}

_PRONOUNS = frozenset("我你他她它咱您")


@dataclass
class Hit:
    memory_id: str
    text: str
    ts: float
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    kind: str = ""
    slot: str = ""
    subject: str = ""
    orbit: str = "confirmed"
    evidence: float = -1.0

    @property
    def certainty(self) -> float:
        return self.score if self.evidence < 0 else self.evidence

    @property
    def confidence(self) -> str:
        if self.certainty >= 0.75:
            return "high"
        if self.certainty >= 0.45:
            return "medium"
        return "low"

    @property
    def layer(self) -> str:
        return self.kind

    @property
    def routes(self) -> list[str]:
        return self.sources


RetrievalHit = Hit


def _query_attributes(query: str) -> set[str]:
    text = str(query or "")
    return {
        name for name, pattern in _ATTRIBUTE_TRIGGERS.items()
        if pattern.search(text)
    }


def _slot_named(query: str, slot: str) -> bool:
    if not slot:
        return False
    base = slot.split(".")[-1]
    return any(cue in query for cue in _SLOT_CUES.get(base, ()))


def _evidence_score(row, query: str, vector_raw: float, route_count: int) -> float:
    value = 0.0
    if vector_raw >= _VECTOR_STRONG:
        value = 0.85
    elif vector_raw >= _VECTOR_FLOOR:
        value = 0.55
    if route_count >= 3:
        value = max(value, 0.8)
    elif route_count == 2:
        value = max(value, 0.6)
    if _slot_named(query, str(row["slot"] or "")):
        value = max(value, 0.9)
    if "time" in query or _TIME_HINT_RE.search(query):
        value = max(value, 0.78)
    memory_value = str(row["value"] or "").strip()
    if len(memory_value) >= 2 and memory_value in query:
        value = max(value, 0.95)
    return round(min(value, 1.0), 3)


def _query_terms(query: str, soul: SoulSpec = DEFAULT_SOUL) -> set[str]:
    stop = _GENERIC | _JUNK_BIGRAMS | soul.all_generic_pronouns()
    terms = token_set(normalize(query)) - stop
    terms = {term for term in terms if not (_PRONOUNS & set(term))}
    terms |= {word for word in _SLOT_WORDS if word in query}
    return {term for term in terms if len(term) >= 2}


def _overlaps(terms: set[str], text: str) -> bool:
    haystack = token_set(normalize(text))
    return bool(terms & haystack)


def _time_content_terms(query: str, soul: SoulSpec = DEFAULT_SOUL) -> set[str]:
    """Extract topic content terms after stripping time hints and activity-inquiry frames."""
    stripped = _TIME_HINT_RE.sub(" ", str(query or ""))
    stripped = _DAYS_RE.sub(" ", stripped)
    stripped = _HOURS_RE.sub(" ", stripped)
    stripped = _ACTIVITY_INQUIRY_RE.sub(" ", stripped)
    for name in soul.all_generic_pronouns():
        stripped = stripped.replace(name, " ")
    return _query_terms(stripped, soul) - _TIME_WORDS - _ACTIVITY_INQUIRY_BIGRAMS


def _time_window(query: str, now: float,
                 tz=None) -> tuple[float, float] | None:
    """Turn "昨天/3天前/上周" into a concrete timestamp window.

    Day-scale windows use real calendar midnights in ``tz`` (system local time
    when None), so a server running in UTC still answers "today" for a user in
    Asia/Shanghai.
    """
    from .clock import day_bounds

    text = str(query or "")
    match = _DAYS_RE.search(text)
    if match:
        days = int(match.group(1))
        return day_bounds(now, tz, offset_days=-days)
    match = _HOURS_RE.search(text)
    if match:
        hours = int(match.group(1))
        end = now - hours * 3600
        return end - 3600, end + 3600
    lowered = text.lower()
    if "昨天" in text or "昨晚" in text or "yesterday" in lowered:
        return day_bounds(now, tz, offset_days=-1)
    if "前天" in text:
        return day_bounds(now, tz, offset_days=-2)
    if "上周" in text or "last week" in lowered:
        return now - 14 * 86400, now - 7 * 86400
    if "上个月" in text or "last month" in lowered:
        return now - 60 * 86400, now - 30 * 86400
    if "今天" in text or "今晚" in text or "today" in lowered or "tonight" in lowered:
        start, _end = day_bounds(now, tz, offset_days=0)
        return start, now + 3600
    if "刚才" in text or "刚刚" in text or "just now" in lowered or "earlier" in lowered:
        return now - 3600, now + 60
    if "最近" in text or "这几天" in text or "recently" in lowered:
        return now - 7 * 86400, now + 60
    if _TIME_HINT_RE.search(text):
        return now - 30 * 86400, now + 60
    return None


class HybridRetriever:
    """Four-way recall fused with weighted reciprocal rank fusion (RRF)."""

    def __init__(self, store, conv_key: str, *, embedder=None, logger=None,
                 mood=None, soul: SoulSpec = DEFAULT_SOUL, tz=None):
        self.store = store
        self.conv_key = conv_key
        self.embedder = embedder
        self.logger = logger or (lambda message: None)
        self.mood = mood
        self.soul = soul
        self.tz = tz

    def _fts(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = _query_terms(query, self.soul) - _ACTIVITY_INQUIRY_BIGRAMS
        if not terms:
            return []
        search_terms = set(terms)
        for term in terms:
            synonym = _SLOT_SYNONYMS.get(term)
            if synonym:
                search_terms.add(synonym)
        try:
            rows = self.store.search_fts_terms(sorted(search_terms), limit=limit)
        except Exception as exc:  # noqa: BLE001
            self.logger("fts recall failed: " + type(exc).__name__)
            return []
        if not rows:
            return []
        try:
            weights = self.store.idf(terms)
        except Exception:  # noqa: BLE001
            weights = {}
        self._preload([memory_id for memory_id, _raw in rows])
        scored = []
        for position, (memory_id, _raw) in enumerate(rows):
            row = self._row(memory_id)
            text = str(row["text"] or "") if row is not None else ""
            present = token_set(text) & terms
            weight = sum(weights.get(term, 1.0) for term in present)
            scored.append((weight, -position, memory_id))
        scored.sort(reverse=True)
        return [(memory_id, weight) for weight, _pos, memory_id in scored]

    def _vector(self, query: str, limit: int) -> list[tuple[str, float]]:
        try:
            if self.embedder is not None:
                vectors = self.embedder([query])
                if not vectors:
                    return []
                query_vec = vectors[0]
            else:
                # Built-in signed feature hashing: the vector route still works
                # offline, with no model download and no embedding service.
                query_vec = hash_vector(query)
            ids, matrix = self.store.vectors()
            if not ids or matrix is None or len(matrix) == 0:
                return []
            if vector_width(matrix) != len(query_vec):
                return []
            return rank_vectors(query_vec, ids, matrix, limit, _VECTOR_FLOOR)
        except Exception as exc:  # noqa: BLE001
            self.logger("vector recall failed: " + type(exc).__name__)
            return []

    def _graph(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = _query_terms(query, self.soul) - _ACTIVITY_INQUIRY_BIGRAMS
        if not terms:
            return []
        marks = ",".join("?" for _ in terms)
        try:
            rows = self.store.db.query(
                "SELECT m.memory_id, SUM(m.weight) AS hits "
                "FROM mentions m JOIN entities e ON e.id = m.entity_id "
                "JOIN memories mem ON mem.id = m.memory_id "
                "WHERE e.norm IN ({}) AND mem.conv_key=? "
                "AND mem.status='active' AND mem.orbit<>'voided' "
                "GROUP BY m.memory_id ORDER BY hits DESC LIMIT ?".format(marks),
                (*sorted(terms), self.conv_key, limit),
            )
            return [(str(row["memory_id"]), float(row["hits"])) for row in rows]
        except Exception as exc:  # noqa: BLE001
            self.logger("graph recall failed: " + type(exc).__name__)
            return []

    def _time(self, query: str, limit: int, now: float) -> list[tuple[str, float]]:
        """Time-window and prospective recall.

        Handles both:
        1. Prospective queries ("你明天干嘛", "接下来做什么") -> active
           ``prospective`` and ``next`` memories.
        2. Past/today window queries ("昨天", "我们今天干了啥") -> memories in
           the window.  Small talk ("今天怎么样") is still rejected because
           "怎么样" is not an activity inquiry phrase.
        """
        if _PROSPECTIVE_QUERY_RE.search(query):
            try:
                rows = self.store.db.query(
                    "SELECT id FROM memories WHERE conv_key=? AND archived=0 "
                    "AND status='active' AND orbit<>'voided' "
                    "AND (layer='prospective' OR slot IN ('next', 'bot.next')) "
                    "ORDER BY ts DESC LIMIT ?",
                    (self.conv_key, limit),
                )
                return [(str(row["id"]), 1.0) for row in rows]
            except Exception:
                return []

        window = _time_window(query, now, self.tz)
        if window is None:
            return []
        explicit_time = (
            bool(_TIME_HINT_RE.search(query))
            or bool(_DAYS_RE.search(query))
            or bool(_HOURS_RE.search(query))
        )
        if not explicit_time:
            return []
        content = _time_content_terms(query, self.soul)
        if not content:
            residue = _TIME_HINT_RE.sub("", query)
            residue = _DAYS_RE.sub("", residue)
            residue = _HOURS_RE.sub("", residue)
            residue = _ACTIVITY_INQUIRY_RE.sub("", residue)
            for name in self.soul.all_generic_pronouns():
                residue = residue.replace(name, "")
            residue = re.sub(
                r"[\s，,。！？!?、；;：:了过有说的吗呢吧啊呀么]+", "", residue
            )
            if residue:
                return []
        start, end = window
        try:
            rows = self.store.db.query(
                "SELECT id, text, slot, ts FROM memories WHERE conv_key=? AND archived=0 "
                "AND status='active' AND orbit<>'voided' "
                "AND ts>=? AND ts<=? ORDER BY ts DESC LIMIT ?",
                (self.conv_key, start, end, limit),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger("time recall failed: " + type(exc).__name__)
            return []
        if not content:
            # A bare "what did we do today" asks for the day's episodes and
            # scene, not every identity slot written today (name, school...).
            transient = {
                "location", "activity", "mood", "with", "next",
                "bot.location", "bot.activity", "bot.mood", "bot.with", "bot.next",
                "user.location", "user.activity", "user.mood",
            }
            rows = [
                row for row in rows
                if not row["slot"] or row["slot"] in transient
            ]
            # Prefer episodic and slotless memories when summarizing a day
            ordered = sorted(
                rows,
                key=lambda r: (1 if not r["slot"] else 0, float(r["ts"] or 0)),
                reverse=True,
            )
            return [(str(row["id"]), 1.0) for row in ordered]
        out = []
        for row in rows:
            text = str(row["text"] or "")
            if not _overlaps(content, text):
                continue
            out.append((str(row["id"]), 1.0))
        return out

    def _row_has_attribute(self, memory_id: str, attrs: set[str]) -> bool:
        row = self._row(memory_id)
        if row is None:
            return False
        text = str(row["text"] or "") + " " + str(row["value"] or "")
        return any(
            word in text
            for attr in attrs
            for word in _ATTRIBUTE_EVIDENCE.get(attr, ())
        )

    def _row_overlaps(self, memory_id: str, content: set[str]) -> bool:
        row = self._row(memory_id)
        if row is None:
            return False
        text = str(row["text"] or "")
        if _overlaps(content, text):
            return True
        slot = str(row["slot"] or "").split(".")[-1]
        label = str(row["value"] or "")
        for term in content:
            synonym = _SLOT_SYNONYMS.get(term)
            if not synonym:
                continue
            if slot and (slot == synonym or label == synonym):
                return True
            evidence = _SLOT_EVIDENCE.get(synonym, ())
            if any(word in text for word in evidence):
                return True
        return False

    def _preload(self, memory_ids) -> None:
        cache = getattr(self, "_row_cache", None)
        if cache is None:
            cache = self._row_cache = {}
        missing = [mid for mid in memory_ids if mid not in cache]
        if not missing:
            return
        try:
            rows = self.store.db.query(
                "SELECT * FROM memories WHERE id IN ({})".format(
                    ",".join("?" for _ in missing)
                ),
                tuple(missing),
            )
            for row in rows:
                cache[str(row["id"])] = row
        except Exception:
            pass
        for mid in missing:
            cache.setdefault(mid, None)

    def _row(self, memory_id: str):
        cache = getattr(self, "_row_cache", None)
        if cache is None:
            cache = self._row_cache = {}
        if memory_id in cache:
            return cache[memory_id]
        try:
            row = self.store.db.query_one(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            )
        except Exception:
            row = None
        cache[memory_id] = row
        return row

    def search(self, query: str, *, limit: int = 6, now: float | None = None,
               per_route: int = 40) -> list[Hit]:
        now = time.time() if now is None else float(now)
        if not normalize(query):
            return []
        self._row_cache = {}
        has_window = _time_window(query, now, self.tz) is not None
        prospective = bool(_PROSPECTIVE_QUERY_RE.search(query))
        content = (
            _query_terms(query, self.soul)
            - _TIME_WORDS
            - _ACTIVITY_INQUIRY_BIGRAMS
        )
        if not content and not has_window and not prospective:
            return []
        routes = {
            "fts": self._fts(query, per_route),
            "vector": self._vector(query, per_route),
            "graph": self._graph(query, per_route),
            "time": self._time(query, per_route, now),
        }
        if not content and (has_window or prospective):
            # "What did we do today?" names no topic: the answer is the day's
            # timeline, not every memory that happens to share a pronoun.
            routes = {"fts": [], "vector": [], "graph": [], "time": routes["time"]}
        attrs = _query_attributes(query)
        if attrs:
            self._preload({
                memory_id for ranked in routes.values() for memory_id, _raw in ranked
            })
            for name in ("fts", "graph", "time"):
                routes[name] = [
                    pair for pair in routes[name]
                    if self._row_has_attribute(pair[0], attrs)
                ]
            routes["vector"] = [
                pair for pair in routes["vector"]
                if pair[1] >= _VECTOR_STRONG or self._row_has_attribute(pair[0], attrs)
            ]
        elif content:
            candidates = {pair[0] for pair in routes["vector"]}
            candidates |= {pair[0] for pair in routes["graph"]}
            self._preload(candidates)
            routes["vector"] = [
                pair for pair in routes["vector"]
                if pair[1] >= _VECTOR_STRONG or self._row_overlaps(pair[0], content)
            ]
            routes["graph"] = [
                pair for pair in routes["graph"]
                if self._row_overlaps(pair[0], content)
            ]
        scores: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        vector_raw: dict[str, float] = {}
        for memory_id, raw in routes["vector"]:
            vector_raw[memory_id] = max(vector_raw.get(memory_id, 0.0), float(raw))
        for name, ranked in routes.items():
            weight = _WEIGHTS.get(name, 0.5)
            for position, (memory_id, _raw) in enumerate(ranked):
                scores[memory_id] = scores.get(memory_id, 0.0) + (
                    weight / (_RRF_K + position + 1)
                )
                sources.setdefault(memory_id, []).append(name)
        if not scores:
            return []
        best = max(scores.values())
        rows = self._load(list(scores))
        hits = []
        for memory_id, row in rows.items():
            base = scores.get(memory_id, 0.0) / best
            score = base * _recency_weight(float(row["ts"] or 0), now)
            hits.append(Hit(
                memory_id=memory_id,
                text=str(row["text"] or ""),
                ts=float(row["ts"] or 0),
                score=score,
                sources=sources.get(memory_id, []),
                kind=str(row["kind"] or ""),
                slot=str(row["slot"] or ""),
                subject=str(row["subject"] or ""),
                orbit=str(row["orbit"] or "confirmed"),
                evidence=_evidence_score(
                    row, query, vector_raw.get(memory_id, 0.0),
                    len(sources.get(memory_id, [])),
                ),
            ))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def _load(self, memory_ids: list[str]) -> dict:
        if not memory_ids:
            return {}
        try:
            rows = self.store.db.query(
                "SELECT * FROM memories WHERE id IN ({}) AND archived=0 "
                "AND status='active' AND orbit<>'voided'".format(
                    ",".join("?" for _ in memory_ids)
                ),
                tuple(memory_ids),
            )
            return {str(row["id"]): row for row in rows}
        except Exception as exc:  # noqa: BLE001
            self.logger("hit load failed: " + type(exc).__name__)
            return {}


def _recency_weight(ts: float, now: float) -> float:
    if ts <= 0:
        return 0.6
    age_days = max(0.0, (now - ts) / 86400.0)
    return 0.55 + 0.45 * math.exp(-age_days / 30.0)


def render_hits(hits: list[Hit], *, limit: int = 6, tz=None) -> str:
    if not hits:
        return ""
    from .clock import stamp as _stamp
    lines = []
    seen: set[str] = set()
    for hit in hits:
        key = normalize(hit.text)[:120]
        if key in seen:
            continue
        seen.add(key)
        stamp = _stamp(hit.ts or time.time(), tz)
        marker = "" if hit.confidence == "high" else "（记不太清）"
        orbit_tag = " [待定]" if hit.orbit == "pending" else ""
        lines.append(f"- [{stamp}]{orbit_tag}{marker} {hit.text[:120]}")
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return "【相关记忆】\n" + "\n".join(lines)
