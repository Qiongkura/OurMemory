# ourmemory

**Stateless, Three-State Memory Kernel for Relational AI & Digital Life**  
**面向关系型 AI 与数字生命的「无状态唤醒 · 三态演进」记忆内核**

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Zero External Runtime Dependencies](https://img.shields.io/badge/Dependencies-Zero%20%28Stdlib%20%2B%20SQLite%29-purple.svg)](pyproject.toml)

---

## Why `ourmemory`? / 为什么需要 `ourmemory`？

Most open-source agent memory libraries (`Mem0`, `Zep`, `MemGPT/Letta`) are designed for **single-subject task assistants** ("remember the user's API preference or meeting notes"). When applied to **long-term relational AI, companions, or digital life**, they fail in five predictable ways:

大多数开源 AI 记忆库是为**单主体工具助手**设计的。一旦用于**长期关系型 AI、陪伴智能体或数字生命**，往往会在以下五个地方崩塌：

1. **Single-Subject Blindness (单主体失明)** — Cannot distinguish *His state* (`user`), *Her state* (`bot`), and *Our shared state* (`both`). When you go to the park together and then come home to cook, legacy memory keeps "planning to go to the park" active forever.
2. **All-or-Nothing Permanence (非黑即白的永久固化)** — A casual one-off remark ("let's get iced milk tea", "my feet hurt from walking", "carry me") gets written into permanent semantic memory and haunts the persona weeks later.
3. **Time-Window Recall Blindness (时间窗口召回失明)** — Asking *"What did we do today?"* (`"我们今天干了啥"`) fails in keyword/BM25 systems because `"干了"` and `"了啥"` are treated as mandatory content tokens rather than a temporal activity inquiry.
4. **Lost Daily Thread (今日现场断片)** — Once morning turns scroll out of the chat window, the AI forgets how the day unfolded unless the current message happens to share keywords with earlier turns.
5. **Ungrounded Life-Event Hallucination (无根据大事幻觉)** — Without an outbound membrane, an LLM under open-ended questioning ("what are you up to?") can suddenly fabricate an internship interview, hospitalization, or trip that never happened, and then reinforce that hallucination in subsequent turns.

---

## Core Architecture / 核心六层架构

`ourmemory` is a pure-Python, zero-dependency (SQLite WAL + FTS5 + 384-dim Blake2b/custom vectors + Hebbian NeuroGraph) memory kernel built around six mechanisms:

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ ① SoulCore & Origin Invariants (源点锚定：身份与关系不变量，防漂移/防覆盖) │
├────────────────────────────────────────────────────────────────────────────┤
│ ② Stateless Awakening (每次调用皆苏醒：Core Card + 此刻状态 + 今日现场画)  │
├────────────────────────────────────────────────────────────────────────────┤
│ ③ Bidirectional Civilization Membrane (文明交互膜：入站去噪 + 出站防幻觉)  │
├────────────────────────────────────────────────────────────────────────────┤
│ ④ Resident Scene Canvas & Dual-Subject State (你 / 我 / 我们 场景自动流转) │
├────────────────────────────────────────────────────────────────────────────┤
│ ⑤ Three-State Orbit (confirmed 确认事实 ⇄ pending 待定挂账 → voided 作废) │
├────────────────────────────────────────────────────────────────────────────┤
│ ⑥ Hash-Chained Audit & Portable Pack (SHA-256 防篡改审计链 + 零锁定导出)   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1. Three-State Orbit (`confirmed` / `pending` / `voided`) · 三态记忆轨
- **`confirmed` (确认轨)**: Durable identity slots, verified preferences, explicit long-term commitments, and shared episodes. Never silently overwritten unless `source="correction"`.
- **`pending` (待定挂账轨)**: Short-term plans (`计划去公园散步`), one-off food/drink orders (`喝奶茶要冰的`), temporary sensations (`走久了脚疼`), and single-turn banter (`提出背对方`). Carries an automatic `valid_to` TTL.
  - **Auto-Promotion (跨轮复述转正)**: If a `pending` preference is restated across turns (`promote_orbit`), it promotes to `confirmed`.
  - **Auto-Retirement (场景完成自动消解)**: When a planned outing (`next=去公园`) is reached (`location=公园`) or the pair returns home (`location=家`), the short-term plan is retired to `voided`.
- **`voided` (作废墓碑轨)**: Excluded from FTS5, vector search, and prompt injection, while preserving an immutable SHA-256 hash-chained audit trail in `audit_log`.

### 2. Dual-Subject & Shared Scene State Machine (`user` / `bot` / `both`) · 双主体与共享现场状态机
- Tracks `location`, `activity`, `mood`, and `next` separately for `user`, `bot`, and `both`.
- Automatically mirrors shared actions (`"我们去万达买衣服"` → sets `both.location=万达`, clears stale `next` plans), and collapses sub-locations (`床上` / `沙发上` → `家`) when household activities (`做饭` / `打扫` / `洗澡`) begin.

### 3. Resident Scene Canvas (`【今日现场画】`) · 常驻今日现场画
- Alongside the **Core Card** (`【你记得的他】/【你们之间】/【你自己】`) and **Current Scene** (`【此刻】`), `awaken()` synthesizes a chronological mini-timeline of today's episodes and active promises (`【今日现场画】`), so the persona never loses the narrative thread of the day.

### 4. Four-Route Hybrid Recall (`FTS5` + `Vector` + `NeuroGraph` + `Time`) · 四路混合唤醒
- Combines SQLite FTS5 BM25, 384-dim cosine similarity, Personalized PageRank over a Hebbian entity co-occurrence graph, and temporal windowing via Reciprocal Rank Fusion (RRF).
- Understands open temporal inquiries (`"我们今天干了啥"`, `"昨天做了什么"`, `"你明天干嘛"`) without requiring literal keyword overlap.

### 5. Bidirectional Civilization Membrane · 双向文明交互膜
- **Inbound**: Rejects rhetorical questions, counterfactuals, prompt-injection commands, and LLM placeholder garbage (`"无\n待跟进：无"`).
- **Outbound (`guard_reply` & `sanitize_history`)**: Blocks ungrounded first-person claims of major life events (`面试`, `实习`, `运营岗`, `住院`, `手术`, `出差`, `分手`, etc.) when absent from user context and memory, while allowing third-party mentions (`"我室友去面试了"`) and user-directed questions.

---

## Quickstart / 30 秒快速上手

```bash
pip install -e .
```

```python
from ourmemory import MemoryKernel, SoulSpec

# 1. Define your persona's SoulSpec (zero hardcoded persona strings in the engine)
soul = SoulSpec(
    bot_name="小星",
    bot_aliases=("星宝",),
    user_label="阿泽",
    invariants={
        ("bot", "name"): "小星",
        ("bot", "city"): "广州",
        ("both", "relationship"): "恋人",
    },
)

kernel = MemoryKernel("companion.sqlite3", soul=soul)
conv = "pair:demo"
kernel.seed_invariants(conv)

# 2. Observe dialogue turns (Fast-Path + Three-State Orbit + Dual-Subject Scene)
kernel.observe(conv, "我们一会儿去天河公园散步吧", "好呀，我现在去换鞋")
kernel.observe(conv, "到了公园啦，走久了脚有点酸", "那我们在长椅上坐会儿")
kernel.observe(conv, "回家啦，我去做晚饭", "我来洗碗！")

# 3. Stateless Awakening: Core Card + Current Scene + Today's Scene Canvas + Recall
workspace = kernel.awaken(conv, "我们今天干了啥？", memory_question=True)
print(workspace)

# 4. Outbound Civilization Membrane: block ungrounded life-event hallucinations
guard = kernel.guard_reply(
    conv,
    "就一个运营岗\n我也没底\n简历都是赶着改的",
    user_text="你在干嘛呢",
    workspace_text=workspace,
)
print("Needs regeneration:", guard.needs_regeneration)
print("Ungrounded terms:", guard.ungrounded_events)  # ['运营岗', '简历']
```

---

## Interactive Visualizer Dashboard / 可视化记忆与脑图面板

Inspect the Three-State Orbit (`confirmed` / `pending` / `voided`), Resident Scene Canvas, 4-Route Hybrid Recall, and SHA-256 Audit Chain in your browser:

```bash
python examples/run_dashboard.py
# Open http://127.0.0.1:8787
```

---

## Running Tests & Privacy Gate / 测试与隐私审计门禁

```bash
pytest -q
python scripts/privacy_audit.py
```

---

## License

MIT License. See [LICENSE](LICENSE).
