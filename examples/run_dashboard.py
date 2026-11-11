# -*- coding: utf-8 -*-
"""Launch the ourmemory interactive dashboard with a sample conversation."""

from __future__ import annotations

from ourmemory import MemoryKernel, SoulSpec
from ourmemory.dashboard import run_dashboard


def seed_demo(db_path: str) -> SoulSpec:
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
    kernel = MemoryKernel(db_path, soul=soul)
    conv = "demo:companion"
    if not kernel.store.active_slots(conv):
        kernel.seed_invariants(conv)
        kernel.observe(conv, "我叫阿泽，在广州工作", "记住啦阿泽")
        kernel.observe(conv, "我们下午去天河公园散步吧", "好呀，走吧")
        kernel.observe(conv, "到公园啦，喝杯冰奶茶", "好喝！")
        kernel.observe(conv, "回家做晚饭啦", "我帮你打下手")
    kernel.close()
    return soul


if __name__ == "__main__":
    db_file = "demo_ourmemory.sqlite3"
    demo_soul = seed_demo(db_file)
    run_dashboard(db_file, port=8787, soul=demo_soul)
