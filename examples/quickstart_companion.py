# -*- coding: utf-8 -*-
"""Minimal end-to-end demo: a day out, a recall, and a blocked fabrication."""

from __future__ import annotations

from ourmemory import MemoryKernel, SoulSpec


def main() -> None:
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

    kernel = MemoryKernel(":memory:", soul=soul)
    conv = "private:demo"
    kernel.seed_invariants(conv)

    kernel.observe(conv, "我们下午去公园散步吧", "好呀，我去换鞋")
    kernel.observe(conv, "到公园啦", "今天的风好舒服")
    kernel.observe(conv, "回家啦，我们做晚饭", "我来洗菜")

    print(kernel.awaken(conv, "我们今天干了啥？", memory_question=True))

    print("\n--- outbound guard ---")
    draft = "就一个运营岗，我也没底，简历都是赶着改的"
    checked = kernel.guard_reply(conv, draft, user_text="你在忙什么呢")
    print("needs_regeneration:", checked.needs_regeneration)
    print("ungrounded_events:", checked.ungrounded_events)
    print("cleaned_reply:", checked.cleaned_reply)


if __name__ == "__main__":
    main()
