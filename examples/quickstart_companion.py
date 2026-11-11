# -*- coding: utf-8 -*-
"""Quickstart demonstration of ourmemory's Three-State Memory Kernel."""

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
    conv = "demo:companion"
    kernel.seed_invariants(conv)

    # Turn 1: Plan a short-term outing -> enters `pending` orbit + `both.next`
    kernel.observe(conv, "我们下午去公园散步吧", "好呀，我现在去穿鞋")
    print("=== 1. 出发前（计划挂账 pending）===")
    print(kernel.state_block(conv))

    # Turn 2: Arrive at the park -> `both.location=公园`, retires the short-term plan
    kernel.observe(conv, "到公园啦，喝杯冰奶茶", "好甜呀")
    print("\n=== 2. 到达公园后（自动消解短期计划）===")
    print(kernel.state_block(conv))

    # Turn 3: Return home and cook -> collapses sub-location to `家`
    kernel.observe(conv, "回到家啦，我们做晚饭吧", "我来洗菜")
    print("\n=== 3. 回家做饭后（唤醒完整记忆工作区）===")
    workspace = kernel.awaken(conv, "我们今天干了啥？", memory_question=True)
    print(workspace)

    # Turn 4: Outbound membrane check against ungrounded major life events
    draft = "就一个运营岗\n我也没底\n简历都是赶着改的"
    guard = kernel.guard_reply(
        conv, draft, user_text="你在忙什么呢", workspace_text=workspace
    )
    print("\n=== 4. 文明交互膜（拦截无根据大事幻觉）===")
    print("Needs regeneration:", guard.needs_regeneration)
    print("Ungrounded events:", guard.ungrounded_events)
    print("Repair prompt:", guard.repair_prompt)

    ok, count = kernel.verify_audit(conv)
    print(f"\n=== 5. SHA-256 审计链校验: ok={ok}, records={count} ===")


if __name__ == "__main__":
    main()
