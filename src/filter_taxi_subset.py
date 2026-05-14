"""
filter_taxi_subset.py
为 "零样本跨域 DST on taxi" 评测口径生成测试子集。

CLAUDE.md 定义的零样本跨域：源域 = hotel/restaurant/attraction/train，
目标域 = taxi。评测口径要求 **仅在 taxi 槽上计算 JGA / Slot F1**，否则
hotel/restaurant 等源域样本会稀释目标域真实表现，且与 T5DST / TransferQA
的对照前提不一致。

本脚本从 data/processed/test.json 中筛出 taxi 子集，并把每轮 gold belief
裁剪为仅含 `taxi-*` 槽，写入 data/processed/test_taxi.json。
推理脚本（inference.py）只需把 TEST_PATH 切到这个文件即可。

筛选模式（--mode）:
  dialogues  (默认)  保留所有 *涉及 taxi* 的对话的全部轮次。
                     与 T5DST/TransferQA 一致：模型必须在用户尚未提到 taxi
                     的早期轮次正确输出空 taxi 状态，这本身就考验了零样本能力。
  turns              更严格：只保留 belief 本身含 taxi 槽的轮次。

prompt 端不做任何裁剪：模型仍看到全部 5 域的 schema，仅评测口径限定于 taxi。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, TEST_PATH

TARGET_DOMAIN = "taxi"
DEFAULT_OUTPUT = DATA_DIR / "test_taxi.json"


def filter_subset(test_path: Path, output_path: Path, mode: str) -> dict:
    with open(test_path, encoding="utf-8") as f:
        all_samples = json.load(f)

    # 第一遍：找出 taxi 出现过的 dial_id（dialogues 模式才需要）
    taxi_dialogs: set[str] = set()
    for s in all_samples:
        if any(k.startswith(f"{TARGET_DOMAIN}-") for k in s["belief"]):
            taxi_dialogs.add(s["dial_id"])

    # 第二遍：按模式筛样本，裁剪 gold belief
    out_samples = []
    for s in all_samples:
        if mode == "dialogues":
            keep = s["dial_id"] in taxi_dialogs
        elif mode == "turns":
            keep = any(k.startswith(f"{TARGET_DOMAIN}-") for k in s["belief"])
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        if not keep:
            continue

        # 同时过滤掉 *-booked 这类非字符串残留（来自 MultiWOZ metadata 的预订回执
        # list 结构；老版 normalize_belief 漏过滤了，evaluator 是靠 normalize_value
        # 隐式 drop 的，这里显式去掉以保证子集自洽）。
        gold_taxi = {
            k: v for k, v in s["belief"].items()
            if k.startswith(f"{TARGET_DOMAIN}-")
            and k != f"{TARGET_DOMAIN}-booked"
            and isinstance(v, str)
        }
        out_samples.append({
            "dial_id": s["dial_id"],
            "turn_id": s["turn_id"],
            "history": s["history"],
            "belief":  gold_taxi,
            "_target_domain": TARGET_DOMAIN,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_samples, f, ensure_ascii=False, indent=2)

    stats = {
        "input_total_turns":   len(all_samples),
        "input_total_dialogs": len({s["dial_id"] for s in all_samples}),
        "taxi_dialogs":        len(taxi_dialogs),
        "output_turns":        len(out_samples),
        "output_dialogs":      len({s["dial_id"] for s in out_samples}),
        "turns_with_nonempty_gold":
            sum(1 for s in out_samples if s["belief"]),
        "mode":   mode,
        "output": str(output_path),
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  default=str(TEST_PATH),
                    help="预处理后的全域测试集（默认 data/processed/test.json）")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="输出子集路径（默认 data/processed/test_taxi.json）")
    ap.add_argument("--mode", choices=["dialogues", "turns"], default="dialogues",
                    help="dialogues = 涉及 taxi 的对话全部轮次（推荐，与 baseline 一致）；"
                         "turns = 仅 belief 含 taxi 槽的轮次")
    args = ap.parse_args()

    stats = filter_subset(Path(args.input), Path(args.output), args.mode)

    print("\n" + "=" * 60)
    print(f"  零样本 taxi 子集生成完毕  (mode = {stats['mode']})")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")
    print("=" * 60)
    print(f"\n下一步：把 utils.TEST_PATH 切到 {args.output}，或在 dst_pipeline")
    print("中传入 --target_domain taxi 以激活子集模式。")


if __name__ == "__main__":
    main()
