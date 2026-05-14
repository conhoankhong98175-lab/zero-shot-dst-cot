"""
dst_pipeline.py  v2
一键执行完整实验流程的编排脚本。

用法:
    # 单变体冒烟测试（50 条）
    python src/dst_pipeline.py --mode pilot

    # 单变体正式全量
    python src/dst_pipeline.py --mode full

    # 实验一 + 实验二：七组变体全量跑 + 自动生成横向对比
    python src/dst_pipeline.py --mode ablation

    # 指定套件与变体
    python src/dst_pipeline.py --mode single --variant standard --max_samples 100 --tag pilot100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ablation_runner import SUITES, run_suite
from evaluator import evaluate, print_report
from inference import run_inference
from prompt_builder import list_variants
from utils import RESULTS_DIR, eval_report_path, prediction_path, resolve_test_path

LOG_PATH = RESULTS_DIR / "experiment_log.jsonl"


# ── 单变体流水线 ─────────────────────────────────────────────
def run_single(*, variant: str, tag: str, max_samples: int, delay: float,
               resume: bool, target_domain: str | None = None) -> dict:
    test_path = resolve_test_path(target_domain)
    print(f"\n{'=' * 62}")
    print(f"🔬 单变体模式: {variant}  |  tag = {tag}  |  N = {max_samples or 'ALL'}")
    print(f"   test ← {test_path.name}  (target_domain={target_domain or 'all'})")
    print("=" * 62)

    pred_path = prediction_path(variant, tag)
    eval_path = eval_report_path(variant, tag)

    print("\n[Step 1/2] 推理...")
    run_inference(
        test_path   = test_path,
        output_path = pred_path,
        variant     = variant,
        max_samples = max_samples,
        delay       = delay,
        resume      = resume,
    )

    print("\n[Step 2/2] 评测...")
    summary, detailed = evaluate(pred_path, target_domain=target_domain)
    print_report(summary)

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "summary": summary, "details": detailed},
                  f, ensure_ascii=False, indent=2)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp":     datetime.now().isoformat(),
            "mode":          "single",
            "variant":       variant,
            "tag":           tag,
            "max_samples":   max_samples,
            "target_domain": target_domain or "all",
            "summary":       summary,
        }, ensure_ascii=False) + "\n")

    print(f"\n📁 预测: {pred_path}")
    print(f"📁 评测: {eval_path}")
    print(f"📋 日志: {LOG_PATH}")
    print("✅ 单变体流水线完成。")
    return summary


# ── 预设快捷模式 ─────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "pilot": {
        "variant":     "cot_full",
        "tag":         "pilot50",
        "max_samples": 50,
        "delay":       0.5,
    },
    "full": {
        "variant":     "cot_full",
        "tag":         "full",
        "max_samples": 0,
        "delay":       0.5,
    },
}


# ── 顶层分发 ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["pilot", "full", "single", "ablation"],
                    default="pilot",
                    help="pilot/full = 快捷单变体; single = 自定义单变体; "
                         "ablation = 批量跑 7 组变体并生成对比表")
    ap.add_argument("--variant", choices=list_variants(), default=None,
                    help="mode=single 时指定变体")
    ap.add_argument("--suite", choices=list(SUITES), default="all",
                    help="mode=ablation 时指定套件（main/ablation/all）")
    ap.add_argument("--tag", default=None,
                    help="文件名 tag，覆盖预设默认值")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="样本数，0 = 全量；覆盖预设默认值")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--target_domain", default=None,
                    help="零样本目标域；'taxi' 启用 test_taxi.json 子集与 taxi-only 评测")
    args = ap.parse_args()

    resume = not args.no_resume

    if args.mode in PRESETS:
        preset = PRESETS[args.mode]
        run_single(
            variant       = args.variant or preset["variant"],
            tag           = args.tag or preset["tag"],
            max_samples   = preset["max_samples"] if args.max_samples is None else args.max_samples,
            delay         = args.delay,
            resume        = resume,
            target_domain = args.target_domain,
        )

    elif args.mode == "single":
        if not args.variant:
            ap.error("--mode single 需要显式指定 --variant")
        run_single(
            variant       = args.variant,
            tag           = args.tag or "custom",
            max_samples   = 0 if args.max_samples is None else args.max_samples,
            delay         = args.delay,
            resume        = resume,
            target_domain = args.target_domain,
        )

    elif args.mode == "ablation":
        run_suite(
            suite         = args.suite,
            tag           = args.tag or "full",
            max_samples   = 0 if args.max_samples is None else args.max_samples,
            delay         = args.delay,
            resume        = resume,
            target_domain = args.target_domain,
        )


if __name__ == "__main__":
    main()
