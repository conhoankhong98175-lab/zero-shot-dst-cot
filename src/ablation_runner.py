"""
ablation_runner.py
一键运行实验一（主对比）与实验二（消融）的所有 prompt 变体。

用法:
    # 仅跑主对比三组（实验一）
    python src/ablation_runner.py --suite main --max_samples 100

    # 仅跑消融四组（实验二）
    python src/ablation_runner.py --suite ablation --max_samples 100

    # 全部七组一起跑（推荐用于正式实验）
    python src/ablation_runner.py --suite all --max_samples 0 --tag full

每个变体独立落盘，失败不影响其它变体。结束后自动调用 compare_ablations 生成对比表。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator import evaluate, print_report
from inference import run_inference
from prompt_builder import VARIANTS
from utils import RESULTS_DIR, eval_report_path, prediction_path, resolve_test_path

# ── 套件定义 ─────────────────────────────────────────────────
SUITES: dict[str, list[str]] = {
    "main":     ["standard", "fn_style", "cot_basic", "cot_full"],
    "ablation": ["cot_full", "ab_no_s1", "ab_no_s2", "ab_no_s3", "ab_no_s4"],
    "all":      ["standard", "fn_style", "cot_basic", "cot_full",
                 "ab_no_s1", "ab_no_s2", "ab_no_s3", "ab_no_s4"],
}

LOG_PATH = RESULTS_DIR / "experiment_log.jsonl"


# ── 单变体完整流程：推理 → 评测 ─────────────────────────────
def _run_one_variant(variant: str, *, tag: str, max_samples: int, delay: float,
                     resume: bool, target_domain: str | None = None) -> dict | None:
    """运行单个变体，返回 summary 或 None（失败）。"""
    pred_path = prediction_path(variant, tag)
    eval_path = eval_report_path(variant, tag)
    test_path = resolve_test_path(target_domain)

    print(f"\n{'█' * 72}")
    print(f"▶ VARIANT: {variant}  —  {VARIANTS[variant]['description']}")
    print(f"  test  ← {test_path}  (target_domain={target_domain or 'all'})")
    print(f"  pred  → {pred_path}")
    print(f"  eval  → {eval_path}")
    print(f"{'█' * 72}")

    t0 = time.time()

    # Step 1: 推理（带断点续传）
    try:
        run_inference(
            test_path   = test_path,
            output_path = pred_path,
            variant     = variant,
            max_samples = max_samples,
            delay       = delay,
            resume      = resume,
        )
    except Exception as e:
        print(f"❌ [{variant}] 推理阶段异常: {e}")
        traceback.print_exc()
        return None

    # Step 2: 评测
    try:
        summary, detailed = evaluate(pred_path, target_domain=target_domain)
    except Exception as e:
        print(f"❌ [{variant}] 评测阶段异常: {e}")
        traceback.print_exc()
        return None

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "summary": summary, "details": detailed},
                  f, ensure_ascii=False, indent=2)

    summary_with_meta = {
        **summary,
        "variant":      variant,
        "description":  VARIANTS[variant]["description"],
        "elapsed_sec":  round(time.time() - t0, 2),
    }
    print_report(summary)
    return summary_with_meta


# ── 套件执行器 ───────────────────────────────────────────────
def run_suite(suite: str,
              tag: str = "full",
              max_samples: int = 0,
              delay: float = 0.5,
              resume: bool = True,
              target_domain: str | None = None) -> dict[str, dict]:
    """运行一整个套件，返回 {variant: summary}。"""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}; choose from {list(SUITES)}")

    variants = SUITES[suite]
    print(f"\n🔬 Suite: {suite}  |  variants: {variants}")
    print(f"   tag = {tag}  |  max_samples = {max_samples or 'ALL'}  "
          f"|  target_domain = {target_domain or 'all'}")

    all_summaries: dict[str, dict] = {}

    for variant in variants:
        summary = _run_one_variant(
            variant,
            tag=tag, max_samples=max_samples, delay=delay, resume=resume,
            target_domain=target_domain,
        )
        if summary is not None:
            all_summaries[variant] = summary
            _append_log(variant, tag, max_samples, summary)
        else:
            print(f"⚠️  Variant {variant} 未能生成 summary，已跳过。")

    # 生成横向对比表
    if all_summaries:
        try:
            from compare_ablations import build_comparison
            cmp_path = build_comparison(all_summaries, tag=tag)
            print(f"\n📊 横向对比表已生成: {cmp_path}")
        except Exception as e:
            print(f"⚠️  对比表生成失败: {e}")
            traceback.print_exc()

    return all_summaries


def _append_log(variant: str, tag: str, max_samples: int, summary: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp":   datetime.now().isoformat(),
        "variant":     variant,
        "tag":         tag,
        "max_samples": max_samples,
        "summary":     {k: v for k, v in summary.items() if k != "details"},
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), default="all",
                    help="main = 实验一; ablation = 实验二; all = 七组全跑")
    ap.add_argument("--tag", default="full",
                    help="文件名后缀（如 pilot100 / full）")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="每个变体跑多少条；0 = 全量")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--no_resume", action="store_true",
                    help="禁用断点续传，强制从零开始")
    ap.add_argument("--target_domain", default=None,
                    help="零样本目标域；'taxi' 启用 test_taxi.json 子集与 taxi-only 评测")
    args = ap.parse_args()

    run_suite(
        suite        = args.suite,
        tag          = args.tag,
        max_samples  = args.max_samples,
        delay        = args.delay,
        resume       = not args.no_resume,
        target_domain= args.target_domain,
    )
