"""
inference.py  v2 (ablation-ready)
Zero-shot DST 推理主流程，支持多 prompt 变体 (OpenAI 官方 API)。

用法:
    # 主方法（四步 CoT）冒烟测试
    python src/inference.py --variant cot_full --max_samples 50

    # Standard Prompting 基线，全量
    python src/inference.py --variant standard --max_samples 0 \\
        --output results/preds_standard_full.json

    # 自动命名（推荐）：省略 --output，让 utils.prediction_path 生成标准路径
    python src/inference.py --variant ab_no_s4 --max_samples 100 --tag pilot100

依赖环境变量:
    OPENAI_API_KEY  —  OpenAI 的官方 API 密钥
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI, RateLimitError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from parser import extract_belief, extract_reasoning
from prompt_builder import VARIANTS, build_prompt, list_variants
from utils import TEST_PATH, prediction_path


# ── API 客户端初始化 ──────────────────────────────────────────
def _init_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 环境变量 OPENAI_API_KEY 未找到！")
        print("   请运行: $env:OPENAI_API_KEY='your_api_key_here'  (PowerShell)")
        print("   或运行: export OPENAI_API_KEY='your_api_key_here' (Linux/macOS)")
        sys.exit(1)
    print(f"📡 API Key 已读取: {api_key[:4]}... (长度: {len(api_key)})")
    # httpx 0.28+ 移除了 proxies 参数，手动构造 http_client 绕过 SDK 内部冲突
    http_client = httpx.Client(timeout=60.0)
    return OpenAI(api_key=api_key, http_client=http_client)


_client: OpenAI | None = None   # 延迟初始化，避免被 import 时就触发
MODEL = "gpt-4o-mini"

# CoT 变体需要更多 token 容纳推理链；standard / fn_style 输出简短无需那么多
_MAX_TOKENS: dict[str, int] = {
    "standard":  512,
    "fn_style":  512,
    "cot_basic": 1024,
    "cot_full":  2048,
    "ab_no_s1":  2048,
    "ab_no_s2":  2048,
    "ab_no_s3":  2048,
    "ab_no_s4":  2048,
}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = _init_client()
    return _client


# ── LLM 调用（含指数退避重试）────────────────────────────────
def call_llm(system_prompt: str,
             user_message: str,
             variant: str = "cot_full",
             temperature: float = 0.0,
             max_retries: int = 3) -> str:
    """
    调用 LLM API，失败自动重试。
    - 限速错误（429）: 等待 60s 后重试
    - 其他 API 错误: 指数退避（1s / 2s / 4s）
    temperature=0 确保输出确定性，满足评测可复现要求。
    """
    client = _get_client()
    max_tokens = _MAX_TOKENS.get(variant, 2048)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            wait = 60
            print(f"\n[retry {attempt + 1}/{max_retries}] Rate limit hit. "
                  f"Waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n[retry {attempt + 1}/{max_retries}] API error: {e}. "
                  f"Waiting {wait}s...")
            time.sleep(wait)
    print("\n⚠️  该样本 API 调用失败，记录空预测。")
    return ""


# ── 主推理循环 ────────────────────────────────────────────────
def run_inference(test_path: Path,
                  output_path: Path,
                  variant: str = "cot_full",
                  max_samples: int = 50,
                  delay: float = 0.5,
                  resume: bool = True) -> list:
    """
    对 test_path 中的样本逐条调用 LLM，结果写入 output_path。

    Args:
        test_path:    data/processed/test.json
        output_path:  results/preds_{variant}_{tag}.json
        variant:      prompt_builder.VARIANTS 中的任一键
        max_samples:  ≤0 表示全量；否则取前 N 条（调试用）
        delay:        每次 API 调用后的等待秒数，防止触发 rate limit
        resume:       若 output_path 已存在，从中断处继续（容错关键）
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; valid: {list_variants()}")

    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)

    if max_samples and max_samples > 0:
        test_data = test_data[:max_samples]
        print(f"📌 调试模式：仅处理前 {max_samples} 条样本")
    else:
        print(f"🚀 全量模式：处理全部 {len(test_data)} 条样本")

    # --- 断点续传：读取已有结果，按 (dial_id, turn_id) 去重 ---
    done_keys: set[tuple[str, int]] = set()
    results: list = []
    if resume and output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                results = json.load(f)
            done_keys = {(r["dial_id"], r["turn_id"]) for r in results}
            print(f"♻️  发现已有结果 {len(done_keys)} 条，将跳过这些样本。")
        except (json.JSONDecodeError, KeyError):
            print("⚠️  已有结果文件损坏，重新开始。")
            results = []

    print(f"🧪 Variant: {variant}  —  {VARIANTS[variant]['description']}")

    pending = [s for s in test_data if (s["dial_id"], s["turn_id"]) not in done_keys]
    if not pending:
        print("✅ 所有样本均已完成，无需重复推理。")
        return results

    save_every = 20   # 每 N 条落盘一次，防止崩溃后丢失
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(tqdm(pending, desc=f"推理中[{variant}]")):
        system_prompt, user_message = build_prompt(sample, variant=variant)
        raw_output  = call_llm(system_prompt, user_message, variant=variant)
        pred_belief = extract_belief(raw_output)
        reasoning   = extract_reasoning(raw_output)

        results.append({
            "dial_id":     sample["dial_id"],
            "turn_id":     sample["turn_id"],
            "variant":     variant,
            "gold_belief": sample["belief"],
            "pred_belief": pred_belief,
            "reasoning":   reasoning,
            "raw_output":  raw_output,
        })

        if (idx + 1) % save_every == 0:
            _atomic_write(results, output_path)

        time.sleep(delay)

    _atomic_write(results, output_path)
    print(f"\n✅ 推理完成！共 {len(results)} 条 (本轮新增 {len(pending)})，"
          f"结果已保存至: {output_path}")
    return results


def _atomic_write(data: list, path: Path) -> None:
    """原子写：先写 .tmp 再 rename，防止崩溃时文件半截。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ── CLI 入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zero-shot DST Inference (ablation-ready)")
    ap.add_argument("--variant", default="cot_full", choices=list_variants(),
                    help="prompt 变体；默认 cot_full (四步完整 CoT)")
    ap.add_argument("--input", default=str(TEST_PATH),
                    help="预处理后的测试集路径")
    ap.add_argument("--output", default=None,
                    help="预测结果输出路径；省略则按 variant+tag 自动命名")
    ap.add_argument("--tag", default="pilot50",
                    help="输出文件标签（如 pilot50 / full），仅在未显式指定 --output 时使用")
    ap.add_argument("--max_samples", type=int, default=50,
                    help="调试样本数；0 或负数表示全量")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="API 调用间隔（秒），防止 rate limit")
    ap.add_argument("--no_resume", action="store_true",
                    help="禁用断点续传，强制从零开始")
    args = ap.parse_args()

    output_path = Path(args.output) if args.output else prediction_path(args.variant, args.tag)

    run_inference(
        test_path   = Path(args.input),
        output_path = output_path,
        variant     = args.variant,
        max_samples = args.max_samples,
        delay       = args.delay,
        resume      = not args.no_resume,
    )
