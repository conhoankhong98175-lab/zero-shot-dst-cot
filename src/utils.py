"""
utils.py
跨模块共享的工具函数和常量。
"""

from __future__ import annotations

from pathlib import Path

# ── 项目路径常量 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results"

TEST_PATH      = DATA_DIR / "test.json"
TEST_TAXI_PATH = DATA_DIR / "test_taxi.json"   # 零样本跨域子集（src/filter_taxi_subset.py）
TRAIN_PATH     = DATA_DIR / "train.json"
VAL_PATH       = DATA_DIR / "val.json"

# MultiWOZ 2.1 五大领域
DOMAINS = ["hotel", "restaurant", "attraction", "taxi", "train"]


def resolve_test_path(target_domain: str | None = None) -> Path:
    """根据是否启用零样本目标域返回对应测试集。
    target_domain == 'taxi'  → data/processed/test_taxi.json
    target_domain is None    → data/processed/test.json （全域）
    """
    if target_domain == "taxi":
        if not TEST_TAXI_PATH.exists():
            raise FileNotFoundError(
                f"零样本 taxi 子集不存在: {TEST_TAXI_PATH}\n"
                f"请先运行：python src/filter_taxi_subset.py"
            )
        return TEST_TAXI_PATH
    if target_domain in (None, "", "all"):
        return TEST_PATH
    raise ValueError(
        f"暂未支持的 target_domain={target_domain!r}；"
        f"当前仅实现 'taxi' 与全域。"
    )


# ── 结果文件命名约定 ─────────────────────────────────────────
# 统一入口，避免 variant 命名在多处手写造成漂移。
def prediction_path(variant: str, tag: str = "full") -> Path:
    """
    推理原始输出的落盘路径。

    tag: "pilot50" / "full" / 其它自定义标签，用于区分样本规模。
    """
    return RESULTS_DIR / f"preds_{variant}_{tag}.json"


def eval_report_path(variant: str, tag: str = "full") -> Path:
    """对应预测文件的评测报告路径。"""
    return RESULTS_DIR / f"eval_{variant}_{tag}.json"


def ablation_summary_path(tag: str = "full") -> Path:
    """消融实验横向对比表的输出路径。"""
    return RESULTS_DIR / f"ablation_summary_{tag}.json"
