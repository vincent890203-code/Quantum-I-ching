"""Model A / B / C / D / E 固定定義 — 單一真相來源（Single Source of Truth）.

此模組定義五個模型的名稱與特徵列表，供 inspect_model_features、visualize_feature_funnel
及相關腳本匯入使用。修改特徵時請僅在此處更新。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (顯示名稱, 特徵名稱列表)；model_id 即鍵 A/B/C/D/E
MODELS: Dict[str, Tuple[str, List[str]]] = {
    "A": (
        "Model A (Full I-Ching)",
        [
            "Close",
            "Volume",
            "RVOL",
            "Daily_Return",
            "Yang_Count_Main",
            "Yang_Count_Future",
            "Moving_Lines_Count",
            "Energy_Delta",
            "Conflict_Score",
        ],
    ),
    "B": (
        "Model B (Baseline)",
        ["Close", "Volume", "RVOL", "Daily_Return"],
    ),
    "C": (
        "Model C (Lean)",
        [
            "Close",
            "Volume",
            "RVOL",
            "Daily_Return",
            "Moving_Lines_Count",
            "Energy_Delta",
        ],
    ),
    "D": (
        "Model D (Pure Quantum)",
        ["Daily_Return", "Moving_Lines_Count", "Energy_Delta"],
    ),
    "E": (
        "Model E (Momentum Only)",
        ["Daily_Return"],
    ),
}

# 依 A→B→C→D→E 順序
MODEL_ORDER = ["A", "B", "C", "D", "E"]


def get_display_name(model_id: str) -> str:
    """回傳模型顯示名稱."""
    if model_id not in MODELS:
        return model_id
    return MODELS[model_id][0]


def get_features(model_id: str) -> List[str]:
    """回傳該模型特徵名稱列表."""
    if model_id not in MODELS:
        return []
    return list(MODELS[model_id][1])


def get_all_stages() -> List[Tuple[str, str, List[str]]]:
    """回傳 (model_id, display_name, features) 列表，順序 A→E."""
    return [(mid, MODELS[mid][0], MODELS[mid][1]) for mid in MODEL_ORDER]
