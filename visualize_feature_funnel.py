"""Feature Selection Funnel 視覺化.

依 model_definitions.py 固定之 Model A/B/C/D/E，繪製特徵一覽：
五個模型各顯示特徵數與完整特徵列表，便於實際檢視。

使用 Plotly 或 Matplotlib。儲存為 output/feature_funnel.png，並複製至 feature_funnel.png。
"""

from __future__ import annotations

import os
import shutil
from typing import List, Tuple

from model_definitions import MODELS, MODEL_ORDER, get_all_stages

# 漸層：灰 → 藍 → 紅（A→E）
COLORS = ["#95a5a6", "#7eb8d4", "#3498db", "#2980b9", "#c0392b"]
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "feature_funnel.png")
OUTPUT_ROOT = "feature_funnel.png"


def _stages() -> List[Tuple[str, int, str]]:
    """(顯示名稱, 特徵數, 特徵列表字串) 依 A→E."""
    out: List[Tuple[str, int, str]] = []
    for i, mid in enumerate(MODEL_ORDER):
        name, feats = MODELS[mid][0], MODELS[mid][1]
        n = len(feats)
        out.append((name, n, ", ".join(feats)))
    return out


def _plot_plotly() -> bool:
    """使用 Plotly 繪製漏斗圖並儲存 PNG。成功回傳 True，否則 False."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False

    stages = _stages()
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    texts = [f"{s[1]} features<br>{s[2]}" for s in stages]
    colors = [COLORS[i] for i in range(len(stages))]

    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            text=texts,
            textposition="inside",
            textfont=dict(size=11, color="white"),
            marker=dict(
                color=colors,
                line=dict(width=1, color="rgba(255,255,255,0.5)"),
            ),
            connector=dict(line=dict(color="rgba(128,128,128,0.3)", width=1)),
        )
    )

    fig.update_layout(
        title=dict(
            text="Model A / B / C / D / E — Feature Definitions (Fixed)",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=200, r=80, t=80, b=60),
        font=dict(family="sans-serif", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        height=680,
        funnelgap=0.2,
        funnelgroupgap=0.0,
    )
    fig.update_xaxes(title_text="Feature Count", gridcolor="rgba(0,0,0,0.08)")

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.write_image(OUTPUT_PATH, scale=2)
        return True
    except Exception:
        return False


def _plot_matplotlib() -> None:
    """使用 Matplotlib 繪製 Model A–E 特徵條圖並儲存 PNG."""
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stages = _stages()
    n = len(stages)
    max_val = max(s[1] for s in stages)
    y_pos = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (name, count, feats) in enumerate(stages):
        color = COLORS[i]
        ax.barh(
            y_pos[i],
            count,
            left=0,
            height=0.6,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            align="center",
        )
        ax.text(
            count / 2,
            y_pos[i],
            f"  {name}  —  {count}  ",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
        ax.text(
            max_val + 1.2,
            y_pos[i],
            feats,
            ha="left",
            va="center",
            fontsize=9,
            color="#333",
        )

    ax.set_xlim(0, max_val + 22)
    ax.set_ylim(-0.8, n)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in stages], fontsize=10)
    ax.set_xlabel("Feature Count", fontsize=11)
    ax.set_title(
        "Model A / B / C / D / E — Feature Definitions (Fixed)",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """產生 Model A–E 特徵圖並儲存至 output/feature_funnel.png 與 feature_funnel.png。"""
    print("[INFO] 繪製 Model A/B/C/D/E 特徵一覽（依 model_definitions.py）...")
    if _plot_plotly():
        print(f"[INFO] 已使用 Plotly 儲存: {OUTPUT_PATH}")
    else:
        print("[WARNING] Plotly 匯出 PNG 失敗（可能需 pip install kaleido），改用 Matplotlib。")
        _plot_matplotlib()
        print(f"[INFO] 已使用 Matplotlib 儲存: {OUTPUT_PATH}")
    try:
        shutil.copy(OUTPUT_PATH, OUTPUT_ROOT)
        print(f"[INFO] 已複製至: {OUTPUT_ROOT}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
