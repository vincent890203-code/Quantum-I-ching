"""視覺化儀表板：Model D (Pure Quantum) vs Buy & Hold.

產出一張高品質的「績效比較 Dashboard」：
- 成長能力：Total Return / CAGR
- 風險調整品質：Sharpe / Calmar
- 風險輪廓：Max Drawdown
-（可選）載入現有的 Equity Curve 圖作為下半部展示

輸出檔案：
- performance_dashboard_model_d_vs_bh.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


def add_value_labels(ax, fmt: str = "{:.1f}", offset: float = 0.02) -> None:
    """在 bar 上方（或下方）標註數值."""
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        x = p.get_x() + p.get_width() / 2.0
        # 若為負值，標在 bar 下方
        if height >= 0:
            va = "bottom"
            y = height + offset * ax.get_ylim()[1]
        else:
            va = "top"
            y = height - offset * abs(ax.get_ylim()[0])
        ax.text(
            x,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=11,
            color="#333333",
        )


def main() -> None:
    # ===== 1. 固定輸入資料（來自已驗證回測） =====
    data = {
        "Metric": [
            "Total Return (%)",
            "CAGR (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Calmar Ratio",
        ],
        # 數值來自最新的 quant_report_model_d.py Tear Sheet
        # Model D：Total Return 242.6%, CAGR 28.83%, Sharpe 1.905, MaxDD -23.37%
        # Calmar Ratio 以 CAGR / |MaxDD| 近似計算 ≈ 0.2883 / 0.2337 ≈ 1.23
        "Model D (Pure Quantum)": [242.60, 28.83, 1.91, -23.37, 1.23],
        # Buy & Hold：沿用先前已驗證的基準數值（TSMC 現貨長期持有）
        "Buy & Hold (Benchmark)": [207.01, 25.15, 0.85, -54.43, 0.46],
    }
    df = pd.DataFrame(data)

    # 分組：成長 / 風險調整 / 風險
    growth_metrics = ["Total Return (%)", "CAGR (%)"]
    quality_metrics = ["Sharpe Ratio", "Calmar Ratio"]
    risk_metrics = ["Max Drawdown (%)"]

    df_growth = df[df["Metric"].isin(growth_metrics)].copy()
    df_quality = df[df["Metric"].isin(quality_metrics)].copy()
    df_risk = df[df["Metric"].isin(risk_metrics)].copy()

    # ===== 2. 設定風格與畫布 =====
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(
        2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.35, wspace=0.3
    )

    # Colors
    color_model = "#2ecc71"  # 綠色：Model D
    color_bench = "#95a5a6"  # 灰色：Bench

    # ===== 3. Plot 1: Growth Engine =====
    ax_growth = fig.add_subplot(gs[0, 0])

    metrics_order = df_growth["Metric"].tolist()
    x = np.arange(len(metrics_order))
    width = 0.35

    y_model = df_growth["Model D (Pure Quantum)"].values
    y_bench = df_growth["Buy & Hold (Benchmark)"].values

    # 控制 Y 軸範圍，避免標籤跑出圖外
    max_growth = float(max(max(y_model), max(y_bench)))
    ax_growth.set_ylim(0, max_growth * 1.25)

    ax_growth.bar(
        x - width / 2,
        y_model,
        width,
        label="Model D (Pure Quantum)",
        color=color_model,
        alpha=0.9,
    )
    ax_growth.bar(
        x + width / 2,
        y_bench,
        width,
        label="Buy & Hold (Benchmark)",
        color=color_bench,
        alpha=0.9,
    )

    ax_growth.set_xticks(x)
    ax_growth.set_xticklabels(metrics_order, rotation=0, fontsize=11)
    ax_growth.set_ylabel("Percentage (%)")
    ax_growth.set_title("Growth Potential", fontsize=15, fontweight="bold")
    ax_growth.legend(fontsize=11, frameon=False)

    # 加上文字標籤（% 格式）
    for i, v in enumerate(y_model):
        ax_growth.text(
            x[i] - width / 2,
            v + max_growth * 0.03,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2c3e50",
        )
    for i, v in enumerate(y_bench):
        ax_growth.text(
            x[i] + width / 2,
            v + max_growth * 0.03,
            f"{v:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2c3e50",
        )

    # ===== 4. Plot 2: Risk-Adjusted Quality (Sharpe & Calmar) =====
    ax_quality = fig.add_subplot(gs[0, 1])
    metrics_q = df_quality["Metric"].tolist()
    xq = np.arange(len(metrics_q))
    yq_model = df_quality["Model D (Pure Quantum)"].values
    yq_bench = df_quality["Buy & Hold (Benchmark)"].values

    # 控制 Y 軸範圍，避免 Sharpe / Calmar 標籤跑出圖外
    max_quality = float(max(max(yq_model), max(yq_bench)))
    ax_quality.set_ylim(0, max_quality * 1.35)

    ax_quality.bar(
        xq - width / 2,
        yq_model,
        width,
        label="Model D (Pure Quantum)",
        color=color_model,
        alpha=0.9,
    )
    ax_quality.bar(
        xq + width / 2,
        yq_bench,
        width,
        label="Buy & Hold (Benchmark)",
        color=color_bench,
        alpha=0.9,
    )

    ax_quality.set_xticks(xq)
    ax_quality.set_xticklabels(metrics_q, rotation=0, fontsize=11)
    ax_quality.set_ylabel("Ratio")
    ax_quality.set_title("Risk-Adjusted Quality (Sharpe & Calmar)", fontsize=15, fontweight="bold")
    ax_quality.legend(fontsize=11, frameon=False)

    for i, v in enumerate(yq_model):
        ax_quality.text(
            xq[i] - width / 2,
            v + max_quality * 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2c3e50",
        )
    for i, v in enumerate(yq_bench):
        ax_quality.text(
            xq[i] + width / 2,
            v + max_quality * 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2c3e50",
        )

    # ===== 5. Plot 3: Risk Profile (Max Drawdown) =====
    ax_risk = fig.add_subplot(gs[0, 2])
    metric_r = df_risk["Metric"].tolist()
    xr = np.arange(len(metric_r))
    yr_model = df_risk["Model D (Pure Quantum)"].values
    yr_bench = df_risk["Buy & Hold (Benchmark)"].values

    # 我們希望視覺上「越低越好」，所以用正值表示 Drawdown 的絕對值
    yr_model_abs = np.abs(yr_model)
    yr_bench_abs = np.abs(yr_bench)

    # 控制 Y 軸範圍，避免最大回撤標籤跑出圖外
    max_dd_abs = float(max(max(yr_model_abs), max(yr_bench_abs)))
    ax_risk.set_ylim(0, max_dd_abs * 1.3)

    ax_risk.bar(
        xr - width / 2,
        yr_model_abs,
        width,
        label="Model D (Pure Quantum)",
        color=color_model,
        alpha=0.9,
    )
    ax_risk.bar(
        xr + width / 2,
        yr_bench_abs,
        width,
        label="Buy & Hold (Benchmark)",
        color=color_bench,
        alpha=0.9,
    )

    ax_risk.set_xticks(xr)
    ax_risk.set_xticklabels(["Max Drawdown (%)"], rotation=0, fontsize=11)
    ax_risk.set_ylabel("Drawdown Magnitude (%)")
    ax_risk.set_title("Risk Profile (Max Drawdown)", fontsize=15, fontweight="bold")
    # 圖例移到左上角，避免與柱子/標籤重疊
    ax_risk.legend(fontsize=11, frameon=False, loc="upper left")

    # 在柱子上標註原始負值（例如 -27.9%）
    ax_risk.text(
        xr[0] - width / 2,
        yr_model_abs[0] + max_dd_abs * 0.05,
        f"{yr_model[0]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#2c3e50",
    )
    ax_risk.text(
        xr[0] + width / 2,
        yr_bench_abs[0] + max_dd_abs * 0.05,
        f"{yr_bench[0]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#2c3e50",
    )

    # ===== 6. Plot 4: Equity Curve 或 Survival Mode 說明 =====
    ax_bottom = fig.add_subplot(gs[1, :])
    equity_path = "output/model_d_equity_curve.png"
    if os.path.exists(equity_path):
        img = plt.imread(equity_path)
        ax_bottom.imshow(img)
        ax_bottom.axis("off")
        ax_bottom.set_title(
            "Equity Curve: Model D (Pure Quantum) vs Buy & Hold (with Survival Mode)",
            fontsize=15,
            fontweight="bold",
        )
    else:
        ax_bottom.axis("off")
        text = (
            "Equity Curve Placeholder\n\n"
            "• Survival Mode 風控：當策略從高點回撤超過 20% 時，自動將部位縮減至 20%，"
            "避免出現 Buy & Hold 那種 -50% 以上的大型回撤。\n"
            "• 這讓 Model D 在保留長期成長性的同時，仍能有效壓低下行情境的風險。"
        )
        ax_bottom.text(
            0.02,
            0.98,
            text,
            ha="left",
            va="top",
            fontsize=13,
            color="#2c3e50",
            wrap=True,
        )

    # ===== 7. 全局標題與輸出 =====
    fig.suptitle(
        "Quantum Advantage: Model D vs. Buy & Hold",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = "performance_dashboard_model_d_vs_bh.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Performance Dashboard 已輸出至: {output_path}")


if __name__ == "__main__":
    main()

