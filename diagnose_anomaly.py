"""診斷 Model D 回測異常：檢查部位大小、波動與單期報酬.

目的：
- 針對 Model D (Pure Quantum) 的 Walk-Forward 回測，檢查是否存在：
  1) 部位倍率 > 1 的槓桿錯誤
  2) 波動極低導致 position_size 異常放大
  3) 單期報酬異常巨大（例如單日 +50% 以上）

輸出：
- 圖：debug_anomaly_metrics.png
  * 上：Position Size（實際使用的部位比例）
  * 中：Rolling Volatility vs TARGET_VOL
  * 下：策略單期報酬（pnl per 5-day window）
- 終端列印：
  * Max / Min Position Size
  * Max / Min Single-Period Return
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import MarketDataLoader
from market_encoder import MarketEncoder
from quant_report_model_d import (
    prepare_pure_quantum_tabular,
    run_walk_forward_proba,
    generate_trading_pnl,
    TARGET_VOL,
    MAX_DRAWDOWN_TRIGGER,
    SURVIVAL_POS_CAP,
    set_random_seed,
)


def main() -> None:
    print("=" * 80)
    print("Diagnose Anomaly - Model D Position Sizing & Risk Control")
    print("=" * 80)

    set_random_seed(42)

    # 1. 載入並編碼市場資料（與 Model D 報告一致）
    loader = MarketDataLoader()
    print("\n[INFO] 載入 TSMC (2330.TW) 市場資料...")
    raw_data = loader.fetch_data(tickers=["2330.TW"], market_type="TW")
    if raw_data.empty:
        print("[ERROR] 無法獲取市場資料")
        return

    print("[INFO] 編碼為易經卦象...")
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)
    if encoded_data.empty:
        print("[ERROR] 編碼後資料為空")
        return

    # 2. 準備 Pure Quantum tabular 資料
    X, y, actual_returns, dates = prepare_pure_quantum_tabular(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
    )

    # 3. Walk-Forward 機率預測（OOS）
    print("\n[INFO] Walk-Forward 驗證（用於診斷）...")
    y_proba_oos, y_true_oos, actual_oos, dates_oos = run_walk_forward_proba(
        X,
        y,
        actual_returns,
        dates,
        train_months=12,
        test_months=1,
        step_months=1,
        model_name="Model D (Pure Quantum)",
    )

    if y_proba_oos.size == 0:
        print("[WARNING] 無 OOS 樣本，無法診斷。")
        return

    # 4. 根據機率產生原始策略 PnL（5 日視窗報酬）
    signals, pnl_raw = generate_trading_pnl(
        y_proba=y_proba_oos,
        actual_returns=actual_oos,
        threshold=0.5,
    )

    # 5. 準備日報酬與 20 日滾動波動度
    close_series = encoded_data["Close"].astype(float)
    idx = close_series.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close_series.index = idx

    # 以日期為索引
    date_index = close_series.index.normalize()
    daily_close = close_series.groupby(date_index).last()
    daily_close = daily_close.replace([np.inf, -np.inf], np.nan).dropna()
    daily_ret_full = (
        daily_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    )

    # 將 OOS 日期索引標準化：去除時區與時間，僅保留日期，與 daily_ret_full 對齊
    dates_oos_idx = dates_oos
    if getattr(dates_oos_idx, "tz", None) is not None:
        dates_oos_idx = dates_oos_idx.tz_localize(None)
    dates_oos_dates = dates_oos_idx.normalize()

    # 20 日滾動波動度（以日報酬為基礎），並使用 bfill 補早期 NaN
    rolling_vol = (
        daily_ret_full.rolling(window=20, min_periods=5).std().fillna(method="bfill")
    )
    # 使用「前一日」波動度估計，並在日期粒度上對齊到 OOS 交易視窗
    vol_oos = (
        rolling_vol.shift(1)
        .reindex(dates_oos_dates)
        .fillna(method="ffill")
        .fillna(0.0)
    )

    # 6. 根據 quant_report_model_d 的最新風控邏輯重建 pos_size 與 pnl_strategy
    pos_size = np.ones_like(pnl_raw, dtype=float)
    vol_values = vol_oos.to_numpy()
    eff_vol_5d = vol_values * np.sqrt(5.0)
    valid_vol_mask = eff_vol_5d > 0
    # Vol Targeting：部位不得大於 1.0
    pos_size[valid_vol_mask] = np.minimum(1.0, TARGET_VOL / eff_vol_5d[valid_vol_mask])

    # Survival Mode：回撤 < -20% 時，當日部位 cap 在 20%，
    # 若回撤改善至 > -10%，則解除生存模式。
    equity = 1.0
    peak = 1.0
    in_survival_mode = False
    for i in range(len(pnl_raw)):
        current_dd = equity / peak - 1.0
        if current_dd <= MAX_DRAWDOWN_TRIGGER:
            in_survival_mode = True
        if in_survival_mode and current_dd > -0.10:
            in_survival_mode = False
        if in_survival_mode:
            pos_size[i] = min(pos_size[i], SURVIVAL_POS_CAP)

        ret_i = pnl_raw[i] * pos_size[i]
        equity *= (1.0 + ret_i)
        peak = max(peak, equity)

    pnl_strategy = pnl_raw * pos_size

    # 7. 計算診斷指標
    max_pos = float(np.max(pos_size))
    min_pos = float(np.min(pos_size))
    max_ret = float(np.max(pnl_strategy))
    min_ret = float(np.min(pnl_strategy))

    print("\n[DIAGNOSTIC] Position Size & Return Summary")
    print(f"  Max Position Size : {max_pos:.4f}")
    print(f"  Min Position Size : {min_pos:.4f}")
    print(f"  Max Single-Period Return : {max_ret*100:.2f}%")
    print(f"  Min Single-Period Return : {min_ret*100:.2f}%")

    # 8. 視覺化三個關鍵序列
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Top: Position Size
    ax1 = axes[0]
    ax1.plot(dates_oos, pos_size, color="#2ecc71", linewidth=1.5)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Max Leverage = 1.0")
    ax1.axhline(SURVIVAL_POS_CAP, color="orange", linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Survival Cap = {SURVIVAL_POS_CAP:.1f}")
    ax1.set_ylabel("Position Size")
    ax1.set_title("Position Size Over Time (Model D)")
    ax1.legend(loc="upper right", fontsize=9)

    # Middle: Rolling Vol vs TARGET_VOL
    ax2 = axes[1]
    ax2.plot(dates_oos, vol_oos, color="#3498db", linewidth=1.5, label="Rolling Vol (20d, daily)")
    ax2.axhline(TARGET_VOL / np.sqrt(5.0), color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                label=f"Implied Daily Target (≈ {TARGET_VOL/np.sqrt(5.0):.3f})")
    ax2.set_ylabel("Volatility")
    ax2.set_title("Rolling Volatility vs Target Volatility")
    ax2.legend(loc="upper right", fontsize=9)

    # Bottom: Strategy Single-Period Returns
    ax3 = axes[2]
    ax3.plot(dates_oos, pnl_strategy * 100.0, color="#e74c3c", linewidth=1.0)
    ax3.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax3.set_ylabel("Return per 5-Day Window (%)")
    ax3.set_title("Strategy Single-Period Returns (per OOS trade window)")
    ax3.set_xlabel("Date")

    plt.tight_layout()
    out_path = "debug_anomaly_metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 診斷圖已輸出至: {out_path}")


if __name__ == "__main__":
    main()

