"""Cross-Sectional Validation for Model D (Pure Quantum).

目的：
- 檢驗 Model D（Pure Quantum + Survival Mode 風控）在不同資產類型上的泛化能力：
  1) 高波動成長股（High Beta）
  2) 低波動防禦股（Low Volatility）
  3) 指數型商品（Benchmark Index）

作法：
- 針對每一檔標的執行與 `quant_report_model_d.py` 相同的：
  * 易經特徵抽取 + Pure Quantum 特徵建構
  * Walk-Forward 訓練／預測
  * Volatility Targeting + Survival Mode 風控
  * 策略與 Buy & Hold 的績效計算

輸出：
- 不繪製個別圖，只在終端列印比較表：
  Ticker | Type | Strat CAGR | B&H CAGR | Strat Sharpe | B&H Sharpe
       | Win Rate | Max DD (Strat) | Max DD (B&H)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader import MarketDataLoader
from market_encoder import MarketEncoder
from quant_report_model_d import (
    TARGET_VOL,
    MAX_DRAWDOWN_TRIGGER,
    SURVIVAL_POS_CAP,
    prepare_pure_quantum_tabular,
    run_walk_forward_proba,
    generate_trading_pnl,
    compute_performance_metrics,
    set_random_seed,
)


@dataclass
class AssetConfig:
    """單一資產的設定."""

    ticker: str
    asset_type: str  # 例如 "High Beta", "Low Vol", "Index"
    market_type: Optional[str] = None  # "US" / "TW" / None(使用預設)


def infer_market_type(ticker: str) -> str:
    """根據 ticker 粗略判斷市場類型.

    - 結尾為 '.TW' 視為台股
    - 其他視為美股 / 一般股票
    """
    if ticker.endswith(".TW"):
        return "TW"
    return "US"


def apply_vol_target_and_survival_mode(
    pnl_raw: np.ndarray,
    dates_oos: pd.DatetimeIndex,
    daily_ret_full: pd.Series,
) -> np.ndarray:
    """套用與 Model D 主報告一致的 Vol Target + Survival Mode 風控.

    Args:
        pnl_raw: 未經風控縮放前的策略單期報酬（5 日視窗報酬）。
        dates_oos: OOS 交易視窗對應的日期索引（DatatimeIndex）。
        daily_ret_full: 該標的的「日頻簡單報酬」序列（index 為日期）。

    Returns:
        套用部位控制後的策略日報酬序列。
    """
    # 20 日滾動波動度（以日報酬為基礎），並使用 bfill 補早期 NaN
    rolling_vol = (
        daily_ret_full.rolling(window=20, min_periods=5).std().fillna(method="bfill")
    )

    # 將 OOS 日期壓平成日期，並去除時區，確保與 daily_ret_full / rolling_vol 對齊
    dates_oos_idx = dates_oos
    if getattr(dates_oos_idx, "tz", None) is not None:
        dates_oos_idx = dates_oos_idx.tz_localize(None)
    dates_oos_dates = dates_oos_idx.normalize()

    # 使用 t-1 的波動度估計，並在「日期」粒度上對齊到 OOS 交易視窗
    vol_oos = (
        rolling_vol.shift(1)
        .reindex(dates_oos_dates)
        .fillna(method="ffill")
        .fillna(0.0)
    )

    pos_size = np.ones_like(pnl_raw, dtype=float)
    vol_values = vol_oos.to_numpy()

    # 將日波動度換算為 5 日持有期間的預期波動度
    eff_vol_5d = vol_values * np.sqrt(5.0)
    valid_vol_mask = eff_vol_5d > 0

    # 動態倉位：position_size = min(1.0, target_vol / current_vol_5d)
    pos_size[valid_vol_mask] = np.minimum(1.0, TARGET_VOL / eff_vol_5d[valid_vol_mask])

    # Drawdown Survival Mode（具「進出」邏輯）：
    # - 當回撤 <= -20% 時，進入生存模式
    # - 當回撤恢復至 > -10% 時，離開生存模式
    # - 生存模式下當期部位上限為 SURVIVAL_POS_CAP（例如 0.2）
    equity = 1.0
    peak = 1.0
    in_survival_mode = False

    for i in range(len(pnl_raw)):
        current_drawdown = equity / peak - 1.0

        # 1. 觸發生存模式：回撤 <= -20%
        if current_drawdown <= MAX_DRAWDOWN_TRIGGER:
            in_survival_mode = True

        # 2. 復原條件：回撤改善至 > -10%，離開生存模式
        if in_survival_mode and current_drawdown > -0.10:
            in_survival_mode = False

        # 3. 套用生存模式部位上限
        if in_survival_mode:
            pos_size[i] = min(pos_size[i], SURVIVAL_POS_CAP)

        # 4. 以調整後的部位計算當日報酬，更新權益與高點
        ret_i = pnl_raw[i] * pos_size[i]
        equity *= 1.0 + ret_i
        peak = max(peak, equity)

    return pnl_raw * pos_size


def build_buyhold_returns(
    encoded_data: pd.DataFrame,
    dates_oos: pd.DatetimeIndex,
) -> np.ndarray:
    """根據編碼後資料建立 Buy & Hold 的日報酬（與策略 OOS 日期對齊）."""
    close_series = encoded_data["Close"].astype(float)

    # 統一為 tz-naive，並壓平成日期級別
    idx = close_series.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close_series.index = idx

    date_index = close_series.index.normalize()
    daily_close = close_series.groupby(date_index).last()

    daily_close = daily_close.replace([np.inf, -np.inf], np.nan).dropna()
    daily_ret_full = (
        daily_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    )

    # OOS 日期同樣壓平成日期，以確保能與 daily_ret_full 對齊
    dates_oos_idx = dates_oos
    if getattr(dates_oos_idx, "tz", None) is not None:
        dates_oos_idx = dates_oos_idx.tz_localize(None)
    dates_oos_dates = dates_oos_idx.normalize()

    # 使用與策略相同的 OOS 日期索引（日期粒度），對齊 Buy & Hold 的日報酬
    daily_ret_oos = daily_ret_full.reindex(dates_oos_dates).fillna(0.0)
    return daily_ret_oos.to_numpy(), daily_ret_full


def test_strategy(config: AssetConfig) -> Optional[Dict[str, float]]:
    """對單一資產執行 Pure Quantum + 風控邏輯，回傳摘要指標.

    Returns:
        dict，包含策略與 Buy & Hold 的核心績效；若資料不足則回傳 None。
    """
    ticker = config.ticker
    market_type = config.market_type or infer_market_type(ticker)

    loader = MarketDataLoader()
    encoder = MarketEncoder()

    print(f"\n{'=' * 80}")
    print(f"[INFO] 測試標的: {ticker} ({config.asset_type}), market_type={market_type}")
    print(f"{'=' * 80}")

    # 1) 載入原始市場資料
    raw_data = loader.fetch_data(tickers=[ticker], market_type=market_type)
    if raw_data.empty:
        print(f"[WARNING] {ticker} 資料為空，略過。")
        return None

    # 2) 產生易經卦象與 Ritual_Sequence
    encoded_data = encoder.generate_hexagrams(raw_data)
    if encoded_data.empty:
        print(f"[WARNING] {ticker} 編碼後資料為空，略過。")
        return None

    # 3) 準備 Pure Quantum tabular 資料
    X, y, actual_returns, dates = prepare_pure_quantum_tabular(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
    )

    # 4) Walk-Forward OOS 機率預測
    y_proba_oos, y_true_oos, actual_oos, dates_oos = run_walk_forward_proba(
        X=X,
        y=y,
        actual_returns=actual_returns,
        dates=dates,
        train_months=12,
        test_months=1,
        step_months=1,
        model_name=f"Model D (Pure Quantum) - {ticker}",
    )

    if y_proba_oos.size == 0:
        print(f"[WARNING] {ticker} 無 OOS 樣本，略過。")
        return None

    # 5) 產生策略 PnL（未加風控）
    signals, pnl_strategy_raw = generate_trading_pnl(
        y_proba=y_proba_oos,
        actual_returns=actual_oos,
        threshold=0.5,
    )

    # 6) Buy & Hold 報酬 + 風控後的策略報酬
    returns_buyhold, daily_ret_full = build_buyhold_returns(
        encoded_data=encoded_data,
        dates_oos=dates_oos,
    )

    pnl_strategy = apply_vol_target_and_survival_mode(
        pnl_raw=pnl_strategy_raw,
        dates_oos=dates_oos,
        daily_ret_full=daily_ret_full,
    )

    # 尺度檢查：確保長度一致
    min_len = min(len(pnl_strategy), len(returns_buyhold), len(dates_oos))
    pnl_strategy = pnl_strategy[:min_len]
    returns_buyhold = returns_buyhold[:min_len]
    dates_oos = dates_oos[:min_len]
    signals = signals[:min_len]

    # 7) 計算策略與 Buy & Hold 績效
    metrics_strategy = compute_performance_metrics(
        dates=dates_oos,
        strategy_returns=pnl_strategy,
        signals=signals,
    )

    # Buy & Hold 視為「天天持有」，signals 全為 1
    bh_signals = np.ones_like(returns_buyhold, dtype=int)
    metrics_bh = compute_performance_metrics(
        dates=dates_oos,
        strategy_returns=returns_buyhold,
        signals=bh_signals,
    )

    result = {
        "Ticker": ticker,
        "Type": config.asset_type,
        "Strat_CAGR": metrics_strategy["CAGR"],
        "BH_CAGR": metrics_bh["CAGR"],
        "Strat_Sharpe": metrics_strategy["Sharpe"],
        "BH_Sharpe": metrics_bh["Sharpe"],
        "WinRate": metrics_strategy["Win Rate"],
        "MaxDD_Strat": metrics_strategy["Max Drawdown"],
        "MaxDD_BH": metrics_bh["Max Drawdown"],
        "TotalRet_Strat": metrics_strategy["Total Return"],
        "TotalRet_BH": metrics_bh["Total Return"],
    }

    print(
        f"[SUMMARY] {ticker} | Strat CAGR: {metrics_strategy['CAGR']*100:.2f}% | "
        f"B&H CAGR: {metrics_bh['CAGR']*100:.2f}% | "
        f"Sharpe (Strat/BH): {metrics_strategy['Sharpe']:.2f}/{metrics_bh['Sharpe']:.2f}"
    )

    return result


def main() -> None:
    """針對多檔資產跑 Model D 策略，輸出橫斷面比較表."""
    set_random_seed(42)

    # 若偏好台股，可改為：
    #   2330.TW: 高波動科技
    #   2412.TW: 低波動電信
    #   0050.TW: 市場指數 ETF
    universe: List[AssetConfig] = [
        AssetConfig(ticker="2330.TW", asset_type="High Beta (Tech)", market_type="TW"),
        AssetConfig(ticker="2412.TW", asset_type="Low Vol (Telecom)", market_type="TW"),
        AssetConfig(ticker="0050.TW", asset_type="Index ETF", market_type="TW"),
    ]

    results: List[Dict[str, float]] = []
    for cfg in universe:
        res = test_strategy(cfg)
        if res is not None:
            results.append(res)

    if not results:
        print("[ERROR] 所有標的皆無有效結果，請檢查資料或參數設定。")
        return

    df = pd.DataFrame(results)

    # 只選擇要顯示的欄位，並格式化輸出
    display_cols = [
        "Ticker",
        "Type",
        "Strat_CAGR",
        "BH_CAGR",
        "Strat_Sharpe",
        "BH_Sharpe",
        "WinRate",
        "MaxDD_Strat",
        "MaxDD_BH",
    ]
    df_display = df[display_cols].copy()

    # 轉成百分比顯示的欄位
    pct_cols = ["Strat_CAGR", "BH_CAGR", "WinRate", "MaxDD_Strat", "MaxDD_BH"]
    for col in pct_cols:
        if "MaxDD" in col:
            # MaxDD 已經是負值，直接 *100
            df_display[col] = df_display[col] * 100.0
        else:
            df_display[col] = df_display[col] * 100.0

    print("\n" + "=" * 80)
    print("Cross-Sectional Validation - Model D (Pure Quantum) vs Buy & Hold")
    print("=" * 80)
    print(
        "Columns: Ticker | Type | Strat CAGR% | B&H CAGR% | "
        "Sharpe (Strat/BH) | WinRate% | MaxDD% (Strat/BH)"
    )
    print("-" * 80)

    for _, row in df_display.iterrows():
        print(
            f"{row['Ticker']:>8} | {row['Type']:<18} | "
            f"{row['Strat_CAGR']:6.2f}% | {row['BH_CAGR']:6.2f}% | "
            f"{row['Strat_Sharpe']:.2f}/{row['BH_Sharpe']:.2f} | "
            f"{row['WinRate']:6.2f}% | "
            f"{row['MaxDD_Strat']:6.2f}% / {row['MaxDD_BH']:6.2f}%"
        )

    print("=" * 80)
    print("註：所有結果皆為 Walk-Forward OOS，並套用與 2330.TW 相同的風控邏輯。")


if __name__ == "__main__":
    main()

