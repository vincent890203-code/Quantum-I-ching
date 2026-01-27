"""Screen High-Beta Candidates for Model D Quantum Portfolio.

目的：
- 在台灣 50 成分股中，篩選出最適合套用 Model D（Pure Quantum + Survival Mode 風控）
  的高波動標的，作為「Quantum Portfolio」的候選名單。

流程摘要：
1. 定義 0050.TW 前 20 大成分股（以常見大型權值股為近似，不追求精確權重）。
2. 過去三年計算每檔股票的：
   - 年化波動度（Annualized Volatility）
   - Beta（相對 0050.TW 的市場 Beta）
   - 流動性（平均每日成交金額，Close * Volume）
3. 篩選條件：
   - 平均成交金額 > 5 億新台幣
   - 排除金融／防禦型產業（以 ticker 列表近似）
   - 依年化波動度由高到低排序，取前 5 名
4. 對這 5 檔股票執行 Model D 的 `test_strategy`，輸出策略 vs Buy & Hold 的比較表。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader import MarketDataLoader
from multi_asset_validation import AssetConfig, test_strategy, set_random_seed


# 近似 0050.TW 前 20 大成分股（僅做示意與篩選用途，不追求精確權重與完整性）
# 資料來源：常見權值股，包含電子、半導體、電腦、通訊等高 Beta 族群與部分金融股
UNIVERSE_0050_TOP20: Dict[str, str] = {
    "2330.TW": "Semiconductor",  # TSMC
    "2317.TW": "Electronics",  # Hon Hai
    "2454.TW": "Semiconductor",  # Mediatek
    "2308.TW": "Electronics",  # Delta
    "2382.TW": "Computer",  # Quanta
    "2303.TW": "Semiconductor",  # UMC
    "2881.TW": "Financials",
    "2882.TW": "Financials",
    "2883.TW": "Financials",
    "2885.TW": "Financials",
    "2886.TW": "Financials",
    "2412.TW": "Telecom",  # Chunghwa Telecom（防禦型）
    "1301.TW": "Materials",
    "1303.TW": "Materials",
    "1326.TW": "Materials",
    "2002.TW": "Materials",
    "1101.TW": "Materials",
    "1216.TW": "Consumer Defensive",
    "2207.TW": "Consumer Cyclical",
    "2603.TW": "Shipping",
}

# 需排除的防禦性／金融類股（以 ticker 為主，避免 Quantum 策略套在過於平靜的資產上）
EXCLUDED_TICKERS = {
    "2412.TW",  # Telecom
    "1216.TW",  # 食品
    "2881.TW",
    "2882.TW",
    "2883.TW",
    "2885.TW",
    "2886.TW",
}

LIQUIDITY_THRESHOLD = 500_000_000  # 平均每日成交金額下限：5 億 TWD


@dataclass
class CandidateMetrics:
    ticker: str
    sector: str
    ann_vol: float
    beta: float
    avg_turnover: float


def compute_daily_returns(df: pd.DataFrame) -> pd.Series:
    """從 OHLCV DataFrame 計算日報酬（以收盤價簡單報酬）."""
    close = df["Close"].astype(float)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return ret


def restrict_last_n_years(df: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """僅保留最近 N 年資料."""
    if df.empty:
        return df
    end = df.index.max()
    start = end - pd.DateOffset(years=years)
    return df.loc[start:end].copy()


def compute_beta(
    asset_ret: pd.Series,
    market_ret: pd.Series,
) -> float:
    """計算資產相對市場的 Beta 值."""
    aligned_asset, aligned_mkt = asset_ret.align(market_ret, join="inner")
    if len(aligned_asset) < 30:
        return np.nan
    cov = np.cov(aligned_asset, aligned_mkt, ddof=1)[0, 1]
    var_mkt = np.var(aligned_mkt, ddof=1)
    if var_mkt == 0:
        return np.nan
    return float(cov / var_mkt)


def gather_candidate_metrics(
    loader: MarketDataLoader,
    market_ret_3y: pd.Series,
) -> List[CandidateMetrics]:
    """對 0050 前 20 檔股票計算 3 年期 Vol / Beta / 流動性."""
    candidates: List[CandidateMetrics] = []

    for ticker, sector in UNIVERSE_0050_TOP20.items():
        # 1) 下載個股資料（台股）
        df = loader.fetch_data(tickers=[ticker], market_type="TW")
        if df.empty or "Close" not in df.columns or "Volume" not in df.columns:
            continue

        df_3y = restrict_last_n_years(df, years=3)
        if df_3y.empty:
            continue

        # 2) 計算日報酬與年化波動度
        ret = compute_daily_returns(df_3y)
        if ret.empty:
            continue
        ann_vol = float(ret.std(ddof=1) * np.sqrt(252.0))

        # 3) 計算 Beta（相對 0050 日報酬）
        beta = compute_beta(ret, market_ret_3y)

        # 4) 計算平均每日成交金額（TWD）
        turnover = (df_3y["Close"].astype(float) * df_3y["Volume"].astype(float)).replace(
            [np.inf, -np.inf], np.nan
        )
        avg_turnover = float(turnover.mean(skipna=True))

        candidates.append(
            CandidateMetrics(
                ticker=ticker,
                sector=sector,
                ann_vol=ann_vol,
                beta=beta,
                avg_turnover=avg_turnover,
            )
        )

    return candidates


def select_top_5_high_beta(candidates: List[CandidateMetrics]) -> List[CandidateMetrics]:
    """根據流動性與行業過濾後，依年化波動度挑出前 5 名."""
    filtered = [
        c
        for c in candidates
        if c.avg_turnover >= LIQUIDITY_THRESHOLD and c.ticker not in EXCLUDED_TICKERS
    ]

    if not filtered:
        return []

    # 依年化波動度由高到低排序
    filtered.sort(key=lambda x: x.ann_vol, reverse=True)
    return filtered[:5]


def main() -> None:
    """篩選高 Beta 台股，並執行 Model D 回測比較."""
    set_random_seed(42)

    loader = MarketDataLoader()

    # 1) 市場指數：0050.TW，作為 Beta 參考與時間對齊
    mkt_df = loader.fetch_data(tickers=["0050.TW"], market_type="TW")
    if mkt_df.empty or "Close" not in mkt_df.columns:
        print("[ERROR] 無法取得 0050.TW 資料，無法計算 Beta。")
        return

    mkt_df_3y = restrict_last_n_years(mkt_df, years=3)
    market_ret_3y = compute_daily_returns(mkt_df_3y)

    # 2) 對 0050 前 20 檔股票計算 Vol / Beta / 流動性
    candidates = gather_candidate_metrics(loader, market_ret_3y)
    if not candidates:
        print("[ERROR] 無可用候選股票，請檢查資料來源。")
        return

    # 3) 依條件選出前 5 檔高 Beta / 高波動、且流動性充足的標的
    top5 = select_top_5_high_beta(candidates)
    if not top5:
        print("[ERROR] 依條件篩選後沒有任何股票符合高波動 + 高流動性。")
        return

    print("\n" + "=" * 80)
    print("Step 1 - Screening High-Beta / High-Volatility Candidates (Past 3 Years)")
    print("=" * 80)
    print(
        "Columns: Ticker | Sector | Ann.Vol% | Beta vs 0050 | Avg Turnover (TWD, M)"
    )
    print("-" * 80)
    for c in top5:
        print(
            f"{c.ticker:>8} | {c.sector:<18} | "
            f"{c.ann_vol*100:6.2f}% | {c.beta:6.2f} | {c.avg_turnover/1e6:10.1f}M"
        )

    # 4) 對這 5 檔股票執行 Model D 策略回測（使用 multi_asset_validation.test_strategy）
    print("\n" + "=" * 80)
    print("Step 2 - Running Model D Strategy on Selected Candidates")
    print("=" * 80)

    results: List[Dict[str, float]] = []
    for c in top5:
        cfg = AssetConfig(ticker=c.ticker, asset_type=f"High Beta (Screened)", market_type="TW")
        res = test_strategy(cfg)
        if res is not None:
            results.append(res)

    if not results:
        print("[ERROR] 無任何標的完成 Model D 回測。")
        return

    df = pd.DataFrame(results)
    # 只保留關鍵欄位，並轉成百分比
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

    pct_cols = ["Strat_CAGR", "BH_CAGR", "WinRate", "MaxDD_Strat", "MaxDD_BH"]
    for col in pct_cols:
        df_display[col] = df_display[col] * 100.0

    print("\n" + "=" * 80)
    print("Step 3 - Cross-Sectional Performance: Model D vs Buy & Hold")
    print("=" * 80)
    print(
        "Columns: Ticker | Strat CAGR% | B&H CAGR% | "
        "Sharpe (Strat/BH) | WinRate% | MaxDD% (Strat/BH)"
    )
    print("-" * 80)

    for _, row in df_display.iterrows():
        print(
            f"{row['Ticker']:>8} | "
            f"{row['Strat_CAGR']:6.2f}% | {row['BH_CAGR']:6.2f}% | "
            f"{row['Strat_Sharpe']:.2f}/{row['BH_Sharpe']:.2f} | "
            f"{row['WinRate']:6.2f}% | "
            f"{row['MaxDD_Strat']:6.2f}% / {row['MaxDD_BH']:6.2f}%"
        )

    print("=" * 80)
    print("註：以上結果皆為 Walk-Forward OOS，並套用與 2330.TW 相同的 Pure Quantum Risk Control。")


if __name__ == "__main__":
    main()

