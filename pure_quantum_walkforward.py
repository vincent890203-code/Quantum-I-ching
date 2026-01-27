"""Pure Quantum Model Walk-Forward Backtest.

目的：
- 嚴格檢驗「純易經訊號」是否具獨立預測力（不依賴價格水準 / 趨勢）。

特徵限制（Pure Quantum）：
- 僅使用 3 個特徵：
  1) Moving_Lines_Count  （易經：動爻數量）
  2) Energy_Delta        （易經：能量變化）
  3) Daily_Return        （市場動能，較為平穩的 stationary 特徵）

排除：
- Close, Volume, RVOL 等所有價格水準 / 成交量水準類特徵，降低「順著牛市趨勢」的風險。

模型：
- XGBoost 二元分類器，超參數與先前 Light Model 相同。

標籤：
- 與既有 PnL 設計一致：
  y = 1  若 |Return(T -> T+5)| > volatility_threshold
  y = 0  否則

回測：
- Walk-Forward 時間序列驗證（train 12M / test 1M / step 1M）
- 在每個測試視窗內：
  - 若預測 y_hat = 1，則進場「Long Volatility」交易：
      * 單筆 PnL = |Return(T -> T+5)|        （此處忽略交易成本，專注在訊號本身 alpha）
  - 若預測 y_hat = 0，則不交易（PnL = 0）

評估：
- Cumulative Return：Pure Quantum vs Buy & Hold（單純持有標的）
- Win Rate：有交易日中 PnL > 0 的比例
- Sharpe Ratio：以每日 PnL 為基礎（假設 252 交易日/年）
- 繪製 Equity Curve。
"""

import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


# =============================================================================
# Matplotlib 中文字型設定
# =============================================================================

def setup_chinese_font() -> None:
    """設置 matplotlib 使用中文字體."""
    chinese_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Microsoft JhengHei",
        "KaiTi",
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_to_use = None

    for font in chinese_fonts:
        if font in available_fonts:
            font_to_use = font
            break

    if font_to_use:
        plt.rcParams["font.sans-serif"] = [font_to_use]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[INFO] 已設置中文字體: {font_to_use}")
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("[WARNING] 未找到中文字體，可能無法正確顯示中文")


setup_chinese_font()


# Walk-Forward 參數（與現有系統對齊）
WALK_FORWARD_TRAIN_MONTHS = 12
WALK_FORWARD_TEST_MONTHS = 1
WALK_FORWARD_STEP_MONTHS = 1


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# 資料準備：僅保留 Moving_Lines_Count / Energy_Delta / Daily_Return
# =============================================================================

def prepare_pure_quantum_tabular(
    encoded_data: pd.DataFrame,
    prediction_window: int = 5,
    volatility_threshold: float = 0.03,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """準備 Pure Quantum 模型使用的表格資料.

    僅使用 3 個特徵：
        - Daily_Return
        - Moving_Lines_Count
        - Energy_Delta
    """
    print("\n[INFO] 準備 Pure Quantum tabular 資料...")

    required_cols = ["Close", "Daily_Return", "Ritual_Sequence"]
    missing = [c for c in required_cols if c not in encoded_data.columns]
    if missing:
        raise ValueError(f"編碼資料缺少必要欄位: {missing}")

    processor = DataProcessor()

    # 1) 從 Ritual_Sequence 中萃取易經特徵
    print("[INFO] 從 Ritual_Sequence 提取易經特徵 (Moving_Lines_Count, Energy_Delta)...")
    all_iching_features = [
        "Yang_Count_Main",
        "Yang_Count_Future",
        "Moving_Lines_Count",
        "Energy_Delta",
        "Conflict_Score",
    ]
    iching_features_list: list[list[float]] = []

    for idx, ritual_seq in enumerate(encoded_data["Ritual_Sequence"]):
        if pd.isna(ritual_seq) or len(str(ritual_seq)) != 6:
            iching_features_list.append([0.0] * len(all_iching_features))
        else:
            try:
                feats = processor.extract_iching_features(str(ritual_seq))
                iching_features_list.append(feats.tolist())
            except (ValueError, TypeError) as e:
                print(f"[WARNING] 無法提取易經特徵: {ritual_seq}, 錯誤: {e}")
                iching_features_list.append([0.0] * len(all_iching_features))

        if (idx + 1) % 1000 == 0:
            print(f"[INFO] 已處理 {idx + 1}/{len(encoded_data)} 筆易經特徵")

    iching_df_full = pd.DataFrame(
        iching_features_list,
        columns=all_iching_features,
        index=encoded_data.index,
    )

    # 僅保留 Moving_Lines_Count 與 Energy_Delta
    iching_pure = iching_df_full[["Moving_Lines_Count", "Energy_Delta"]].copy()

    # 2) 構建特徵矩陣 X：Daily_Return + 兩個 I-Ching 特徵
    X = pd.concat(
        [
            encoded_data[["Daily_Return"]].copy(),
            iching_pure,
        ],
        axis=1,
    )

    # 3) 構建標籤：T->T+5 絕對報酬是否大於閾值（與原系統一致）
    max_idx = len(encoded_data) - prediction_window
    future_prices = encoded_data["Close"].shift(-prediction_window)
    current_prices = encoded_data["Close"]
    future_returns = (future_prices - current_prices) / current_prices

    y = (future_returns.abs() > volatility_threshold).astype(int)
    actual_returns = future_returns.copy()

    X = X.iloc[:max_idx].copy()
    y = y.iloc[:max_idx].copy()
    actual_returns = actual_returns.iloc[:max_idx].copy()
    dates_before_filter = encoded_data.index[:max_idx]

    # 4) 移除 NaN 行
    valid_mask = ~(X.isna().any(axis=1) | y.isna() | actual_returns.isna())
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    actual_returns = actual_returns[valid_mask].copy()
    dates = dates_before_filter[valid_mask]

    print("[INFO] Pure Quantum 資料準備完成:")
    print(f"  總樣本數: {len(X)}")
    print(f"  特徵欄位: {list(X.columns)}")
    print(f"  標籤分布: 高波動={int(y.sum())}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")

    return X, y, actual_returns, dates


def build_xgb_classifier(feature_names: List[str]) -> xgb.XGBClassifier:
    """建立 XGBoost 分類器（超參數與 Light Model 一致）."""
    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    model = xgb.XGBClassifier(**params)
    # booster 的 feature_names 會在 fit 之後再設定
    return model


# =============================================================================
# Walk-Forward 驗證
# =============================================================================

def run_walk_forward_binary(
    X: pd.DataFrame,
    y: pd.Series,
    actual_returns: pd.Series,
    dates: pd.DatetimeIndex,
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    test_months: int = WALK_FORWARD_TEST_MONTHS,
    step_months: int = WALK_FORWARD_STEP_MONTHS,
    model_name: str = "Pure Quantum",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Walk-Forward 驗證：回傳 OOS 的二元預測與對應未來報酬.

    Returns:
        predictions_oos: 串接後的 OOS 預測 (0/1)
        actual_oos:      對應的 T->T+5 實際報酬
        dates_oos:       對應的日期索引
    """
    dti = pd.DatetimeIndex(dates)
    periods = pd.Series(dti.to_period("M"), index=range(len(dti)))
    unique_periods = sorted(periods.unique())
    n_periods = len(unique_periods)

    if n_periods < train_months + test_months:
        raise ValueError(
            f"資料月數 {n_periods} 不足 Walk-Forward 所需 "
            f"train_months + test_months = {train_months + test_months}"
        )

    preds_list: List[np.ndarray] = []
    actual_list: List[np.ndarray] = []
    dates_list: List[pd.DatetimeIndex] = []
    feature_names = list(X.columns)

    i = 0
    fold = 0
    while i + train_months + test_months <= n_periods:
        train_periods = unique_periods[i : i + train_months]
        test_periods = unique_periods[i + train_months : i + train_months + test_months]

        train_mask = periods.isin(train_periods).to_numpy()
        test_mask = periods.isin(test_periods).to_numpy()

        X_train = X.iloc[train_mask].copy()
        y_train = y.iloc[train_mask].copy()
        X_test = X.iloc[test_mask].copy()
        actual_test = actual_returns.iloc[test_mask].copy()
        dates_test = dti[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            i += step_months
            continue

        model = build_xgb_classifier(feature_names)
        model.fit(X_train, y_train, verbose=False)
        model.get_booster().feature_names = feature_names

        preds_fold = model.predict(X_test)

        preds_list.append(preds_fold)
        actual_list.append(actual_test.values)
        dates_list.append(dates_test)

        fold += 1
        if fold <= 3 or fold % 12 == 0:
            print(
                f"[Walk-Forward] {model_name} fold {fold}: "
                f"train {train_periods[0]}~{train_periods[-1]}, "
                f"test {test_periods[0]}~{test_periods[-1]}, "
                f"n_test={len(X_test)}"
            )

        i += step_months

    if not preds_list:
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    predictions_oos = np.concatenate(preds_list)
    actual_oos = np.concatenate(actual_list)
    dates_oos = pd.DatetimeIndex(np.concatenate([d.values for d in dates_list]))

    print(
        f"[Walk-Forward] {model_name} 共 {fold} 個 fold，OOS 樣本數 = {len(predictions_oos)}"
    )
    return predictions_oos, actual_oos, dates_oos


# =============================================================================
# PnL 與績效計算
# =============================================================================

def simulate_pure_quantum_pnl(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> np.ndarray:
    """根據二元預測與實際 T->T+5 報酬，計算每日 PnL.

    策略邏輯（純訊號檢驗版，方向性報酬）：
    - 若預測 = 1（高波動）：進場
        * 單筆 PnL = Return(T->T+5)
    - 若預測 = 0：不交易（PnL = 0）

    回傳：
        pnl: 每個樣本對應的日 PnL。
    """
    pnl = np.zeros_like(actual_returns, dtype=float)
    mask = predictions == 1
    # 僅在有訊號時承擔未來報酬（可能為正或負）
    pnl[mask] = actual_returns[mask]
    return pnl


def compute_win_rate_and_sharpe(
    pnl: np.ndarray,
    predictions: np.ndarray,
    annualization: int = 252,
) -> Tuple[float, float]:
    """計算 Win Rate 與 Sharpe Ratio.

    Win Rate:
        - 僅計算「有交易的日子」（predictions == 1）中，PnL > 0 的比例。

    Sharpe Ratio:
        - 使用「每日 PnL」序列（含 0，代表未交易日）
        - Sharpe = mean(pnl) / std(pnl) * sqrt(annualization)
    """
    # Win Rate（以有交易日為母數）
    trade_mask = predictions == 1
    trades = pnl[trade_mask]
    if trades.size == 0:
        win_rate = 0.0
    else:
        wins = (trades > 0).sum()
        win_rate = wins / trades.size

    # Sharpe Ratio
    mu = pnl.mean()
    sigma = pnl.std(ddof=1)
    if sigma == 0:
        sharpe = 0.0
    else:
        sharpe = mu / sigma * np.sqrt(annualization)

    return win_rate, sharpe


def plot_equity_curve(
    dates: pd.DatetimeIndex,
    pnl_quantum: np.ndarray,
    returns_buyhold: np.ndarray,
    save_path: str = "data/pure_quantum_equity_curve.png",
) -> None:
    """繪製 Pure Quantum vs Buy & Hold 累積收益曲線."""
    print("\n[INFO] 繪製 Equity Curve (Pure Quantum vs Buy & Hold)...")

    cum_quantum = np.cumsum(pnl_quantum)
    cum_buyhold = np.cumsum(returns_buyhold)

    total_quantum = cum_quantum[-1] if len(cum_quantum) > 0 else 0.0
    total_buyhold = cum_buyhold[-1] if len(cum_buyhold) > 0 else 0.0

    print("[RESULT] 累積收益（以 OOS 區間為準）：")
    print(f"  Pure Quantum: {total_quantum:.4f} ({total_quantum*100:.2f}%)")
    print(f"  Buy & Hold : {total_buyhold:.4f} ({total_buyhold*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.set_facecolor("white")

    # Pure Quantum：紅色
    ax.plot(
        dates,
        cum_quantum,
        label=f"Pure Quantum (總收益: {total_quantum*100:.2f}%)",
        linewidth=2.5,
        color="#FF6B6B",
        alpha=0.9,
    )

    # Buy & Hold：綠色虛線
    ax.plot(
        dates,
        cum_buyhold,
        label=f"Buy & Hold (總收益: {total_buyhold*100:.2f}%)",
        linewidth=2,
        color="#2ECC71",
        linestyle="--",
        alpha=0.8,
    )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.set_title(
        "Pure Quantum Model vs Buy & Hold (Walk-Forward OOS)",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#333333",
    )
    ax.set_xlabel("日期", fontsize=13, color="#333333")
    ax.set_ylabel("累積收益 (Cumulative Return)", fontsize=13, color="#333333")

    ax.legend(
        loc="best",
        fontsize=11,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        labelcolor="#333333",
    )
    ax.grid(True, alpha=0.3, linestyle="--", color="gray")
    ax.tick_params(colors="#333333")
    fig.autofmt_xdate()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[INFO] Equity Curve 圖已保存至: {save_path}")
    plt.close()


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    """執行 Pure Quantum 模型的 Walk-Forward 回測."""
    print("=" * 80)
    print("Pure Quantum Model Walk-Forward Backtest")
    print("=" * 80)

    set_random_seed(42)

    # 1. 載入並編碼市場資料
    print("\n[INFO] 載入 TSMC (2330.TW) 市場資料...")
    loader = MarketDataLoader()
    raw_data = loader.fetch_data(tickers=["2330.TW"], market_type="TW")

    if raw_data.empty:
        print("[ERROR] 無法獲取 2330.TW 的市場資料")
        return

    print("[INFO] 編碼為易經卦象...")
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)

    if encoded_data.empty:
        print("[ERROR] 編碼後的資料為空")
        return

    # 2. 準備 Pure Quantum tabular 資料
    X_pure, y_pure, actual_returns, dates = prepare_pure_quantum_tabular(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
    )

    # 3. Walk-Forward 回測（使用全部資料做 rolling OOS）
    print("\n[INFO] Walk-Forward 驗證（train 12M / test 1M / step 1M）...")
    preds_oos, actual_oos, dates_oos = run_walk_forward_binary(
        X_pure,
        y_pure,
        actual_returns,
        dates,
        train_months=WALK_FORWARD_TRAIN_MONTHS,
        test_months=WALK_FORWARD_TEST_MONTHS,
        step_months=WALK_FORWARD_STEP_MONTHS,
        model_name="Pure Quantum",
    )

    if preds_oos.size == 0:
        print("[WARNING] Walk-Forward 未產生任何 OOS 樣本，請檢查資料長度是否足夠。")
        return

    # 4. 計算策略 PnL 與 Buy & Hold
    pnl_quantum = simulate_pure_quantum_pnl(preds_oos, actual_oos)
    returns_buyhold = actual_oos  # Buy & Hold：單純持有標的

    # 5. Win Rate & Sharpe Ratio
    win_rate, sharpe = compute_win_rate_and_sharpe(pnl_quantum, preds_oos)

    print("\n[RESULT] 策略績效指標（Walk-Forward OOS）：")
    print(f"  Win Rate (有交易日): {win_rate:.2%}")
    print(f"  Sharpe Ratio      : {sharpe:.4f}")

    # 6. Equity Curve 圖
    plot_equity_curve(
        dates_oos,
        pnl_quantum,
        returns_buyhold,
        save_path="data/pure_quantum_equity_curve.png",
    )

    print("\n" + "=" * 80)
    print("完成：Pure Quantum 模型 Walk-Forward 回測")
    print("=" * 80)
    print("\n[SUCCESS] 輸出結果：")
    print("  - data/pure_quantum_equity_curve.png (Equity Curve: Pure Quantum vs Buy & Hold)")
    print(f"  - Win Rate (trade days): {win_rate:.2%}")
    print(f"  - Sharpe Ratio         : {sharpe:.4f}")


if __name__ == "__main__":
    main()

