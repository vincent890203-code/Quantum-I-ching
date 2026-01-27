"""Quant Performance Report (Tear Sheet) for Model D - Pure Quantum.

模型：Model D (Pure Quantum Strategy)
特徵：['Moving_Lines_Count', 'Energy_Delta', 'Daily_Return']

目標：
- 使用 Walk-Forward Validation 產生 Model D 的完整量化績效報告：
  1) ROC Curve + AUC
  2) Cumulative Returns (Equity Curve): Model D Strategy vs Buy & Hold
  3) Underwater Plot (Drawdown)
  4) 關鍵績效指標：CAGR / Sharpe / Max Drawdown / Win Rate / Profit Factor

說明：
- 訓練與回測皆使用滾動視窗（train 12M / test 1M / step 1M），避免 look-ahead bias。
- 策略邏輯（純訊號檢驗版，方向性報酬）：
    * 若 y_proba >= 0.5 -> 進場，單日報酬 = Return_5d
    * 若 y_proba < 0.5 -> 不交易，報酬 = 0
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)

from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


# =============================================================================
# Matplotlib 字型設定（支援中文）
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


WALK_FORWARD_TRAIN_MONTHS = 12
WALK_FORWARD_TEST_MONTHS = 1
WALK_FORWARD_STEP_MONTHS = 1

DEFAULT_THRESHOLD = 0.5     # 訊號閾值（可日後優化）
TARGET_VOL = 0.015          # 目標日波動（1.5%）
MAX_DRAWDOWN_TRIGGER = -0.20  # 進入「生存模式」的回撤門檻（-20%）
SURVIVAL_POS_CAP = 0.2        # 生存模式下的最大部位（20%）


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# 資料準備：Pure Quantum 特徵
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

    Returns:
        X: 特徵 DataFrame
        y: 標籤 Series (0/1)
        actual_returns: 未來 T->T+prediction_window 的實際報酬
        dates: 對應每列的日期索引
    """
    print("\n[INFO] 準備 Pure Quantum tabular 資料 (Model D)...")

    required_cols = ["Close", "Daily_Return", "Ritual_Sequence"]
    missing = [c for c in required_cols if c not in encoded_data.columns]
    if missing:
        raise ValueError(f"編碼資料缺少必要欄位: {missing}")

    processor = DataProcessor()

    # 1) 由 Ritual_Sequence 提取易經特徵
    print("[INFO] 從 Ritual_Sequence 提取易經特徵...")
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
    iching_pure = iching_df_full[["Moving_Lines_Count", "Energy_Delta"]].copy()

    # 2) 構建特徵 X：Daily_Return + 兩個 I-Ching 特徵
    X = pd.concat(
        [
            encoded_data[["Daily_Return"]].copy(),
            iching_pure,
        ],
        axis=1,
    )

    # 3) 標籤與未來報酬
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
    """建立 XGBoost 分類器（與 Model D 一致的超參數）."""
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
    return model


# =============================================================================
# Walk-Forward: 機率預測 (y_proba) + OOS 樣本
# =============================================================================

def run_walk_forward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    actual_returns: pd.Series,
    dates: pd.DatetimeIndex,
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    test_months: int = WALK_FORWARD_TEST_MONTHS,
    step_months: int = WALK_FORWARD_STEP_MONTHS,
    model_name: str = "Model D (Pure Quantum)",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Walk-Forward 驗證：回傳 OOS 的機率預測與標籤與未來報酬."""
    dti = pd.DatetimeIndex(dates)
    periods = pd.Series(dti.to_period("M"), index=range(len(dti)))
    unique_periods = sorted(periods.unique())
    n_periods = len(unique_periods)

    if n_periods < train_months + test_months:
        raise ValueError(
            f"資料月數 {n_periods} 不足 Walk-Forward 所需 "
            f"train_months + test_months = {train_months + test_months}"
        )

    proba_list: List[np.ndarray] = []
    y_true_list: List[np.ndarray] = []
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
        y_test = y.iloc[test_mask].copy()
        actual_test = actual_returns.iloc[test_mask].copy()
        dates_test = dti[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            i += step_months
            continue

        model = build_xgb_classifier(feature_names)
        model.fit(X_train, y_train, verbose=False)
        try:
            model.get_booster().feature_names = feature_names
        except Exception:
            pass

        proba_fold = model.predict_proba(X_test)[:, 1]

        proba_list.append(proba_fold)
        y_true_list.append(y_test.values)
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

    if not proba_list:
        return np.array([]), np.array([]), np.array([]), pd.DatetimeIndex([])

    y_proba_oos = np.concatenate(proba_list)
    y_true_oos = np.concatenate(y_true_list)
    actual_oos = np.concatenate(actual_list)
    dates_oos = pd.DatetimeIndex(np.concatenate([d.values for d in dates_list]))

    print(
        f"[Walk-Forward] {model_name} 共 {fold} 個 fold，OOS 樣本數 = {len(y_proba_oos)}"
    )
    return y_proba_oos, y_true_oos, actual_oos, dates_oos


# =============================================================================
# 策略邏輯與績效計算
# =============================================================================

def generate_trading_pnl(
    y_proba: np.ndarray,
    actual_returns: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """根據機率與閾值產生交易訊號與每日 PnL（方向性報酬）.
    
    策略：
        - y_proba >= threshold -> signal = 1, 當日 PnL = actual_return
        - y_proba < threshold  -> signal = 0, 當日 PnL = 0
    
    注意：
        - 僅使用模型在時間 t 的預測（y_proba[t]）決定是否參與
          未來視窗 t->t+5 的實際報酬（actual_returns[t]），
          不會使用 y_true 或未來資訊產生訊號。
    """
    signals = (y_proba >= threshold).astype(int)
    pnl = np.zeros_like(actual_returns, dtype=float)
    mask = signals == 1
    # 僅在有訊號時承擔未來報酬（可能為正或負）
    pnl[mask] = actual_returns[mask]
    return signals, pnl


def compute_equity_curve(returns: np.ndarray) -> np.ndarray:
    """根據每日報酬率建立權益曲線（從 1.0 開始）."""
    return np.cumprod(1.0 + returns)


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    """根據權益曲線計算 drawdown 序列（負值表示回撤）."""
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd


def compute_performance_metrics(
    dates: pd.DatetimeIndex,
    strategy_returns: np.ndarray,
    signals: np.ndarray,
) -> dict:
    """計算 CAGR / Sharpe / Max Drawdown / Win Rate / Profit Factor."""
    n_days = len(strategy_returns)
    if n_days == 0:
        return {}

    equity = compute_equity_curve(strategy_returns)
    total_return = equity[-1] - 1.0

    # CAGR
    years = n_days / 252.0
    cagr = (equity[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    # Sharpe Ratio（假設每日報酬為獨立樣本）
    mu = strategy_returns.mean()
    sigma = strategy_returns.std(ddof=1)
    if sigma == 0:
        sharpe = 0.0
    else:
        sharpe = mu / sigma * np.sqrt(252.0)

    # Max Drawdown
    dd = compute_drawdown(equity)
    max_drawdown = dd.min()  # 負值，代表最大回撤

    # Win Rate（僅計算有交易日）
    trade_mask = signals == 1
    trade_returns = strategy_returns[trade_mask]
    if trade_returns.size == 0:
        win_rate = 0.0
        profit_factor = 0.0
    else:
        wins = (trade_returns > 0).sum()
        win_rate = wins / trade_returns.size

        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = -trade_returns[trade_returns < 0].sum()
        if gross_loss <= 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Num Days": n_days,
        "Num Trades": int(trade_mask.sum()),
        "Start Date": dates[0].strftime("%Y-%m-%d"),
        "End Date": dates[-1].strftime("%Y-%m-%d"),
    }
    return metrics


# =============================================================================
# 視覺化：ROC / Equity Curve / Underwater Plot
# =============================================================================

def ensure_output_dir(path: str = "output") -> str:
    """確保 output 目錄存在."""
    os.makedirs(path, exist_ok=True)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str,
) -> float:
    """繪製 ROC 曲線並保存，回傳 AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#FF6B6B", lw=2, label=f"Model D (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="No Skill")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Model D (Pure Quantum)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, linestyle="--", color="gray")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] ROC Curve 已保存至: {save_path}")
    return auc


def plot_equity_curve(
    dates: pd.DatetimeIndex,
    strategy_returns: np.ndarray,
    buyhold_returns: np.ndarray,
    save_path: str,
) -> None:
    """繪製策略 vs Buy & Hold 的累積收益曲線."""
    equity_strategy = compute_equity_curve(strategy_returns)
    equity_buyhold = compute_equity_curve(buyhold_returns)

    total_ret_strategy = equity_strategy[-1] - 1.0
    total_ret_buyhold = equity_buyhold[-1] - 1.0
    diff = total_ret_strategy - total_ret_buyhold

    plt.figure(figsize=(12, 7))
    plt.plot(
        dates,
        equity_strategy - 1.0,
        label=f"Model D Strategy (Total: {total_ret_strategy*100:.2f}%)",
        color="#FF6B6B",
        lw=2.5,
    )
    plt.plot(
        dates,
        equity_buyhold - 1.0,
        label=f"Buy & Hold (Total: {total_ret_buyhold*100:.2f}%)",
        color="#2ECC71",
        lw=2,
        linestyle="--",
    )
    plt.axhline(0, color="black", lw=0.8, alpha=0.3)
    plt.title("Cumulative Returns - Model D Strategy vs Buy & Hold (Walk-Forward OOS)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3, linestyle="--", color="gray")
    plt.legend(loc="best")

    text_color = "#2ECC71" if diff > 0 else "#E74C3C"
    text = f"Final Difference (Model D - Buy & Hold): {diff*100:.2f}%"
    plt.text(
        0.02,
        0.98,
        text,
        transform=plt.gca().transAxes,
        fontsize=11,
        color="#333333",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor=text_color,
            linewidth=2,
        ),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Equity Curve 已保存至: {save_path}")


def plot_underwater(
    dates: pd.DatetimeIndex,
    strategy_returns: np.ndarray,
    save_path: str,
) -> None:
    """繪製 Underwater Plot（Drawdown 百分比隨時間）."""
    equity = compute_equity_curve(strategy_returns)
    dd = compute_drawdown(equity) * 100.0  # 轉為百分比

    plt.figure(figsize=(12, 4))
    plt.fill_between(dates, dd, 0, color="#3498DB", alpha=0.6)
    plt.axhline(0, color="black", lw=0.8, alpha=0.5)
    plt.title("Underwater Plot (Drawdown %) - Model D Strategy")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3, linestyle="--", color="gray")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Underwater Plot 已保存至: {save_path}")


# =============================================================================
# 主流程：生成 Tear Sheet
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("Quant Performance Report - Model D (Pure Quantum, Walk-Forward OOS)")
    print("=" * 80)

    set_random_seed(42)

    # 1. 載入並編碼市場資料
    print("\n[INFO] 載入 TSMC (2330.TW) 市場資料...")
    loader = MarketDataLoader()
    raw_data = loader.fetch_data(tickers=["2330.TW"], market_type="TW")

    if raw_data.empty:
        print("[ERROR] 無法獲取市場資料，請檢查 data_loader 與網路環境。")
        return

    print("[INFO] 編碼為易經卦象...")
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)

    if encoded_data.empty:
        print("[ERROR] 編碼後資料為空，無法生成報告。")
        return

    # 2. 準備 Pure Quantum tabular 資料
    X, y, actual_returns, dates = prepare_pure_quantum_tabular(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
    )

    # 3. Walk-Forward 機率預測（OOS）
    print("\n[INFO] Walk-Forward 驗證（train 12M / test 1M / step 1M）...")
    y_proba_oos, y_true_oos, actual_oos, dates_oos = run_walk_forward_proba(
        X,
        y,
        actual_returns,
        dates,
        train_months=WALK_FORWARD_TRAIN_MONTHS,
        test_months=WALK_FORWARD_TEST_MONTHS,
        step_months=WALK_FORWARD_STEP_MONTHS,
        model_name="Model D (Pure Quantum)",
    )

    if y_proba_oos.size == 0:
        print("[WARNING] Walk-Forward 未產生任何 OOS 樣本，請檢查資料長度是否足夠。")
        return

    # 4. 產生策略 PnL 與 Buy & Hold 報酬
    # 4.1 策略報酬（以 5 日視窗報酬作為單筆交易報酬）
    signals, pnl_strategy_raw = generate_trading_pnl(
        y_proba=y_proba_oos,
        actual_returns=actual_oos,
        threshold=DEFAULT_THRESHOLD,
    )

    # 4.2 Buy & Hold 報酬：使用每日簡單報酬，並在 OOS 日期範圍內累積
    #     cumulative_return = (1 + daily_returns).cumprod() - 1
    close_series = encoded_data["Close"].astype(float)
    # 將索引統一為 tz-naive，並壓平成日期級別，以避免時間部分造成 reindex 失配
    idx = close_series.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close_series.index = idx

    # 以「日期」為索引，取每日最後一筆收盤價來計算日報酬
    date_index = close_series.index.normalize()
    daily_close = close_series.groupby(date_index).last()
    daily_ret_full = daily_close.pct_change().fillna(0.0)  # 已為小數，例如 0.015 代表 1.5%

    # OOS 日期同樣壓平成日期，以確保能與 daily_ret_full 對齊
    if getattr(dates_oos, "tz", None) is not None:
        dates_oos = dates_oos.tz_localize(None)
    dates_oos_dates = dates_oos.normalize()

    # 使用與策略相同的 OOS 日期索引（日期粒度），對齊 Buy & Hold 的日報酬
    daily_ret_oos = daily_ret_full.reindex(dates_oos_dates).fillna(0.0)
    returns_buyhold = daily_ret_oos.to_numpy()

    # 4.3 Volatility Targeting：根據 20 日滾動波動度縮放部位大小（僅對策略，Baseline 不縮放）
    #     使用「前一日」的 20 日日波動度估計，並考慮 5 日持有期間的放大效果：
    #     sigma_d  = rolling 20d std of daily returns
    #     sigma_5  ≈ sigma_d * sqrt(5)
    #     Position_Size_t = min(1.0, Target_Vol / sigma_5_{t-1})
    rolling_vol = daily_ret_full.rolling(window=20, min_periods=5).std()
    # 使用 t-1 的波動度（shift(1)），再對齊到 OOS 日期
    vol_oos = rolling_vol.shift(1).reindex(dates_oos)
    pos_size = np.ones_like(pnl_strategy_raw, dtype=float)
    vol_values = vol_oos.to_numpy()
    # 將日波動度換算為 5 日持有期間的預期波動度
    eff_vol_5d = vol_values * np.sqrt(5.0)
    valid_vol_mask = eff_vol_5d > 0
    # 動態倉位：position_size = min(1.0, target_vol / current_vol_5d)
    pos_size[valid_vol_mask] = np.minimum(1.0, TARGET_VOL / eff_vol_5d[valid_vol_mask])
    # 對 NaN 或 0 波動度，維持部位大小 = 1.0（不加槓桿也不縮放，等同全現金或全倉取決於 signal）

    # 4.4 Drawdown Survival Mode：
    #     若策略從高點回撤超過 20%，則將當期部位 cap 在 20%（而非完全退出），
    #     讓策略以「最小倉位」持續參與市場，隨著回撤收斂自然恢復風險承擔能力。
    equity = 1.0
    peak = 1.0
    for i in range(len(pnl_strategy_raw)):
        # 根據目前累積回撤決定是否需要進入「生存模式」
        current_drawdown = equity / peak - 1.0
        if current_drawdown <= MAX_DRAWDOWN_TRIGGER:
            # 進入生存模式：將當日部位上限壓到 SURVIVAL_POS_CAP
            pos_size[i] = min(pos_size[i], SURVIVAL_POS_CAP)

        # 以調整後的部位計算當日報酬，更新權益與高點
        ret_i = pnl_strategy_raw[i] * pos_size[i]
        equity *= (1.0 + ret_i)
        peak = max(peak, equity)

    pnl_strategy = pnl_strategy_raw * pos_size

    # 5. 計算績效指標（僅對策略，Baseline 只作為比較線）
    metrics = compute_performance_metrics(
        dates=dates_oos,
        strategy_returns=pnl_strategy,
        signals=signals,
    )

    # 6. 視覺化輸出
    output_dir = ensure_output_dir("output")

    # Figure A: ROC Curve
    roc_path = os.path.join(output_dir, "model_d_roc_curve.png")
    auc = plot_roc_curve(
        y_true=y_true_oos,
        y_proba=y_proba_oos,
        save_path=roc_path,
    )

    # Figure B: Cumulative Returns (Equity Curve)
    equity_path = os.path.join(output_dir, "model_d_equity_curve.png")
    plot_equity_curve(
        dates=dates_oos,
        strategy_returns=pnl_strategy,
        buyhold_returns=returns_buyhold,
        save_path=equity_path,
    )

    # Figure C: Underwater Plot
    dd_path = os.path.join(output_dir, "model_d_underwater.png")
    plot_underwater(
        dates=dates_oos,
        strategy_returns=pnl_strategy,
        save_path=dd_path,
    )

    # 7. 印出結構化摘要表
    print("\n" + "=" * 80)
    print("Model D - Pure Quantum Strategy (Walk-Forward OOS Tear Sheet)")
    print("=" * 80)
    print(f"Data Range     : {metrics['Start Date']} ~ {metrics['End Date']}")
    print(f"OOS Days       : {metrics['Num Days']}")
    print(f"Num Trades     : {metrics['Num Trades']}")
    print("-" * 80)
    print(f"Total Return   : {metrics['Total Return']*100:8.2f}%")
    print(f"CAGR           : {metrics['CAGR']*100:8.2f}%")
    print(f"Sharpe Ratio   : {metrics['Sharpe']:8.3f}")
    print(f"Max Drawdown   : {metrics['Max Drawdown']*100:8.2f}%")
    print(f"Win Rate       : {metrics['Win Rate']*100:8.2f}%")
    print(f"Profit Factor  : {metrics['Profit Factor']:8.3f}")
    print(f"AUC (ROC)      : {auc:8.3f}")
    print("-" * 80)
    print("Figures saved:")
    print(f"  - ROC Curve        : {roc_path}")
    print(f"  - Equity Curve     : {equity_path}")
    print(f"  - Underwater Plot  : {dd_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

