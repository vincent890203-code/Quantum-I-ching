"""Ablation Study: Model D (Pure Quantum) vs Model E (Momentum Only).

目標：
- 比較兩種模型在相同 Walk-Forward 設定與相同風控邏輯下的績效差異：
  * Model D：Pure Quantum（特徵 = [Daily_Return, Moving_Lines_Count, Energy_Delta]）
  * Model E：Momentum Only（特徵 = [Daily_Return]）

指標：
- Total Return
- Sharpe Ratio
- Max Drawdown
- Calmar Ratio = TotalReturn / |MaxDrawdown|

結論判斷：
- 若 Model D Sharpe 明顯較高或 Max DD 明顯較低，表示 I-Ching 特徵在風控 / 過濾波動上有實質貢獻。
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


WALK_FORWARD_TRAIN_MONTHS = 12
WALK_FORWARD_TEST_MONTHS = 1
WALK_FORWARD_STEP_MONTHS = 1

DEFAULT_THRESHOLD = 0.5      # 訊號閾值
TARGET_VOL = 0.015           # 目標日波動（與最新 Model D 一致：1.5%）
MAX_DRAWDOWN_TRIGGER = -0.20  # 進入「生存模式」的回撤門檻（-20%）
SURVIVAL_POS_CAP = 0.2        # 生存模式下最大部位（20%）


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# 資料準備：同一組 y / actual_returns / dates，產生兩組 X_D, X_E
# =============================================================================

def prepare_features_for_models(
    encoded_data: pd.DataFrame,
    prediction_window: int = 5,
    volatility_threshold: float = 0.03,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """建立 Model D / Model E 的輸入特徵，對齊同一組 y 與 actual_returns.

    Model D: X_D = [Daily_Return, Moving_Lines_Count, Energy_Delta]
    Model E: X_E = [Daily_Return]
    """
    required_cols = ["Close", "Daily_Return", "Ritual_Sequence"]
    missing = [c for c in required_cols if c not in encoded_data.columns]
    if missing:
        raise ValueError(f"編碼資料缺少必要欄位: {missing}")

    processor = DataProcessor()

    # 提取完整 I-Ching 特徵
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
            except (ValueError, TypeError):
                iching_features_list.append([0.0] * len(all_iching_features))

    iching_df_full = pd.DataFrame(
        iching_features_list,
        columns=all_iching_features,
        index=encoded_data.index,
    )
    iching_pure = iching_df_full[["Moving_Lines_Count", "Energy_Delta"]].copy()

    # Model D 特徵
    X_D = pd.concat(
        [
            encoded_data[["Daily_Return"]].copy(),
            iching_pure,
        ],
        axis=1,
    )

    # Model E 特徵（僅 Momentum）
    X_E = encoded_data[["Daily_Return"]].copy()

    # 標籤與未來報酬（與 Model D 相同定義）
    max_idx = len(encoded_data) - prediction_window
    future_prices = encoded_data["Close"].shift(-prediction_window)
    current_prices = encoded_data["Close"]
    future_returns = (future_prices - current_prices) / current_prices

    y = (future_returns.abs() > volatility_threshold).astype(int)
    actual_returns = future_returns.copy()

    # 截掉最後 prediction_window 筆
    X_D = X_D.iloc[:max_idx].copy()
    X_E = X_E.iloc[:max_idx].copy()
    y = y.iloc[:max_idx].copy()
    actual_returns = actual_returns.iloc[:max_idx].copy()
    dates_before_filter = encoded_data.index[:max_idx]

    # 以 X_D 為基準做 valid_mask，確保兩個模型看到同一批樣本
    valid_mask = ~(X_D.isna().any(axis=1) | y.isna() | actual_returns.isna())
    X_D = X_D[valid_mask].copy()
    X_E = X_E[valid_mask].copy()
    y = y[valid_mask].copy()
    actual_returns = actual_returns[valid_mask].copy()
    dates = dates_before_filter[valid_mask]

    return X_D, X_E, y, actual_returns, dates


def build_xgb_classifier(feature_names: List[str]) -> xgb.XGBClassifier:
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


def run_walk_forward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    actual_returns: pd.Series,
    dates: pd.DatetimeIndex,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """與 quant_report_model_d 中相同的 Walk-Forward 機率預測."""
    dti = pd.DatetimeIndex(dates)
    periods = pd.Series(dti.to_period("M"), index=range(len(dti)))
    unique_periods = sorted(periods.unique())
    n_periods = len(unique_periods)

    if n_periods < WALK_FORWARD_TRAIN_MONTHS + WALK_FORWARD_TEST_MONTHS:
        raise ValueError(
            f"資料月數 {n_periods} 不足 Walk-Forward 所需 "
            f"{WALK_FORWARD_TRAIN_MONTHS + WALK_FORWARD_TEST_MONTHS}"
        )

    proba_list: List[np.ndarray] = []
    y_true_list: List[np.ndarray] = []
    actual_list: List[np.ndarray] = []
    dates_list: List[pd.DatetimeIndex] = []
    feature_names = list(X.columns)

    i = 0
    fold = 0
    while i + WALK_FORWARD_TRAIN_MONTHS + WALK_FORWARD_TEST_MONTHS <= n_periods:
        train_periods = unique_periods[i : i + WALK_FORWARD_TRAIN_MONTHS]
        test_periods = unique_periods[
            i + WALK_FORWARD_TRAIN_MONTHS : i + WALK_FORWARD_TRAIN_MONTHS + WALK_FORWARD_TEST_MONTHS
        ]

        train_mask = periods.isin(train_periods).to_numpy()
        test_mask = periods.isin(test_periods).to_numpy()

        X_train = X.iloc[train_mask].copy()
        y_train = y.iloc[train_mask].copy()
        X_test = X.iloc[test_mask].copy()
        y_test = y.iloc[test_mask].copy()
        actual_test = actual_returns.iloc[test_mask].copy()
        dates_test = dti[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            i += WALK_FORWARD_STEP_MONTHS
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

        i += WALK_FORWARD_STEP_MONTHS

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


def generate_trading_pnl(
    y_proba: np.ndarray,
    actual_returns: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """與 Model D 一致：方向性報酬，signal=1 時承擔 T->T+5 報酬."""
    signals = (y_proba >= threshold).astype(int)
    pnl = np.zeros_like(actual_returns, dtype=float)
    mask = signals == 1
    pnl[mask] = actual_returns[mask]
    return signals, pnl


def compute_equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd


def compute_performance_metrics(
    dates: pd.DatetimeIndex,
    strategy_returns: np.ndarray,
    signals: np.ndarray,
) -> Dict[str, float]:
    n_days = len(strategy_returns)
    if n_days == 0:
        return {}

    equity = compute_equity_curve(strategy_returns)
    total_return = equity[-1] - 1.0

    years = n_days / 252.0
    cagr = (equity[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    mu = strategy_returns.mean()
    sigma = strategy_returns.std(ddof=1)
    sharpe = 0.0 if sigma == 0 else mu / sigma * np.sqrt(252.0)

    dd = compute_drawdown(equity)
    max_dd = dd.min()

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

    calmar = 0.0
    if max_dd < 0:
        calmar = total_return / abs(max_dd)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Calmar": calmar,
        "Num Days": n_days,
        "Num Trades": int(trade_mask.sum()),
        "Start Date": dates[0].strftime("%Y-%m-%d"),
        "End Date": dates[-1].strftime("%Y-%m-%d"),
    }


def apply_risk_control_and_metrics(
    model_name: str,
    y_proba_oos: np.ndarray,
    actual_oos: np.ndarray,
    dates_oos: pd.DatetimeIndex,
    encoded_data: pd.DataFrame,
) -> Dict[str, float]:
    """套用與最新 Model D 相同的風控（Vol Targeting + Survival Mode），並回傳績效指標."""
    if y_proba_oos.size == 0:
        return {}

    # 1) 產生原始策略報酬（5 日視窗）
    signals, pnl_raw = generate_trading_pnl(
        y_proba=y_proba_oos,
        actual_returns=actual_oos,
        threshold=DEFAULT_THRESHOLD,
    )

    # 2) 準備日報酬序列，用於估計波動度（邏輯與 quant_report_model_d 保持一致）
    close_series = encoded_data["Close"].astype(float)
    idx = close_series.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close_series.index = idx

    date_index = close_series.index.normalize()
    daily_close = close_series.groupby(date_index).last()
    # 先處理 NaN / inf，再計算日報酬
    daily_close = daily_close.replace([np.inf, -np.inf], np.nan).dropna()
    daily_ret_full = (
        daily_close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    )

    # OOS 日期同樣壓平成日期，以確保能與 daily_ret_full / rolling_vol 對齊
    dates_oos_idx = dates_oos
    if getattr(dates_oos_idx, "tz", None) is not None:
        dates_oos_idx = dates_oos_idx.tz_localize(None)
    dates_oos_dates = dates_oos_idx.normalize()

    # 3) Volatility Targeting（與 Model D 一致）：20 日滾動波動度 + bfill/ffill
    rolling_vol = (
        daily_ret_full.rolling(window=20, min_periods=5).std().fillna(method="bfill")
    )
    vol_oos = (
        rolling_vol.shift(1)
        .reindex(dates_oos_dates)
        .fillna(method="ffill")
        .fillna(0.0)
    )

    pos_size = np.ones_like(pnl_raw, dtype=float)
    vol_values = vol_oos.to_numpy()
    eff_vol_5d = vol_values * np.sqrt(5.0)
    valid_vol_mask = eff_vol_5d > 0
    pos_size[valid_vol_mask] = np.minimum(1.0, TARGET_VOL / eff_vol_5d[valid_vol_mask])

    # 4) Drawdown Survival Mode（具「進出」邏輯）：
    #    - 回撤 <= MAX_DRAWDOWN_TRIGGER(-20%)：進入生存模式，部位上限 SURVIVAL_POS_CAP
    #    - 回撤恢復至 > -10%：解除生存模式
    equity = 1.0
    peak = 1.0
    in_survival_mode = False
    for i in range(len(pnl_raw)):
        current_dd = equity / peak - 1.0

        # 1. 觸發生存模式：回撤 <= -20%
        if current_dd <= MAX_DRAWDOWN_TRIGGER:
            in_survival_mode = True

        # 2. 復原條件：回撤改善至 > -10%，離開生存模式
        if in_survival_mode and current_dd > -0.10:
            in_survival_mode = False

        # 3. 生存模式下壓縮當日部位上限
        if in_survival_mode:
            pos_size[i] = min(pos_size[i], SURVIVAL_POS_CAP)

        # 4. 以調整後部位計算報酬，更新權益與高點
        ret_i = pnl_raw[i] * pos_size[i]
        equity *= (1.0 + ret_i)
        peak = max(peak, equity)

    pnl_final = pnl_raw * pos_size

    # 5) 計算績效
    metrics = compute_performance_metrics(
        dates=dates_oos,
        strategy_returns=pnl_final,
        signals=signals,
    )
    print(
        f"[RESULT] {model_name}: TotalReturn={metrics['Total Return']*100:.2f}%, "
        f"Sharpe={metrics['Sharpe']:.3f}, MaxDD={metrics['Max Drawdown']*100:.2f}%, "
        f"Calmar={metrics['Calmar']:.3f}"
    )
    return metrics


def plot_ablation_dashboard(metrics_D: Dict[str, float], metrics_E: Dict[str, float]) -> None:
    """將 Model D / E 的結果畫成視覺化對照圖."""
    sns.set_theme(style="whitegrid")

    data = {
        "Metric": [
            "Total Return (%)",
            "CAGR (%)",
            "Sharpe Ratio",
            "Profit Factor",
            "Max Drawdown (%)",
            "Calmar Ratio",
        ],
        "Model D (Quantum)": [
            metrics_D["Total Return"] * 100.0,
            metrics_D["CAGR"] * 100.0,
            metrics_D["Sharpe"],
            metrics_D["Profit Factor"],
            metrics_D["Max Drawdown"] * 100.0,
            metrics_D["Calmar"],
        ],
        "Model E (Momentum)": [
            metrics_E["Total Return"] * 100.0,
            metrics_E["CAGR"] * 100.0,
            metrics_E["Sharpe"],
            metrics_E["Profit Factor"],
            metrics_E["Max Drawdown"] * 100.0,
            metrics_E["Calmar"],
        ],
    }
    df = pd.DataFrame(data)

    df_growth = df[df["Metric"].isin(["Total Return (%)", "CAGR (%)"])].copy()
    df_quality = df[df["Metric"].isin(["Sharpe Ratio", "Profit Factor"])].copy()
    df_risk = df[df["Metric"].isin(["Max Drawdown (%)"])].copy()
    df_calmar = df[df["Metric"].isin(["Calmar Ratio"])].copy()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    color_D = "#e74c3c"
    color_E = "#1abc9c"
    width = 0.35

    # Growth Potential
    ax_g = fig.add_subplot(gs[0, 0])
    x = np.arange(len(df_growth))
    yD = df_growth["Model D (Quantum)"].values
    yE = df_growth["Model E (Momentum)"].values
    ax_g.bar(x - width / 2, yD, width, color=color_D, label="Model D (Quantum)")
    ax_g.bar(x + width / 2, yE, width, color=color_E, label="Model E (Momentum)")
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(df_growth["Metric"], rotation=0)
    ax_g.set_ylabel("Percentage (%)")
    ax_g.set_title("Growth Potential")
    ymax = max(yD.max(), yE.max())
    ax_g.set_ylim(0, ymax * 1.25)
    for i, v in enumerate(yD):
        ax_g.text(x[i] - width / 2, v + ymax * 0.03, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(yE):
        ax_g.text(x[i] + width / 2, v + ymax * 0.03, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    # Strategy Quality
    ax_q = fig.add_subplot(gs[0, 1])
    xq = np.arange(len(df_quality))
    qD = df_quality["Model D (Quantum)"].values
    qE = df_quality["Model E (Momentum)"].values
    ax_q.bar(xq - width / 2, qD, width, color=color_D, label="Model D (Quantum)")
    ax_q.bar(xq + width / 2, qE, width, color=color_E, label="Model E (Momentum)")
    ax_q.set_xticks(xq)
    ax_q.set_xticklabels(df_quality["Metric"], rotation=0)
    ax_q.set_ylabel("Ratio")
    ax_q.set_title("Strategy Quality")
    qmax = max(qD.max(), qE.max())
    ax_q.set_ylim(0, qmax * 1.3)
    for i, v in enumerate(qD):
        ax_q.text(xq[i] - width / 2, v + qmax * 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(qE):
        ax_q.text(xq[i] + width / 2, v + qmax * 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Risk (Max Drawdown)
    ax_r = fig.add_subplot(gs[0, 2])
    xr = np.arange(len(df_risk))
    dD = abs(df_risk["Model D (Quantum)"].values[0])
    dE = abs(df_risk["Model E (Momentum)"].values[0])
    ax_r.bar(xr - width / 2, dD, width, color=color_D, label="Model D (Quantum)")
    ax_r.bar(xr + width / 2, dE, width, color=color_E, label="Model E (Momentum)")
    ax_r.set_xticks(xr)
    ax_r.set_xticklabels(["Max Drawdown (%)"])
    ax_r.set_ylabel("Drawdown Magnitude (%)")
    ax_r.set_title("Risk (Drawdown)")
    dmax = max(dD, dE)
    ax_r.set_ylim(0, dmax * 1.3)
    ax_r.text(
        xr[0] - width / 2,
        dD + dmax * 0.05,
        f"{df_risk['Model D (Quantum)'].values[0]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax_r.text(
        xr[0] + width / 2,
        dE + dmax * 0.05,
        f"{df_risk['Model E (Momentum)'].values[0]:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax_r.legend(fontsize=9, frameon=False, loc="upper right")

    # Calmar Ratio（水平條）
    ax_c = fig.add_subplot(gs[1, :])
    labels = ["Model D (Quantum)", "Model E (Momentum)"]
    scores = [
        df_calmar["Model D (Quantum)"].values[0],
        df_calmar["Model E (Momentum)"].values[0],
    ]
    y_pos = np.arange(len(labels))
    ax_c.barh(y_pos, scores, color=[color_D, color_E])
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(labels)
    ax_c.set_xlabel("Calmar Ratio")
    ax_c.set_title("Return-to-Drawdown Efficiency (Calmar Ratio)")
    for i, v in enumerate(scores):
        ax_c.text(v * 1.01, i, f"{v:.1f}", va="center", fontsize=10)

    fig.suptitle(
        "Ablation Study: Quantum I-Ching (Model D) vs Momentum Only (Model E)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "ablation_de_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Ablation Dashboard 已輸出至: {out_path}")


def main() -> None:
    print("=" * 80)
    print("Ablation Study - Model D (Pure Quantum) vs Model E (Momentum Only)")
    print("=" * 80)

    set_random_seed(42)

    # 1. 載入並編碼市場資料
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

    # 2. 建立 Model D / Model E 特徵（同一組 y / actual_returns / dates）
    X_D, X_E, y, actual_returns, dates = prepare_features_for_models(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
    )

    # 3. Walk-Forward OOS 預測（兩個模型共用同一組 y / actual_returns / dates）
    print("\n[INFO] Walk-Forward 驗證 - Model D (Pure Quantum)...")
    y_proba_D, y_true_D, actual_D, dates_D = run_walk_forward_proba(
        X_D, y, actual_returns, dates, model_name="Model D (Pure Quantum)"
    )

    print("\n[INFO] Walk-Forward 驗證 - Model E (Momentum Only)...")
    y_proba_E, y_true_E, actual_E, dates_E = run_walk_forward_proba(
        X_E, y, actual_returns, dates, model_name="Model E (Momentum Only)"
    )

    # 確認兩者的 OOS 樣本對齊（日期 / 標籤）
    if len(dates_D) != len(dates_E) or not np.array_equal(dates_D.values, dates_E.values):
        print("[WARNING] Model D / E 的 OOS 日期未完全對齊，後續比較可能略有偏差。")
    if len(y_true_D) == len(y_true_E) and not np.array_equal(y_true_D, y_true_E):
        print("[WARNING] Model D / E 的 OOS 標籤略有差異（可能因有效樣本遮罩不同）。")

    # 4. 套用相同風控邏輯並計算績效
    print("\n[INFO] 套用相同風控邏輯並計算績效指標...")
    metrics_D = apply_risk_control_and_metrics(
        "Model D (Pure Quantum)", y_proba_D, actual_D, dates_D, encoded_data
    )
    metrics_E = apply_risk_control_and_metrics(
        "Model E (Momentum Only)", y_proba_E, actual_E, dates_E, encoded_data
    )

    # 5. 輸出並排比較表
    print("\n" + "=" * 80)
    print("Ablation Result: Model D vs Model E")
    print("=" * 80)
    headers = [
        "Model",
        "TotalRet(%)",
        "CAGR(%)",
        "Sharpe",
        "MaxDD(%)",
        "Calmar",
        "WinRate(%)",
        "PF",
        "Trades",
    ]
    row_D = [
        "Model D (Quantum)",
        f"{metrics_D['Total Return']*100:10.2f}",
        f"{metrics_D['CAGR']*100:8.2f}",
        f"{metrics_D['Sharpe']:6.3f}",
        f"{metrics_D['Max Drawdown']*100:8.2f}",
        f"{metrics_D['Calmar']:6.3f}",
        f"{metrics_D['Win Rate']*100:8.2f}",
        f"{metrics_D['Profit Factor']:6.3f}",
        f"{metrics_D['Num Trades']:6d}",
    ]
    row_E = [
        "Model E (Momentum)",
        f"{metrics_E['Total Return']*100:10.2f}",
        f"{metrics_E['CAGR']*100:8.2f}",
        f"{metrics_E['Sharpe']:6.3f}",
        f"{metrics_E['Max Drawdown']*100:8.2f}",
        f"{metrics_E['Calmar']:6.3f}",
        f"{metrics_E['Win Rate']*100:8.2f}",
        f"{metrics_E['Profit Factor']:6.3f}",
        f"{metrics_E['Num Trades']:6d}",
    ]

    print(" | ".join(f"{h:>14s}" for h in headers))
    print("-" * 120)
    print(" | ".join(row_D))
    print(" | ".join(row_E))
    print("=" * 80)

    print("\n解讀建議：")
    print("- 若 Model D 的 Sharpe 明顯高於 Model E，或 Calmar / MaxDD 明顯更好，")
    print("  則代表 I-Ching 特徵在『過濾高風險環境 / 優化風控』上有實質貢獻。")
    print("- 若兩者指標非常接近，尤其 Sharpe / Calmar 幾乎一樣，")
    print("  則代表目前設定下，I-Ching 特徵多半只是噪音，主要 alpha 來自 Momentum + Risk Control。")

    # 6. 視覺化 Ablation Dashboard
    plot_ablation_dashboard(metrics_D, metrics_E)


if __name__ == "__main__":
    main()

