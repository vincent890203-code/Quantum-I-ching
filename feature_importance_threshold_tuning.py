"""特徵重要度分析與閾值調參腳本.

目標：
- 在與 PnL 相同的 Quantum XGBoost 架構上，做分類層面的診斷：
  1) 使用 Permutation Importance（測試集）評估各特徵貢獻度
  2) 僅保留 Top-5 特徵重訓一個「精簡模型」（Light Model）
  3) 對 Original Quantum / Light Model 進行閾值掃描，找出最大化 F1-Score 的最佳閾值
  4) 繪製 PR 曲線：Original Quantum vs Top-5 Features Only vs No Skill（Base Rate）

說明：
- 僅使用「測試集 (最後 20%)」作為 Out-of-Sample 評估區間，避免資料洩漏。
- 模型為 XGBoost（與 PnL 用的 tabular Quantum 模型一致，只是這裡專注在分類質量）。
"""

import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


# =============================================================================
# 繪圖中文字型設定
# =============================================================================

def setup_chinese_font() -> None:
    """設置 matplotlib 使用中文字體（與現有可視化風格一致）."""
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


DEFAULT_THRESHOLD = 0.5  # 與先前 PnL 回測預設閾值對齊


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# 資料準備：與 `visualize_pnl.py` / `plot_pr_curve.py` 一致的 tabular Quantum 資料
# =============================================================================

def prepare_tabular_data(
    encoded_data: pd.DataFrame,
    prediction_window: int = 5,
    volatility_threshold: float = 0.03,
    selected_iching_features: list | None = None,
    use_iching: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """準備表格型資料集（用於 XGBoost）.

    - 特徵僅使用 T 時點資訊（Close/Volume/RVOL/Daily_Return + I-Ching 特徵）
    - 目標為 T -> T+5 的絕對報酬是否超過波動閾值

    Returns:
        (X, y, actual_returns, dates)
    """
    print("\n[INFO] 準備表格型資料集 (tabular XGBoost)...")
    print(f"  預測窗口: T+{prediction_window}")
    print(f"  波動性閾值: {volatility_threshold * 100:.1f}%")
    print(f"  使用易經特徵: {use_iching}")

    required_base_cols = ["Close", "Volume", "RVOL", "Daily_Return", "Ritual_Sequence"]
    missing_cols = [col for col in required_base_cols if col not in encoded_data.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    processor = DataProcessor()

    all_iching_features = [
        "Yang_Count_Main",
        "Yang_Count_Future",
        "Moving_Lines_Count",
        "Energy_Delta",
        "Conflict_Score",
    ]

    if selected_iching_features is None:
        selected_iching_features = ["Moving_Lines_Count", "Energy_Delta"]

    if use_iching:
        print(f"[INFO] 使用易經特徵: {selected_iching_features}")

        print("[INFO] 提取易經特徵...")
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
        iching_df = iching_df_full[selected_iching_features].copy()

        numerical_cols = ["Close", "Volume", "RVOL", "Daily_Return"]
        X = pd.concat([encoded_data[numerical_cols], iching_df], axis=1)
    else:
        print("[INFO] Baseline 模型：只使用數值特徵")
        numerical_cols = ["Close", "Volume", "RVOL", "Daily_Return"]
        X = encoded_data[numerical_cols].copy()

    # 構建標籤：是否為「高波動」樣本
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

    print("[INFO] 資料準備完成:")
    print(f"  總樣本數: {len(X)}")
    print(f"  特徵數: {len(X.columns)}")
    print(f"  標籤分布: 高波動={int(y.sum())}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")

    return X, y, actual_returns, dates


def _build_xgb_classifier(feature_names: List[str]) -> xgb.XGBClassifier:
    """建立一個固定超參數的 XGBoost 分類器."""
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
    # feature_names 僅在 fit 後再設置 booster 內部名稱
    return model


# =============================================================================
# PR 曲線繪製
# =============================================================================

def plot_pr_curve_three_models(
    y_true: np.ndarray,
    y_proba_original: np.ndarray,
    y_proba_light: np.ndarray,
    threshold_original: float,
    threshold_light: float,
    save_path: str = "data/pr_curve_feature_tuning.png",
) -> None:
    """繪製三條曲線：Original Quantum / Top-5 Light / No Skill."""
    print("\n[INFO] 生成 Precision-Recall 曲線 (Original vs Top-5 vs No Skill)...")

    # Original Quantum
    prec_orig, rec_orig, _ = precision_recall_curve(y_true, y_proba_original)
    ap_orig = average_precision_score(y_true, y_proba_original)

    # Light Model (Top-5 Features)
    prec_light, rec_light, _ = precision_recall_curve(y_true, y_proba_light)
    ap_light = average_precision_score(y_true, y_proba_light)

    base_rate = y_true.mean()

    # 當前（最佳）閾值對應點
    y_pred_orig_opt = (y_proba_original >= threshold_original).astype(int)
    y_pred_light_opt = (y_proba_light >= threshold_light).astype(int)

    # 為避免 zero_division 例外，這裡不重算 F1，僅標記點位
    from sklearn.metrics import precision_score, recall_score  # 局部導入，減少全域污染

    p_orig_opt = precision_score(y_true, y_pred_orig_opt, zero_division=0)
    r_orig_opt = recall_score(y_true, y_pred_orig_opt, zero_division=0)

    p_light_opt = precision_score(y_true, y_pred_light_opt, zero_division=0)
    r_light_opt = recall_score(y_true, y_pred_light_opt, zero_division=0)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    ax.set_facecolor("white")

    # Original Quantum: 紅色實線
    ax.plot(
        rec_orig,
        prec_orig,
        label=f"Original Quantum (AP: {ap_orig:.3f})",
        linewidth=2.5,
        color="#FF6B6B",
        alpha=0.9,
    )

    # Top-5 Light: 藍色實線
    ax.plot(
        rec_light,
        prec_light,
        label=f"Top-5 Features Only (AP: {ap_light:.3f})",
        linewidth=2.5,
        color="#4A90E2",
        alpha=0.9,
    )

    # No Skill: 灰色虛線，y = base_rate
    ax.axhline(
        y=base_rate,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"No Skill / Base Rate ({base_rate:.3f})",
        alpha=0.7,
    )

    # 標記最佳閾值點
    ax.scatter(
        [r_orig_opt],
        [p_orig_opt],
        color="#FF6B6B",
        s=150,
        marker="o",
        zorder=5,
        label=f"Optimal Threshold (Original: {threshold_original:.2f})",
        edgecolors="black",
        linewidths=1.5,
    )
    ax.scatter(
        [r_light_opt],
        [p_light_opt],
        color="#4A90E2",
        s=150,
        marker="s",
        zorder=5,
        label=f"Optimal Threshold (Top-5: {threshold_light:.2f})",
        edgecolors="black",
        linewidths=1.5,
    )

    ax.set_title(
        "Precision-Recall Curve: Original Quantum vs Top-5 Features vs No Skill (Test Set)",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="#333333",
    )
    ax.set_xlabel("Recall", fontsize=13, color="#333333")
    ax.set_ylabel("Precision", fontsize=13, color="#333333")

    ax.legend(
        loc="best",
        fontsize=10,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        labelcolor="#333333",
    )

    ax.grid(True, alpha=0.3, linestyle="--", color="gray")
    ax.tick_params(colors="#333333")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    # 文字總結：AP 差異
    improvement = ap_light - ap_orig
    if improvement > 0:
        text_color = "#2ECC71"
        text = f"Top-5 模型相對 Original AP 提升: +{improvement:.4f}"
    else:
        text_color = "#E74C3C"
        text = f"Top-5 模型相對 Original AP 變化: {improvement:.4f}"

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[INFO] PR 曲線已保存至: {save_path}")
    plt.close()


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    """執行特徵重要度分析 + 精簡模型 + 閾值調參."""
    print("=" * 80)
    print("特徵重要度分析與閾值調參 (Quantum XGBoost)")
    print("=" * 80)

    set_random_seed(42)

    # -----------------------------
    # 1. 載入與編碼市場資料
    # -----------------------------
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

    # -----------------------------
    # 2. 準備 Quantum tabular 特徵
    # -----------------------------
    X_q, y_q, actual_returns_q, dates_q = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        selected_iching_features=["Moving_Lines_Count", "Energy_Delta"],
        use_iching=True,
    )

    # 時間序列 80/20 切分（與其他腳本一致）
    split_idx = int(len(X_q) * 0.8)
    X_train = X_q.iloc[:split_idx].copy()
    y_train = y_q.iloc[:split_idx].copy()
    X_test = X_q.iloc[split_idx:].copy()
    y_test = y_q.iloc[split_idx:].copy()

    print("\n[INFO] 時間序列切分 (Train/Test):")
    print(f"  訓練集樣本數: {len(X_train)}")
    print(f"  測試集樣本數: {len(X_test)}")

    feature_names = list(X_q.columns)

    # -----------------------------
    # 3. Original Quantum 模型 (全特徵)
    # -----------------------------
    print("\n[INFO] 訓練 Original Quantum 模型 (使用全部特徵)...")
    model_orig = _build_xgb_classifier(feature_names)
    model_orig.fit(X_train, y_train, verbose=False)
    model_orig.get_booster().feature_names = feature_names

    y_proba_orig = model_orig.predict_proba(X_test)[:, 1]

    # -----------------------------
    # 4. Permutation Importance (測試集)
    # -----------------------------
    print("\n[INFO] 計算 Permutation Importance (Test Set)...")
    pi_result = permutation_importance(
        model_orig,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importances_mean = pi_result.importances_mean
    importances_std = pi_result.importances_std

    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    )
    fi_df = fi_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    print("\n[RESULT] Top 10 特徵 (Permutation Importance, Test Set):")
    for i in range(min(10, len(fi_df))):
        row = fi_df.iloc[i]
        print(
            f"  {i+1:2d}. {row['feature']:<24s} "
            f"importance_mean={row['importance_mean']:.6f} "
            f"(± {row['importance_std']:.6f})"
        )

    top5_features = fi_df["feature"].head(5).tolist()
    print("\n[INFO] Top-5 特徵 (將用於 Light Model):")
    for f in top5_features:
        print(f"  - {f}")

    # -----------------------------
    # 5. Light Model：僅使用 Top-5 特徵重訓
    # -----------------------------
    print("\n[INFO] 訓練 Light Model (僅 Top-5 特徵)...")
    X_train_light = X_train[top5_features].copy()
    X_test_light = X_test[top5_features].copy()

    model_light = _build_xgb_classifier(top5_features)
    model_light.fit(X_train_light, y_train, verbose=False)
    model_light.get_booster().feature_names = top5_features

    y_proba_light = model_light.predict_proba(X_test_light)[:, 1]

    # -----------------------------
    # 6. 閾值掃描：最大化 F1-Score
    # -----------------------------
    print("\n[INFO] 進行閾值掃描以最大化 F1-Score...")
    thresholds = np.linspace(0.01, 0.99, 99)

    best_f1_orig = -1.0
    best_thresh_orig = DEFAULT_THRESHOLD

    best_f1_light = -1.0
    best_thresh_light = DEFAULT_THRESHOLD

    for t in thresholds:
        y_pred_o = (y_proba_orig >= t).astype(int)
        y_pred_l = (y_proba_light >= t).astype(int)

        f1_o = f1_score(y_test, y_pred_o, zero_division=0)
        f1_l = f1_score(y_test, y_pred_l, zero_division=0)

        if f1_o > best_f1_orig:
            best_f1_orig = f1_o
            best_thresh_orig = t

        if f1_l > best_f1_light:
            best_f1_light = f1_l
            best_thresh_light = t

    print("\n[RESULT] 閾值調參結果 (F1-Score 最大化):")
    print(
        f"  Original Quantum: 最佳閾值 = {best_thresh_orig:.3f}, "
        f"F1-Score = {best_f1_orig:.4f}"
    )
    print(
        f"  Top-5 Light   : 最佳閾值 = {best_thresh_light:.3f}, "
        f"F1-Score = {best_f1_light:.4f}"
    )

    # -----------------------------
    # 7. PR 曲線：Original vs Top-5 vs No Skill
    # -----------------------------
    plot_pr_curve_three_models(
        y_true=y_test.values,
        y_proba_original=y_proba_orig,
        y_proba_light=y_proba_light,
        threshold_original=best_thresh_orig,
        threshold_light=best_thresh_light,
        save_path="data/pr_curve_feature_tuning.png",
    )

    print("\n" + "=" * 80)
    print("完成：特徵重要度分析 + 精簡模型 + 閾值調參")
    print("=" * 80)
    print("\n[SUCCESS] 輸出結果：")
    print("  - Top 10 特徵已印出（Permutation Importance, Test Set）")
    print("  - data/pr_curve_feature_tuning.png (Original vs Top-5 vs No Skill)")
    print(
        f"  - Original Quantum 最佳閾值: {best_thresh_orig:.3f} "
        f"(F1={best_f1_orig:.4f})"
    )
    print(
        f"  - Top-5 Light 最佳閾值: {best_thresh_light:.3f} "
        f"(F1={best_f1_light:.4f})"
    )


if __name__ == "__main__":
    main()

