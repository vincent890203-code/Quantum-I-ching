"""生成 Precision-Recall (PR) 曲線比較圖.

此腳本比較三個策略的分類性能：
- Model A (Quantum Strategy): 使用複雜模型的預測機率
- Model B (Baseline Strategy): 使用簡單 ML baseline 的預測機率
- Model C (Buy & Hold / No Skill): 基準線（正類比例）

使用 Out-of-Sample (Test) 資料進行評估。
"""

import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    precision_score,
    recall_score
)

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder

# 配置 matplotlib 中文字體
def setup_chinese_font():
    """設置 matplotlib 使用中文字體."""
    chinese_fonts = [
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'Microsoft JhengHei',
        'KaiTi',
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_to_use = None
    
    for font in chinese_fonts:
        if font in available_fonts:
            font_to_use = font
            break
    
    if font_to_use:
        plt.rcParams['font.sans-serif'] = [font_to_use]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[INFO] 已設置中文字體: {font_to_use}")
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("[WARNING] 未找到中文字體，可能無法正確顯示中文")

setup_chinese_font()

# Walk-Forward 驗證參數
WALK_FORWARD_TRAIN_MONTHS = 12
WALK_FORWARD_TEST_MONTHS = 1
WALK_FORWARD_STEP_MONTHS = 1

# 預設閾值（用於 PnL 回測）
DEFAULT_THRESHOLD = 0.5


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)


def prepare_tabular_data(
    encoded_data: pd.DataFrame,
    prediction_window: int = 5,
    volatility_threshold: float = 0.03,
    selected_iching_features: list = None,
    use_iching: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """準備表格型資料集（用於 XGBoost）.
    
    Args:
        encoded_data: 包含易經卦象的編碼資料。
        prediction_window: 預測窗口長度（預設 5 天）。
        volatility_threshold: 波動性閾值（預設 0.03，即 3%）。
        selected_iching_features: 要使用的易經特徵列表。
        use_iching: 是否使用易經特徵（False 時為 Baseline 模型）。
    
    Returns:
        (特徵 DataFrame, 標籤 Series, 實際收益率 Series, 日期索引) 四元組。
    """
    print(f"\n[INFO] 準備表格型資料集...")
    print(f"  預測窗口: T+{prediction_window}")
    print(f"  波動性閾值: {volatility_threshold * 100}%")
    print(f"  使用易經特徵: {use_iching}")
    
    required_base_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return', 'Ritual_Sequence']
    missing_cols = [col for col in required_base_cols if col not in encoded_data.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")
    
    processor = DataProcessor()
    
    all_iching_features = [
        'Yang_Count_Main', 'Yang_Count_Future', 'Moving_Lines_Count',
        'Energy_Delta', 'Conflict_Score'
    ]
    
    if selected_iching_features is None:
        selected_iching_features = ['Moving_Lines_Count', 'Energy_Delta']
    
    if use_iching:
        print(f"[INFO] 使用易經特徵: {selected_iching_features}")
        
        print("[INFO] 提取易經特徵...")
        iching_features_list = []
        for idx, ritual_seq in enumerate(encoded_data['Ritual_Sequence']):
            if pd.isna(ritual_seq) or len(str(ritual_seq)) != 6:
                iching_features_list.append([0.0] * len(all_iching_features))
            else:
                try:
                    iching_features = processor.extract_iching_features(str(ritual_seq))
                    iching_features_list.append(iching_features.tolist())
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] 無法提取易經特徵: {ritual_seq}, 錯誤: {e}")
                    iching_features_list.append([0.0] * len(all_iching_features))
            
            if (idx + 1) % 1000 == 0:
                print(f"[INFO] 已處理 {idx + 1}/{len(encoded_data)} 筆易經特徵")
        
        iching_df_full = pd.DataFrame(
            iching_features_list,
            columns=all_iching_features,
            index=encoded_data.index
        )
        
        iching_df = iching_df_full[selected_iching_features].copy()

        numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        X = pd.concat([
            encoded_data[numerical_cols],
            iching_df
        ], axis=1)
    else:
        print("[INFO] Baseline 模型：只使用數值特徵")
        numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        X = encoded_data[numerical_cols].copy()
    
    max_idx = len(encoded_data) - prediction_window
    
    future_prices = encoded_data['Close'].shift(-prediction_window)
    current_prices = encoded_data['Close']
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
    
    print(f"[INFO] 資料準備完成:")
    print(f"  總樣本數: {len(X)}")
    print(f"  特徵數: {len(X.columns)}")
    print(f"  標籤分布: 高波動={y.sum()}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")
    
    return X, y, actual_returns, dates


def _fit_xgb_on_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
) -> xgb.XGBClassifier:
    """在給定的訓練集上擬合 XGBoost，不接觸測試集."""
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    model.get_booster().feature_names = feature_names
    return model


def run_walk_forward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    actual_returns: pd.Series,
    dates: pd.DatetimeIndex,
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    test_months: int = WALK_FORWARD_TEST_MONTHS,
    step_months: int = WALK_FORWARD_STEP_MONTHS,
    model_name: str = "Quantum",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Walk-Forward 驗證：返回預測機率（而非二進制預測）.
    
    Args:
        X: 特徵 DataFrame。
        y: 標籤 Series。
        actual_returns: 實際收益率 Series。
        dates: 對應每行的日期。
        train_months: 訓練視窗月數。
        test_months: 測試視窗月數。
        step_months: 每步前滑月數。
        model_name: 日誌用名稱。
    
    Returns:
        (y_proba_oos, y_true_oos, dates_oos) 串接後的 OOS 機率預測、真實標籤、日期。
    """
    dti = pd.DatetimeIndex(dates)
    periods = pd.Series(dti.to_period('M'), index=range(len(dti)))
    uper = sorted(periods.unique())
    n_periods = len(uper)
    if n_periods < train_months + test_months:
        raise ValueError(
            f"資料月數 {n_periods} 不足 Walk-Forward 所需 "
            f"train_months + test_months = {train_months + test_months}"
        )

    proba_list: List[np.ndarray] = []
    y_true_list: List[np.ndarray] = []
    dates_list: List[pd.DatetimeIndex] = []
    feature_names = list(X.columns)

    i = 0
    fold = 0
    while i + train_months + test_months <= n_periods:
        train_periods = uper[i : i + train_months]
        test_periods = uper[i + train_months : i + train_months + test_months]
        train_mask = periods.isin(train_periods).to_numpy()
        test_mask = periods.isin(test_periods).to_numpy()

        X_train = X.iloc[train_mask].copy()
        y_train = y.iloc[train_mask].copy()
        X_test = X.iloc[test_mask].copy()
        y_test = y.iloc[test_mask].copy()
        dates_test = dti[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            i += step_months
            continue

        model = _fit_xgb_on_subset(X_train, y_train, feature_names)
        # 獲取正類的機率（第二列）
        proba_fold = model.predict_proba(X_test)[:, 1]

        proba_list.append(proba_fold)
        y_true_list.append(y_test.values)
        dates_list.append(dates_test)

        fold += 1
        if fold <= 3 or fold % 12 == 0:
            print(f"[Walk-Forward] {model_name} fold {fold}: "
                  f"train {train_periods[0]}~{train_periods[-1]}, "
                  f"test {test_periods[0]}~{test_periods[-1]}, "
                  f"n_test={len(X_test)}")
        i += step_months

    if not proba_list:
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    y_proba_oos = np.concatenate(proba_list)
    y_true_oos = np.concatenate(y_true_list)
    dates_oos = pd.DatetimeIndex(np.concatenate([d.values for d in dates_list]))
    print(f"[Walk-Forward] {model_name} 共 {fold} 個 fold，OOS 樣本數 = {len(y_proba_oos)}")
    return y_proba_oos, y_true_oos, dates_oos


def train_model_get_proba(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    split_idx: int = None
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series, np.ndarray]:
    """訓練 XGBoost 模型並返回測試集的預測機率.
    
    Args:
        X: 特徵 DataFrame。
        y: 標籤 Series。
        model_name: 模型名稱（用於日誌）。
        split_idx: 分割索引（如果為 None，則使用 80/20 分割）。
    
    Returns:
        (模型, 測試集特徵, 測試集標籤, 測試集預測機率) 四元組。
    """
    print(f"\n[INFO] 訓練 {model_name}...")
    print(f"  特徵: {list(X.columns)}")
    print(f"  樣本數: {len(X)}")
    
    if split_idx is None:
        split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    model.get_booster().feature_names = list(X.columns)
    
    # 獲取測試集的預測機率（正類機率）
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    print(f"[INFO] {model_name} 訓練完成")
    print(f"  訓練集樣本數: {len(X_train)}")
    print(f"  測試集樣本數: {len(X_test)}")
    
    return model, X_test, y_test, y_proba_test


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba_quantum: np.ndarray,
    y_proba_baseline: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    save_path: str = "data/pr_curve_comparison.png",
) -> None:
    """繪製 Precision-Recall 曲線比較圖.
    
    Args:
        y_true: 真實標籤。
        y_proba_quantum: Quantum 模型的預測機率。
        y_proba_baseline: Baseline 模型的預測機率。
        threshold: 當前使用的閾值（用於標記）。
        save_path: 保存路徑。
    """
    print(f"\n[INFO] 生成 Precision-Recall 曲線比較圖...")
    
    # 計算 PR 曲線
    precision_quantum, recall_quantum, thresholds_quantum = precision_recall_curve(
        y_true, y_proba_quantum
    )
    precision_baseline, recall_baseline, thresholds_baseline = precision_recall_curve(
        y_true, y_proba_baseline
    )
    
    # 計算 Average Precision (AP) 分數
    ap_quantum = average_precision_score(y_true, y_proba_quantum)
    ap_baseline = average_precision_score(y_true, y_proba_baseline)
    
    # 計算基準線（No Skill）：水平線在 y = sum(y_true) / len(y_true)
    base_rate = y_true.mean()
    
    # 找到當前閾值對應的點
    # 對於 Quantum 模型：使用閾值計算二進制預測，然後計算 precision 和 recall
    y_pred_quantum = (y_proba_quantum >= threshold).astype(int)
    precision_at_threshold_quantum = precision_score(y_true, y_pred_quantum, zero_division=0)
    recall_at_threshold_quantum = recall_score(y_true, y_pred_quantum, zero_division=0)
    
    # 對於 Baseline 模型
    y_pred_baseline = (y_proba_baseline >= threshold).astype(int)
    precision_at_threshold_baseline = precision_score(y_true, y_pred_baseline, zero_division=0)
    recall_at_threshold_baseline = recall_score(y_true, y_pred_baseline, zero_division=0)
    
    print(f"[INFO] Average Precision 分數:")
    print(f"  Quantum Strategy: {ap_quantum:.4f}")
    print(f"  Baseline Strategy: {ap_baseline:.4f}")
    print(f"  基準線 (No Skill): {base_rate:.4f}")
    print(f"\n[INFO] 當前閾值 ({threshold}) 對應的點:")
    print(f"  Quantum: Precision={precision_at_threshold_quantum:.4f}, Recall={recall_at_threshold_quantum:.4f}")
    print(f"  Baseline: Precision={precision_at_threshold_baseline:.4f}, Recall={recall_at_threshold_baseline:.4f}")
    
    # 繪製圖表
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Quantum Strategy (Red)
    ax.plot(recall_quantum, precision_quantum,
            label=f'Quantum Strategy (AP: {ap_quantum:.3f})',
            linewidth=2.5, color='#FF6B6B', alpha=0.9)
    
    # Baseline Strategy (Blue)
    ax.plot(recall_baseline, precision_baseline,
            label=f'Baseline Strategy (AP: {ap_baseline:.3f})',
            linewidth=2.5, color='#4A90E2', alpha=0.9)
    
    # No Skill / Buy & Hold (Dashed Grey)
    ax.axhline(y=base_rate, color='gray', linestyle='--', linewidth=2,
               label=f'No Skill / Buy & Hold (Base Rate: {base_rate:.3f})', alpha=0.7)
    
    # 標記當前閾值點
    ax.scatter([recall_at_threshold_quantum], [precision_at_threshold_quantum],
               color='#FF6B6B', s=150, marker='o', zorder=5,
               label=f'Current Threshold (Quantum, {threshold})', edgecolors='black', linewidths=1.5)
    ax.scatter([recall_at_threshold_baseline], [precision_at_threshold_baseline],
               color='#4A90E2', s=150, marker='s', zorder=5,
               label=f'Current Threshold (Baseline, {threshold})', edgecolors='black', linewidths=1.5)
    
    # 設置標題和標籤
    ax.set_title('Precision-Recall Curve: Quantum vs Baseline vs No Skill (Out-of-Sample)', 
                 fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('Recall', fontsize=13, color='#333333')
    ax.set_ylabel('Precision', fontsize=13, color='#333333')
    
    # 設置圖例
    ax.legend(loc='best', fontsize=10, framealpha=0.9, facecolor='white', 
              edgecolor='#cccccc', labelcolor='#333333')
    
    # 設置網格
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.tick_params(colors='#333333')
    
    # 設置軸範圍
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # 分析結果文字
    improvement = ap_quantum - ap_baseline
    if improvement > 0:
        text_color = '#2ECC71'
        text = f'Quantum 優於 Baseline: +{improvement:.4f}'
    else:
        text_color = '#E74C3C'
        text = f'Quantum 劣於 Baseline: {improvement:.4f}'
    
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, fontsize=11, color='#333333',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor=text_color, linewidth=2))
    
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] PR 曲線比較圖已保存至: {save_path}")
    
    plt.close()
    
    # 輸出分析
    print("\n" + "=" * 80)
    print("分類性能分析")
    print("=" * 80)
    print(f"\nAverage Precision (AP) 分數:")
    print(f"  Model A (Quantum Strategy): {ap_quantum:.4f}")
    print(f"  Model B (Baseline Strategy): {ap_baseline:.4f}")
    print(f"  Model C (No Skill / Buy & Hold): {base_rate:.4f}")
    print(f"\n差異分析:")
    print(f"  Quantum vs Baseline: {improvement:+.4f} ({improvement/ap_baseline*100:+.2f}%)")
    
    if ap_quantum > ap_baseline:
        print(f"\n✓ Model A (Quantum) 的 AP 分數顯著高於 Model B (Baseline)")
        print(f"  這表明 Quantum 特徵確實增加了預測價值。")
        if ap_quantum > ap_baseline * 1.1:  # 10% 以上的提升
            print(f"  提升幅度 ({improvement/ap_baseline*100:.1f}%) 相當可觀，複雜度是合理的。")
        else:
            print(f"  提升幅度 ({improvement/ap_baseline*100:.1f}%) 較小，需要權衡複雜度與收益。")
    elif ap_quantum < ap_baseline:
        print(f"\n✗ Model A (Quantum) 的 AP 分數低於 Model B (Baseline)")
        print(f"  這表明 Quantum 特徵可能沒有增加預測價值，甚至可能造成過擬合。")
    else:
        print(f"\n≈ Model A (Quantum) 和 Model B (Baseline) 的 AP 分數相近")
        print(f"  這表明 Quantum 特徵可能沒有顯著增加預測價值。")
    
    # 檢查曲線是否嚴格在上方
    print(f"\n曲線位置分析:")
    # 在相同的 recall 值上比較 precision
    recall_common = np.linspace(0, 1, 100)
    precision_quantum_interp = np.interp(recall_common, recall_quantum[::-1], precision_quantum[::-1])
    precision_baseline_interp = np.interp(recall_common, recall_baseline[::-1], precision_baseline[::-1])
    
    quantum_above = np.sum(precision_quantum_interp > precision_baseline_interp)
    quantum_below = np.sum(precision_quantum_interp < precision_baseline_interp)
    
    print(f"  在 100 個均勻分布的 recall 點上:")
    print(f"    Quantum > Baseline: {quantum_above} 個點")
    print(f"    Quantum < Baseline: {quantum_below} 個點")
    
    if quantum_above > quantum_below * 2:
        print(f"  ✓ Quantum 曲線在大部分區域都高於 Baseline，確認 Quantum 特徵的價值。")
    elif quantum_below > quantum_above * 2:
        print(f"  ✗ Baseline 曲線在大部分區域都高於 Quantum，Quantum 特徵可能沒有幫助。")
    else:
        print(f"  ≈ 兩條曲線交叉較多，需要進一步分析。")
    
    print("=" * 80)


def main() -> None:
    """主函數：生成 PR 曲線比較圖."""
    print("=" * 80)
    print("Precision-Recall 曲線比較 - 分類性能評估")
    print("=" * 80)
    
    set_random_seed(42)
    
    # 載入資料
    print("\n[INFO] 載入 TSMC (2330.TW) 市場資料...")
    loader = MarketDataLoader()
    raw_data = loader.fetch_data(tickers=["2330.TW"], market_type="TW")
    
    if raw_data.empty:
        print("[ERROR] 無法獲取 2330.TW 的市場資料")
        return
    
    # 編碼為易經卦象
    print("[INFO] 編碼為易經卦象...")
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)
    
    if encoded_data.empty:
        print("[ERROR] 編碼後的資料為空")
        return
    
    # 準備 Quantum 資料
    X_quantum, y_quantum, actual_returns_quantum, dates_quantum = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        selected_iching_features=['Moving_Lines_Count', 'Energy_Delta'],
        use_iching=True
    )
    
    # 準備 Baseline 資料
    X_baseline, y_baseline, actual_returns_baseline, dates_baseline = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        use_iching=False
    )
    
    # 確保使用相同的索引範圍
    min_len = min(len(X_quantum), len(X_baseline))
    X_quantum = X_quantum.iloc[:min_len].reset_index(drop=True)
    y_quantum = y_quantum.iloc[:min_len].reset_index(drop=True)
    dates_quantum = dates_quantum[:min_len]
    X_baseline = X_baseline.iloc[:min_len].reset_index(drop=True)
    y_baseline = y_baseline.iloc[:min_len].reset_index(drop=True)
    dates_baseline = dates_baseline[:min_len]
    
    # 確保標籤一致
    assert (y_quantum.values == y_baseline.values).all(), "Quantum 和 Baseline 的標籤不一致"
    
    # 使用相同的分割點
    split_idx = int(min_len * 0.8)
    
    # ----- Walk-Forward 驗證（獲取 OOS 機率）-----
    print("\n[INFO] Walk-Forward 驗證（獲取 OOS 預測機率）...")
    try:
        y_proba_wf_quantum, y_true_wf_quantum, dates_wf_quantum = run_walk_forward_proba(
            X_quantum, y_quantum, actual_returns_quantum, dates_quantum,
            train_months=WALK_FORWARD_TRAIN_MONTHS,
            test_months=WALK_FORWARD_TEST_MONTHS,
            step_months=WALK_FORWARD_STEP_MONTHS,
            model_name="Quantum",
        )
        y_proba_wf_baseline, y_true_wf_baseline, dates_wf_baseline = run_walk_forward_proba(
            X_baseline, y_baseline, actual_returns_baseline, dates_baseline,
            train_months=WALK_FORWARD_TRAIN_MONTHS,
            test_months=WALK_FORWARD_TEST_MONTHS,
            step_months=WALK_FORWARD_STEP_MONTHS,
            model_name="Baseline",
        )
        
        # 確保 Walk-Forward 的標籤一致
        assert len(y_true_wf_quantum) == len(y_true_wf_baseline), "Walk-Forward 標籤長度不一致"
        assert (y_true_wf_quantum == y_true_wf_baseline).all(), "Walk-Forward 標籤不一致"
        
        print(f"\n[INFO] Walk-Forward OOS 樣本數: {len(y_proba_wf_quantum)}")
        
        # 使用 Walk-Forward OOS 資料繪製 PR 曲線
        plot_pr_curve(
            y_true_wf_quantum,
            y_proba_wf_quantum,
            y_proba_wf_baseline,
            threshold=DEFAULT_THRESHOLD,
            save_path="data/pr_curve_comparison_oos.png",
        )
        
    except ValueError as e:
        print(f"[WARNING] Walk-Forward 跳過（資料不足）: {e}")
        print("[INFO] 改用 Single-Split 測試集...")
        
        # 如果 Walk-Forward 失敗，使用 Single-Split 測試集
        _, X_test_quantum, y_test_quantum, y_proba_test_quantum = train_model_get_proba(
            X_quantum, y_quantum, 
            model_name="Quantum",
            split_idx=split_idx
        )
        
        _, X_test_baseline, y_test_baseline, y_proba_test_baseline = train_model_get_proba(
            X_baseline, y_baseline, 
            model_name="Baseline",
            split_idx=split_idx
        )
        
        # 確保測試集標籤一致
        assert len(y_test_quantum) == len(y_test_baseline), "測試集長度不一致"
        assert (y_test_quantum.values == y_test_baseline.values).all(), "測試集標籤不一致"
        
        print(f"\n[INFO] Single-Split 測試集樣本數: {len(y_test_quantum)}")
        
        # 使用 Single-Split 測試集繪製 PR 曲線
        plot_pr_curve(
            y_test_quantum.values,
            y_proba_test_quantum,
            y_proba_test_baseline,
            threshold=DEFAULT_THRESHOLD,
            save_path="data/pr_curve_comparison_oos.png",
        )
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print("\n[SUCCESS] PR 曲線比較圖已生成:")
    print("  - data/pr_curve_comparison_oos.png (Out-of-Sample PR 曲線比較)")


if __name__ == "__main__":
    main()
