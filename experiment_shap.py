"""Quantum I-Ching 專案 SHAP 可解釋性分析模組.

使用 SHAP (SHapley Additive exPlanations) 來解釋精簡版 XGBoost 模型的預測。
重點：驗證易經特徵（Moving_Lines_Count, Energy_Delta）是否真的在學習有意義的模式。

分析內容：
1. Summary Plot: 全局特徵重要性和方向性
2. Dependence Plot for Moving_Lines_Count: 驗證「更多動爻 = 更高波動性」假設
3. Dependence Plot for Energy_Delta: 驗證能量變化對波動性的影響
4. Pearson 相關性分析：量化易經特徵與 SHAP 值的關係
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr

# 可選導入 SHAP 和 matplotlib
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[ERROR] shap 未安裝，請執行: pip install shap")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib 未安裝，將跳過圖表繪製")
    print("  安裝指令: pip install matplotlib")

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現.
    
    Args:
        seed: 隨機種子值。
    """
    random.seed(seed)
    np.random.seed(seed)


def prepare_tabular_data(
    encoded_data: pd.DataFrame,
    prediction_window: int = 5,
    volatility_threshold: float = 0.03,
    selected_iching_features: list = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """準備表格型資料集（用於 XGBoost）.
    
    將時間序列資料轉換為表格格式，每個樣本使用 T-0 的特徵預測 T+5 的目標。
    
    Args:
        encoded_data: 包含易經卦象的編碼資料（必須有 Ritual_Sequence）。
        prediction_window: 預測窗口長度（預設 5 天）。
        volatility_threshold: 波動性閾值（預設 0.03，即 3%）。
        selected_iching_features: 要使用的易經特徵列表。
    
    Returns:
        (特徵 DataFrame, 標籤 Series) 二元組。
    """
    print(f"\n[INFO] 準備表格型資料集...")
    print(f"  預測窗口: T+{prediction_window}")
    print(f"  波動性閾值: {volatility_threshold * 100}%")
    
    # 確保必要的基礎欄位存在
    required_base_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return', 'Ritual_Sequence']
    missing_cols = [col for col in required_base_cols if col not in encoded_data.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")
    
    # 使用 DataProcessor 來提取易經特徵
    processor = DataProcessor()
    
    # 定義所有可用的易經特徵
    all_iching_features = [
        'Yang_Count_Main', 'Yang_Count_Future', 'Moving_Lines_Count',
        'Energy_Delta', 'Conflict_Score'
    ]
    
    # 如果未指定，使用預設的精簡特徵
    if selected_iching_features is None:
        selected_iching_features = ['Moving_Lines_Count', 'Energy_Delta']
    
    # 驗證選定的特徵是否有效
    invalid_features = [f for f in selected_iching_features if f not in all_iching_features]
    if invalid_features:
        raise ValueError(f"無效的易經特徵: {invalid_features}")
    
    print(f"[INFO] 使用易經特徵: {selected_iching_features}")
    
    # 提取易經特徵
    print("[INFO] 提取易經特徵...")
    iching_features_list = []
    for idx, ritual_seq in enumerate(encoded_data['Ritual_Sequence']):
        if pd.isna(ritual_seq) or len(str(ritual_seq)) != 6:
            # 如果無效，使用零特徵
            iching_features_list.append([0.0] * len(all_iching_features))
        else:
            try:
                iching_features = processor.extract_iching_features(str(ritual_seq))
                iching_features_list.append(iching_features.tolist())
            except (ValueError, TypeError) as e:
                print(f"[WARNING] 無法提取易經特徵: {ritual_seq}, 錯誤: {e}")
                iching_features_list.append([0.0] * len(all_iching_features))
        
        # 每處理1000筆顯示進度
        if (idx + 1) % 1000 == 0:
            print(f"[INFO] 已處理 {idx + 1}/{len(encoded_data)} 筆易經特徵")
    
    # 轉換為 DataFrame（包含所有特徵）
    iching_df_full = pd.DataFrame(
        iching_features_list,
        columns=all_iching_features,
        index=encoded_data.index
    )
    
    # 只選擇指定的易經特徵
    iching_df = iching_df_full[selected_iching_features].copy()
    
    # 合併數值特徵和選定的易經特徵
    numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
    X = pd.concat([
        encoded_data[numerical_cols],
        iching_df
    ], axis=1)
    
    # 計算目標標籤（T+5 波動性突破）
    max_idx = len(encoded_data) - prediction_window
    
    # 計算未來收益率
    future_prices = encoded_data['Close'].shift(-prediction_window)
    current_prices = encoded_data['Close']
    future_returns = (future_prices - current_prices) / current_prices
    
    # 創建標籤：1 = 高波動（|Return_5d| > threshold），0 = 低波動
    y = (future_returns.abs() > volatility_threshold).astype(int)
    
    # 移除最後 prediction_window 行（無法計算未來收益率）
    X = X.iloc[:max_idx].copy()
    y = y.iloc[:max_idx].copy()
    
    # 移除包含 NaN 的行
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    print(f"[INFO] 資料準備完成:")
    print(f"  總樣本數: {len(X)}")
    print(f"  特徵數: {len(X.columns)}")
    print(f"  標籤分布: 高波動={y.sum()}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")
    
    return X, y


def train_lean_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> xgb.XGBClassifier:
    """訓練精簡版 XGBoost 模型（與 Model C 相同配置）.
    
    Args:
        X_train: 訓練特徵。
        y_train: 訓練標籤。
        X_test: 測試特徵。
        y_test: 測試標籤。
    
    Returns:
        訓練好的 XGBoost 模型。
    """
    print(f"\n[INFO] 訓練精簡版 XGBoost 模型...")
    print(f"  特徵: {list(X_train.columns)}")
    
    # 精簡模式超參數（與 Model C 相同）
    params = {
        'n_estimators': 100,
        'max_depth': 3,  # 更淺的樹
        'learning_rate': 0.05,  # 更低的學習率
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    # 創建 XGBoost 分類器
    model = xgb.XGBClassifier(**params)
    
    # 訓練模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 強制設置特徵名稱到 booster（關鍵步驟）
    model.get_booster().feature_names = list(X_train.columns)
    
    print(f"[INFO] 模型訓練完成")
    
    return model


def analyze_shap(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """執行 SHAP 分析.
    
    Args:
        model: 訓練好的 XGBoost 模型。
        X_test: 測試特徵。
        y_test: 測試標籤。
    
    Returns:
        (SHAP Explainer, SHAP Values) 二元組。
    """
    if not HAS_SHAP:
        raise ImportError("SHAP 未安裝，請執行: pip install shap")
    
    print(f"\n[INFO] 執行 SHAP 分析...")
    print(f"  測試集樣本數: {len(X_test)}")
    
    # 創建 TreeExplainer（適用於 XGBoost）
    explainer = shap.TreeExplainer(model)
    
    # 計算 SHAP 值（使用測試集）
    # 注意：對於大型數據集，可以只使用樣本子集
    shap_values = explainer.shap_values(X_test)
    
    print(f"[INFO] SHAP 值計算完成")
    print(f"  SHAP 值形狀: {shap_values.shape}")
    
    return explainer, shap_values


def plot_shap_summary(
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    save_path: str = "data/shap_summary.png"
) -> None:
    """繪製 SHAP Summary Plot.
    
    Args:
        explainer: SHAP Explainer。
        X_test: 測試特徵。
        shap_values: SHAP 值。
        save_path: 保存路徑。
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib 未安裝，跳過 Summary Plot")
        return
    
    print(f"\n[INFO] 繪製 SHAP Summary Plot...")
    
    # 創建 Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
    plt.title("SHAP Summary Plot - Feature Importance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存圖表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Summary Plot 已保存至: {save_path}")
    plt.close()


def plot_shap_dependence(
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    feature_name: str,
    save_path: str
) -> None:
    """繪製 SHAP Dependence Plot.
    
    Args:
        explainer: SHAP Explainer。
        X_test: 測試特徵。
        shap_values: SHAP 值。
        feature_name: 要分析的特徵名稱。
        save_path: 保存路徑。
    """
    if not HAS_MATPLOTLIB:
        print(f"[WARNING] matplotlib 未安裝，跳過 {feature_name} Dependence Plot")
        return
    
    if feature_name not in X_test.columns:
        print(f"[WARNING] 特徵 {feature_name} 不存在於測試集中")
        return
    
    print(f"\n[INFO] 繪製 {feature_name} 的 Dependence Plot...")
    
    # 獲取特徵索引
    feature_idx = list(X_test.columns).index(feature_name)
    
    # 創建 Dependence Plot
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X_test,
        show=False
    )
    plt.title(f"SHAP Dependence Plot - {feature_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存圖表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Dependence Plot 已保存至: {save_path}")
    plt.close()


def calculate_correlation(
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    feature_name: str
) -> Tuple[float, float]:
    """計算特徵值與 SHAP 值的 Pearson 相關性.
    
    Args:
        X_test: 測試特徵。
        shap_values: SHAP 值。
        feature_name: 要分析的特徵名稱。
    
    Returns:
        (相關係數, p-value) 二元組。
    """
    if feature_name not in X_test.columns:
        raise ValueError(f"特徵 {feature_name} 不存在於測試集中")
    
    # 獲取特徵索引
    feature_idx = list(X_test.columns).index(feature_name)
    
    # 提取特徵值和對應的 SHAP 值
    feature_values = X_test[feature_name].values
    feature_shap_values = shap_values[:, feature_idx]
    
    # 計算 Pearson 相關性
    correlation, p_value = pearsonr(feature_values, feature_shap_values)
    
    return correlation, p_value


def interpret_correlation(correlation: float, p_value: float) -> str:
    """解釋相關性結果.
    
    Args:
        correlation: Pearson 相關係數。
        p_value: p 值。
    
    Returns:
        解釋文字。
    """
    abs_corr = abs(correlation)
    
    # 判斷相關性強度
    if abs_corr > 0.7:
        strength = "非常強"
    elif abs_corr > 0.5:
        strength = "強"
    elif abs_corr > 0.3:
        strength = "中等"
    elif abs_corr > 0.1:
        strength = "弱"
    else:
        strength = "極弱"
    
    # 判斷方向
    direction = "正" if correlation > 0 else "負"
    
    # 判斷統計顯著性
    if p_value < 0.001:
        significance = "極顯著 (p < 0.001)"
    elif p_value < 0.01:
        significance = "非常顯著 (p < 0.01)"
    elif p_value < 0.05:
        significance = "顯著 (p < 0.05)"
    else:
        significance = "不顯著 (p >= 0.05)"
    
    # 生成解釋
    interpretation = f"{strength}{direction}相關關係"
    
    if abs_corr > 0.5 and p_value < 0.05:
        conclusion = "✅ 驗證易經理論：特徵與波動性預測有明確的因果關係"
    elif abs_corr > 0.3 and p_value < 0.05:
        conclusion = "⚠️ 部分驗證：特徵有一定影響，但關係較弱"
    elif p_value >= 0.05:
        conclusion = "❌ 統計不顯著：可能是隨機噪音，易經特徵可能無用"
    else:
        conclusion = "❓ 關係不明確：需要進一步分析"
    
    return f"{interpretation} (r={correlation:.4f}, {significance})\n  結論: {conclusion}"


def main() -> None:
    """主函數：執行 SHAP 可解釋性分析."""
    print("=" * 80)
    print("Quantum I-Ching SHAP 可解釋性分析")
    print("=" * 80)
    print("\n目標：驗證易經特徵是否真的在學習有意義的模式")
    print("方法：SHAP (SHapley Additive exPlanations)")
    print("模型：精簡版 XGBoost (Moving_Lines_Count + Energy_Delta)")
    print("預測目標：波動性突破（|Return_5d| > 3%）\n")
    
    if not HAS_SHAP:
        print("[ERROR] 請先安裝 SHAP: pip install shap")
        return
    
    set_random_seed(42)
    
    # 載入資料
    print("[INFO] 載入市場資料...")
    loader = MarketDataLoader()
    default_symbol = (
        settings.TARGET_TICKERS[0] if settings.TARGET_TICKERS else "NVDA"
    )
    raw_data = loader.fetch_data(tickers=[default_symbol])
    
    if raw_data.empty:
        print(f"[ERROR] 無法獲取 {default_symbol} 的市場資料")
        return
    
    # 編碼為易經卦象
    print("[INFO] 編碼為易經卦象...")
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)
    
    if encoded_data.empty:
        print("[ERROR] 編碼後的資料為空")
        return
    
    # 準備表格型資料（精簡版：只使用表現最好的易經特徵）
    X, y = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        selected_iching_features=['Moving_Lines_Count', 'Energy_Delta']
    )
    
    # 分割資料（時間序列分割，不隨機打亂）
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    
    print(f"\n[INFO] 資料分割:")
    print(f"  訓練集: {len(X_train)} 樣本")
    print(f"  測試集: {len(X_test)} 樣本")
    
    # 訓練精簡版模型
    print("\n" + "=" * 80)
    print("訓練精簡版 XGBoost 模型")
    print("=" * 80)
    model = train_lean_model(X_train, y_train, X_test, y_test)
    
    # 執行 SHAP 分析
    print("\n" + "=" * 80)
    print("SHAP 分析")
    print("=" * 80)
    explainer, shap_values = analyze_shap(model, X_test, y_test)
    
    # 繪製圖表
    print("\n" + "=" * 80)
    print("生成可視化圖表")
    print("=" * 80)
    
    # Plot 1: Summary Plot
    plot_shap_summary(
        explainer,
        X_test,
        shap_values,
        save_path="data/shap_summary.png"
    )
    
    # Plot 2: Dependence Plot for Moving_Lines_Count
    plot_shap_dependence(
        explainer,
        X_test,
        shap_values,
        feature_name="Moving_Lines_Count",
        save_path="data/shap_dependence_moving_lines.png"
    )
    
    # Plot 3: Dependence Plot for Energy_Delta
    plot_shap_dependence(
        explainer,
        X_test,
        shap_values,
        feature_name="Energy_Delta",
        save_path="data/shap_dependence_energy_delta.png"
    )
    
    # 計算並解釋相關性
    print("\n" + "=" * 80)
    print("Pearson 相關性分析")
    print("=" * 80)
    
    # Moving_Lines_Count 相關性
    print("\n[INFO] Moving_Lines_Count 與 SHAP 值的相關性:")
    corr_moving, p_moving = calculate_correlation(X_test, shap_values, "Moving_Lines_Count")
    print(f"  相關係數: {corr_moving:.4f}")
    print(f"  p 值: {p_moving:.4e}")
    interpretation_moving = interpret_correlation(corr_moving, p_moving)
    print(f"  解釋: {interpretation_moving}")
    
    # Energy_Delta 相關性
    print("\n[INFO] Energy_Delta 與 SHAP 值的相關性:")
    corr_energy, p_energy = calculate_correlation(X_test, shap_values, "Energy_Delta")
    print(f"  相關係數: {corr_energy:.4f}")
    print(f"  p 值: {p_energy:.4e}")
    interpretation_energy = interpret_correlation(corr_energy, p_energy)
    print(f"  解釋: {interpretation_energy}")
    
    # 總結
    print("\n" + "=" * 80)
    print("分析總結")
    print("=" * 80)
    
    print("\n[INFO] 易經特徵解釋性分析結果:")
    print(f"  1. Moving_Lines_Count:")
    print(f"     - 相關性: {corr_moving:.4f} (p={p_moving:.4e})")
    if corr_moving > 0.5 and p_moving < 0.05:
        print(f"     - ✅ 強烈支持「更多動爻 = 更高波動性」假設")
    elif corr_moving > 0 and p_moving < 0.05:
        print(f"     - ⚠️ 部分支持假設，但關係較弱")
    else:
        print(f"     - ❌ 無法驗證假設，可能是隨機噪音")
    
    print(f"\n  2. Energy_Delta:")
    print(f"     - 相關性: {corr_energy:.4f} (p={p_energy:.4e})")
    if abs(corr_energy) > 0.5 and p_energy < 0.05:
        print(f"     - ✅ 能量變化與波動性預測有明確關係")
    elif abs(corr_energy) > 0.3 and p_energy < 0.05:
        print(f"     - ⚠️ 能量變化有一定影響，但關係較弱")
    else:
        print(f"     - ❌ 無法驗證能量變化與波動性的關係")
    
    print("\n[INFO] 所有圖表已保存至 data/ 資料夾:")
    print("  - data/shap_summary.png")
    print("  - data/shap_dependence_moving_lines.png")
    print("  - data/shap_dependence_energy_delta.png")


if __name__ == "__main__":
    main()
