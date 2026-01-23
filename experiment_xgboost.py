"""Quantum I-Ching 專案 XGBoost 實驗模組.

使用 XGBoost（梯度提升）來驗證易經特徵的預測能力。
重點：分析特徵重要性，確認易經特徵是否被模型使用。

實驗設計：
1. Model A: 包含易經特徵（數值特徵 + 易經特徵）
2. Model B: 基準模型（僅數值特徵）
3. 比較兩者的 AUC、Precision、Recall
4. 繪製特徵重要性圖，確認易經特徵的貢獻
"""

import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 可選導入 matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib 未安裝，將跳過特徵重要性圖繪製")
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
    selected_iching_features: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """準備表格型資料集（用於 XGBoost）.
    
    將時間序列資料轉換為表格格式，每個樣本使用 T-0 的特徵預測 T+5 的目標。
    
    Args:
        encoded_data: 包含易經卦象的編碼資料（必須有 Ritual_Sequence）。
        prediction_window: 預測窗口長度（預設 5 天）。
        volatility_threshold: 波動性閾值（預設 0.03，即 3%）。
        selected_iching_features: 要使用的易經特徵列表（預設 None，使用全部 5 個）。
    
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
    
    # 如果指定了選定的特徵，只使用這些；否則使用全部
    if selected_iching_features is None:
        selected_iching_features = all_iching_features
    
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
    # 需要確保有足夠的未來資料
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


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost",
    use_iching_features: bool = True,
    lean_mode: bool = False
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """訓練 XGBoost 模型.
    
    Args:
        X_train: 訓練特徵。
        y_train: 訓練標籤。
        X_test: 測試特徵。
        y_test: 測試標籤。
        model_name: 模型名稱（用於顯示）。
        use_iching_features: 是否使用易經特徵。
        lean_mode: 是否使用精簡模式（更嚴格的超參數以防止過擬合）。
    
    Returns:
        (訓練好的模型, 評估指標字典) 二元組。
    """
    print(f"\n[INFO] 訓練 {model_name} 模型...")
    print(f"  使用易經特徵: {use_iching_features}")
    print(f"  精簡模式: {lean_mode}")
    
    # 如果不需要易經特徵，只使用前 4 個數值特徵
    if not use_iching_features:
        numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        X_train = X_train[numerical_cols].copy()
        X_test = X_test[numerical_cols].copy()
        print(f"  特徵: {numerical_cols}")
    else:
        print(f"  特徵: {list(X_train.columns)}")
    
    # 根據模式設置超參數
    if lean_mode:
        # 精簡模式：更嚴格的超參數以防止過擬合
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
    else:
        # 標準模式
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
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
    
    # 預測
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 計算評估指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    
    print(f"[INFO] {model_name} 評估結果:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    return model, metrics


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list,
    save_path: str = "data/feature_importance.png",
    title: str = "XGBoost Feature Importance"
) -> None:
    """繪製特徵重要性圖.
    
    Args:
        model: 訓練好的 XGBoost 模型（必須已設置 feature_names）。
        feature_names: 特徵名稱列表（用於驗證）。
        save_path: 保存路徑。
        title: 圖表標題。
    """
    # 獲取特徵重要性（gain）- 使用正確的特徵名稱
    importance = model.get_booster().get_score(importance_type='gain')
    
    # 如果 importance 為空或只有 f0, f1 等，說明特徵名稱未正確設置
    if not importance:
        print("[WARNING] 無法獲取特徵重要性，嘗試使用 feature_names")
        # 嘗試從 booster 獲取特徵名稱
        booster_feature_names = model.get_booster().feature_names
        if booster_feature_names:
            print(f"[DEBUG] Booster feature_names: {booster_feature_names}")
        else:
            print("[ERROR] Booster 中沒有特徵名稱")
            return
    
    # 排序特徵重要性（按重要性值降序）
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # 顯示前 5 個最重要的特徵（無論是否有 matplotlib）
    print(f"\n[INFO] 前 5 個最重要的特徵:")
    for i, (name, score) in enumerate(sorted_importance[:5], 1):
        print(f"  {i}. {name}: {score:.4f}")
    
    # 如果 matplotlib 可用，繪製圖表
    if HAS_MATPLOTLIB:
        # 準備繪圖數據（取前 10 個特徵）
        top_n = min(10, len(sorted_importance))
        names = [x[0] for x in sorted_importance[:top_n]]
        scores = [x[1] for x in sorted_importance[:top_n]]
        
        # 繪製水平條形圖（反轉順序以便最重要的在頂部）
        plt.figure(figsize=(10, 6))
        plt.barh(names[::-1], scores[::-1], color='steelblue')
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 保存圖表
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[INFO] 特徵重要性圖已保存至: {save_path}")
        plt.close()
    else:
        print(f"\n[INFO] 跳過圖表繪製（matplotlib 未安裝）")
        print(f"  特徵重要性數據已顯示在上方")


def main() -> None:
    """主函數：執行 XGBoost 實驗."""
    print("=" * 80)
    print("Quantum I-Ching XGBoost 實驗（精簡版優化）")
    print("=" * 80)
    print("\n目標：驗證易經特徵的預測能力並分析特徵重要性")
    print("方法：XGBoost 梯度提升分類器")
    print("預測目標：波動性突破（|Return_5d| > 3%）")
    print("\n實驗設計：")
    print("  - Model A: 完整版（所有 5 個易經特徵）")
    print("  - Model B: 基準模型（僅數值特徵）")
    print("  - Model C: 精簡版（只使用 Moving_Lines_Count + Energy_Delta）\n")
    
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
    
    # 準備表格型資料（完整版：所有易經特徵）
    X_full, y = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        selected_iching_features=None  # 使用全部 5 個易經特徵
    )
    
    # 準備表格型資料（精簡版：只使用表現最好的易經特徵）
    X_lean, y_lean = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        selected_iching_features=['Moving_Lines_Count', 'Energy_Delta']  # 只使用這兩個
    )
    
    # 分割資料（時間序列分割，不隨機打亂）
    split_idx_full = int(len(X_full) * 0.8)
    split_idx_lean = int(len(X_lean) * 0.8)
    
    # 完整版資料分割
    X_train_full = X_full.iloc[:split_idx_full].copy()
    y_train_full = y.iloc[:split_idx_full].copy()
    X_test_full = X_full.iloc[split_idx_full:].copy()
    y_test_full = y.iloc[split_idx_full:].copy()
    
    # 精簡版資料分割
    X_train_lean = X_lean.iloc[:split_idx_lean].copy()
    y_train_lean = y_lean.iloc[:split_idx_lean].copy()
    X_test_lean = X_lean.iloc[split_idx_lean:].copy()
    y_test_lean = y_lean.iloc[split_idx_lean:].copy()
    
    print(f"\n[INFO] 資料分割:")
    print(f"  完整版 - 訓練集: {len(X_train_full)} 樣本, 測試集: {len(X_test_full)} 樣本")
    print(f"  精簡版 - 訓練集: {len(X_train_lean)} 樣本, 測試集: {len(X_test_lean)} 樣本")
    
    # 訓練 Model A: 完整版（所有易經特徵）
    print("\n" + "=" * 80)
    print("Model A: 完整版（所有 5 個易經特徵）")
    print("=" * 80)
    model_a, metrics_a = train_xgboost_model(
        X_train_full, y_train_full, X_test_full, y_test_full,
        model_name="Quantum XGBoost (Full)",
        use_iching_features=True,
        lean_mode=False
    )
    
    # 繪製 Model A 的特徵重要性
    plot_feature_importance(
        model_a,
        feature_names=list(X_full.columns),
        save_path="data/feature_importance_with_iching.png",
        title="Feature Importance (With All I-Ching Features)"
    )
    
    # 訓練 Model B: 基準模型（僅數值特徵）
    print("\n" + "=" * 80)
    print("Model B: 基準模型（僅數值特徵）")
    print("=" * 80)
    model_b, metrics_b = train_xgboost_model(
        X_train_full, y_train_full, X_test_full, y_test_full,
        model_name="Baseline XGBoost",
        use_iching_features=False,
        lean_mode=False
    )
    
    # 繪製 Model B 的特徵重要性
    plot_feature_importance(
        model_b,
        feature_names=['Close', 'Volume', 'RVOL', 'Daily_Return'],
        save_path="data/feature_importance_baseline.png",
        title="Feature Importance (Baseline - Numerical Features Only)"
    )
    
    # 訓練 Model C: 精簡版（只使用表現最好的易經特徵）
    print("\n" + "=" * 80)
    print("Model C: 精簡版（Moving_Lines_Count + Energy_Delta）")
    print("=" * 80)
    model_c, metrics_c = train_xgboost_model(
        X_train_lean, y_train_lean, X_test_lean, y_test_lean,
        model_name="Quantum XGBoost (Lean)",
        use_iching_features=True,
        lean_mode=True  # 使用精簡模式（更嚴格的超參數）
    )
    
    # 繪製 Model C 的特徵重要性
    plot_feature_importance(
        model_c,
        feature_names=list(X_lean.columns),
        save_path="data/feature_importance_lean.png",
        title="Feature Importance (Lean - Top 2 I-Ching Features Only)"
    )
    
    # 比較結果（三個模型）
    print("\n" + "=" * 80)
    print("模型比較結果")
    print("=" * 80)
    print(f"\n{'指標':<15} {'Model A (完整)':<20} {'Model B (基準)':<20} {'Model C (精簡)':<20}")
    print("-" * 80)
    
    for metric in ['accuracy', 'precision', 'recall', 'auc']:
        val_a = metrics_a[metric]
        val_b = metrics_b[metric]
        val_c = metrics_c[metric]
        print(f"{metric.capitalize():<15} {val_a:<20.4f} {val_b:<20.4f} {val_c:<20.4f}")
    
    # 計算差異
    print("\n" + "=" * 80)
    print("與基準模型 (Model B) 的差異")
    print("=" * 80)
    print(f"\n{'指標':<15} {'Model A vs B':<20} {'Model C vs B':<20}")
    print("-" * 80)
    
    for metric in ['accuracy', 'precision', 'recall', 'auc']:
        diff_a = metrics_a[metric] - metrics_b[metric]
        diff_c = metrics_c[metric] - metrics_b[metric]
        diff_a_str = f"{diff_a:+.4f} ({diff_a/metrics_b[metric]*100:+.2f}%)"
        diff_c_str = f"{diff_c:+.4f} ({diff_c/metrics_b[metric]*100:+.2f}%)"
        print(f"{metric.capitalize():<15} {diff_a_str:<20} {diff_c_str:<20}")
    
    # 結論
    print("\n" + "=" * 80)
    print("結論")
    print("=" * 80)
    
    # 比較 Model A vs Baseline
    if metrics_a['auc'] > metrics_b['auc']:
        improvement_a = ((metrics_a['auc'] - metrics_b['auc']) / metrics_b['auc']) * 100
        print(f"[INFO] Model A (完整版易經特徵) vs Baseline:")
        print(f"  AUC 提升: {improvement_a:.2f}%")
    else:
        decline_a = ((metrics_b['auc'] - metrics_a['auc']) / metrics_b['auc']) * 100
        print(f"[INFO] Model A (完整版易經特徵) vs Baseline:")
        print(f"  AUC 下降: {decline_a:.2f}% (可能因噪音特徵導致過擬合)")
    
    # 比較 Model C vs Baseline
    if metrics_c['auc'] > metrics_b['auc']:
        improvement_c = ((metrics_c['auc'] - metrics_b['auc']) / metrics_b['auc']) * 100
        print(f"\n[SUCCESS] Model C (精簡版易經特徵) vs Baseline:")
        print(f"  AUC 提升: {improvement_c:.2f}%")
        print(f"  結論：移除噪音特徵後，精簡版模型優於基準模型！")
    else:
        decline_c = ((metrics_b['auc'] - metrics_c['auc']) / metrics_b['auc']) * 100
        print(f"\n[INFO] Model C (精簡版易經特徵) vs Baseline:")
        print(f"  AUC 下降: {decline_c:.2f}%")
    
    # 比較 Model C vs Model A
    if metrics_c['auc'] > metrics_a['auc']:
        improvement_ca = ((metrics_c['auc'] - metrics_a['auc']) / metrics_a['auc']) * 100
        print(f"\n[SUCCESS] Model C (精簡版) vs Model A (完整版):")
        print(f"  AUC 提升: {improvement_ca:.2f}%")
        print(f"  結論：特徵選擇成功！精簡版優於完整版。")
    else:
        print(f"\n[INFO] Model C (精簡版) vs Model A (完整版):")
        print(f"  完整版仍略優，但精簡版更簡潔且不易過擬合")
    
    # 檢查易經特徵重要性（Model C）
    print("\n" + "=" * 80)
    print("精簡版模型 (Model C) 特徵重要性分析")
    print("=" * 80)
    importance_c = model_c.get_booster().get_score(importance_type='gain')
    sorted_importance_c = sorted(importance_c.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n[INFO] 特徵重要性排名（全部特徵）:")
    for i, (feat, imp) in enumerate(sorted_importance_c, 1):
        is_iching = feat in ['Moving_Lines_Count', 'Energy_Delta']
        marker = "[易經]" if is_iching else ""
        print(f"  {i}. {feat}: {imp:.4f} {marker}")
    
    # 驗證易經特徵的重要性
    lean_iching_features = ['Moving_Lines_Count', 'Energy_Delta']
    lean_top_features = [feat for feat, _ in sorted_importance_c[:len(X_lean.columns)]]
    iching_in_lean = sum(1 for feat in lean_top_features if feat in lean_iching_features)
    
    print(f"\n[INFO] 易經特徵在精簡版模型中的排名:")
    for feat in lean_iching_features:
        rank = next((i+1 for i, (f, _) in enumerate(sorted_importance_c) if f == feat), None)
        if rank:
            print(f"  {feat}: 排名第 {rank}")
    
    if iching_in_lean >= 1:
        print("\n[SUCCESS] 精簡版易經特徵確實在模型中被使用！")
    else:
        print("\n[WARNING] 精簡版易經特徵的重要性較低")


if __name__ == "__main__":
    main()
