"""生成最終 ROC 曲線比較圖（用於簡報）.

此腳本比較三個 XGBoost 模型版本，展示特徵選擇過程：
- Model A (Full): 所有特徵（市場數據 + 所有 5 個易經特徵）- 有噪音
- Model B (Baseline): 僅市場數據 - 基準模型
- Model C (Lean): 市場數據 + 2 個最佳易經特徵 - 最終產品
"""

import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder

# 配置 matplotlib 中文字體
def setup_chinese_font():
    """設置 matplotlib 使用中文字體."""
    chinese_fonts = [
        'Microsoft YaHei',      # 微軟雅黑
        'SimHei',               # 黑體
        'SimSun',               # 宋體
        'Microsoft JhengHei',   # 微軟正黑體
        'KaiTi',                # 楷體
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


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)


def prepare_features(
    encoded_data: pd.DataFrame,
    feature_set: str = "baseline"
) -> Tuple[pd.DataFrame, pd.Series]:
    """準備不同特徵集的資料.
    
    Args:
        encoded_data: 包含易經卦象的編碼資料（必須有 Ritual_Sequence）。
        feature_set: 特徵集類型 ("baseline", "lean", "full")。
    
    Returns:
        (特徵 DataFrame, 標籤 Series) 二元組。
    """
    print(f"\n[INFO] 準備 {feature_set.upper()} 特徵集...")
    
    # 確保必要的基礎欄位存在
    required_base_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return', 'Ritual_Sequence']
    missing_cols = [col for col in required_base_cols if col not in encoded_data.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")
    
    # 基礎數值特徵
    numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
    X = encoded_data[numerical_cols].copy()
    
    # 如果需要易經特徵，提取它們
    if feature_set in ["lean", "full"]:
        processor = DataProcessor()
        
        # 定義所有可用的易經特徵
        all_iching_features = [
            'Yang_Count_Main', 'Yang_Count_Future', 'Moving_Lines_Count',
            'Energy_Delta', 'Conflict_Score'
        ]
        
        # 提取易經特徵
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
        
        # 轉換為 DataFrame
        iching_df = pd.DataFrame(
            iching_features_list,
            columns=all_iching_features,
            index=encoded_data.index
        )
        
        if feature_set == "lean":
            # Model C: 只使用 2 個最佳特徵
            selected_features = ['Moving_Lines_Count', 'Energy_Delta']
            iching_df = iching_df[selected_features].copy()
            print(f"[INFO] 使用精簡易經特徵: {selected_features}")
        else:
            # Model A: 使用所有 5 個特徵
            print(f"[INFO] 使用完整易經特徵: {all_iching_features}")
        
        # 合併特徵
        X = pd.concat([X, iching_df], axis=1)
    
    # 計算目標標籤（T+5 波動性突破）
    prediction_window = 5
    volatility_threshold = 0.03
    max_idx = len(encoded_data) - prediction_window
    
    # 計算未來收益率
    future_prices = encoded_data['Close'].shift(-prediction_window)
    current_prices = encoded_data['Close']
    future_returns = (future_prices - current_prices) / current_prices
    
    # 創建標籤：1 = 高波動（|Return_5d| > threshold），0 = 低波動
    y = (future_returns.abs() > volatility_threshold).astype(int)
    
    # 移除最後 prediction_window 行
    X = X.iloc[:max_idx].copy()
    y = y.iloc[:max_idx].copy()
    
    # 移除包含 NaN 的行
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    print(f"[INFO] {feature_set.upper()} 特徵集準備完成:")
    print(f"  總樣本數: {len(X)}")
    print(f"  特徵數: {len(X.columns)}")
    print(f"  特徵列表: {list(X.columns)}")
    print(f"  標籤分布: 高波動={y.sum()}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")
    
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model"
) -> xgb.XGBClassifier:
    """訓練 XGBoost 模型（使用固定超參數）.
    
    Args:
        X_train: 訓練特徵 DataFrame。
        y_train: 訓練標籤 Series。
        model_name: 模型名稱（用於日誌）。
    
    Returns:
        訓練好的 XGBoost 模型。
    """
    print(f"\n[INFO] 訓練 {model_name}...")
    
    # 固定超參數（確保公平比較）
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
    
    # 創建並訓練模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # 強制設置特徵名稱
    model.get_booster().feature_names = list(X_train.columns)
    
    print(f"[INFO] {model_name} 訓練完成")
    
    return model


def plot_roc_curves(
    y_test: pd.Series,
    y_pred_proba_a: np.ndarray,
    y_pred_proba_b: np.ndarray,
    y_pred_proba_c: np.ndarray,
    save_path: str = "data/final_roc_curves.png"
) -> None:
    """繪製三個模型的 ROC 曲線比較圖.
    
    Args:
        y_test: 測試集真實標籤。
        y_pred_proba_a: Model A 的預測機率。
        y_pred_proba_b: Model B 的預測機率。
        y_pred_proba_c: Model C 的預測機率。
        save_path: 保存路徑。
    """
    print(f"\n[INFO] 計算 ROC 曲線...")
    
    # 計算 ROC 曲線
    fpr_a, tpr_a, _ = roc_curve(y_test, y_pred_proba_a)
    fpr_b, tpr_b, _ = roc_curve(y_test, y_pred_proba_b)
    fpr_c, tpr_c, _ = roc_curve(y_test, y_pred_proba_c)
    
    # 計算 AUC
    auc_a = auc(fpr_a, tpr_a)
    auc_b = auc(fpr_b, tpr_b)
    auc_c = auc(fpr_c, tpr_c)
    
    print(f"[INFO] AUC 分數:")
    print(f"  Model A (Full): {auc_a:.4f}")
    print(f"  Model B (Baseline): {auc_b:.4f}")
    print(f"  Model C (Lean): {auc_c:.4f}")
    
    # 繪製 ROC 曲線
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Model A: 紅色虛線
    ax.plot(fpr_a, tpr_a, 
            label=f'Model A: Full I-Ching (AUC = {auc_a:.3f})',
            color='#E74C3C', linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Model B: 藍色實線
    ax.plot(fpr_b, tpr_b,
            label=f'Model B: Baseline (AUC = {auc_b:.3f})',
            color='#3498DB', linestyle='-', linewidth=2.5, alpha=0.8)
    
    # Model C: 綠色實線
    ax.plot(fpr_c, tpr_c,
            label=f'Model C: Lean I-Ching (AUC = {auc_c:.3f})',
            color='#2ECC71', linestyle='-', linewidth=2.5, alpha=0.8)
    
    # 對角線（隨機分類器）
    ax.plot([0, 1], [0, 1], 
            color='#95A5A6', linestyle=':', linewidth=1.5, alpha=0.5,
            label='Random Classifier (AUC = 0.500)')
    
    # 設置標題和標籤
    ax.set_title('ROC Curves: Feature Selection Process Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    
    # 設置圖例
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # 設置網格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 設置座標軸範圍
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # 添加文字說明
    improvement_c = auc_c - auc_b
    if improvement_c > 0:
        text_color = '#2ECC71'
        text = f'Model C 優於 Baseline: +{improvement_c:.4f} ({improvement_c/auc_b*100:.2f}%)'
    else:
        text_color = '#E74C3C'
        text = f'Model C 劣於 Baseline: {improvement_c:.4f} ({improvement_c/auc_b*100:.2f}%)'
    
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8, edgecolor=text_color, linewidth=2))
    
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] ROC 曲線圖已保存至: {save_path}")
    
    plt.close()


def main() -> None:
    """主函數：生成最終 ROC 曲線比較圖."""
    print("=" * 80)
    print("最終 ROC 曲線比較 - 特徵選擇過程")
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
    
    # 準備三個特徵集
    X_baseline, y_baseline = prepare_features(encoded_data, feature_set="baseline")
    X_lean, y_lean = prepare_features(encoded_data, feature_set="lean")
    X_full, y_full = prepare_features(encoded_data, feature_set="full")
    
    # 確保標籤一致（應該相同，因為目標定義相同）
    assert len(y_baseline) == len(y_lean) == len(y_full), "標籤長度不一致"
    assert (y_baseline.values == y_lean.values).all() and (y_lean.values == y_full.values).all(), "標籤不一致"
    y = y_baseline  # 使用統一的標籤
    
    # 分割訓練集和測試集（80/20，時間序列分割，不隨機打亂）
    split_idx = int(len(X_baseline) * 0.8)
    
    # Model B (Baseline) 資料分割
    X_train_b = X_baseline.iloc[:split_idx].copy()
    X_test_b = X_baseline.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    # Model C (Lean) 資料分割
    X_train_c = X_lean.iloc[:split_idx].copy()
    X_test_c = X_lean.iloc[split_idx:].copy()
    
    # Model A (Full) 資料分割
    X_train_a = X_full.iloc[:split_idx].copy()
    X_test_a = X_full.iloc[split_idx:].copy()
    
    print(f"\n[INFO] 資料分割:")
    print(f"  訓練集: {len(X_train_b)} 樣本")
    print(f"  測試集: {len(X_test_b)} 樣本")
    
    # 訓練三個模型
    model_a = train_model(X_train_a, y_train, model_name="Model A (Full I-Ching)")
    model_b = train_model(X_train_b, y_train, model_name="Model B (Baseline)")
    model_c = train_model(X_train_c, y_train, model_name="Model C (Lean I-Ching)")
    
    # 獲取預測機率
    print("\n[INFO] 進行模型預測...")
    y_pred_proba_a = model_a.predict_proba(X_test_a)[:, 1]
    y_pred_proba_b = model_b.predict_proba(X_test_b)[:, 1]
    y_pred_proba_c = model_c.predict_proba(X_test_c)[:, 1]
    
    # 繪製 ROC 曲線
    plot_roc_curves(
        y_test,
        y_pred_proba_a,
        y_pred_proba_b,
        y_pred_proba_c,
        save_path="data/final_roc_curves.png"
    )
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print("\n[SUCCESS] 最終 ROC 曲線比較圖已生成:")
    print("  - data/final_roc_curves.png")
    print(f"\n[INFO] 模型參數:")
    print(f"  - n_estimators: 100")
    print(f"  - max_depth: 3")
    print(f"  - learning_rate: 0.05")
    print(f"  - 測試集樣本數: {len(y_test)}")


if __name__ == "__main__":
    main()
