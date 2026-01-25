"""生成 Model C 的評估圖表（混淆矩陣、ROC 曲線）並進行 Bitcoin 測試.

此腳本會：
1. 載入 TSMC (2330.TW) 數據並訓練 Model C
2. 生成混淆矩陣圖
3. 生成 ROC 曲線圖（比較 Model C vs Baseline）
4. 進行 Bitcoin 測試（快速檢查易經特徵在加密貨幣上的表現）
"""

import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder

# 配置 matplotlib 中文字體
def setup_chinese_font():
    """設置 matplotlib 使用中文字體."""
    # Windows 常見中文字體列表（按優先順序）
    chinese_fonts = [
        'Microsoft YaHei',      # 微軟雅黑
        'SimHei',               # 黑體
        'SimSun',               # 宋體
        'Microsoft JhengHei',   # 微軟正黑體
        'KaiTi',                # 楷體
    ]
    
    # 查找可用的中文字體
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_to_use = None
    
    for font in chinese_fonts:
        if font in available_fonts:
            font_to_use = font
            break
    
    if font_to_use:
        plt.rcParams['font.sans-serif'] = [font_to_use]
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        print(f"[INFO] 已設置中文字體: {font_to_use}")
    else:
        # 如果找不到中文字體，嘗試使用系統預設字體
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("[WARNING] 未找到中文字體，可能無法正確顯示中文")

# 在導入時設置字體
setup_chinese_font()


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
) -> Tuple[pd.DataFrame, pd.Series]:
    """準備表格型資料集（用於 XGBoost）.
    
    將時間序列資料轉換為表格格式，每個樣本使用 T-0 的特徵預測 T+5 的目標。
    
    Args:
        encoded_data: 包含易經卦象的編碼資料（必須有 Ritual_Sequence）。
        prediction_window: 預測窗口長度（預設 5 天）。
        volatility_threshold: 波動性閾值（預設 0.03，即 3%）。
        selected_iching_features: 要使用的易經特徵列表。
        use_iching: 是否使用易經特徵（False 時為 Baseline 模型）。
    
    Returns:
        (特徵 DataFrame, 標籤 Series) 二元組。
    """
    print(f"\n[INFO] 準備表格型資料集...")
    print(f"  預測窗口: T+{prediction_window}")
    print(f"  波動性閾值: {volatility_threshold * 100}%")
    print(f"  使用易經特徵: {use_iching}")
    
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
    
    # 提取易經特徵（如果需要）
    if use_iching:
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
    else:
        # Baseline 模型：只使用數值特徵
        print("[INFO] Baseline 模型：只使用數值特徵")
        numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        X = encoded_data[numerical_cols].copy()
    
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


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model C",
    split_idx: int = None
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series]:
    """訓練 XGBoost 模型.
    
    Args:
        X: 特徵 DataFrame。
        y: 標籤 Series。
        model_name: 模型名稱（用於日誌）。
        split_idx: 分割索引（如果為 None，則使用 80/20 分割）。
    
    Returns:
        (模型, 測試集特徵, 測試集標籤) 三元組。
    """
    print(f"\n[INFO] 訓練 {model_name}...")
    print(f"  特徵: {list(X.columns)}")
    print(f"  樣本數: {len(X)}")
    
    # 分割訓練集和測試集（80/20）
    if split_idx is None:
        split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Model C 超參數
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
    
    # 創建 XGBoost 分類器
    model = xgb.XGBClassifier(**params)
    
    # 訓練模型
    model.fit(X_train, y_train, verbose=False)
    
    # 強制設置特徵名稱到 booster
    model.get_booster().feature_names = list(X.columns)
    
    print(f"[INFO] {model_name} 訓練完成")
    print(f"  訓練集樣本數: {len(X_train)}")
    print(f"  測試集樣本數: {len(X_test)}")
    
    return model, X_test, y_test


def plot_confusion_matrix(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "data/confusion_matrix_2330.png"
) -> None:
    """繪製混淆矩陣.
    
    Args:
        model: 訓練好的 XGBoost 模型。
        X_test: 測試集特徵。
        y_test: 測試集標籤。
        save_path: 保存路徑。
    """
    print(f"\n[INFO] 生成混淆矩陣圖...")
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    
    # 繪製混淆矩陣
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['低波動', '高波動'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    ax.set_title('混淆矩陣 - Model C (TSMC 2330.TW)', fontsize=14, fontweight='bold')
    ax.set_xlabel('預測標籤', fontsize=12)
    ax.set_ylabel('真實標籤', fontsize=12)
    
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 混淆矩陣圖已保存至: {save_path}")
    
    # 顯示分類報告
    print("\n[INFO] 分類報告:")
    print(classification_report(y_test, y_pred, target_names=['低波動', '高波動']))
    
    plt.close()


def plot_roc_curve(
    model_c: xgb.XGBClassifier,
    model_baseline: xgb.XGBClassifier,
    X_test_c: pd.DataFrame,
    X_test_baseline: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "data/roc_curve_2330.png"
) -> None:
    """繪製 ROC 曲線（比較 Model C vs Baseline）.
    
    Args:
        model_c: Model C（使用易經特徵）。
        model_baseline: Baseline 模型（不使用易經特徵）。
        X_test_c: Model C 的測試集特徵。
        X_test_baseline: Baseline 的測試集特徵。
        y_test: 測試集標籤。
        save_path: 保存路徑。
    """
    print(f"\n[INFO] 生成 ROC 曲線圖...")
    
    # 獲取預測機率
    y_pred_proba_c = model_c.predict_proba(X_test_c)[:, 1]
    y_pred_proba_baseline = model_baseline.predict_proba(X_test_baseline)[:, 1]
    
    # 計算 ROC 曲線
    fpr_c, tpr_c, _ = roc_curve(y_test, y_pred_proba_c)
    fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)
    
    # 計算 AUC
    auc_c = roc_auc_score(y_test, y_pred_proba_c)
    auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
    
    # 繪製 ROC 曲線
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr_c, tpr_c, label=f'Model C (I-Ching) - AUC = {auc_c:.3f}', 
            linewidth=2, color='#2E86AB')
    ax.plot(fpr_baseline, tpr_baseline, label=f'Baseline (No I-Ching) - AUC = {auc_baseline:.3f}', 
            linewidth=2, color='#A23B72', linestyle='--')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='隨機猜測 (AUC = 0.500)')
    
    ax.set_xlabel('假陽性率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
    ax.set_title('ROC 曲線 - Model C vs Baseline (TSMC 2330.TW)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] ROC 曲線圖已保存至: {save_path}")
    print(f"[INFO] Model C AUC: {auc_c:.4f}")
    print(f"[INFO] Baseline AUC: {auc_baseline:.4f}")
    print(f"[INFO] AUC 提升: {auc_c - auc_baseline:.4f}")
    
    plt.close()


def test_bitcoin(
    ticker: str = "BTC-USD"
) -> None:
    """測試 Bitcoin 數據（快速檢查易經特徵在加密貨幣上的表現）.
    
    Args:
        ticker: 加密貨幣代號（預設 BTC-USD）。
    """
    print("\n" + "=" * 80)
    print(f"Bitcoin 測試: {ticker}")
    print("=" * 80)
    
    try:
        # 載入資料
        print(f"[INFO] 載入 {ticker} 市場資料...")
        loader = MarketDataLoader()
        raw_data = loader.fetch_data(tickers=[ticker], market_type="CRYPTO")
        
        if raw_data.empty:
            print(f"[WARNING] 無法獲取 {ticker} 的市場資料，跳過 Bitcoin 測試")
            return
        
        # 編碼為易經卦象
        print("[INFO] 編碼為易經卦象...")
        encoder = MarketEncoder()
        encoded_data = encoder.generate_hexagrams(raw_data)
        
        if encoded_data.empty:
            print("[WARNING] 編碼後的資料為空，跳過 Bitcoin 測試")
            return
        
        # 準備表格型資料（Model C）
        X, y = prepare_tabular_data(
            encoded_data,
            prediction_window=5,
            volatility_threshold=0.03,
            selected_iching_features=['Moving_Lines_Count', 'Energy_Delta'],
            use_iching=True
        )
        
        # 訓練模型
        model, X_test, y_test = train_model(X, y, model_name="Model C (Bitcoin)")
        
        # 計算 Recall
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
        else:
            recall = 0.0
        
        print(f"\n[INFO] Bitcoin ({ticker}) 測試結果:")
        print(f"  Recall: {recall:.4f}")
        
        # 顯示特徵重要性
        feature_importance = model.feature_importances_
        feature_names = list(X.columns)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n[INFO] 特徵重要性排序:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['Feature']:<25} {row['Importance']:.4f}")
        
        # 檢查 Moving_Lines_Count 的排名
        importance_df_reset = importance_df.reset_index(drop=True)
        moving_lines_mask = importance_df_reset['Feature'] == 'Moving_Lines_Count'
        if moving_lines_mask.any():
            moving_lines_rank = importance_df_reset[moving_lines_mask].index[0] + 1
        else:
            moving_lines_rank = len(importance_df_reset) + 1  # 如果找不到，設為最後一名
        total_features = len(importance_df_reset)
        print(f"\n[INFO] Moving_Lines_Count 排名: {moving_lines_rank}/{total_features}")
        
        if moving_lines_rank <= 2:
            print(f"[INFO] ✓ Moving_Lines_Count 在 Bitcoin 上表現優異（前 2 名）")
        elif moving_lines_rank <= 3:
            print(f"[INFO] ✓ Moving_Lines_Count 在 Bitcoin 上表現良好（前 3 名）")
        else:
            print(f"[INFO] Moving_Lines_Count 在 Bitcoin 上排名較低")
        
    except Exception as e:
        print(f"[ERROR] Bitcoin 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """主函數：生成評估圖表並進行 Bitcoin 測試."""
    print("=" * 80)
    print("Model C 評估圖表生成與 Bitcoin 測試")
    print("=" * 80)
    
    set_random_seed(42)
    
    # ===== 任務 1: TSMC 評估圖表 =====
    print("\n" + "=" * 80)
    print("任務 1: 生成 TSMC (2330.TW) 評估圖表")
    print("=" * 80)
    
    # 載入資料
    print("[INFO] 載入 TSMC (2330.TW) 市場資料...")
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
    
    # 準備 Model C 資料
    X_c, y_c = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        selected_iching_features=['Moving_Lines_Count', 'Energy_Delta'],
        use_iching=True
    )
    
    # 準備 Baseline 資料
    X_baseline, y_baseline = prepare_tabular_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
        use_iching=False
    )
    
    # 確保使用相同的索引範圍（使用較小的數據集大小）
    min_len = min(len(X_c), len(X_baseline))
    X_c = X_c.iloc[:min_len].reset_index(drop=True)
    y_c = y_c.iloc[:min_len].reset_index(drop=True)
    X_baseline = X_baseline.iloc[:min_len].reset_index(drop=True)
    y_baseline = y_baseline.iloc[:min_len].reset_index(drop=True)
    
    # 確保標籤一致
    assert (y_c.values == y_baseline.values).all(), "Model C 和 Baseline 的標籤不一致"
    
    # 使用相同的分割點確保測試集一致
    split_idx = int(min_len * 0.8)
    
    # 訓練 Model C
    model_c, X_test_c, y_test_c = train_model(
        X_c, y_c, 
        model_name="Model C (I-Ching)",
        split_idx=split_idx
    )
    
    # 訓練 Baseline（使用相同的分割點）
    model_baseline, X_test_baseline, y_test_baseline = train_model(
        X_baseline, y_baseline, 
        model_name="Baseline (No I-Ching)",
        split_idx=split_idx
    )
    
    # 確保測試集標籤一致
    assert len(y_test_c) == len(y_test_baseline), "測試集長度不一致"
    assert (y_test_c.values == y_test_baseline.values).all(), "測試集標籤不一致"
    y_test = y_test_c
    
    # 生成混淆矩陣
    plot_confusion_matrix(model_c, X_test_c, y_test, save_path="data/confusion_matrix_2330.png")
    
    # 生成 ROC 曲線
    plot_roc_curve(
        model_c, model_baseline,
        X_test_c, X_test_baseline,
        y_test,
        save_path="data/roc_curve_2330.png"
    )
    
    # ===== 任務 2: Bitcoin 測試 =====
    print("\n" + "=" * 80)
    print("任務 2: Bitcoin 測試（快速檢查）")
    print("=" * 80)
    
    test_bitcoin(ticker="BTC-USD")
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print("\n[SUCCESS] 評估圖表已生成:")
    print("  - data/confusion_matrix_2330.png")
    print("  - data/roc_curve_2330.png")


if __name__ == "__main__":
    main()
