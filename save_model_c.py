"""保存精簡版 XGBoost 模型（Model C）供 Dashboard 使用.

此腳本會訓練 Model C（精簡版 XGBoost）並保存到 `data/volatility_model.json`。
模型使用所有可用資料進行訓練，用於實時波動性預測。
"""

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from config import settings
from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
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


def train_and_save_model(
    X: pd.DataFrame,
    y: pd.Series,
    save_path: str = "data/volatility_model.json"
) -> xgb.XGBClassifier:
    """訓練精簡版 XGBoost 模型並保存.
    
    使用所有可用資料訓練 Model C（精簡版配置）。
    
    Args:
        X: 特徵 DataFrame。
        y: 標籤 Series。
        save_path: 模型保存路徑。
    
    Returns:
        訓練好的 XGBoost 模型。
    """
    print(f"\n[INFO] 訓練精簡版 XGBoost 模型（使用全部資料）...")
    print(f"  特徵: {list(X.columns)}")
    print(f"  樣本數: {len(X)}")
    
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
    
    # 訓練模型（使用全部資料，不分割）
    model.fit(X, y, verbose=False)
    
    # 強制設置特徵名稱到 booster（關鍵步驟）
    model.get_booster().feature_names = list(X.columns)
    
    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    
    print(f"[INFO] 模型已保存至: {save_path}")
    
    # 顯示模型資訊
    print(f"\n[INFO] 模型配置:")
    print(f"  特徵順序: {list(X.columns)}")
    print(f"  超參數: {params}")
    
    return model


def main() -> None:
    """主函數：訓練並保存 Model C."""
    print("=" * 80)
    print("保存精簡版 XGBoost 模型（Model C）")
    print("=" * 80)
    print("\n目標：訓練波動性預測模型供 Dashboard 使用")
    print("模型：精簡版 XGBoost (Moving_Lines_Count + Energy_Delta)")
    print("預測目標：波動性突破（|Return_5d| > 3%）\n")
    
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
    
    # 訓練並保存模型（使用全部資料）
    print("\n" + "=" * 80)
    print("訓練模型（使用全部資料）")
    print("=" * 80)
    model = train_and_save_model(X, y, save_path="data/volatility_model.json")
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print("\n[SUCCESS] 模型已成功保存至 data/volatility_model.json")
    print("  現在可以在 dashboard.py 中使用此模型進行實時預測")


if __name__ == "__main__":
    main()
