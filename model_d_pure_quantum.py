"""Model D - Pure Quantum 生產模型訓練與推論腳本.

目標：
- 最終確立「Pure Quantum」特徵組合作為生產用 Model D：
  features = ['Moving_Lines_Count', 'Energy_Delta', 'Daily_Return']

功能：
1. 從完整歷史資料（至最新日期）訓練 XGBoost 模型（僅用上述三個特徵）
2. 將訓練完成的模型儲存為 `model_d_pure_quantum.json`
3. 提供推論函數 `predict_next_day(current_iching_data, current_return)`：
   - 輸入：當日易經儀式序列 (例如 "987896") 與當日日報酬率
   - 輸出：交易訊號 0/1
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import MarketDataLoader
from data_processor import DataProcessor
from market_encoder import MarketEncoder


# 模型檔名（JSON 為 XGBoost 原生格式，方便跨平台載入）
MODEL_D_PATH = "model_d_pure_quantum.json"


def set_random_seed(seed: int = 42) -> None:
    """設置隨機種子，確保實驗可重現."""
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# 資料準備：Pure Quantum 特徵（全資料，用於最終訓練）
# =============================================================================

def prepare_pure_quantum_training_data(
    encoded_data: pd.DataFrame,
    prediction_window: int = 5,
    volatility_threshold: float = 0.03,
) -> Tuple[pd.DataFrame, pd.Series]:
    """從完整編碼資料中建立 Model D 的訓練資料.

    特徵：
        - Daily_Return
        - Moving_Lines_Count
        - Energy_Delta

    標籤：
        y = 1  若 |Return(T -> T+prediction_window)| > volatility_threshold
        y = 0  否則

    僅回傳 X, y（不做 train/test split，因為這是最終生產模型，使用全部歷史資料訓練）。
    """
    print("\n[INFO] 準備 Model D 訓練資料 (Pure Quantum Features)...")

    required_cols = ["Close", "Daily_Return", "Ritual_Sequence"]
    missing = [c for c in required_cols if c not in encoded_data.columns]
    if missing:
        raise ValueError(f"編碼資料缺少必要欄位: {missing}")

    processor = DataProcessor()

    # 1) 由 Ritual_Sequence 提取完整易經特徵
    all_iching_features = [
        "Yang_Count_Main",
        "Yang_Count_Future",
        "Moving_Lines_Count",
        "Energy_Delta",
        "Conflict_Score",
    ]
    iching_features_list: list[list[float]] = []

    print("[INFO] 從 Ritual_Sequence 提取易經特徵 (用於 Model D)...")
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

    # 2) 僅保留 Moving_Lines_Count & Energy_Delta（Pure Quantum）
    iching_pure = iching_df_full[["Moving_Lines_Count", "Energy_Delta"]].copy()

    # 3) 構建特徵矩陣 X：Daily_Return + 兩個 I-Ching 特徵
    X = pd.concat(
        [
            encoded_data[["Daily_Return"]].copy(),
            iching_pure,
        ],
        axis=1,
    )

    # 4) 構建標籤 y：T->T+prediction_window 的絕對報酬是否大於閾值
    max_idx = len(encoded_data) - prediction_window
    future_prices = encoded_data["Close"].shift(-prediction_window)
    current_prices = encoded_data["Close"]
    future_returns = (future_prices - current_prices) / current_prices

    y = (future_returns.abs() > volatility_threshold).astype(int)

    # 截掉最後 prediction_window 筆（沒有未來價格）
    X = X.iloc[:max_idx].copy()
    y = y.iloc[:max_idx].copy()

    # 移除 NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()

    print("[INFO] Model D 訓練資料準備完成:")
    print(f"  特徵欄位: {list(X.columns)}")
    print(f"  總樣本數: {len(X)}")
    print(f"  標籤分布: 高波動={int(y.sum())}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")

    return X, y


# =============================================================================
# 模型建立 / 儲存 / 載入
# =============================================================================

def build_xgb_classifier(feature_names: List[str]) -> xgb.XGBClassifier:
    """建立 XGBoost 分類器（超參數與 Pure Quantum / Light Model 一致）."""
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


def train_and_save_model_d() -> None:
    """訓練 Model D（Pure Quantum）並儲存模型檔."""
    print("=" * 80)
    print("訓練 Model D - Pure Quantum (全歷史資料)")
    print("=" * 80)

    set_random_seed(42)

    # 1. 載入完整市場資料並編碼
    print("\n[INFO] 載入 TSMC (2330.TW) 市場資料（全期間）...")
    loader = MarketDataLoader()
    raw_data = loader.fetch_data(tickers=["2330.TW"], market_type="TW")

    if raw_data.empty:
        raise RuntimeError("無法取得市場資料，請檢查 data_loader 與網路環境。")

    print("[INFO] 編碼為易經卦象...")
    encoder = MarketEncoder()
    encoded_data = encoder.generate_hexagrams(raw_data)

    if encoded_data.empty:
        raise RuntimeError("編碼後資料為空，無法訓練 Model D。")

    # 2. 準備 Pure Quantum 特徵與標籤
    X, y = prepare_pure_quantum_training_data(
        encoded_data,
        prediction_window=5,
        volatility_threshold=0.03,
    )

    feature_names = list(X.columns)

    # 3. 建立並訓練 XGBoost 模型（使用全部樣本）
    print("\n[INFO] 開始訓練 Model D (使用全部樣本)...")
    model = build_xgb_classifier(feature_names)
    model.fit(X, y, verbose=False)

    # 確保 booster 內部紀錄特徵名稱
    try:
        model.get_booster().feature_names = feature_names
    except Exception:
        # 某些版本可能無此屬性，忽略即可
        pass

    # 4. 儲存模型
    model.save_model(MODEL_D_PATH)
    print(f"\n[SUCCESS] Model D 已訓練完成並儲存至: {MODEL_D_PATH}")
    print(f"  使用特徵: {feature_names}")
    print(f"  總訓練樣本數: {len(X)}")


def load_model_d() -> xgb.XGBClassifier:
    """載入已訓練好的 Model D（Pure Quantum）."""
    if not os.path.exists(MODEL_D_PATH):
        raise FileNotFoundError(
            f"找不到 {MODEL_D_PATH}。\n"
            "請先執行 train_and_save_model_d() 或直接在 CLI 執行：\n"
            "  python model_d_pure_quantum.py"
        )
    model = xgb.XGBClassifier()
    model.load_model(MODEL_D_PATH)
    return model


# =============================================================================
# 推論函數：predict_next_day
# =============================================================================

def _extract_pure_features_from_ritual(
    ritual_sequence: str,
    current_return: float,
) -> pd.DataFrame:
    """從當日 Ritual Sequence + 當日日報酬率組出 Model D 所需特徵.

    Args:
        ritual_sequence: 長度為 6 的儀式數字序列（例如 "987896"）
        current_return : 當日日報酬率 (float)，例如 0.0123 代表 +1.23%

    Returns:
        單筆樣本的 DataFrame，欄位順序為：
            ['Daily_Return', 'Moving_Lines_Count', 'Energy_Delta']
    """
    processor = DataProcessor()

    if ritual_sequence is None or len(str(ritual_sequence)) != 6:
        # 若輸入異常，給予保守的零特徵
        print(
            "[WARNING] ritual_sequence 無效，將使用零向量作為易經特徵 "
            "(Moving_Lines_Count=0, Energy_Delta=0)。"
        )
        moving_lines = 0.0
        energy_delta = 0.0
    else:
        feats = processor.extract_iching_features(str(ritual_sequence))
        # feats 順序：Yang_Count_Main, Yang_Count_Future, Moving_Lines_Count, Energy_Delta, Conflict_Score
        moving_lines = float(feats[2])
        energy_delta = float(feats[3])

    data = {
        "Daily_Return": [float(current_return)],
        "Moving_Lines_Count": [moving_lines],
        "Energy_Delta": [energy_delta],
    }
    X_new = pd.DataFrame(data)
    return X_new


def predict_next_day(
    current_iching_data: str,
    current_return: float,
    threshold: float = 0.5,
) -> int:
    """使用 Model D 對「下一個 T->T+5 視窗」產生交易訊號.

    Args:
        current_iching_data: 當日 Ritual Sequence 字串（例如 "987896"）
        current_return:     當日日報酬率 (float)，例如 0.005 代表 +0.5%
        threshold:          將機率轉成 0/1 的閾值，預設 0.5

    Returns:
        交易訊號 (0 或 1)：
            - 1: 代表預期未來 |Return(T->T+5)| 可能高於閾值，建議進行「Long Volatility」交易
            - 0: 代表不交易（或保持中立）
    """
    # 載入模型
    model = load_model_d()

    # 準備特徵（與訓練時完全對齊）
    X_new = _extract_pure_features_from_ritual(
        ritual_sequence=current_iching_data,
        current_return=current_return,
    )

    # 模型輸出機率，取正類機率（y=1）
    proba = model.predict_proba(X_new)[:, 1][0]
    signal = int(proba >= threshold)

    print(
        f"[PREDICT] p(y=1 | features) = {proba:.4f}, "
        f"threshold = {threshold:.2f} -> signal = {signal}"
    )
    return signal


if __name__ == "__main__":
    # 直接執行此檔：訓練並儲存 Model D
    train_and_save_model_d()

