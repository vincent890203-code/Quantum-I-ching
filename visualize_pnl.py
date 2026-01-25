"""生成策略累積收益（PnL）比較圖表.

此腳本模擬 Long Straddle 策略，展示 Model C (I-Ching) vs Baseline 的財務表現。
策略邏輯：
- 如果模型預測高波動（1）：買入 Straddle，成本為 COST_PER_TRADE
- 如果模型預測低波動（0）：不做任何事
- 淨 PnL = abs(實際收益率) - COST_PER_TRADE（如果預測為 1）
"""

import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import xgboost as xgb

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

# 策略參數
COST_PER_TRADE = 0.01  # 1.0% 的 Straddle 成本（交易成本 + theta decay proxy）


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
    
    將時間序列資料轉換為表格格式，每個樣本使用 T-0 的特徵預測 T+5 的目標。
    
    Args:
        encoded_data: 包含易經卦象的編碼資料（必須有 Ritual_Sequence）。
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
    
    # 保存實際收益率（用於 PnL 計算）
    actual_returns = future_returns.copy()
    
    # 移除最後 prediction_window 行（無法計算未來收益率）
    X = X.iloc[:max_idx].copy()
    y = y.iloc[:max_idx].copy()
    actual_returns = actual_returns.iloc[:max_idx].copy()
    
    # 獲取對應的日期索引（在過濾 NaN 之前）
    dates_before_filter = encoded_data.index[:max_idx]
    
    # 移除包含 NaN 的行
    valid_mask = ~(X.isna().any(axis=1) | y.isna() | actual_returns.isna())
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    actual_returns = actual_returns[valid_mask].copy()
    
    # 保存日期索引（對應於有效數據）
    dates = dates_before_filter[valid_mask]
    
    print(f"[INFO] 資料準備完成:")
    print(f"  總樣本數: {len(X)}")
    print(f"  特徵數: {len(X.columns)}")
    print(f"  標籤分布: 高波動={y.sum()}, 低波動={(y == 0).sum()}")
    print(f"  高波動比例: {y.mean():.2%}")
    
    return X, y, actual_returns, dates


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


def simulate_strategy_pnl(
    predictions: np.ndarray,
    actual_returns: pd.Series,
    cost_per_trade: float = COST_PER_TRADE
) -> np.ndarray:
    """模擬策略的每日 PnL.
    
    策略邏輯（Long Straddle Proxy）：
    - 如果預測為 1（高波動）：買入 Straddle
        * 成本：COST_PER_TRADE
        * 利潤：abs(實際收益率)
        * 淨 PnL = abs(實際收益率) - COST_PER_TRADE
    - 如果預測為 0：不做任何事（PnL = 0）
    
    Args:
        predictions: 模型預測（0 或 1）。
        actual_returns: 實際收益率（T+5）。
        cost_per_trade: 每次交易的成本（預設 1.0%）。
    
    Returns:
        每日 PnL 陣列。
    """
    pnl = np.zeros(len(predictions))
    
    for i in range(len(predictions)):
        if predictions[i] == 1:
            # 預測高波動：買入 Straddle
            # 淨 PnL = abs(實際收益率) - 成本
            pnl[i] = abs(actual_returns.iloc[i]) - cost_per_trade
        else:
            # 預測低波動：不做任何事
            pnl[i] = 0.0
    
    return pnl


def plot_strategy_comparison(
    dates: pd.DatetimeIndex,
    pnl_iching: np.ndarray,
    pnl_baseline: np.ndarray,
    pnl_buyhold: np.ndarray,
    save_path: str = "data/pnl_comparison.png"
) -> None:
    """繪製策略累積收益比較圖.
    
    Args:
        dates: 日期索引。
        pnl_iching: I-Ching 策略的每日 PnL。
        pnl_baseline: Baseline 策略的每日 PnL。
        pnl_buyhold: Buy & Hold 策略的每日 PnL。
        save_path: 保存路徑。
    """
    print(f"\n[INFO] 生成策略累積收益比較圖...")
    
    # 計算累積收益
    cumulative_iching = np.cumsum(pnl_iching)
    cumulative_baseline = np.cumsum(pnl_baseline)
    cumulative_buyhold = np.cumsum(pnl_buyhold)
    
    # 計算總收益
    total_iching = cumulative_iching[-1]
    total_baseline = cumulative_baseline[-1]
    total_buyhold = cumulative_buyhold[-1]
    
    print(f"[INFO] 總累積收益:")
    print(f"  Quantum I-Ching Strategy: {total_iching:.4f} ({total_iching*100:.2f}%)")
    print(f"  Baseline Strategy: {total_baseline:.4f} ({total_baseline*100:.2f}%)")
    print(f"  Buy & Hold: {total_buyhold:.4f} ({total_buyhold*100:.2f}%)")
    print(f"[INFO] I-Ching 相對 Baseline 提升: {(total_iching - total_baseline)*100:.2f}%")
    
    # 繪製圖表（深色風格）
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # 繪製累積收益線（符合要求的顏色）
    # Quantum: Red/Gold (#FF6B6B / #FFD700)
    ax.plot(dates, cumulative_iching, 
            label=f'Quantum I-Ching Strategy (總收益: {total_iching*100:.2f}%)',
            linewidth=2.5, color='#FF6B6B', alpha=0.9)
    # Baseline: Blue/Grey (#4A90E2 / #95A5A6)
    ax.plot(dates, cumulative_baseline,
            label=f'Baseline Strategy (總收益: {total_baseline*100:.2f}%)',
            linewidth=2.5, color='#4A90E2', alpha=0.9)
    # Buy & Hold: Green Dashed (#2ECC71)
    ax.plot(dates, cumulative_buyhold,
            label=f'Buy & Hold (總收益: {total_buyhold*100:.2f}%)',
            linewidth=2, color='#2ECC71', linestyle='--', alpha=0.7)
    
    # 添加零線
    ax.axhline(y=0, color='white', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # 設置標題和標籤
    ax.set_title('Cumulative Return: Quantum Volatility Strategy vs Baseline', 
                 fontsize=16, fontweight='bold', pad=20, color='white')
    ax.set_xlabel('日期', fontsize=13, color='white')
    ax.set_ylabel('累積收益 (Cumulative PnL)', fontsize=13, color='white')
    
    # 設置圖例（深色背景）
    ax.legend(loc='best', fontsize=11, framealpha=0.9, facecolor='#2d2d2d', 
              edgecolor='white', labelcolor='white')
    
    # 設置深色網格
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    ax.tick_params(colors='white')
    
    # 格式化 x 軸日期
    fig.autofmt_xdate()
    
    # 添加文字說明
    improvement = total_iching - total_baseline
    if improvement > 0:
        text_color = '#2ECC71'
        text = f'Quantum 策略優於 Baseline: +{improvement*100:.2f}%'
    else:
        text_color = '#E74C3C'
        text = f'Quantum 策略劣於 Baseline: {improvement*100:.2f}%'
    
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, fontsize=11, color='white',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='#2d2d2d', alpha=0.9, edgecolor=text_color, linewidth=2))
    
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"[INFO] 策略比較圖已保存至: {save_path}")
    
    plt.close()


def main() -> None:
    """主函數：生成策略累積收益比較圖."""
    print("=" * 80)
    print("策略累積收益（PnL）比較 - Long Straddle 策略")
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
    
    # 準備 Model C 資料
    X_c, y_c, actual_returns_c, dates_c = prepare_tabular_data(
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
    
    # 確保使用相同的索引範圍（使用較小的數據集大小）
    min_len = min(len(X_c), len(X_baseline))
    X_c = X_c.iloc[:min_len].reset_index(drop=True)
    y_c = y_c.iloc[:min_len].reset_index(drop=True)
    actual_returns_c = actual_returns_c.iloc[:min_len].reset_index(drop=True)
    dates_c = dates_c[:min_len]
    X_baseline = X_baseline.iloc[:min_len].reset_index(drop=True)
    y_baseline = y_baseline.iloc[:min_len].reset_index(drop=True)
    actual_returns_baseline = actual_returns_baseline.iloc[:min_len].reset_index(drop=True)
    dates_baseline = dates_baseline[:min_len]
    
    # 確保標籤和實際收益率一致
    assert (y_c.values == y_baseline.values).all(), "Model C 和 Baseline 的標籤不一致"
    assert (actual_returns_c.values == actual_returns_baseline.values).all(), "實際收益率不一致"
    
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
    
    # 獲取測試集的實際收益率和日期索引
    actual_returns_test = actual_returns_c.iloc[split_idx:].reset_index(drop=True)
    test_dates = dates_c[split_idx:split_idx + len(y_test_c)]
    
    # 模型預測
    print("\n[INFO] 進行模型預測...")
    predictions_c = model_c.predict(X_test_c)
    predictions_baseline = model_baseline.predict(X_test_baseline)
    
    print(f"[INFO] Model C 預測統計:")
    print(f"  預測高波動次數: {predictions_c.sum()}/{len(predictions_c)} ({predictions_c.mean()*100:.1f}%)")
    print(f"[INFO] Baseline 預測統計:")
    print(f"  預測高波動次數: {predictions_baseline.sum()}/{len(predictions_baseline)} ({predictions_baseline.mean()*100:.1f}%)")
    
    # 模擬策略 PnL
    print(f"\n[INFO] 模擬策略 PnL (COST_PER_TRADE = {COST_PER_TRADE*100:.2f}%)...")
    pnl_iching = simulate_strategy_pnl(predictions_c, actual_returns_test, COST_PER_TRADE)
    pnl_baseline = simulate_strategy_pnl(predictions_baseline, actual_returns_test, COST_PER_TRADE)
    pnl_buyhold = actual_returns_test.values  # Buy & Hold 就是實際收益率
    
    # 繪製策略比較圖
    plot_strategy_comparison(
        test_dates,
        pnl_iching,
        pnl_baseline,
        pnl_buyhold,
        save_path="data/pnl_comparison.png"
    )
    
    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print("\n[SUCCESS] 策略累積收益比較圖已生成:")
    print("  - data/pnl_comparison.png")
    print(f"\n[INFO] 策略參數:")
    print(f"  - COST_PER_TRADE: {COST_PER_TRADE*100:.2f}%")
    print(f"  - 測試集樣本數: {len(y_test_c)}")


if __name__ == "__main__":
    main()
