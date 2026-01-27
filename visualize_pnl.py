"""生成策略累積收益（PnL）比較圖表.

此腳本模擬 Long Straddle 策略，展示 Model C (I-Ching) vs Baseline 的財務表現。
策略邏輯：
- 如果模型預測高波動（1）：買入 Straddle，成本為 COST_PER_TRADE
- 如果模型預測低波動（0）：不做任何事
- 淨 PnL = abs(實際收益率) - COST_PER_TRADE（如果預測為 1）

Look-Ahead Bias / Data Leakage 設計（本檔已審計）：
- 特徵僅用 T 時刻及之前資訊：Close/Volume/RVOL/Daily_Return 來自 market_encoder，
  RVOL 與 Daily_Return 為 rolling/pct_change，不含未來。
- 目標為嚴格未來：y = |Return(T -> T+5)| > threshold，特徵在 T、目標為 T+5 收益。
- 未使用任何 scaler；若日後加入 StandardScaler/MinMaxScaler，必須僅在 train 上 fit，
  再對 test 做 transform，禁止在全量資料上 fit。
"""

import os
import random
from typing import List, Tuple

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
COST_PER_TRADE = 0.01  # 1.0% 的 Straddle 成本（theta decay proxy，不包含手續費與滑價）
FEE_BPS = 10  # 每筆買或賣的手續費 10 bps = 0.1%
SLIPPAGE_BPS = 5  # 每筆執行的滑價 5 bps = 0.05%（買高、賣低）
FEE_PER_SIDE = FEE_BPS / 10000  # 0.001
SLIPPAGE_PER_SIDE = SLIPPAGE_BPS / 10000  # 0.0005
# 每筆「買入並持有至出場」的總成本：買手續費 + 賣手續費 + 買滑價 + 賣滑價
COST_PER_ROUND_TRIP = 2 * FEE_PER_SIDE + 2 * SLIPPAGE_PER_SIDE  # 0.003 = 30 bps

# Walk-Forward 驗證參數（滾動視窗，嚴格避免 overfitting）
WALK_FORWARD_TRAIN_MONTHS = 12  # 每次訓練使用 12 個月
WALK_FORWARD_TEST_MONTHS = 1    # 每次測試 1 個月
WALK_FORWARD_STEP_MONTHS = 1    # 每次前滑 1 個月


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

        # 合併數值特徵和選定的易經特徵（皆為 T 時刻及以前：RVOL/Daily_Return 來自 rolling/pct_change）
        numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        X = pd.concat([
            encoded_data[numerical_cols],
            iching_df
        ], axis=1)
    else:
        # Baseline 模型：只使用數值特徵（同上，皆 point-in-time）
        print("[INFO] Baseline 模型：只使用數值特徵")
        numerical_cols = ['Close', 'Volume', 'RVOL', 'Daily_Return']
        X = encoded_data[numerical_cols].copy()
    
    # 計算目標標籤（T+5 波動性突破）— 嚴格無 look-ahead：
    # 特徵 X[i] 對應時刻 T，目標 y[i] = 1 iff |(Close[T+5]-Close[T])/Close[T]| > threshold。
    # 僅使用 shift(-prediction_window) 取得未來收盤價，絕不在特徵中代入 T+1 及以後的資訊。
    max_idx = len(encoded_data) - prediction_window
    
    # 未來收益率：Return(T -> T+5)，僅用於建構 y，不用於特徵
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
    
    # 時間序列分割（80/20），禁止 shuffle，以避開 look-ahead
    # 若日後加入 StandardScaler/MinMaxScaler：僅在 X_train 上 fit，再對 X_train/X_test transform
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


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    actual_returns: pd.Series,
    dates: pd.DatetimeIndex,
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    test_months: int = WALK_FORWARD_TEST_MONTHS,
    step_months: int = WALK_FORWARD_STEP_MONTHS,
    model_name: str = "Quantum",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Walk-Forward 驗證：滾動視窗訓練，每次訓練 12 個月、測試 1 個月、前滑 1 個月.
    
    約束：訓練時絕不使用測試期資料。
    
    Args:
        X: 特徵 DataFrame（行順序需與 dates 一致）。
        y: 標籤 Series。
        actual_returns: 實際收益率 Series。
        dates: 對應每行的日期（長度 = len(X)）。
        train_months: 訓練視窗月數。
        test_months: 測試視窗月數。
        step_months: 每步前滑月數。
        model_name: 日誌用名稱。
    
    Returns:
        (predictions_oos, actual_returns_oos, dates_oos) 串接後的 OOS 預測與對應實際收益、日期。
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

    preds_list: List[np.ndarray] = []
    actual_list: List[np.ndarray] = []
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
        actual_test = actual_returns.iloc[test_mask].copy()
        dates_test = dti[test_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            i += step_months
            continue

        model = _fit_xgb_on_subset(X_train, y_train, feature_names)
        preds_fold = model.predict(X_test)

        preds_list.append(preds_fold)
        actual_list.append(actual_test.values)
        dates_list.append(dates_test)

        fold += 1
        if fold <= 3 or fold % 12 == 0:
            print(f"[Walk-Forward] {model_name} fold {fold}: "
                  f"train {train_periods[0]}~{train_periods[-1]}, "
                  f"test {test_periods[0]}~{test_periods[-1]}, "
                  f"n_test={len(X_test)}")
        i += step_months

    if not preds_list:
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    predictions_oos = np.concatenate(preds_list)
    actual_oos = np.concatenate(actual_list)
    dates_oos = pd.DatetimeIndex(np.concatenate([d.values for d in dates_list]))
    print(f"[Walk-Forward] {model_name} 共 {fold} 個 fold，OOS 樣本數 = {len(predictions_oos)}")
    return predictions_oos, actual_oos, dates_oos


def simulate_strategy_pnl(
    predictions: np.ndarray,
    actual_returns: pd.Series,
    cost_per_trade: float = COST_PER_TRADE
) -> np.ndarray:
    """模擬策略的每日 PnL（僅含 Straddle 成本，不含手續費與滑價）.
    
    策略邏輯（Long Straddle Proxy）：
    - 如果預測為 1（高波動）：買入 Straddle
        * 成本：COST_PER_TRADE
        * 利潤：abs(實際收益率)
        * Gross PnL = abs(實際收益率) - COST_PER_TRADE
    - 如果預測為 0：不做任何事（PnL = 0）
    
    Args:
        predictions: 模型預測（0 或 1）。
        actual_returns: 實際收益率（T+5）。
        cost_per_trade: 每次交易的 Straddle 成本（預設 1.0%）。
    
    Returns:
        每日 Gross PnL 陣列。
    """
    pnl = np.zeros(len(predictions))
    for i in range(len(predictions)):
        if predictions[i] == 1:
            pnl[i] = abs(actual_returns.iloc[i]) - cost_per_trade
        else:
            pnl[i] = 0.0
    return pnl


def simulate_strategy_pnl_gross_and_net(
    predictions: np.ndarray,
    actual_returns: pd.Series,
    cost_per_trade: float = COST_PER_TRADE,
    fee_per_side: float = FEE_PER_SIDE,
    slippage_per_side: float = SLIPPAGE_PER_SIDE,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """模擬策略的每日 Gross 與 Net PnL，並回傳交易次數.
    
    每筆交易（預測=1 時）：買入 + 持有至 T+5 出場。
    - Transaction fee: fee_per_side 於買、fee_per_side 於賣（預設各 0.1% = 10 bps）
    - Slippage: slippage_per_side 於買（買高）、slippage_per_side 於賣（賣低）（預設各 0.05% = 5 bps）
    
    Args:
        predictions: 模型預測（0 或 1）。
        actual_returns: 實際收益率（T+5）。
        cost_per_trade: Straddle 成本（theta decay proxy）。
        fee_per_side: 單邊手續費（買或賣）。
        slippage_per_side: 單邊滑價（買貴、賣低）。
    
    Returns:
        (gross_pnl, net_pnl, num_trades) 其中 gross_pnl、net_pnl 為每日陣列，num_trades 為預測=1 的次數。
    """
    n = len(predictions)
    cost_round_trip = 2 * fee_per_side + 2 * slippage_per_side  # 每筆 round-trip 的手續費+滑價
    num_trades = int(predictions.sum())
    
    gross_pnl = np.zeros(n)
    net_pnl = np.zeros(n)
    for i in range(n):
        if predictions[i] == 1:
            gross_pnl[i] = abs(actual_returns.iloc[i]) - cost_per_trade
            net_pnl[i] = gross_pnl[i] - cost_round_trip
        else:
            gross_pnl[i] = 0.0
            net_pnl[i] = 0.0
    
    return gross_pnl, net_pnl, num_trades


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
    
    # 繪製圖表（白色背景風格）
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
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
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # 設置標題和標籤
    ax.set_title('Cumulative Return: Quantum Volatility Strategy vs Baseline', 
                 fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('日期', fontsize=13, color='#333333')
    ax.set_ylabel('累積收益 (Cumulative PnL)', fontsize=13, color='#333333')
    
    # 設置圖例（白色背景）
    ax.legend(loc='best', fontsize=11, framealpha=0.9, facecolor='white', 
              edgecolor='#cccccc', labelcolor='#333333')
    
    # 設置網格
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.tick_params(colors='#333333')
    
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
            transform=ax.transAxes, fontsize=11, color='#333333',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor=text_color, linewidth=2))
    
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] 策略比較圖已保存至: {save_path}")
    
    plt.close()


def plot_gross_net_baseline(
    dates: pd.DatetimeIndex,
    gross_pnl: np.ndarray,
    net_pnl: np.ndarray,
    baseline_pnl: np.ndarray,
    num_trades_quantum: int,
    num_trades_baseline: int,
    save_path: str = "data/pnl_comparison_gross_net.png",
) -> None:
    """繪製 Gross Return vs Net Return vs Baseline 比較圖（含交易成本與滑價）.
    
    Args:
        dates: 日期索引。
        gross_pnl: 策略每日 Gross PnL（不含手續費與滑價）。
        net_pnl: 策略每日 Net PnL（含手續費 0.1% 買+賣、滑價 0.05% 買+賣）。
        baseline_pnl: Baseline 策略的每日 Net PnL（同成本假設）。
        num_trades_quantum: 策略交易次數（預測=1 的次數）。
        num_trades_baseline: Baseline 交易次數。
        save_path: 保存路徑。
    """
    print(f"\n[INFO] 生成 Gross vs Net vs Baseline 比較圖...")
    cum_gross = np.cumsum(gross_pnl)
    cum_net = np.cumsum(net_pnl)
    cum_baseline = np.cumsum(baseline_pnl)
    
    total_gross = cum_gross[-1]
    total_net = cum_net[-1]
    total_baseline = cum_baseline[-1]
    
    print(f"[INFO] 總累積收益 (含成本假設):")
    print(f"  Quantum Gross (不含手續費/滑價): {total_gross:.4f} ({total_gross*100:.2f}%)")
    print(f"  Quantum Net (含 10bps 費+5bps 滑價): {total_net:.4f} ({total_net*100:.2f}%)")
    print(f"  Baseline Net: {total_baseline:.4f} ({total_baseline*100:.2f}%)")
    print(f"[INFO] 交易次數 — Quantum: {num_trades_quantum}, Baseline: {num_trades_baseline}")
    if num_trades_quantum > 0:
        cost_drag = total_gross - total_net
        print(f"[INFO] 手續費+滑價造成的拖累: {cost_drag:.4f} ({cost_drag*100:.2f}%)")

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')

    ax.plot(dates, cum_gross,
            label=f'Quantum Gross (總收益: {total_gross*100:.2f}%)',
            linewidth=2.5, color='#FF6B6B', alpha=0.9)
    ax.plot(dates, cum_net,
            label=f'Quantum Net (總收益: {total_net*100:.2f}%)',
            linewidth=2.5, color='#E74C3C', linestyle='-', alpha=0.95)
    ax.plot(dates, cum_baseline,
            label=f'Baseline Net (總收益: {total_baseline*100:.2f}%)',
            linewidth=2, color='#4A90E2', linestyle='--', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.set_title('Cumulative Return: Gross vs Net (after fees & slippage) vs Baseline',
                 fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('日期', fontsize=13, color='#333333')
    ax.set_ylabel('累積收益 (Cumulative PnL)', fontsize=13, color='#333333')
    ax.legend(loc='best', fontsize=11, framealpha=0.9, facecolor='white',
              edgecolor='#cccccc', labelcolor='#333333')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.tick_params(colors='#333333')
    fig.autofmt_xdate()

    text_parts = [f'Quantum 交易次數: {num_trades_quantum}', f'Baseline 交易次數: {num_trades_baseline}']
    text = '\n'.join(text_parts)
    ax.text(0.02, 0.98, text,
            transform=ax.transAxes, fontsize=10, color='#333333',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.9, edgecolor='#4A90E2', linewidth=1.5))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Gross vs Net vs Baseline 圖已保存至: {save_path}")
    plt.close()


def plot_walk_forward_vs_single_split(
    dates_single: pd.DatetimeIndex,
    pnl_single: np.ndarray,
    dates_wf: pd.DatetimeIndex,
    pnl_wf: np.ndarray,
    label_single: str = "Single-Split (last 20% OOS)",
    label_wf: str = "Walk-Forward (rolling OOS)",
    save_path: str = "data/pnl_walk_forward_vs_single.png",
) -> None:
    """繪製 Walk-Forward OOS 與 Single-Split OOS 累積收益比較圖.
    
    若 Walk-Forward 明顯低於 Single-Split，表示模型有 overfitting。
    
    Args:
        dates_single: Single-Split 的日期索引。
        pnl_single: Single-Split 的每日 PnL（最後 20% 測試集）。
        dates_wf: Walk-Forward 的日期索引（串接各 fold 的 OOS 日期）。
        pnl_wf: Walk-Forward 的每日 PnL（串接 OOS 預測）。
        label_single: 圖例中 Single-Split 的名稱。
        label_wf: 圖例中 Walk-Forward 的名稱。
        save_path: 保存路徑。
    """
    cum_single = np.cumsum(pnl_single)
    cum_wf = np.cumsum(pnl_wf)
    total_single = cum_single[-1] if len(cum_single) > 0 else 0.0
    total_wf = cum_wf[-1] if len(cum_wf) > 0 else 0.0
    drop = total_single - total_wf

    print(f"\n[INFO] Walk-Forward vs Single-Split:")
    print(f"  Single-Split 累積收益: {total_single:.4f} ({total_single*100:.2f}%)")
    print(f"  Walk-Forward 累積收益: {total_wf:.4f} ({total_wf*100:.2f}%)")
    print(f"  落差: {drop:.4f} ({drop*100:.2f}%)")
    if drop > 0.05 and total_single != 0:
        print(f"  [提示] Walk-Forward 明顯低於 Single-Split，可能存在 overfitting。")

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    if len(dates_single) > 0 and len(pnl_single) > 0:
        ax.plot(dates_single, cum_single, label=f'{label_single} (總收益: {total_single*100:.2f}%)',
                linewidth=2.5, color='#4A90E2', alpha=0.9)
    if len(dates_wf) > 0 and len(pnl_wf) > 0:
        ax.plot(dates_wf, cum_wf, label=f'{label_wf} (總收益: {total_wf*100:.2f}%)',
                linewidth=2.5, color='#E74C3C', linestyle='-', alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.set_title('Walk-Forward OOS vs Single-Split OOS (若 WF 明顯偏低，可能 overfitting)',
                 fontsize=16, fontweight='bold', pad=20, color='#333333')
    ax.set_xlabel('日期', fontsize=13, color='#333333')
    ax.set_ylabel('累積收益 (Cumulative PnL)', fontsize=13, color='#333333')
    ax.legend(loc='best', fontsize=11, framealpha=0.9, facecolor='white', edgecolor='#cccccc', labelcolor='#333333')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.tick_params(colors='#333333')
    fig.autofmt_xdate()
    text = f'落差 (Single−WF): {drop*100:.2f}%'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10, color='#333333',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333333', linewidth=1))
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Walk-Forward vs Single-Split 圖已保存至: {save_path}")
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
    
    # 使用相同的分割點確保測試集一致（Single-Split）
    split_idx = int(min_len * 0.8)

    # ----- Walk-Forward 驗證（滾動視窗，嚴格 OOS）-----
    print("\n[INFO] Walk-Forward 驗證（train 12 月 / test 1 月 / step 1 月）...")
    try:
        preds_wf_c, actual_wf_c, dates_wf_c = run_walk_forward(
            X_c, y_c, actual_returns_c, dates_c,
            train_months=WALK_FORWARD_TRAIN_MONTHS,
            test_months=WALK_FORWARD_TEST_MONTHS,
            step_months=WALK_FORWARD_STEP_MONTHS,
            model_name="Quantum",
        )
        preds_wf_baseline, actual_wf_baseline, dates_wf_baseline = run_walk_forward(
            X_baseline, y_baseline, actual_returns_baseline, dates_baseline,
            train_months=WALK_FORWARD_TRAIN_MONTHS,
            test_months=WALK_FORWARD_TEST_MONTHS,
            step_months=WALK_FORWARD_STEP_MONTHS,
            model_name="Baseline",
        )
        actual_wf_series_c = pd.Series(actual_wf_c)
        actual_wf_series_b = pd.Series(actual_wf_baseline)
        _, pnl_wf_net_c, _ = simulate_strategy_pnl_gross_and_net(
            preds_wf_c, actual_wf_series_c,
            cost_per_trade=COST_PER_TRADE,
            fee_per_side=FEE_PER_SIDE,
            slippage_per_side=SLIPPAGE_PER_SIDE,
        )
        _, pnl_wf_net_b, _ = simulate_strategy_pnl_gross_and_net(
            preds_wf_baseline, actual_wf_series_b,
            cost_per_trade=COST_PER_TRADE,
            fee_per_side=FEE_PER_SIDE,
            slippage_per_side=SLIPPAGE_PER_SIDE,
        )
    except ValueError as e:
        print(f"[WARNING] Walk-Forward 跳過（資料不足）: {e}")
        preds_wf_c = np.array([])
        dates_wf_c = pd.DatetimeIndex([])
        pnl_wf_net_c = np.array([])

    # 訓練 Model C（Single-Split）
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

    # 模擬策略 PnL（Gross / Net，含手續費與滑價）
    print(f"\n[INFO] 模擬策略 PnL (COST_PER_TRADE = {COST_PER_TRADE*100:.2f}%, "
          f"手續費 = {FEE_BPS} bps/邊, 滑價 = {SLIPPAGE_BPS} bps/邊)...")
    gross_iching, net_iching, num_trades_iching = simulate_strategy_pnl_gross_and_net(
        predictions_c, actual_returns_test,
        cost_per_trade=COST_PER_TRADE,
        fee_per_side=FEE_PER_SIDE,
        slippage_per_side=SLIPPAGE_PER_SIDE,
    )
    _, net_baseline, num_trades_baseline = simulate_strategy_pnl_gross_and_net(
        predictions_baseline, actual_returns_test,
        cost_per_trade=COST_PER_TRADE,
        fee_per_side=FEE_PER_SIDE,
        slippage_per_side=SLIPPAGE_PER_SIDE,
    )

    # 總交易次數（若過高，手續費可能侵蝕策略）
    print(f"\n[INFO] 總交易次數:")
    print(f"  Quantum I-Ching: {num_trades_iching} 筆")
    print(f"  Baseline: {num_trades_baseline} 筆")
    if num_trades_iching > 0 or num_trades_baseline > 0:
        n_bars = len(actual_returns_test)
        print(f"  測試區間天數: {n_bars}")
        print(f"  若交易頻率過高，手續費與滑價可能明顯侵蝕策略收益。")

    # 新圖：Gross vs Net vs Baseline
    plot_gross_net_baseline(
        test_dates,
        gross_iching,
        net_iching,
        net_baseline,
        num_trades_iching,
        num_trades_baseline,
        save_path="data/pnl_comparison_gross_net.png",
    )

    # Walk-Forward vs Single-Split 比較圖（若 WF 明顯偏低則可能 overfitting）
    if len(preds_wf_c) > 0 and len(dates_wf_c) > 0 and len(pnl_wf_net_c) > 0:
        plot_walk_forward_vs_single_split(
            test_dates,
            net_iching,
            dates_wf_c,
            pnl_wf_net_c,
            label_single="Single-Split (last 20% OOS)",
            label_wf="Walk-Forward (rolling OOS)",
            save_path="data/pnl_walk_forward_vs_single.png",
        )

    # 原有比較圖（Quantum / Baseline / Buy & Hold，沿用原邏輯）
    pnl_iching_legacy = simulate_strategy_pnl(predictions_c, actual_returns_test, COST_PER_TRADE)
    pnl_baseline_legacy = simulate_strategy_pnl(predictions_baseline, actual_returns_test, COST_PER_TRADE)
    pnl_buyhold = actual_returns_test.values
    plot_strategy_comparison(
        test_dates,
        pnl_iching_legacy,
        pnl_baseline_legacy,
        pnl_buyhold,
        save_path="data/pnl_comparison.png",
    )

    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)
    print("\n[SUCCESS] 策略累積收益比較圖已生成:")
    print("  - data/pnl_comparison.png (Quantum vs Baseline vs Buy & Hold)")
    print("  - data/pnl_comparison_gross_net.png (Gross vs Net vs Baseline，含手續費與滑價)")
    if len(preds_wf_c) > 0:
        print("  - data/pnl_walk_forward_vs_single.png (Walk-Forward OOS vs Single-Split OOS)")
    print(f"\n[INFO] 策略參數:")
    print(f"  - COST_PER_TRADE (Straddle): {COST_PER_TRADE*100:.2f}%")
    print(f"  - 手續費: {FEE_BPS} bps 每邊 (買/賣各 {FEE_BPS} bps)")
    print(f"  - 滑價: {SLIPPAGE_BPS} bps 每邊 (買高/賣低各 {SLIPPAGE_BPS} bps)")
    print(f"  - 每筆 round-trip 成本: {COST_PER_ROUND_TRIP*10000:.0f} bps")
    print(f"  - 測試集樣本數: {len(y_test_c)}")


if __name__ == "__main__":
    main()
