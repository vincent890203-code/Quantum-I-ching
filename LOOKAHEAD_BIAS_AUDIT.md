# Look-Ahead Bias / Data Leakage 審計報告

本文件記錄對 `visualize_pnl.py` 的審計結果，確保無「用未來資訊預測當下」的資料洩漏。

---

## 審計範圍

1. **Normalization/Scaling**：Scaler 是否僅在訓練集上 fit？
2. **Feature Engineering**：RVOL、Returns 等是否誤用當前/未來時刻資料？
3. **Target Labeling**：特徵在 T、目標是否嚴格為 T+1（或 T+5）等未來資料？

---

## 1. Normalization / Scaling

**結論：✅ 無洩漏**

- `visualize_pnl.py` **未使用** StandardScaler、MinMaxScaler 或任何 scaler。
- 特徵直接取自 `encoded_data` 與易經特徵，未做標準化。
- **若日後加入 scaler**：必須僅在 **train** 上 `fit`，再對 train / test 做 `transform`，禁止在全集或 test 上 fit。

---

## 2. Feature Engineering

**結論：✅ 無 look-ahead**

| 特徵 | 來源 | 說明 |
|------|------|------|
| Close, Volume | `encoded_data` | 當日收盤/成交量，僅 T 時刻 |
| RVOL | `market_encoder` | Volume / Volume_MA20，Volume_MA20 為 `rolling(20).mean()`，只用 T 及以前 |
| Daily_Return | `market_encoder` | `Close.pct_change()` = (Close_t - Close_{t-1})/Close_{t-1}，已實現報酬 |
| 易經特徵 | `DataProcessor.extract_iching_features(ritual_seq)` | 僅用當期 Ritual_Sequence，不涉及未來價格 |

目標建構使用 `Close.shift(-prediction_window)` 取得**未來收盤價**，僅用於計算 forward return 並建 y，**未**當作特徵輸入。

---

## 3. Target Labeling

**結論：✅ 對齊正確**

- **目標定義**：`future_returns = (Close.shift(-5) - Close) / Close`  
  即 Return(T → T+5) = (Close_{T+5} - Close_T) / Close_T。
- **標籤**：`y = (future_returns.abs() > volatility_threshold).astype(int)`  
  表示「在 T 時點預測：未來 5 日報酬絕對值是否 > 門檻」。
- **對齊**：對樣本 i（對應日期 T），  
  - 特徵 X[i] = (Close_T, Volume_T, RVOL_T, Daily_Return_T, …)  
  - 目標 y[i] = 1 iff |Return(T → T+5)| > threshold  
  故為「用 T 的資訊預測 T→T+5 的波動」，無用未來預測當下。

---

## 4. 程式碼防護性註解（已加入）

- 模組 docstring：簡述「特徵僅用 T 及以前、目標為 T+5、未使用 scaler；若加 scaler 僅在 train fit」。
- 目標建構區：註明「嚴格無 look-ahead」「shift(-prediction_window) 僅用於建 y」。
- Train/Test 分割區：註明「時間序列分割、禁止 shuffle」「若加 scaler 僅在 X_train 上 fit」。
- 數值特徵：註明 Close/Volume/RVOL/Daily_Return 皆為 T 時刻及以前（rolling / pct_change）。

---

## 5. 總結

- **未發現 look-ahead bias 或資料洩漏**；特徵、目標與時間對齊符合「用當下及過去預測未來」。
- 若其他模組（如 `data_processor`）引入 scaler，需在該處遵守「僅在訓練集 fit」；`visualize_pnl.py` 未呼叫這些 scaler。
- 審計日期：2026-01-27。後續若改動特徵、目標或切割方式，建議重新對照本文件檢查。
