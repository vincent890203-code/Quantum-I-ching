# Quantum I-Ching 專案開發日誌

本檔案記錄專案開發過程中的每一步設計決策與實作步驟。

---

## 2024-12-XX | 專案初始化

### 步驟 1: 建立專案配置模組 (`config.py`)

**日期**: 2024-12-XX  
**狀態**: ✅ 完成

#### 設計目標
建立全專案共用的配置中心，定義全局常數、股票代號、日期範圍，以及易經六十四卦的基礎資料結構。

#### 實作細節

1. **模組結構**
   - 使用 Python `dataclasses` 模組
   - 遵循 Google Python Style Guide
   - 完整的型別提示與文件字串

2. **Settings 類別** (`@dataclass(frozen=True)`)
   - `START_DATE`: `str`，預設 `"2020-01-01"`
   - `END_DATE`: `str`，使用 `default_factory` 動態取得今日日期
   - `TARGET_TICKERS`: `list[str]`，包含 6 檔科技股：`["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMD"]`
   - `YIN_YANG_THRESHOLD`: `float`，預設 `0.0`（用於判斷價格變動的陰陽屬性）

3. **HEXAGRAM_MAP 常數字典**
   - **鍵值**: 二進制字串（`"111111"` 等），其中 `1` 代表陽爻，`0` 代表陰爻
   - **值**: `HexagramDict` 型別的字典，包含：
     - `id`: `int`（1-64）
     - `name`: `str`（英文/拼音名稱，如 "Qian"）
     - `nature`: `str`（繁體中文卦名，如 "乾"）
   - **已實作**: 前 8 卦（八卦重複形成的六爻卦）
     - 乾 (111111), 坤 (000000), 震 (001001), 巽 (110110)
     - 坎 (010010), 離 (101101), 艮 (100100), 兌 (011011)
   - **待完成**: 其餘 56 卦

4. **全局實例**
   - 建立 `settings = Settings()` 供全專案使用

#### 技術決策

- **為什麼使用 frozen dataclass？**
  - 確保配置不可變，符合函數式程式設計原則
  - 防止執行時期意外修改全局設定

- **為什麼使用 `default_factory`？**
  - `END_DATE` 需要動態取得當日日期，使用 `default_factory` 確保每次實例化時都會重新計算
  - 避免在模組載入時就固定日期值

- **為什麼先實作前 8 卦？**
  - 節省 token 使用量
  - 建立資料結構範本，後續可依相同模式填入剩餘 56 卦

#### 檔案結構
```
config.py
├── HexagramDict (TypedDict)
├── Settings (frozen dataclass)
├── HEXAGRAM_MAP (dict constant)
└── settings (Settings instance)
```

---

## 2026-01-20 | Phase 1 - 資料獲取

### 步驟 2: 建立資料載入模組 (`data_loader.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
實作市場資料載入功能，從 Yahoo Finance 獲取歷史股票價格資料，為後續的易經卦象生成提供資料基礎。

#### 實作細節

1. **MarketDataLoader 類別**
   - 使用 `yfinance` 作為資料來源
   - 具備完整的日誌記錄功能
   - 錯誤處理機制：不會因單一股票失敗而導致程式崩潰

2. **核心方法 `fetch_data`**
   - **參數**: `tickers: Optional[list[str]] = None`
     - 如果為 None，自動使用 `config.settings.TARGET_TICKERS`
   - **資料結構**:
     - 多檔股票：返回 MultiIndex DataFrame（columns: [Ticker, OHLCV]）
     - 單檔股票：返回標準 DataFrame（columns: OHLCV）
     - 可輕鬆存取每個股票的 `Close`, `Open`, `High`, `Low` 等欄位
   - **日期範圍**: 使用 `settings.START_DATE` 和 `settings.END_DATE`

3. **錯誤處理**
   - 使用 `try-except` 包裹整個下載過程
   - 所有錯誤都記錄到日誌，不會拋出異常
   - 下載失敗時返回空 DataFrame，避免程式崩潰
   - 檢查空資料並記錄警告

4. **日誌系統**
   - 使用標準 `logging` 模組
   - 自動設定 logger 和 handler（如果尚未設定）
   - 記錄下載開始、成功、警告和錯誤資訊

#### 技術決策

- **為什麼不使用 `group_by='ticker'`？**
  - `group_by='ticker'` 會返回字典而非 DataFrame，不符合需求
  - 預設行為會自動返回 MultiIndex DataFrame（多檔股票時）
  - 第一層是 ticker，第二層是 OHLCV，例如：`data['AAPL']['Close']` 可輕鬆取得蘋果的收盤價

- **為什麼使用 centralized configuration？**
  - 透過 `config.settings` 統一管理股票代號和日期範圍
  - 提高程式碼可維護性，修改配置只需在一個地方
  - 符合單一職責原則

- **為什麼不拋出異常？**
  - 資料獲取是外部的、不可靠的操作
  - 單一股票失敗不應影響其他股票的下載
  - 返回空 DataFrame 讓調用者可以檢查並決定後續處理

#### 資料結構範例

多檔股票（MultiIndex）:
```
                AAPL           NVDA           ...
          Open  High  Low  Close  Open  High  Low  Close
Date
2020-01-01  ...   ...  ...   ...   ...   ...  ...   ...
```

單檔股票:
```
          Open  High  Low  Close  Volume
Date
2020-01-01  ...   ...  ...   ...    ...
```

#### 檔案結構
```
data_loader.py
└── MarketDataLoader
    ├── __init__(self) -> None
    └── fetch_data(self, tickers: Optional[list[str]]) -> pd.DataFrame
```

---

## 2026-01-20 | Phase 1 - 特徵工程（Whale Logic）

### 步驟 3: 建立市場資料編碼模組 (`market_encoder.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
實作「Whale Volume Weighting」邏輯，將股票市場資料轉換為易經四象表示（6, 7, 8, 9），並生成六爻卦象。

#### 實作細節

1. **MarketEncoder 類別**
   - 使用相對成交量（RVOL）和價格變動率判斷市場能量
   - 支援單檔和多檔股票的資料處理
   - 完整的向量化運算提升效能

2. **核心方法**

   - **`_calculate_technical_indicators`**:
     - 計算 `Daily_Return`（日報酬率）
     - 計算 `Volume_MA20`（20日成交量移動平均）
     - 計算 `RVOL`（相對成交量 = Volume / Volume_MA20）
     - 處理除以零的情況（設為 NaN）

   - **`_get_ritual_number`**:
     - 根據報酬率和 RVOL 判斷儀式數字
     - 使用 `np.select` 進行向量化運算（在 `generate_hexagrams` 中）

   - **`generate_hexagrams`**:
     - 處理 MultiIndex 和標準 DataFrame
     - 自動丟棄 Volume_MA20 為 NaN 的列（前20天）
     - 使用滾動窗口（6天）生成卦象

3. **四象編碼邏輯（四象）**

   - **9 (老陽 / 變動之陽)**: `Return > Threshold` AND `RVOL > 2.0`
     - 上漲且成交量異常放大（Whale 級買盤）
   
   - **6 (老陰 / 變動之陰)**: `Return <= Threshold` AND `RVOL > 2.0`
     - 下跌且成交量異常放大（Whale 級賣盤）
   
   - **7 (少陽 / 靜止之陽)**: `Return > Threshold` AND `RVOL <= 2.0`
     - 上漲但成交量正常（普通買盤）
   
   - **8 (少陰 / 靜止之陰)**: `Return <= Threshold` AND `RVOL <= 2.0`
     - 下跌但成交量正常（普通賣盤或靜止狀態）

4. **輸出欄位**

   - `Ritual_Num`: 儀式數字（6, 7, 8, 9）
   - `Ritual_Sequence`: 連續6天的儀式數字序列（字串，例如 "987896"）
     - 順序：最舊的在底部（第1位），最新的在頂部（第6位）
   - `Hexagram_Binary`: 二進制字串（用於查詢 `HEXAGRAM_MAP`）
     - 9 和 7 -> "1"（陽爻）
     - 6 和 8 -> "0"（陰爻）
     - 例如："987896" -> "101010"

#### 技術決策

- **為什麼使用 RVOL（相對成交量）？**
  - 成交量絕對值會因股票而異，使用相對值能統一判斷標準
  - RVOL > 2.0 表示成交量是20日均量的2倍以上，代表異常大量（Whale）
  - 能捕捉市場的「能量」變化，而不只是價格方向

- **為什麼使用四象而非簡單的陰陽？**
  - 四象（老陽、少陽、老陰、少陰）能同時表達方向（陰陽）和能量（變動/靜止）
  - 老陽/老陰代表「變動」，少陽/少陰代表「靜止」
  - 更符合易經的哲學體系，也更能反映市場實際狀態

- **為什麼使用 20 日移動平均？**
  - 20 個交易日約等於一個月，是技術分析中常用的週期
  - 足以平滑短期波動，又能反映中期趨勢

- **為什麼需要至少 26 天資料？**
  - 20 天用於計算 Volume_MA20
  - 6 天用於滾動窗口生成卦象
  - 因此需要至少 26 天才能生成完整卦象

- **向量化運算的考量**
  - 使用 `np.select` 取代逐行 `apply`，大幅提升效能
  - RVOL 計算使用向量化除法，避免迴圈
  - 滾動窗口的卦象生成仍需要使用迴圈（因為涉及字串拼接）

#### 資料處理流程

```
原始資料 (OHLCV)
    ↓
計算技術指標
  - Daily_Return (pct_change)
  - Volume_MA20 (rolling mean)
  - RVOL (Volume / Volume_MA20)
    ↓
丟棄前20天（Volume_MA20 為 NaN）
    ↓
計算儀式數字（6, 7, 8, 9）
    ↓
滾動窗口（6天）生成卦象
  - Ritual_Sequence: "987896"
  - Hexagram_Binary: "101010"
```

#### 檔案結構
```
market_encoder.py
└── MarketEncoder
    ├── _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame
    ├── _get_ritual_number(self, return_val: float, rvol_val: float) -> int
    └── generate_hexagrams(self, df: pd.DataFrame) -> pd.DataFrame
```

---

## 2026-01-20 | Phase 1 - 核心邏輯（解碼器）

### 步驟 4: 建立易經核心邏輯模組 (`iching_core.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
實作易經卦象解釋和未來卦（之卦/Zhi Gua）計算的核心功能，將市場資料編碼的結果轉換為易經預測。

#### 實作細節

1. **IChingCore 類別**
   - 提供卦象查詢、未來卦計算和序列解釋功能
   - 完整的錯誤處理和輸入驗證
   - 符合易經傳統規則

2. **核心方法**

   - **`get_hexagram_name`**:
     - 根據二進制字串查詢 `HEXAGRAM_MAP`
     - 返回卦象資訊（id, name, nature）
     - 查不到時返回預設值：`{"id": 0, "name": "Unknown", "nature": "?"}`
     - 不會因未定義的卦而拋出異常

   - **`calculate_future_hexagram`**:
     - 實作易經變爻規則，計算未來卦（之卦）
     - **變爻規則**：
       * 9 (老陽) -> 0 (陰爻) - 變動
       * 6 (老陰) -> 1 (陽爻) - 變動
       * 7 (少陽) -> 1 (陽爻) - 靜止
       * 8 (少陰) -> 0 (陰爻) - 靜止
     - 返回六位二進制字串（用於查詢 HEXAGRAM_MAP）
     - 驗證輸入長度必須為 6

   - **`interpret_sequence`**:
     - 高階整合方法，統一處理當前卦、未來卦和動爻
     - 返回完整解釋結果字典

3. **輸出結構**

   ```python
   {
       "current_hex": {
           "id": 1,
           "name": "Qian",
           "nature": "乾"
       },
       "future_hex": {
           "id": 3,
           "name": "Zhen",
           "nature": "震"
       },
       "moving_lines": [1, 5, 6]  # 1-based index
   }
   ```

4. **爻位索引規則**
   - 列表索引：`[a, b, c, d, e, f]` 中，`a` 為索引 0（底部），`f` 為索引 5（頂部）
   - 易經編號：第1爻（底部）到第6爻（頂部），使用 1-based index
   - 動爻識別：值為 6 或 9 的爻位（1 到 6）

#### 技術決策

- **為什麼需要未來卦（之卦）？**
  - 在易經中，當前卦（本卦）代表現狀，未來卦（之卦）代表變動後的結果
  - 老陽（9）和老陰（6）會變動，少陽（7）和少陰（8）保持不變
  - 未來卦預示了趨勢的發展方向

- **為什麼使用 1-based index 表示動爻？**
  - 符合易經傳統習慣（爻位從 1 到 6）
  - 提高可讀性，避免與程式語言 0-based index 混淆
  - 方便與易經文獻對照

- **為什麼允許未知卦返回預設值？**
  - HEXAGRAM_MAP 目前只包含前 8 卦作為範例
  - 返回預設值而非拋出異常，讓程式能優雅處理未定義的卦
  - 便於後續擴充，不影響現有功能

- **為什麼要驗證輸入長度？**
  - 易經卦象必須恰好 6 爻，這是基本要求
  - 提前發現錯誤，避免產生錯誤的解釋
  - 提供清晰的錯誤訊息，便於除錯

#### 易經變爻規則說明

```
當前卦（本卦）: [9, 8, 7, 8, 9, 6]
                ↓  ↓  ↓  ↓  ↓  ↓
未來卦（之卦）: [0, 0, 1, 0, 0, 1]
                =  "001001"

規則：
- 9 (老陽) → 變為 0 (陰爻)
- 6 (老陰) → 變為 1 (陽爻)
- 7 (少陽) → 保持 1 (陽爻)
- 8 (少陰) → 保持 0 (陰爻)
```

#### 資料流程

```
Ritual_Sequence: [9, 8, 7, 8, 9, 6]
    ↓
計算當前卦: "101010" (9,8,7,8,9,6 -> 1,0,1,0,1,0)
    ↓
查詢 HEXAGRAM_MAP -> current_hex
    ↓
計算未來卦: "001001" (9->0, 8->0, 7->1, 8->0, 9->0, 6->1)
    ↓
查詢 HEXAGRAM_MAP -> future_hex
    ↓
識別動爻: [1, 5, 6] (位置1,5,6的值為9,9,6)
    ↓
返回完整解釋結果
```

#### 檔案結構
```
iching_core.py
└── IChingCore
    ├── get_hexagram_name(self, binary_string: str) -> Dict[str, any]
    ├── calculate_future_hexagram(self, ritual_sequence: List[int]) -> str
    └── interpret_sequence(self, ritual_sequence: List[int]) -> Dict[str, any]
```

---

## 2026-01-20 | Phase 1 - 系統整合

### 步驟 5: 建立主程式 (`main.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
整合所有模組，建立完整的易經分析流程，從資料獲取到結果視覺化的端到端系統。

#### 實作細節

1. **主程式結構**
   - `main()`: 主執行函數，整合所有模組
   - `print_hexagram_visual()`: ASCII 藝術顯示卦象
   - `format_moving_lines()`: 格式化動爻列表

2. **執行流程**

   - **步驟 1: 資料載入**
     - 初始化 `MarketDataLoader`
     - 下載指定股票的歷史資料
     - 驗證資料完整性

   - **步驟 2: 資料編碼**
     - 初始化 `MarketEncoder`
     - 生成易經卦象（`Ritual_Sequence` 和 `Hexagram_Binary`）
     - 提取最新一筆記錄

   - **步驟 3: 卦象解碼**
     - 初始化 `IChingCore`
     - 解釋儀式數字序列
     - 計算當前卦和未來卦

   - **步驟 4: 結果視覺化**
     - 格式化 CLI 報告
     - 顯示卦象 ASCII 藝術
     - 提供易讀的輸出格式

3. **輸出格式**

   - **標題**: "=== Quantum I-Ching 分析報告: [TICKER] ==="
   - **基本資訊**: 分析日期、資料筆數
   - **儀式序列**: 由下至上的儀式數字列表
   - **二進制編碼**: 卦象的二進制表示
   - **當前卦**: 編號、名稱（中文和英文）
   - **未來卦**: 編號、名稱（中文和英文）
   - **變動分析**: 當前卦 → 未來卦、動爻列表

4. **ASCII 藝術顯示**

   - 陽爻（9/7）: `─────────`
   - 陰爻（6/8）: `───   ───`
   - 動爻標記: `* (變)`
   - 從頂部（第6爻）到底部（第1爻）顯示

5. **錯誤處理**

   - 完整的 `try-except` 包裹
   - 清晰的錯誤訊息
   - 適當的退出碼（`sys.exit(1)`）
   - 驗證資料完整性和格式

#### 技術決策

- **為什麼將 Ritual_Sequence 從字串轉換為列表？**
  - `MarketEncoder` 返回的字串格式（例如 "987896"）需要轉換為整數列表
  - `IChingCore.interpret_sequence()` 需要 `List[int]` 作為輸入
  - 轉換時包含錯誤處理，確保資料完整性

- **為什麼使用 ASCII 藝術顯示卦象？**
  - 提高可讀性，讓非程式設計師也能理解
  - 視覺化呈現易經卦象的結構
  - 清楚標記動爻（變爻）

- **為什麼要提取最新一筆記錄？**
  - 最新資料代表當前市場狀態
  - 為未來預測提供最即時的分析基礎
  - 符合易經「現在 → 未來」的邏輯

- **為什麼使用結構化的輸出格式？**
  - 清晰的資訊層級，易於閱讀
  - 使用表情符號和分隔線提升視覺效果
  - 適合 CLI 環境，也便於後續轉換為其他格式

#### 輸出範例

```
============================================================
  === Quantum I-Ching 分析報告: NVDA ===
============================================================

📅 分析日期: 2026-01-20
📊 資料筆數: 1520 筆

🔮 儀式數字序列（由下至上）:
   [9, 8, 7, 8, 9, 6]
   [第1爻] ← 底部
   [第6爻] ← 頂部

🔢 二進制編碼: 101010
   (1 = 陽爻, 0 = 陰爻)

📖 當前卦（本卦）:
   編號: 1
   名稱: Qian (乾)

🔮 未來卦（之卦）:
   編號: 3
   名稱: Zhen (震)

🔄 變動:
   當前: Qian (乾) → 未來: Zhen (震)
   動爻: 1, 5, 6

卦象視覺化（從上到下）：
──────────────────────────────────────────────────
第 6 爻 陰: ───   ─── * (變)
第 5 爻 陽: ───────── * (變)
第 4 爻 陰: ───   ───
第 3 爻 陽: ─────────
第 2 爻 陰: ───   ───
第 1 爻 陽: ───────── * (變)
──────────────────────────────────────────────────
```

#### 使用方式

```bash
python main.py
```

預設分析 NVDA，可在 `main()` 函數中修改 `ticker` 參數來分析其他股票。

#### 檔案結構
```
main.py
├── main(ticker: str = "NVDA") -> None
├── print_hexagram_visual(ritual_sequence: List[int]) -> None
└── format_moving_lines(moving_lines: List[int]) -> str
```

---

## 待辦事項

- [ ] 完成 `HEXAGRAM_MAP` 剩餘 56 卦的定義
- [x] 建立資料獲取模組（股票價格資料）
- [x] 實作易經卦象生成邏輯
- [x] 實作易經核心解釋邏輯（當前卦、未來卦、動爻）
- [x] 建立系統整合主程式
- [ ] 設計預測模型架構

---

## 2026-01-20 | Phase 1 - 錯誤修正與除錯

### 修正記錄: `main.py` 資料提取錯誤

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

1. **缺少 pandas 匯入**
   - 錯誤: `NameError: name 'pd' is not defined`
   - 原因: 使用了 `pd.isna()` 但未匯入 pandas
   - 修正: 在檔案開頭加入 `import pandas as pd`

2. **資料提取方法不穩定**
   - 問題: 使用 `.get()` 方法從 pandas Series 提取資料時可能失敗
   - 原因: 
     * pandas Series 的 `.get()` 方法在某些情況下可能無法正確運作
     * 需要同時處理 KeyError 和 AttributeError
     * 可能遇到 NaN 值需要特別處理
   - 修正:
     * 先檢查欄位是否存在於 index 中
     * 使用 try-except 包裹，提供多種提取方式
     * 加入 `pd.isna()` 檢查 NaN 值
     * 提供更詳細的錯誤訊息

3. **錯誤訊息不夠詳細**
   - 問題: 當資料提取失敗時，錯誤訊息不足以幫助除錯
   - 修正:
     * 顯示可用欄位列表
     * 顯示實際提取到的值
     * 提供更多上下文資訊

#### 修正後的程式碼

```python
# 提取 Ritual_Sequence（字串，需轉換為列表）
try:
    # 嘗試使用直接索引
    if 'Ritual_Sequence' in latest_row.index:
        ritual_sequence_str = latest_row['Ritual_Sequence']
    else:
        ritual_sequence_str = latest_row.get('Ritual_Sequence', None)
except (KeyError, AttributeError, TypeError) as e:
    print(f"❌ 錯誤: 無法存取 Ritual_Sequence 欄位: {e}")
    print(f"   可用欄位: {list(encoded_data.columns)}")
    sys.exit(1)

if ritual_sequence_str is None or ritual_sequence_str == '' or pd.isna(ritual_sequence_str):
    print("❌ 錯誤: 無法取得儀式數字序列（可能資料不足，需要至少 26 天）")
    print(f"   Ritual_Sequence 值: {ritual_sequence_str}")
    sys.exit(1)
```

#### 學到的經驗

1. **pandas 資料結構處理**
   - 從 DataFrame 提取資料時，應該明確檢查欄位是否存在
   - 使用 `pd.isna()` 而非 `is None` 來檢查 NaN 值
   - 對於 Series，直接索引 `series['key']` 通常比 `.get()` 更可靠

2. **錯誤處理最佳實踐**
   - 提供詳細的錯誤訊息，包含：
     * 實際錯誤類型
     * 可用選項（例如可用欄位列表）
     * 實際值（幫助理解為什麼失敗）
   - 使用多層 try-except 處理不同層級的錯誤

3. **依賴管理**
   - 確保所有使用的模組都已正確匯入
   - 在開發時應該使用 linter 檢查，但實際執行時可能仍有遺漏
   - 建議使用 `requirements.txt` 管理依賴

4. **測試的重要性**
   - 雖然程式碼通過 linter，但實際執行時可能遇到：
     * 缺少依賴套件（`ModuleNotFoundError: No module named 'yfinance'`）
     * 資料結構不符合預期
     * 執行環境差異
   - 需要進行實際執行測試，而不只是靜態檢查

#### 環境設定注意事項

1. **PowerShell 語法**
   - PowerShell 不支援 `&&` 運算符
   - 應使用 `;` 或分開執行命令
   - 範例: `cd "path" ; python main.py`

2. **Python 套件安裝**
   - 確保所有依賴都已安裝：
     ```bash
     pip install yfinance pandas numpy
     ```
   - 建議建立 `requirements.txt` 檔案

#### 後續改進建議

1. 建立 `requirements.txt` 檔案列出所有依賴
2. 加入單元測試，測試資料提取邏輯
3. 建立 `README.md` 說明安裝和使用方式
4. 加入更完善的錯誤處理和日誌記錄

---

## 2026-01-20 | Phase 1 - 終端編碼問題修正

### 修正記錄: Windows 終端編碼與依賴檢查

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

1. **模組匯入階段缺少依賴檢查**
   - 錯誤: `ModuleNotFoundError: No module named 'yfinance'`
   - 原因: 模組在匯入階段就失敗，無法提供清晰的錯誤訊息和安裝指引
   - 修正: 在 `main.py` 開頭加入依賴檢查，在匯入前先檢查所有必要套件

2. **Windows 終端編碼問題（cp950）**
   - 錯誤: `UnicodeEncodeError: 'cp950' codec can't encode character '\u274c'`
   - 原因: Windows PowerShell 預設使用 cp950 編碼，無法顯示 Unicode 表情符號（❌、✅、📅 等）
   - 修正: 移除所有表情符號，改用 ASCII 字符（`[錯誤]`、`[成功]`、`[注意]` 等）

#### 修正後的程式碼

1. **依賴檢查（在模組匯入前）**

```python
# 檢查必要依賴是否已安裝
try:
    import pandas as pd
except ImportError:
    print("[錯誤] 缺少必要套件 'pandas'")
    print("  請執行: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[錯誤] 缺少必要套件 'numpy'")
    print("  請執行: pip install numpy")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("[錯誤] 缺少必要套件 'yfinance'")
    print("  請執行: pip install yfinance")
    print("  或執行: pip install -r requirements.txt")
    sys.exit(1)
```

2. **移除所有表情符號，改用 ASCII 標籤**

- ❌ → `[錯誤]`
- ✅ → `[成功]`
- 📅 → `[分析日期]`
- 📊 → `[資料筆數]`
- 🔮 → `[儀式數字序列]` / `[未來卦]`
- 🔢 → `[二進制編碼]`
- 📖 → `[當前卦]`
- 🔄 → `[變動]`
- ⚠️ → `[注意]`

#### 學到的經驗

1. **依賴檢查最佳實踐**
   - 在程式的最開始就檢查所有依賴，不要等到實際使用時才發現
   - 提供清晰的錯誤訊息和安裝指引
   - 檢查順序：先檢查底層依賴（pandas, numpy），再檢查上層依賴（yfinance）
   - 提供多種安裝方式（直接 pip install 或使用 requirements.txt）

2. **跨平台編碼相容性**
   - Windows 終端（特別是 PowerShell）預設使用 cp950 或 cp936 編碼
   - Unicode 表情符號在某些終端無法顯示
   - **建議**：
     * CLI 程式避免使用表情符號，改用 ASCII 字符
     * 如需使用 Unicode，應先檢查終端編碼能力
     * 使用 `sys.stdout.encoding` 檢查終端編碼

3. **錯誤處理分層**
   - 第一層：模組匯入階段（依賴檢查）
   - 第二層：資料處理階段（資料驗證）
   - 第三層：業務邏輯階段（業務規則驗證）
   - 每一層都應提供適當的錯誤訊息

4. **Windows 環境特殊考量**
   - PowerShell 不支援 `&&` 運算符，應使用 `;` 或分開執行命令
   - 終端編碼可能不同，應避免使用非 ASCII 字符
   - 路徑中包含空格時，應使用引號包裹

#### 編碼相容性檢查（參考）

如果需要檢測終端編碼能力，可以使用：

```python
import sys

# 檢查終端編碼
encoding = sys.stdout.encoding or sys.getdefaultencoding()
print(f"終端編碼: {encoding}")

# 檢查是否支援 UTF-8
if encoding.lower().startswith('utf'):
    # 可以使用 Unicode 字符
    print("✅ 支援 UTF-8")
else:
    # 只使用 ASCII 字符
    print("[OK] 僅支援 ASCII")
```

#### 後續改進建議

1. **建立啟動腳本**
   - 建立 `run.bat` (Windows) 和 `run.sh` (Linux/Mac)
   - 自動檢查並安裝依賴
   - 設定正確的編碼環境

2. **環境檢查工具**
   - 建立 `check_environment.py` 檢查所有依賴和環境設定
   - 在執行主程式前先執行環境檢查

3. **日誌記錄**
   - 使用 logging 模組記錄錯誤，而非直接 print
   - 可以設定編碼處理，避免編碼問題

4. **文件說明**
   - 在 README.md 中明確說明系統需求和安裝步驟
   - 提供常見問題（FAQ）解答編碼和依賴問題

#### 測試結果

修正後的程式在以下環境中測試：

- ✅ 提供清晰的錯誤訊息（即使缺少依賴）
- ✅ 在 Windows cp950 編碼下正常顯示
- ✅ 提供安裝指引

---

## 2026-01-20 | Phase 1 - 最終編碼問題解決

### 修正記錄: Windows 終端中文顯示問題

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

**Windows 終端中文顯示亂碼**
- 錯誤現象: 即使移除了表情符號，中文錯誤訊息在 Windows PowerShell (cp950) 下仍顯示為亂碼
- 錯誤輸出: `[~] ʤ֥nM 'yfinance'`
- 原因分析:
  * Windows PowerShell 預設使用 cp950 編碼
  * 中文無法在 cp950 下正確顯示
  * 程式輸出無法強制終端使用 UTF-8

#### 修正方案

採用**雙重策略**：

1. **設定 UTF-8 編碼（優先）**
   - 在程式開頭嘗試設定 `sys.stdout` 和 `sys.stderr` 為 UTF-8
   - 使用 `sys.stdout.reconfigure(encoding='utf-8')`（Python 3.7+）
   - 如果不可用，設定環境變數 `PYTHONIOENCODING=utf-8`

2. **關鍵訊息使用英文（備援）**
   - 依賴檢查錯誤訊息使用英文
   - 確保在所有終端環境下都能正常顯示
   - 不影響程式的主要中文輸出

#### 修正後的程式碼

```python
# 設定輸出編碼為 UTF-8（處理 Windows 終端編碼問題）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Python < 3.7 或 reconfigure 不可用時，使用環境變數設定
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'

# 檢查必要依賴是否已安裝
# 使用英文錯誤訊息確保在所有終端環境下都能正常顯示
try:
    import pandas as pd
except ImportError:
    print("[ERROR] Missing required package: 'pandas'")
    print("  Please run: pip install pandas")
    print("  Or run: pip install -r requirements.txt")
    sys.exit(1)
```

#### 測試結果

修正後在不同環境下的測試結果：

- ✅ Windows PowerShell (cp950): 英文錯誤訊息正常顯示
- ✅ Windows CMD (cp950): 英文錯誤訊息正常顯示
- ✅ 設定 UTF-8 後: 中文也能正常顯示（如果終端支援）

#### 學到的經驗

1. **編碼處理最佳實踐**
   - **關鍵錯誤訊息使用英文**：確保在所有環境下都能讀取
   - **嘗試設定 UTF-8**：如果終端支援，可以改善中文顯示
   - **提供多種編碼方案**：針對不同 Python 版本和終端類型

2. **錯誤訊息設計原則**
   - **關鍵性分級**：
     * 關鍵訊息（依賴檢查、致命錯誤）：使用英文
     * 一般訊息（分析結果、狀態提示）：可以使用中文
   - **可讀性優先**：即使犧牲多語言支援，也要確保關鍵訊息可讀

3. **跨平台相容性考量**
   - Windows 終端編碼最複雜，需要特別處理
   - 使用 `sys.platform == 'win32'` 檢測 Windows
   - 提供降級方案（環境變數設定）

4. **使用者體驗優先**
   - 即使程式使用中文，關鍵錯誤訊息應使用英文
   - 提供清晰的安裝指引
   - 確保在任何環境下都能理解錯誤原因

#### 編碼設定策略總結

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| 關鍵訊息用英文 | 100% 相容所有終端 | 犧牲多語言一致性 | ✅ 推薦：依賴檢查、致命錯誤 |
| 設定 UTF-8 編碼 | 支援中文顯示 | 需要終端支援 UTF-8 | 一般輸出訊息 |
| 使用環境變數 | 相容舊版 Python | 影響整個 Python 環境 | 備援方案 |

#### 後續改進建議

1. **環境檢查腳本**
   - 建立 `check_env.py` 檢查終端編碼和依賴
   - 在執行主程式前提供環境報告

2. **啟動腳本**
   - 建立 `run.bat` (Windows) 設定 UTF-8 編碼環境
   - 建立 `run.sh` (Linux/Mac) 設定正確的環境變數

3. **README 文件**
   - 說明 Windows 終端編碼設定
   - 提供 PowerShell 設定 UTF-8 的指令

---

## 2026-01-20 | Phase 1 - yfinance API 相容性修正

### 修正記錄: `show_errors` 參數不存在問題

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

**`yfinance.download()` API 不相容**
- 錯誤訊息: `TypeError: download() got an unexpected keyword argument 'show_errors'`
- 原因分析:
  * 不同版本的 `yfinance` 函數簽名不同
  * `show_errors` 參數在某些版本中不存在
  * 使用不支援的參數導致下載失敗，返回空 DataFrame

#### 終端錯誤輸出

```
2026-01-20 11:38:29,586 - data_loader - ERROR - 下載股票資料時發生錯誤: download() got an unexpected keyword argument 'show_errors'
Traceback (most recent call last):
  File "C:\Users\USER\Desktop\I-Ching AI\data_loader.py", line 75, in fetch_data
    data = yf.download(
        tickers=tickers,
        start=settings.START_DATE,
        end=settings.END_DATE,
        progress=False,
        show_errors=False,  # 這個參數不存在
    )
TypeError: download() got an unexpected keyword argument 'show_errors'
[錯誤] 無法獲取 NVDA 的資料
```

#### 修正方案

移除不支援的 `show_errors` 參數，使用 `yfinance` 的標準 API。

#### 修正前的程式碼

```python
data = yf.download(
    tickers=tickers,
    start=settings.START_DATE,
    end=settings.END_DATE,
    progress=False,
    show_errors=False,  # ❌ 此參數不存在
)
```

#### 修正後的程式碼

```python
data = yf.download(
    tickers=tickers,
    start=settings.START_DATE,
    end=settings.END_DATE,
    progress=False,  # 關閉進度條輸出，保持日誌整潔
    # 注意：不同版本的 yfinance 可能參數不同
    # show_errors 參數在某些版本中不存在，已移除
)
```

#### 學到的經驗

1. **第三方套件 API 版本相容性**
   - 不同版本的第三方套件可能有不同的 API
   - 應使用標準且穩定的參數，避免使用版本特定的參數
   - **最佳實踐**：
     * 檢查套件文件確認參數
     * 使用最小參數集合（只使用必需的參數）
     * 測試不同版本的相容性

2. **錯誤處理策略**
   - 當遇到 API 錯誤時，優先檢查參數是否正確
   - 查看錯誤訊息中的函數簽名提示
   - 參考最新版本的文檔

3. **yfinance 常用參數**
   - `tickers`: 股票代號（必需）
   - `start`: 開始日期（可選）
   - `end`: 結束日期（可選）
   - `progress`: 是否顯示進度條（可選，預設 True）
   - `interval`: 資料間隔（可選，預設 '1d'）
   - **不推薦使用未在文檔中明確說明的參數**

4. **依賴版本管理**
   - 在 `requirements.txt` 中指定具體版本範圍
   - 例如: `yfinance>=0.2.0,<1.0.0`
   - 有助於避免 API 變更導致的問題

#### 後續改進建議

1. **更新 requirements.txt**
   - 指定 `yfinance` 的版本範圍
   - 例如: `yfinance>=0.2.0,<0.3.0`

2. **加入 API 相容性測試**
   - 測試不同版本的 yfinance
   - 確保程式碼在不同版本下都能正常工作

3. **改進錯誤訊息**
   - 當遇到 API 錯誤時，提供更詳細的提示
   - 建議檢查 yfinance 版本

4. **文檔檢查**
   - 在實作前先查看最新版本的文檔
   - 確認參數的正確用法

---

## 2026-01-20 | Phase 1 - DataFrame 結構處理修正

### 修正記錄: yfinance 單檔股票 MultiIndex 結構問題

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

**DataFrame 欄位檢查失敗**
- 錯誤訊息: `DataFrame 必須包含 'Close' 和 'Volume' 欄位`
- 原因分析:
  * `yfinance.download()` 在單檔股票時（即使使用列表 `['NVDA']`）仍可能返回 MultiIndex DataFrame
  * `data_loader.py` 的邏輯假設單檔股票返回標準 DataFrame，但實際上可能是 MultiIndex
  * 當返回 MultiIndex 時，原邏輯不會正確提取資料，導致返回空 DataFrame 或錯誤結構
  * `market_encoder.py` 檢查 `df.columns` 時，MultiIndex 中沒有直接的 'Close' 欄位

#### 問題根源

當使用 `yf.download(tickers=['NVDA'], ...)` 時：
- **預期**: 返回標準 DataFrame，欄位為 `['Open', 'High', 'Low', 'Close', 'Volume', ...]`
- **實際**: 返回 MultiIndex DataFrame，結構為 `[(NVDA, Open), (NVDA, High), ...]`

原程式碼邏輯：
```python
if len(tickers) == 1:
    if not isinstance(data.columns, pd.MultiIndex):  # ❌ 假設不會是 MultiIndex
        return data
    # 如果是 MultiIndex，沒有處理，導致資料丟失
```

#### 修正方案

1. **修正 `data_loader.py` 的資料提取邏輯**
   - 正確處理單檔股票返回 MultiIndex 的情況
   - 從 MultiIndex 中提取指定 ticker 的資料
   - 確保返回標準 DataFrame

2. **改進 `market_encoder.py` 的錯誤訊息**
   - 提供更詳細的錯誤資訊
   - 顯示缺少的欄位和實際欄位

#### 修正後的程式碼

**`data_loader.py`**:
```python
# 處理單一股票的情況
# 注意：即使只有一個 ticker，yfinance 也可能返回 MultiIndex DataFrame
if len(tickers) == 1:
    ticker = tickers[0]
    # 如果是 MultiIndex，提取該 ticker 的資料
    if isinstance(data.columns, pd.MultiIndex):
        # 檢查 ticker 是否存在於第一層
        if ticker in data.columns.get_level_values(0):
            ticker_data = data[ticker].copy()
            self.logger.info(
                f"成功下載 {ticker} 的資料，共 {len(ticker_data)} 筆記錄（MultiIndex 結構）"
            )
            return ticker_data
        else:
            self.logger.warning(
                f"Ticker {ticker} 不存在於下載的資料中"
            )
            return pd.DataFrame()
    else:
        # 標準 DataFrame，直接返回
        self.logger.info(f"成功下載 {ticker} 的資料，共 {len(data)} 筆記錄")
        return data
```

**`market_encoder.py`**:
```python
# 確保有 Close 和 Volume 欄位
# 檢查欄位是否存在（支援標準和 MultiIndex DataFrame）
missing_cols = []
if 'Close' not in df.columns:
    missing_cols.append('Close')
if 'Volume' not in df.columns:
    missing_cols.append('Volume')

if missing_cols:
    raise ValueError(
        f"DataFrame 必須包含 'Close' 和 'Volume' 欄位。"
        f"缺少欄位: {missing_cols}。"
        f"實際欄位: {list(df.columns)}"
    )
```

#### 學到的經驗

1. **yfinance API 行為變化**
   - `yfinance.download()` 的行為可能因版本而異
   - 即使只下載單檔股票（使用列表），也可能返回 MultiIndex DataFrame
   - **最佳實踐**: 不要假設返回結構，總是檢查並正確處理

2. **MultiIndex DataFrame 處理**
   - MultiIndex DataFrame 的欄位存取方式不同
   - 使用 `df[ticker]` 可以提取特定 ticker 的所有欄位
   - 提取後變成標準 DataFrame，方便後續處理

3. **錯誤訊息設計**
   - 提供詳細的錯誤資訊幫助除錯
   - 顯示預期值和實際值
   - 列出所有相關資訊（缺少欄位、實際欄位等）

4. **防禦式程式設計**
   - 不要假設第三方套件的行為
   - 檢查資料結構再處理
   - 提供多種情況的處理路徑

#### yfinance 行為說明

| 輸入 | yfinance 版本 | 返回結構 |
|------|--------------|----------|
| `tickers=['NVDA']` | 較新版本 | MultiIndex DataFrame |
| `tickers=['NVDA']` | 較舊版本 | 標準 DataFrame |
| `tickers=['NVDA', 'AAPL']` | 所有版本 | MultiIndex DataFrame |

**建議**: 總是檢查返回結構並正確處理，不要依賴特定版本的行為。

#### 後續改進建議

1. **加入資料結構驗證**
   - 在下載後立即驗證資料結構
   - 確保必要的欄位存在

2. **版本相容性測試**
   - 測試不同版本的 yfinance
   - 確保程式碼在所有版本下都能正常工作

3. **單元測試**
   - 測試單檔和多檔股票的情況
   - 測試 MultiIndex 和標準 DataFrame 的情況

---

## 2026-01-20 | Phase 1 - yfinance 單檔股票下載優化

### 修正記錄: 使用 Ticker 物件替代 download() 方法

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

**Ticker 名稱不匹配問題**
- 錯誤訊息: `Ticker NVDA 不存在於下載的資料中`
- 原因分析:
  * `yfinance.download()` 在單檔股票時可能返回 MultiIndex DataFrame
  * 第一層的 ticker 名稱可能與請求的不完全匹配（大小寫、格式等）
  * 複雜的 MultiIndex 處理邏輯容易出錯

#### 問題根源

當使用 `yf.download(tickers=['NVDA'], ...)` 時：
- 可能返回 MultiIndex DataFrame，但第一層的 ticker 名稱可能不匹配
- 需要檢查、提取、轉換，過程複雜且容易出錯
- 不同版本的 yfinance 行為可能不同

#### 修正方案

對於單檔股票，使用更簡單且可靠的 `yf.Ticker()` 方法：

```python
# 如果只有一個 ticker，嘗試直接使用 Ticker 物件（更可靠）
if len(tickers) == 1:
    ticker = tickers[0]
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(
        start=settings.START_DATE,
        end=settings.END_DATE
    )
    # 直接返回標準 DataFrame，沒有 MultiIndex 問題
```

**優點**:
1. **更簡單**: 直接返回標準 DataFrame，不需要處理 MultiIndex
2. **更可靠**: `Ticker.history()` 方法專為單檔股票設計
3. **更一致**: 行為在不同版本間更穩定
4. **更快速**: 少了 MultiIndex 轉換的開銷

#### 修正後的程式碼

```python
try:
    # 如果只有一個 ticker，嘗試直接使用 Ticker 物件（更可靠）
    if len(tickers) == 1:
        ticker = tickers[0]
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=settings.START_DATE,
                end=settings.END_DATE
            )
            
            if not data.empty:
                self.logger.info(
                    f"成功下載 {ticker} 的資料（使用 Ticker 物件），共 {len(data)} 筆記錄"
                )
                return data
            else:
                self.logger.warning(
                    f"使用 Ticker 物件下載 {ticker} 的資料為空，嘗試使用 download 方法"
                )
        except Exception as e:
            self.logger.debug(
                f"使用 Ticker 物件下載失敗: {e}，嘗試使用 download 方法"
            )
    
    # 如果 Ticker 物件方法失敗或有多個 ticker，使用 download 方法
    data = yf.download(
        tickers=tickers,
        start=settings.START_DATE,
        end=settings.END_DATE,
        progress=False,
    )
    
    # ... 後續處理 ...
```

#### 備援機制

- 如果 `Ticker.history()` 失敗，自動降級到 `download()` 方法
- 保留原有的 MultiIndex 處理邏輯作為備援
- 如果 ticker 名稱不匹配，但有唯一 ticker，自動使用它

#### 學到的經驗

1. **選擇正確的 API**
   - `yf.Ticker().history()`: 適合單檔股票，返回標準 DataFrame
   - `yf.download()`: 適合多檔股票，返回 MultiIndex DataFrame
   - **最佳實踐**: 根據使用場景選擇合適的方法

2. **簡化複雜度**
   - 避免不必要的複雜邏輯
   - 使用專為特定用途設計的 API
   - 減少資料結構轉換的步驟

3. **降級策略**
   - 優先使用簡單可靠的方法
   - 提供備援方案處理特殊情況
   - 記錄所有步驟以便除錯

4. **yfinance API 選擇指南**

| 場景 | 推薦方法 | 原因 |
|------|----------|------|
| 單檔股票 | `yf.Ticker(ticker).history()` | 簡單、可靠、標準 DataFrame |
| 多檔股票 | `yf.download(tickers=[...])` | 支援批量下載 |
| 即時資料 | `yf.Ticker(ticker).info` | 提供詳細資訊 |
| 歷史資料 | `yf.Ticker(ticker).history()` | 單檔最佳選擇 |

#### 後續改進建議

1. **統一使用 Ticker 物件**
   - 對於單檔股票，總是使用 `Ticker.history()`
   - 對於多檔股票，使用 `download()` 批量下載

2. **加入重試機制**
   - 如果下載失敗，自動重試
   - 設定重試次數和延遲

3. **快取機制**
   - 快取已下載的資料
   - 避免重複下載相同資料

---

## 2026-01-20 | Phase 1 - 模組匯入錯誤修正

### 修正記錄: HEXAGRAM_MAP 匯入錯誤

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

**AttributeError: 'Settings' object has no attribute 'HEXAGRAM_MAP'**
- 錯誤訊息: `'Settings' object has no attribute 'HEXAGRAM_MAP'`
- 發生位置: `iching_core.py` 的 `get_hexagram_name()` 方法
- 原因分析:
  * `HEXAGRAM_MAP` 是 `config.py` 模組層級的常數，不是 `Settings` 類別的屬性
  * `iching_core.py` 中錯誤地使用 `settings.HEXAGRAM_MAP` 來存取
  * `Settings` 是 dataclass 實例，只包含定義的欄位（START_DATE, END_DATE 等）

#### 問題根源

在 `config.py` 中：
```python
# HEXAGRAM_MAP 是模組層級的常數
HEXAGRAM_MAP: dict[str, HexagramDict] = {...}

# settings 是 Settings 類別的實例
settings = Settings()  # 只包含 START_DATE, END_DATE, TARGET_TICKERS, YIN_YANG_THRESHOLD
```

在 `iching_core.py` 中（錯誤）：
```python
from config import settings

# ❌ 錯誤：HEXAGRAM_MAP 不是 settings 的屬性
hexagram = settings.HEXAGRAM_MAP.get(...)
```

#### 修正方案

直接從 `config` 模組匯入 `HEXAGRAM_MAP`，而不是透過 `settings` 物件存取。

#### 修正後的程式碼

**`iching_core.py`**:
```python
# 修正前
from config import settings

hexagram = settings.HEXAGRAM_MAP.get(...)  # ❌ 錯誤

# 修正後
from config import HEXAGRAM_MAP, settings

hexagram = HEXAGRAM_MAP.get(...)  # ✅ 正確
```

#### 學到的經驗

1. **模組層級常數 vs 類別屬性**
   - 模組層級常數：定義在模組中，不屬於任何類別
   - 類別屬性：定義在類別中，屬於類別或實例
   - **區別**：
     * `HEXAGRAM_MAP` 是模組層級常數，應直接匯入
     * `settings` 是類別實例，只包含類別定義的欄位

2. **Python 匯入最佳實踐**
   - 明確匯入需要的常數和物件
   - 使用 `from module import constant, class_instance`
   - 避免透過物件存取模組層級常數

3. **程式碼組織原則**
   - 常數（如 `HEXAGRAM_MAP`）應定義在模組層級
   - 配置物件（如 `Settings`）應定義為類別
   - 清楚區分兩者的使用方式

4. **錯誤預防**
   - 使用 IDE 的型別檢查功能
   - 執行前進行靜態分析
   - 編寫單元測試驗證匯入和使用

#### 模組結構說明

```
config.py
├── HexagramDict (TypedDict) - 型別定義
├── Settings (dataclass) - 配置類別
├── HEXAGRAM_MAP (dict) - 模組層級常數 ⭐
└── settings (Settings instance) - 類別實例
```

**存取方式**：
- `HEXAGRAM_MAP`: 直接匯入使用 `from config import HEXAGRAM_MAP`
- `settings`: 匯入實例使用 `from config import settings`
- `settings.START_DATE`: 透過實例存取類別屬性 ✅
- `settings.HEXAGRAM_MAP`: 錯誤！HEXAGRAM_MAP 不是 Settings 的屬性 ❌

#### 後續改進建議

1. **型別檢查**
   - 使用 `mypy` 進行靜態型別檢查
   - 可以提前發現這類錯誤

2. **文件說明**
   - 在模組 docstring 中說明哪些是常數，哪些是類別
   - 提供使用範例

3. **單元測試**
   - 測試模組匯入是否正確
   - 測試常數和物件的存取方式

---

## 2026-01-20 | Phase 1 - 完成六十四卦資料

### 步驟 6: 完成 HEXAGRAM_MAP 完整資料

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
完成 `HEXAGRAM_MAP` 字典，從原本只有前 8 卦的範例資料擴充為完整的 64 卦對照表。

#### 實作細節

1. **資料結構**
   - 鍵值：六位二進制字串（1=陽爻，0=陰爻）
   - 值：`HexagramDict` 型別的字典
     - `id`: 卦象編號（1-64）
     - `name`: 英文名稱（包含拼音和英文解釋）
     - `nature`: 繁體中文卦名

2. **資料內容**
   - 完整的 64 卦，按照易經傳統順序
   - 每卦都包含詳細的英文名稱和繁體中文卦名
   - 所有卦象都已正確編號（1-64）

3. **資料驗證**
   - ✅ 總卦數：64 卦
   - ✅ ID 範圍：1-64（無遺漏）
   - ✅ 無重複 ID
   - ✅ 所有卦象都可正確存取

#### 資料範例

```python
"111111": {"id": 1, "name": "Qian (The Creative)", "nature": "乾"}
"000000": {"id": 2, "name": "Kun (The Receptive)", "nature": "坤"}
"100010": {"id": 3, "name": "Chun (Difficulty at the Beginning)", "nature": "屯"}
...
"010101": {"id": 64, "name": "Wei Ji (Before Completion)", "nature": "未濟"}
```

#### 關鍵改進

1. **從部分資料到完整資料**
   - 原本：只有前 8 卦作為範例
   - 現在：完整的 64 卦，涵蓋所有易經卦象

2. **英文名稱格式統一**
   - 使用格式：`"Name (English Translation)"`
   - 例如：`"Qian (The Creative)"` 而非單純的 `"Qian"`
   - 提供更清楚的解釋和上下文

3. **繁體中文卦名**
   - 所有卦象都包含正確的繁體中文名稱
   - 包括複雜的字元（如 "蠱"、"賁"、"遯" 等）

#### 重要說明

1. **卦象編號與二進制對應**
   - 編號 1-64 對應易經傳統順序
   - 二進制字串代表六爻的陰陽組合
   - 順序：從底部（第1位）到頂部（第6位）

2. **同名問題處理**
   - 注意：第 1 卦 "Qian" 和第 15 卦 "Qian (Modesty)" 都使用了 "Qian"
   - 第 15 卦實際是 "謙" 卦，英文使用 "Qian (Modesty)" 來區分
   - 這是易經中常見的情況（不同卦可能有相似或相同的拼音名稱）

3. **易經卦象順序**
   - 按照易經傳統的 64 卦順序排列
   - 從乾（1）開始，到未濟（64）結束
   - 符合易經的卦序邏輯

#### 後續應用

現在 `HEXAGRAM_MAP` 已完整，可以：
1. 支援所有 64 卦的查詢和解釋
2. 提供完整的卦象名稱（英文和中文）
3. 用於易經分析和預測

#### 資料統計

- 總卦數：64
- 二進制組合：64 種（2^6 = 64）
- 涵蓋範圍：易經所有卦象
- 資料完整性：100%

---

## 2026-01-20 | Phase 1 - RAG 知識庫初始化

### 步驟 7: 建立易經知識庫 JSON 檔案

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
建立 `data/iching_book.json` 檔案，包含完整的 64 卦資料，作為 RAG 系統的知識庫。此檔案將用於後續的檢索增強生成（Retrieval-Augmented Generation）功能。

#### 實作細節

1. **腳本建立** (`scripts/seed_data.py`)
   - 從 `config.py` 匯入 `HEXAGRAM_MAP` 獲取正確的卦象名稱
   - 定義前 8 卦的完整 Judgement 和 Image
   - 程式化生成第 9-64 卦的資料（使用正確名稱，通用文字作為內容）

2. **資料結構**
   - 每個卦象包含 5 個欄位：
     * `hexagram_id`: 卦象編號（1-64）
     * `name`: 英文/拼音名稱（例如 "Qian"）
     * `chinese_name`: 繁體中文卦名（例如 "乾"）
     * `judgement`: 卦辭（前 8 卦完整，9-64 卦通用文字）
     * `image`: 象辭（前 8 卦完整，9-64 卦通用文字）

3. **前 8 卦完整資料**
   - 使用用戶提供的完整 Judgement 和 Image
   - 包含易經的經典解釋
   - 提供深度的哲學和實用指導

4. **第 9-64 卦資料**
   - 使用正確的卦象名稱（從 `HEXAGRAM_MAP` 提取）
   - 使用通用文字作為 Judgement 和 Image（節省空間）
   - 格式：`"The hexagram [Name] ([Chinese]) represents a significant moment..."`

5. **檔案生成**
   - 自動建立 `data/` 目錄（如果不存在）
   - 使用 UTF-8 編碼寫入 JSON
   - 使用 `indent=2` 和 `ensure_ascii=False` 確保可讀性和中文顯示

#### 資料範例

**前 8 卦（完整資料）**:
```json
{
  "hexagram_id": 1,
  "name": "Qian",
  "chinese_name": "乾",
  "judgement": "The Creative works sublime success, furthering through perseverance.",
  "image": "The movement of heaven is full of power. Thus the superior man makes himself strong and untiring."
}
```

**第 9-64 卦（通用文字）**:
```json
{
  "hexagram_id": 29,
  "name": "Kan",
  "chinese_name": "坎",
  "judgement": "The hexagram Kan (坎) represents a significant moment in the cycle of change. Success comes through understanding the situation and acting appropriately.",
  "image": "The image of Kan (坎) reflects the natural patterns of transformation. The superior man observes these patterns and aligns his actions accordingly."
}
```

#### 技術決策

- **為什麼使用程式化生成？**
  - 節省 token 空間，避免手動輸入所有 64 卦的完整內容
  - 前 8 卦作為範例，提供完整的易經解釋
  - 第 9-64 卦使用正確名稱，後續可逐步擴充內容

- **為什麼從 HEXAGRAM_MAP 提取名稱？**
  - 確保名稱一致性
  - 避免手動輸入錯誤
  - 自動同步 `config.py` 中的資料

- **為什麼使用 UTF-8 編碼？**
  - 正確顯示繁體中文字元
  - 符合 JSON 標準
  - 確保跨平台相容性

#### 檔案結構

```
scripts/
└── seed_data.py
    ├── FIRST_8_DETAILS (dict) - 前 8 卦的完整資料
    ├── ICHING_DATA (list) - 完整的 64 卦資料
    └── 檔案寫入邏輯

data/
└── iching_book.json - 生成的 JSON 知識庫檔案
```

#### 驗證結果

- ✅ 總卦數：64
- ✅ ID 範圍：1-64（無遺漏）
- ✅ 前 8 卦：包含完整的 Judgement 和 Image
- ✅ 第 9-64 卦：包含正確的名稱和通用文字
- ✅ JSON 格式：有效且可讀
- ✅ 中文顯示：UTF-8 編碼正確

#### 後續應用

此 JSON 檔案將用於：
1. RAG 系統的知識庫
2. 向量資料庫的資料來源
3. 易經解釋和查詢功能
4. 後續可擴充為完整的 64 卦詳細內容

---

## 備註

- 專案遵循 Google Python Style Guide
- 所有程式碼註解使用繁體中文
- 型別提示為必要項目
- **重要**: 執行前請確保已安裝所有依賴套件（`yfinance`, `pandas`, `numpy`）
- **Windows 相容性**: 
  * 已移除所有表情符號
  * 關鍵錯誤訊息使用英文確保可讀性
  * 自動設定 UTF-8 編碼（如支援）
- **API 相容性**: 
  * 已移除 `show_errors` 參數（某些 yfinance 版本不支援）
  * 使用標準且穩定的 API 參數
- **資料結構處理**:
  * 單檔股票使用 `Ticker.history()` 方法（更簡單可靠）
  * 多檔股票使用 `download()` 方法（支援批量下載）
  * 正確處理 MultiIndex 作為備援方案
- **模組匯入**:
  * `HEXAGRAM_MAP` 是模組層級常數，應直接匯入
  * `settings` 是類別實例，只包含類別定義的欄位
- **HEXAGRAM_MAP**:
  * 已完成完整的 64 卦資料
  * 所有卦象都包含英文名稱和繁體中文卦名
  * 資料已驗證：無遺漏、無重複
- **RAG 知識庫**:
  * 已建立 `data/iching_book.json` 包含 64 卦資料
  * 前 8 卦包含完整的 Judgement 和 Image
  * 第 9-64 卦使用正確名稱和通用文字（可後續擴充）

---

## 2026-01-20 | Phase 2 - RAG 資料載入

### 步驟 8: 建立知識庫載入模組 (`knowledge_loader.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
建立知識庫載入器，將 JSON 格式的易經資料轉換為適合嵌入的文件物件，為 RAG 系統的向量化做準備。

#### 實作細節

1. **IChingDocument 資料類別**
   - 使用 `@dataclass` 定義文件結構
   - `id`: 卦象編號（1-64）
   - `content`: 完整的文字內容，用於語義嵌入
   - `metadata`: 結構化元資料，用於檢索和 UI 顯示

2. **IChingKnowledgeLoader 類別**
   - `__init__`: 儲存 JSON 檔案路徑
   - `load_documents`: 載入並轉換所有卦象資料

3. **文字內容構造策略**
   - **格式**: `"Hexagram [ID]: [Chinese Name] [Name]. Judgement: [Text]. Image: [Text]"`
   - 將 Judgement 和 Image 合併為單一字串
   - 包含卦象 ID、中文名稱和英文名稱
   - 支援未來的 Key Lines 欄位（目前為可選）

4. **元資料結構**
   - `name`: 英文/拼音名稱
   - `chinese_name`: 繁體中文名稱
   - `hexagram_id`: 卦象編號（字串格式，便於 JSON 序列化）

5. **錯誤處理**
   - `FileNotFoundError`: 檔案不存在時提供清晰的錯誤訊息
   - `json.JSONDecodeError`: JSON 格式錯誤時提供詳細資訊
   - `ValueError`: 資料結構驗證失敗時提供具體錯誤

#### 技術決策

- **為什麼將 Judgement 和 Image 合併為單一字串？**
  - 語義嵌入需要完整的上下文
  - 單一文字塊更容易進行向量化
  - 提高檢索的準確性和相關性

- **為什麼保留結構化 metadata？**
  - 方便 UI 顯示和過濾
  - 支援多種檢索方式（按名稱、ID 等）
  - 保持資料的可讀性和可維護性

- **為什麼使用 dataclass？**
  - 簡潔的資料結構定義
  - 自動生成 `__init__`、`__repr__` 等方法
  - 型別提示支援，提高程式碼可讀性

#### 資料流程

```
data/iching_book.json
    ↓
IChingKnowledgeLoader.load_documents()
    ↓
讀取 JSON 並驗證結構
    ↓
遍歷每個卦象
    ↓
構造 content（合併文字）
    ↓
構造 metadata（結構化資料）
    ↓
建立 IChingDocument 物件
    ↓
返回 List[IChingDocument]
```

#### 文件內容範例

```python
IChingDocument(
    id=1,
    content="Hexagram 1: 乾 Qian. Judgement: The Creative works sublime success, furthering through perseverance. Image: The movement of heaven is full of power. Thus the superior man makes himself strong and untiring.",
    metadata={
        "name": "Qian",
        "chinese_name": "乾",
        "hexagram_id": "1"
    }
)
```

#### 驗證結果

- ✅ 成功載入 64 個文件
- ✅ 所有文件都包含完整的 content 和 metadata
- ✅ 文字內容格式正確
- ✅ UTF-8 編碼正確處理中文字元
- ✅ 錯誤處理機制完善

#### 後續應用

此載入器將用於：
1. 向量嵌入前的資料準備
2. 向量資料庫的資料來源
3. RAG 系統的檢索和生成
4. 易經解釋和查詢功能

#### 檔案結構
```
knowledge_loader.py
├── IChingDocument (dataclass)
│   ├── id: int
│   ├── content: str
│   └── metadata: Dict[str, str]
└── IChingKnowledgeLoader
    ├── __init__(self, file_path: str) -> None
    └── load_documents(self) -> List[IChingDocument]
```

---

## 2026-01-20 | Phase 2 - 向量資料庫建立

### 步驟 9: 建立向量資料庫模組 (`vector_store.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
建立向量資料庫模組，使用 ChromaDB 進行本地持久化儲存，並使用 SentenceTransformers 進行嵌入。提供文件的儲存和語義搜尋功能，支援 RAG 系統的檢索增強生成。

#### 實作細節

1. **IChingVectorStore 類別**
   - 使用 ChromaDB 進行本地持久化
   - 使用 SentenceTransformers (all-MiniLM-L6-v2) 進行嵌入
   - 提供文件的儲存和查詢功能

2. **初始化方法 (`__init__`)**
   - 初始化嵌入函數（SentenceTransformerEmbeddingFunction）
   - 建立 ChromaDB 持久化客戶端
   - 取得或建立集合 "iching_knowledge"

3. **文件儲存方法 (`add_documents`)**
   - 將 IChingDocument 物件轉換為 ChromaDB 格式
   - 使用 upsert 操作（如果文件已存在則更新）
   - 保留完整的 metadata 供後續使用

4. **查詢方法 (`query`)**
   - 使用語義搜尋找出相關文件
   - 返回文件內容列表（按相關性排序）
   - 支援自訂返回結果數量

5. **輔助函數 (`build_vector_db`)**
   - 執行完整的資料載入和向量化流程
   - 整合知識載入器和向量資料庫
   - 提供便捷的初始化方式

#### 技術決策

- **為什麼使用 ChromaDB？**
  - 輕量級、易於使用的向量資料庫
  - 支援本地持久化，無需額外服務
  - 內建嵌入函數支援，簡化實作

- **為什麼使用 all-MiniLM-L6-v2 模型？**
  - 輕量級模型，速度快
  - 支援多語言（包括中文）
  - 適合中小型知識庫

- **為什麼使用 upsert 而非 insert？**
  - 支援資料更新，避免重複插入
  - 更適合增量更新場景
  - 提高資料一致性

#### 資料流程

```
IChingDocument 物件列表
    ↓
IChingVectorStore.add_documents()
    ↓
轉換為 ChromaDB 格式
  - ids: 卦象 ID（字串）
  - documents: 文件內容
  - metadatas: 結構化元資料
    ↓
ChromaDB 嵌入和儲存
    ↓
持久化到本地檔案系統
```

#### 查詢流程

```
查詢文字
    ↓
IChingVectorStore.query()
    ↓
ChromaDB 語義搜尋
    ↓
返回相關文件列表（按相似度排序）
```

#### 檔案結構

```
vector_store.py
├── IChingVectorStore
│   ├── __init__(self, persist_directory: str) -> None
│   ├── add_documents(self, documents: List[IChingDocument]) -> None
│   └── query(self, query_text: str, n_results: int) -> List[str]
└── build_vector_db() -> None
```

#### 依賴套件

- `chromadb>=0.4.0`: 向量資料庫
- `sentence-transformers>=2.2.0`: 嵌入模型

#### 使用範例

```python
# 建立向量資料庫
from vector_store import build_vector_db
build_vector_db()

# 或手動使用
from knowledge_loader import IChingKnowledgeLoader
from vector_store import IChingVectorStore

loader = IChingKnowledgeLoader()
documents = loader.load_documents()

vector_store = IChingVectorStore()
vector_store.add_documents(documents)

# 查詢
results = vector_store.query("What is the meaning of creativity?", n_results=3)
```

#### 後續應用

此向量資料庫將用於：
1. RAG 系統的語義搜尋
2. 易經解釋和查詢功能
3. 上下文增強生成
4. 知識檢索和推薦

---

## 2026-01-20 | Phase 2 - RAG 系統整合

### 步驟 10: 建立神諭對話模組 (`oracle_chat.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
建立完整的 RAG 系統，整合市場資料分析、易經卦象解讀和知識庫檢索，使用 Google Gemini API 提供智慧化的金融建議。

#### 實作細節

1. **Oracle 類別**
   - 整合所有組件：市場資料載入、編碼、解碼、向量檢索
   - 使用 Google Gemini API 生成智慧化回答
   - 提供完整的錯誤處理機制

2. **初始化方法 (`__init__`)**
   - 檢查並設定 Google API Key
   - 初始化所有處理組件（MarketDataLoader, MarketEncoder, IChingCore）
   - 初始化向量資料庫（IChingVectorStore）
   - 初始化 Gemini 模型（優先使用 gemini-1.5-flash）

3. **市場卦象獲取 (`_get_market_hexagram`)**
   - 載入股票市場資料
   - 編碼為易經卦象
   - 提取最新的卦象資訊
   - 返回卦象名稱、中文名稱、ID 等完整資訊

4. **易經智慧檢索 (`_get_iching_wisdom`)**
   - 構造語義查詢（結合卦象名稱和用戶問題）
   - 從向量資料庫檢索相關文本
   - 返回最相關的 3 個結果

5. **主要詢問方法 (`ask`)**
   - 整合所有步驟
   - 構造系統提示（包含市場資料、卦象、易經文本、用戶問題）
   - 使用 Gemini API 生成回答
   - 返回智慧化的金融建議

#### 技術決策

- **為什麼使用 Gemini API？**
  - Google 的 Gemini 模型支援多語言（包括中文）
  - 提供良好的 API 介面
  - 支援長文本生成
  - 適合整合 RAG 系統

- **為什麼優先使用 gemini-1.5-flash？**
  - 更快、更便宜的模型
  - 適合即時對話場景
  - 如果不可用，自動降級到 gemini-pro

- **為什麼結合卦象名稱和用戶問題進行檢索？**
  - 提高檢索相關性
  - 同時考慮卦象語境和用戶意圖
  - 返回更精準的易經文本

- **為什麼使用結構化的系統提示？**
  - 明確角色定位（FinTech Master）
  - 提供完整的上下文資訊
  - 指導生成風格（神秘而專業）
  - 確保回答的結構化和可讀性

#### 資料流程

```
用戶問題 + 股票代號
    ↓
Oracle.ask()
    ↓
_get_market_hexagram()
  - 載入市場資料
  - 編碼為卦象
  - 解碼卦象資訊
    ↓
_get_iching_wisdom()
  - 構造查詢
  - 向量資料庫檢索
    ↓
構造系統提示
  - 市場資料
  - 卦象資訊
  - 易經文本
  - 用戶問題
    ↓
Gemini API 生成回答
    ↓
返回智慧化建議
```

#### 系統提示結構

```
角色：FinTech Master（結合華爾街邏輯與易經智慧）

上下文：
- 股票代號
- 觀察到的卦象（從價格行為推導）
- 易經文本（從知識庫檢索）
- 用戶問題

指令：
- 在股票趨勢背景下解釋卦象
- 根據卦象含義提供建議（等待/謹慎 vs 力量/動量）
- 使用易經文本支持建議
- 保持神秘而專業的語調
```

#### 錯誤處理

- **API Key 檢查**：如果未設定，提供清晰的錯誤訊息和設定指引
- **資料獲取錯誤**：處理市場資料獲取失敗的情況
- **API 調用錯誤**：處理 Gemini API 調用失敗的情況
- **優雅降級**：如果 flash 模型不可用，自動使用 pro 模型

#### 檔案結構

```
oracle_chat.py
└── Oracle
    ├── __init__(self) -> None
    ├── _get_market_hexagram(self, ticker: str) -> dict
    ├── _get_iching_wisdom(self, hexagram_name: str, user_question: str) -> str
    └── ask(self, ticker: str, question: str) -> str
```

#### 依賴套件

- `google-generativeai>=0.3.0`: Google Gemini API
- `python-dotenv>=1.0.0`: 環境變數管理

#### 使用範例

```python
from oracle_chat import Oracle

# 初始化神諭系統
oracle = Oracle()

# 詢問
answer = oracle.ask("NVDA", "Should I buy now?")
print(answer)
```

#### 環境設定

需要在 `.env` 檔案中設定：
```
GOOGLE_API_KEY=your_api_key_here
```

#### 後續應用

此神諭系統將用於：
1. 智慧化金融建議
2. 易經卦象解釋
3. 市場趨勢分析
4. 投資決策支援

---

## 2026-01-20 | Phase 2 - Gemini 模型選擇修正

### 修正記錄: 更新 Gemini 模型名稱

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 發現的問題

**404 錯誤：模型不存在**
- 錯誤訊息: `404 models/gemini-1.5-flash is not found for API version v1beta`
- 原因: `gemini-1.5-flash` 模型在當前 API 版本中不可用
- 影響: 無法初始化 Gemini 模型，導致系統無法運行

#### 問題根源

Google Gemini API 的模型名稱和可用性會隨時間變化：
- 舊模型（如 `gemini-1.5-flash`）可能被新模型取代
- 不同 API Key 可能有不同的模型存取權限
- 需要使用當前可用的模型名稱

#### 修正方案

1. **自動模型選擇**
   - 按優先順序嘗試多個模型
   - 如果某個模型不可用，自動嘗試下一個
   - 提供清晰的錯誤訊息

2. **模型優先順序**
   ```python
   model_names = [
       "gemini-2.5-flash",      # 最新、最快、最便宜
       "gemini-pro-latest",      # 通用版本
       "gemini-2.5-pro",         # 更強大的版本
       "gemini-2.0-flash",      # 備用選項
   ]
   ```

3. **改進錯誤處理**
   - 在初始化時提供詳細的調試資訊
   - 在 API 調用失敗時提供具體的錯誤訊息
   - 區分模型不存在錯誤和其他 API 錯誤

#### 修正後的程式碼

```python
# 初始化 Gemini 模型
# 嘗試多個模型，按優先順序：gemini-2.5-flash > gemini-pro-latest > gemini-2.5-pro
model_names = [
    "gemini-2.5-flash",      # 最新、最快、最便宜
    "gemini-pro-latest",      # 通用版本
    "gemini-2.5-pro",         # 更強大的版本
    "gemini-2.0-flash",      # 備用選項
]

self.model = None
self.model_name = None

# 嘗試初始化模型（不進行實際 API 調用）
for model_name in model_names:
    try:
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        print(f"[INFO] Initialized Gemini model: {model_name}")
        break
    except Exception as e:
        # 如果模型不可用，嘗試下一個
        print(f"[DEBUG] Model {model_name} not available: {str(e)[:50]}")
        continue

if self.model is None:
    raise ValueError(
        "無法初始化任何 Gemini 模型。\n"
        "請檢查 API Key 是否正確，或稍後再試。\n"
        "嘗試的模型: " + ", ".join(model_names)
    )
```

#### 可用的模型

根據檢查結果，當前 API Key 可用的模型包括：
- `gemini-2.5-flash` ✅（推薦，最快、最便宜）
- `gemini-2.5-pro` ✅（更強大）
- `gemini-pro-latest` ✅（通用版本）
- `gemini-2.0-flash` ✅（備用選項）
- 以及其他 30+ 個模型

#### 學到的經驗

1. **模型名稱會變化**
   - Google 會定期更新模型名稱
   - 舊模型可能被新模型取代
   - 需要使用當前可用的模型名稱

2. **API Key 權限差異**
   - 不同 API Key 可能有不同的模型存取權限
   - 建議實作自動模型選擇機制
   - 提供多個備用選項

3. **錯誤處理最佳實踐**
   - 提供清晰的錯誤訊息
   - 區分不同類型的錯誤（模型不存在 vs API Key 錯誤）
   - 提供調試資訊幫助除錯

4. **向後相容性**
   - 實作多模型嘗試機制
   - 優先使用最新模型，但保留舊模型作為備用
   - 確保系統在不同環境下都能正常運行

#### 後續改進建議

1. **動態模型檢查**
   - 在初始化時動態檢查可用模型
   - 使用 `genai.list_models()` 獲取當前可用模型列表
   - 自動選擇最適合的模型

2. **模型效能監控**
   - 記錄不同模型的響應時間
   - 根據效能自動選擇最佳模型
   - 提供模型切換功能

3. **配置檔案支援**
   - 允許用戶在配置檔案中指定優先使用的模型
   - 提供模型選擇的靈活性

---

## 2026-01-20 | Phase 2 - 繁體中文輸出支援

### 修正記錄: 將系統提示改為繁體中文

**日期**: 2026-01-20  
**狀態**: ✅ 已修正

#### 變更內容

**系統提示語言切換**
- 將所有系統提示從英文改為繁體中文
- 明確要求 Gemini 使用繁體中文回答
- 保持專業而神秘的語調

#### 修正後的系統提示

```python
system_prompt = f"""你是一位結合華爾街邏輯與易經智慧的金融大師。

**上下文：**
- 股票代號：{ticker}
- 觀察到的卦象：{hexagram_name}（{chinese_name}，編號：{hexagram_id}）
- 易經文本：{retrieved_context}
- 用戶問題：{question}

**指示：**
請在股票趨勢的背景下解釋這個卦象。
- 如果卦象暗示「等待」或謹慎，建議保持耐心和仔細觀察。
- 如果暗示「力量」或動量，建議考慮進場或持有。
- 使用檢索到的易經文本來支持你的建議，引用古代智慧。
- 使用神秘而專業的語調，融合傳統易經哲學與現代金融分析。

**重要：請全程使用繁體中文回答，不要使用英文。**
```

#### 效果

- ✅ 所有回答現在都是繁體中文
- ✅ 保持專業而神秘的語調
- ✅ 更符合中文用戶的使用習慣
- ✅ 易經術語和概念更易理解

---

## 2026-01-20 | Phase 2 - 現代化輸出格式

### 改進記錄: 結構化、現代化的金融分析輸出

**日期**: 2026-01-20  
**狀態**: ✅ 已完成

#### 改進目標

將輸出從過於古風的風格改為現代化的金融分析格式，同時保留易經原文作為參考。

#### 改進內容

1. **語調調整**
   - 從古風文言文改為現代專業金融分析師語調
   - 避免使用「吾」、「汝」、「此乃」等古語
   - 採用 Bloomberg 分析師風格，同時具備易經學者背景

2. **結構化輸出格式**
   - **🚀 投資快訊 (Executive Summary)**: 一句話總結
   - **📜 易經原文 (The Source)**: 引用最相關的 1-2 句易經文本
   - **💡 現代解讀 (Modern Decoding)**: 將易經隱喻轉換為金融術語
   - **📈 操作建議 (Action Plan)**: 具體的操作建議

3. **現代金融術語映射**
   - 「井」卦 → 基礎設施、深度價值、股息、累積
   - 「乾」卦 → 高動量、突破、超買
   - 「需」卦 → 整理、耐心、等待催化劑

#### 新的系統提示結構

```python
system_prompt = f"""You are a sophisticated AI Financial Advisor named 'Quantum I-Ching'.
Your goal is to interpret ancient I-Ching hexagrams into **actionable modern stock market insights**.

**Response Guidelines:**
1. **Tone**: Professional, crisp, and modern. Like a Bloomberg analyst who happens to be an I-Ching scholar.
2. **Structure**:
    * **🚀 投資快訊**: 1-sentence bottom line
    * **📜 易經原文**: Quote relevant I-Ching text
    * **💡 現代解讀**: Translate metaphor to financial terms
    * **📈 操作建議**: Concrete actionable advice
"""
```

#### 輸出範例格式

```markdown
## 🚀 投資快訊
短期整理，長期看多。

## 📜 易經原文
"The Well provides sustenance for all, yet if one fails to draw, or draws from a spoiled well, what profit is there?"

## 💡 現代解讀
井卦象徵深度價值和基礎設施。對於 NVDA，這代表其核心技術實力和在未來基礎設施中的關鍵角色...

## 📈 操作建議
- 建議採用定期定額策略
- 設定止損於 $XXX
- 等待成交量放大確認突破
```

#### 技術決策

- **為什麼使用英文提示？**
  - 英文提示對 LLM 更清晰明確
  - 可以更精確地控制輸出格式和風格
  - 同時要求輸出使用繁體中文

- **為什麼保留易經原文？**
  - 提供傳統智慧的權威性
  - 讓用戶了解建議的來源
  - 保持易經的神秘特質

- **為什麼使用結構化格式？**
  - 提高可讀性
  - 方便 CLI 和 Web UI 顯示
  - 便於後續解析和處理

#### 效果

- ✅ 輸出更現代化、專業化
- ✅ 結構清晰，易於閱讀
- ✅ 保留易經原文作為參考
- ✅ 提供具體可操作建議
- ✅ 適合現代金融分析場景

---

## 2026-01-20 | Phase 3 - 深度學習資料處理

### 步驟 11: 多市場支援與資料處理器

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標
進入 Phase 3 (Deep Learning)，建立支援多市場（美股、台股、加密貨幣）的資料處理管道，並實作適合 LSTM 訓練的資料處理器。

#### 實作細節

1. **更新 `config.py`**
   - 新增 `MARKET_TYPE` 設定：'US'（美股）、'TW'（台股）、'CRYPTO'（加密貨幣）
   - 預設為 'US'
   - 在 `TARGET_TICKERS` 註解中提供不同市場的範例

2. **更新 `data_loader.py`**
   - 新增 `_format_ticker()` 方法，根據市場類型格式化 ticker
   - 台股（TW）：自動添加 `.TW` 後綴
   - 加密貨幣（CRYPTO）：自動添加 `-USD` 後綴
   - 美股（US）：直接使用原始 ticker
   - 在 `fetch_data()` 中整合 ticker 格式化邏輯

3. **建立 `data_processor.py`**
   - `QuantumDataset`: PyTorch Dataset 類別，提供 (features, hexagram_id, label) 三元組
   - `DataProcessor`: 資料處理器類別
     * `prepare_data()`: 準備訓練資料
       - 特徵選擇：'Close', 'Volume', 'RVOL', 'Daily_Return'
       - 卦象 ID 提取：從 'Hexagram_Binary' 或 'Ritual_Sequence' 計算
       - 標籤創建：二分類（上漲=1，下跌=0）
       - 標準化：使用 StandardScaler 標準化數值特徵
       - 序列生成：滑動窗口生成時間序列
     * `split_data()`: 時間序列分割（不隨機打亂）
       - 前 80% 用於訓練，後 20% 用於測試
       - 返回 PyTorch DataLoader

#### 技術決策

- **為什麼支援多市場？**
  - 擴大資料來源，提高模型泛化能力
  - 不同市場有不同的交易特性
  - 提供更靈活的資料獲取選項

- **為什麼使用時間序列分割而非隨機分割？**
  - 時間序列資料具有時間依賴性
  - 隨機分割會造成資料洩漏（未來資訊洩漏到過去）
  - 更符合實際預測場景

- **為什麼將卦象 ID 轉換為 0-indexed？**
  - PyTorch 的 Embedding 層需要 0-indexed 的索引
  - 原始 ID 是 1-64，轉換為 0-63
  - 如果 ID 為 0（未知），保持為 0

- **為什麼只標準化數值特徵？**
  - 卦象 ID 是類別變數，不應標準化
  - 標準化會破壞類別變數的語義
  - 卦象 ID 將用於 Embedding 層

#### 資料流程

```
原始市場資料 (OHLCV)
    ↓
MarketEncoder.generate_hexagrams()
    ↓
包含卦象的 DataFrame
  - Close, Volume, RVOL, Daily_Return
  - Hexagram_Binary / Ritual_Sequence
    ↓
DataProcessor.prepare_data()
    ↓
提取特徵和標籤
  - 標準化數值特徵
  - 計算卦象 ID（0-indexed）
  - 創建二分類標籤
    ↓
生成序列（滑動窗口）
  - X_num: (N, seq_len, num_features)
  - X_hex: (N, seq_len)
  - y: (N, 1)
    ↓
DataProcessor.split_data()
    ↓
PyTorch DataLoader
  - train_loader
  - test_loader
```

#### 檔案結構

```
data_processor.py
├── QuantumDataset (Dataset)
│   ├── __init__(self, X_num, X_hex, y) -> None
│   ├── __len__(self) -> int
│   └── __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]
└── DataProcessor
    ├── __init__(self, sequence_length, prediction_window) -> None
    ├── prepare_data(self, df) -> Tuple[ndarray, ndarray, ndarray]
    └── split_data(self, X_num, X_hex, y, train_split) -> Tuple[DataLoader, DataLoader]
```

#### 資料格式說明

**輸入 DataFrame 必須包含**：
- 數值特徵：'Close', 'Volume', 'RVOL', 'Daily_Return'
- 卦象資訊：'Hexagram_Binary' 或 'Ritual_Sequence'

**輸出格式**：
- `X_num`: (N, sequence_length, 4) - 標準化後的數值特徵
- `X_hex`: (N, sequence_length) - 卦象 ID（0-63）
- `y`: (N, 1) - 二分類標籤（0=下跌, 1=上漲）

#### 使用範例

```python
from data_loader import MarketDataLoader
from market_encoder import MarketEncoder
from data_processor import DataProcessor

# 載入和編碼資料
loader = MarketDataLoader()
raw_data = loader.fetch_data(tickers=["NVDA"])
encoder = MarketEncoder()
encoded_data = encoder.generate_hexagrams(raw_data)

# 處理資料
processor = DataProcessor(sequence_length=10, prediction_window=1)
X_num, X_hex, y = processor.prepare_data(encoded_data)

# 分割資料
train_loader, test_loader = processor.split_data(X_num, X_hex, y)
```

#### 依賴套件

- `torch>=2.0.0`: PyTorch 深度學習框架
- `scikit-learn>=1.3.0`: 標準化工具

#### 後續應用

此資料處理器將用於：
1. LSTM 模型訓練
2. 時間序列預測
3. 多市場資料處理
4. 深度學習實驗

---

## 2026-01-20 | Phase 3 - LSTM Model

* **File**: `model_lstm.py`
* **Architecture**: Hybrid LSTM (Embedding + Numerical).
* **Inputs**: 
    * Hexagram IDs (Embedding Dim=8)
    * Technical Indicators (Close, Vol, RVOL, Return)
* **Objective**: Binary Classification (Up/Down probability).

---

## 2026-01-20 | Phase 3 - Backtesting Engine

* **File**: `backtester.py`
* **Action**: Implemented LSTM-based backtesting engine on unseen test data.
* **Logic**:
    * Use `DataProcessor` to prepare sequences and split train/test (time-series split).
    * Align each test sequence with the corresponding date and close price.
    * Run `QuantumLSTM` in `eval()` mode to generate probabilities and trading signals.
    * Signals are shifted by one day: signal at t controls position at t+1 (no look-ahead bias).
    * Compute `Market_Return`, `Strategy_Return`, and cumulative curves.
    * Save equity curve plot to `data/backtest_result.png` and print total return and win rate.

### 小修正: MarketDataLoader ticker 格式化方法

* **File**: `data_loader.py`
* **Action**: 補上遺漏的 `_format_ticker()` 方法，避免 `AttributeError`。
* **Logic**:
    * 若 `MARKET_TYPE == "TW"`，自動補上 `.TW`。
    * 若 `MARKET_TYPE == "CRYPTO"`，自動補上 `-USD`。
    * 其他情況（含 `US`）：直接回傳原始 ticker。
* **Reason**: 在執行 `python model_lstm.py` 時，`MarketDataLoader.fetch_data()` 會呼叫 `_format_ticker()`；先前方法定義遺漏導致 `AttributeError`，已修正。

---

## 2026-01-20 | Phase 3 - LSTM Training UX 強化

### tqdm 進度條與 Early Stopping

* **File**: `model_lstm.py`
* **Action**: 強化訓練體驗與穩定性。
* **Updates**:
  * 新增 `tqdm` 進度條顯示：
    * 在 `train()` 中對 `train_loader` 與 `val_loader` 使用 `tqdm`，即時顯示 batch 損失。
    * 方便觀察每個 epoch 的訓練/驗證收斂情況。
  * 新增 Early Stopping 機制：
    * 在 `QuantumTrainer.__init__` 中加入 `patience` 與 `min_delta` 參數。
    * 若驗證損失在連續 `patience` 個 epoch 內未改善超過 `min_delta`，提前停止訓練。
    * 避免過度訓練與浪費時間。
  * 新增訓練超參數列印：
    * 在 `train()` 開頭印出 `device`, `learning_rate`, `patience`, `min_delta`, `epochs`。
    * 方便日後回顧與實驗重現。
  * 最佳模型儲存邏輯更新：
    * 僅當 `avg_val_loss < best_val_loss - min_delta` 時才更新最佳模型與重置 early stopping 計數。

### `__main__` 執行入口

* **Action**: 讓 `python model_lstm.py` 可以直接啟動訓練流程。
* **Logic**:
  * 使用 `settings.TARGET_TICKERS[0]` 作為預設標的（若為空則使用 `"NVDA"`）。
  * 使用 `MarketDataLoader` + `MarketEncoder` 生成帶卦象的市場資料。
  * 使用 `DataProcessor(sequence_length=10, prediction_window=1)` 準備訓練/驗證資料。
  * 建立 `QuantumLSTM` 與 `QuantumTrainer`，執行 `train()` 並印出最後一個 epoch 的 Train / Val loss。
* **Result**:
  * 使用者現在只需執行：
    ```bash
    python model_lstm.py
    ```
    即可看到每個 epoch 的：
    * 進度條（Train / Val）
    * `Train Loss` / `Val Loss`
    * Early Stopping 訊息與最佳模型儲存提示。

---

## 2026-01-20 | Phase 4 - Streamlit Dashboard 與前端除錯

### 步驟 12: 建立 Web 儀表板介面 (`dashboard.py`)

**日期**: 2026-01-20  
**狀態**: ✅ 完成

#### 設計目標

- 提供一個 **淺色、專業金融風格** 的 Streamlit 前端，讓使用者可以：
  - 輸入股票代號（以台股為主，同時支援美股／加密貨幣）。
  - 看到最近 60 日 K 線圖。
  - 查看對應的 I-Ching 市場卦象（六爻視覺化）。
  - 取得由 `Oracle`（Gemini + RAG）產生的卜卦解讀。
- 保持良好效能：避免每次互動都重新載入向量資料庫與 Gemini 模型。

#### 主要實作重點

1. **頁面結構與樣式**
   - 使用 `st.set_page_config(layout="wide", page_title="Quantum I-Ching")`。
   - 採用淺色金融風格：
     - 背景 `#f0f2f6`，主要卡片白底＋圓角＋陰影。
     - 深灰文字 `#333333`，提高可讀性。
   - 左側 `Sidebar`：
     - 標題：`🔮 設定 (Settings)`。
     - `user_ticker`：股票代號輸入欄（台股可直接輸入如 `2330`、`2317`）。
     - `question`：問題輸入欄（預設：「Should I buy now? / 我現在該買嗎？」）。
     - 執行按鈕：`Consult the Oracle (卜卦)`。
     - 說明與免責：強調目前以台股／美股為主，內容僅供研究與教育參考，非投資建議。
   - 主畫面：
     - 標題：`Quantum I-Ching 股市卜卦系統`。
     - `col_chart, col_hex = st.columns([2, 1])`：
       - 左側 2/3：K 線圖。
       - 右側 1/3：I-Ching 市場卦象卡片。

2. **Oracle 資源快取**
   - 使用 `@st.cache_resource` 包裝 `get_oracle()`：
     - 僅在第一次載入時初始化 `Oracle`，後續互動共用同一個實例。
     - 避免每次重新載入 Chroma 向量庫與 Gemini 模型。

3. **台股代碼支援與股票名稱顯示**
   - 前端處理 ticker：
     - 若 `user_ticker.isdigit()` → 自動補上 `.TW`（例如 `2330` → `2330.TW`），作為 `backend_ticker`。
     - 其他格式（已含 `.` 或 `-`）直接視為完整代碼。
   - 資料抓取：
     - `raw_df = oracle.data_loader.fetch_data(tickers=[backend_ticker])`。
   - 股票名稱顯示：
     - 使用 `oracle.data_loader._format_ticker(backend_ticker)` 搭配 `yf.Ticker(...).info` 取得 `shortName` / `longName`。
     - K 線圖標題格式：
       - 若有名稱：`{user_ticker} ({stock_name}) - {chinese_name} / {hexagram_name} (最近 60 日價格走勢)`。
       - 否則：`{user_ticker} - {chinese_name} / {hexagram_name} (最近 60 日價格走勢)`。

4. **卦象生成與視覺化**
   - 利用後端現有流程：
     - `encoded_df = oracle.encoder.generate_hexagrams(raw_df)`。
     - 檢查 `Ritual_Sequence` / `Hexagram_Binary` 是否存在且非空。
     - 將 `Ritual_Sequence` 轉為 `List[int]`，長度必須等於 6。
     - 透過 `oracle.core.interpret_sequence(ritual_sequence)` 取得 `current_hex`。
   - 視覺化函式 `draw_hexagram(ritual_seq, binary_string, name)`：
     - 以 HTML/CSS 畫出 6 條爻線，上爻在上、初爻在下。
     - 陽爻（1）：深藍實線 `#004e92`。
     - 陰爻（0）：左右兩段紅橘實線 `#d9534f`，中間留白。
     - 左側標出「上爻、五爻、四爻、三爻、二爻、初爻」。
     - 底部顯示 `Ritual` / `Binary` / `Hexagram` 的 meta 訊息。

5. **K 線圖**
   - 採樣最近 60 日資料：`chart_df = raw_df.tail(60)`。
   - 使用 Plotly `go.Candlestick`：
     - 模板 `template="plotly_white"`。
     - 背景 `paper_bgcolor="#ffffff"`、`plot_bgcolor="#ffffff"`。
     - 陽線綠色 `#22c55e`、陰線紅色 `#ef4444`。
     - 關閉 range slider，`use_container_width=True` 提供自適應寬度。

6. **AI 卜卦解讀區塊**
   - 放置在整個兩欄佈局下方，靠近 K 線圖之後，標題為：
     - `### 🧠 Oracle's Advice / 卜卦解讀`
   - 內容：
     - 呼叫 `ai_answer = oracle.ask(backend_ticker, question or "Should I buy now?")`。
     - 使用 Streamlit 內建 `st.info(ai_answer)`，讓整段 Markdown 文字（含各小標）被單一框線完整包覆。
     - 使用 `st.caption(...)` 顯示免責聲明（僅供參考，非投資建議）。

#### 前端除錯紀錄

1. **暗色 Cyberpunk UI 不易閱讀**
   - 問題：初版使用深色背景＋霓虹風格，字體對比不足、使用者覺得「醜」、「難讀」。
   - 修正：
     - 調整為淺色主題，背景灰＋白底卡片。
     - 使用較大字級與深灰文字色，提高閱讀性。

2. **台股代碼無法抓到資料**
   - 問題：`MARKET_TYPE` 預設為 `"US"`，`_format_ticker("2330")` 回傳 `"2330"`，yfinance 對美股代碼 `2330` 無資料。
   - 修正：
     - 在 `config.py` 將 `MARKET_TYPE` 改為 `"TW"`，`TARGET_TICKERS` 改為台股範例。
     - 另外在前端強制數字代碼自動補 `.TW`，即使後端市場設定未調整，仍可正確抓到台股資料。

3. **NameError: `ticker` 未定義**
   - 問題：在圖表標題中使用了不存在的變數 `ticker`。
   - 修正：
     - 統一使用 `user_ticker`（顯示用）與 `backend_ticker`（實際查詢用），並在圖表標題中改成 `user_ticker`。

4. **卜卦解讀區塊位置與外框問題**
   - 問題 1：`Oracle's Advice` 一開始與卦象放在同欄，導致版面擁擠、捲動體驗差。
   - 問題 2：自訂 HTML 卡片外框（`.oracle-advice`）曾出現只包到標題、未完全包覆所有文字的視覺問題。
   - 修正：
     - 將 AI 解讀移到 K 線圖下方，整行寬度一致，閱讀動線清楚。
     - 改用 `st.info(ai_answer)` 取代客製 HTML 容器，確保整段文字（含 Markdown）自動被完整框線包對齊。

5. **TW 股票名稱顯示與卦象名稱混淆**
   - 說明：
     - K 線圖與左側標題顯示的是 **股票名稱**（例如 `3661.TW (ALCHIP TECHNOLOGIES LIMITD)`，即世芯-KY）。
     - `I-Ching 市場卦象` 內的「卦名：中孚 (Zhong Fu, ID: 61)」是根據最近 6 日市場結構算出的 **易經卦象名稱**，並非公司名稱。
   - 處理：
     - 在介面文案與變數命名上，明確區分「股票名稱」與「卦象名稱」，避免使用者混淆。

6. **Streamlit 互動中斷／未顯示結果的誤會**
   - 問題：使用者以為下方沒有任何結果，其實 AI 卜卦解讀在頁面下方，但一開始版面設計需要額外捲動才看得到。
   - 修正：
     - 重新安排行為順序與版面，確保：
       - 按下「卜卦」後，K 線圖、卦象與 AI 解讀都在第一屏或輕微捲動即可完全看到。

---

## 2025-01-23 | 之卦 (Zhi Gua) 傳統解法與貞／悔架構

### 步驟：Oracle 之卦策略與動爻邏輯

**日期**: 2025-01-23  
**狀態**: ✅ 完成

#### 設計目標

在 `oracle_chat.py` 的 `Oracle` 中實作傳統「之卦」解法：依動爻（6、9）數量選擇不同解釋策略，並引入 **貞 (Zhen) / 悔 (Hui)** 架構，供 Gemini 產出具主客、支撐／阻力、進出場意涵的金融建議。

#### 實作細節

1. **`_get_future_hexagram_name(ritual_sequence)`**
   - 輸入：`ritual_sequence`（`List[int]`，如 `[8,7,9,6,8,8]`）。
   - 規則：6→7（老陰→少陽）、9→8（老陽→少陰），再以奇=1、偶=0 轉二進位，以 `IChingCore.get_hexagram_name`（底層 `HEXAGRAM_MAP`）查之卦名。
   - 輸出：之卦卦名（字串）。

2. **`_resolve_strategy(current_hex_name, ritual_sequence)`**
   - 統計動爻數量（值為 6 或 9）。
   - 依數量回傳 `(strategy_context, search_queries, future_hex_name)`：
     * **0 動爻**：Total Acceptance，查本卦 Judgement / Image。
     * **1 動爻**：Specific Focus，僅查該動爻（如 `Hexagram X Line Y`）。
     * **2 動爻**：Primary vs Secondary，下爻貞（進場／支撐）、上爻悔（出場／阻力），查兩爻。
     * **3 動爻**：Hedging Moment，本卦貞（持有）、之卦悔（風險），查本卦＋之卦 Judgement。
     * **4 或 5 動爻**：Trend Reversal，之卦貞（主趨勢）、本卦悔（歷史），查之卦＋本卦 Judgement。
     * **6 動爻**：Extreme Reversal；若本卦為乾用「Use Nine」、坤用「Use Six」，否則用之卦 Judgement。

3. **`_get_iching_wisdom(search_queries, user_question)` 重構**
   - 改為接受 `List[str]` 的 `search_queries`，依策略產生的查詢逐筆向向量庫檢索，合併並去重後回傳。

4. **`ask(ticker, question)` 更新**
   - 從 `_get_market_hexagram` 取得 `ritual_sequence`。
   - 呼叫 `_resolve_strategy` 取得情境、查詢列表、之卦名。
   - 以 `search_queries` 呼叫 `_get_iching_wisdom`（不再僅用本卦名）。
   - 系統提示中注入：之卦策略情境、**貞 (Zhen) = 主／支撐／長期／進場／持有**、**悔 (Hui) = 客／阻力／短期／出場／風險**，並要求 Gemini 在投資快訊、現代解讀、操作建議中依貞／悔給出結構化建議（例如貞—支撐與持有；悔—止損與減碼）。

#### 邏輯重點

- **貞 (Zhen) vs 悔 (Hui)**：用於動爻 2、3、4、5、6 等需主客或趨勢／風險對照的情境，在提示中明確定義，並要求模型在操作建議中對應到支撐、趨勢、進場（貞）與阻力、風險、出場（悔）。

---

## 2025-01-23 | 易經知識庫：open-iching 來源與簡體轉繁體

### 步驟：資料來源切換、簡繁轉換、向量庫重建

**日期**: 2025-01-23  
**狀態**: ✅ 完成

#### 設計目標

1. 將易經資料來源由失效的 `kwanlin/iching-json` 改為 [john-walks-slow/open-iching](https://github.com/john-walks-slow/open-iching)，取得可用的 64 卦與六爻結構化資料。  
2. 把 `data/iching_complete.json` 內簡體中文轉成繁體，並同步重建 ChromaDB 向量庫。

#### 實作細節

1. **`setup_iching_db.py` 改用 open-iching**
   - **iching**：`https://cdn.jsdelivr.net/gh/john-walks-slow/open-iching@main/iching/iching.json`
   - **象傳（可選）**：`ichuan/xiang.json`，若取得則填入大象／小象，否則為空。
   - **驗證**：總數 64、第 1 卦為「乾」、`lines` 非空且含 `scripture` 或 `name`。
   - **轉換**：將 open-iching 格式（`id`, `name`, `scripture`, `lines` 之 `id`/`name`/`scripture`）對應為統一格式：`number`, `name`, `judgment`, `image`, `lines[{position, meaning, xiang}]`，爻辭為 `{name}：{scripture}`（如「初九：潛龍，勿用。」）。
   - **輸出**：`data/iching_complete.json`（utf-8）。成功時印出：`[OK] Verified I-Ching Data (64 Hexagrams + Lines) saved.`

2. **新增 `convert_iching_s2t.py`**
   - 使用 **OpenCC**（`s2tw`）將 `data/iching_complete.json` 中字串由簡體轉為臺灣繁體。
   - 轉換欄位：`name`, `judgment`, `image`，以及各爻之 `meaning`, `xiang`。
   - 若未安裝 `opencc-python-reimplemented`，腳本會先 `pip install` 再執行。
   - 轉完後呼叫 `IChingKnowledgeLoader().build_vector_db()` 重建 ChromaDB。
   - 執行：`python convert_iching_s2t.py`。

3. **`knowledge_loader.py`**
   - 註解更新：說明 `iching_complete.json` 來自 `setup_iching_db` 自 john-walks-slow/open-iching 下載並轉成之統一格式。
   - 建庫成功時改印 `[OK] Knowledge Base Rebuilt: X documents indexed.`，避免 Windows 終端 cp950 下 `UnicodeEncodeError`（如 ✅）。

#### 簡繁轉換範例

| 簡體 | 繁體 |
|------|------|
| 元亨利贞。 | 元亨利貞。 |
| 潜龙，勿用。 | 潛龍，勿用。 |
| 见龙再田，利见大人 | 見龍再田，利見大人 |
| 飞龙在天 | 飛龍在天 |
| 龙战于野，其血玄黄 | 龍戰於野，其血玄黃 |
| 利牝马之贞 | 利牝馬之貞 |

#### 執行結果

- `python setup_iching_db.py`：下載、驗證、轉換、寫入成功。
- `python convert_iching_s2t.py`：簡轉繁完成，ChromaDB 重建 **450** 份文件（64 主卦 ＋ 386 爻，含乾卦「用九」、坤卦「用六」）。

---

## 2025-01-23 | Dashboard 市場類型選擇器與 US/CRYPTO 資料讀取修正

### 步驟：Dashboard 市場類型切換與 fetch_data market_type 參數

**日期**: 2025-01-23  
**狀態**: ✅ 完成

#### 問題描述

1. **Dashboard 缺少市場類型選擇**：使用者無法在介面上切換 TW/US/CRYPTO，需修改 `config.py`。
2. **US/CRYPTO 市場無法讀取資料**：當 `settings.MARKET_TYPE` 為 "TW"（預設）時，`fetch_data` 內部會用 TW 邏輯格式化所有 ticker，導致 US 股票（如 "NVDA"）被錯誤處理。

#### 問題根源

- `MarketDataLoader.fetch_data()` 內部使用 `settings.MARKET_TYPE` 來格式化 ticker。
- Dashboard 已根據使用者選擇格式化 ticker（如 "NVDA"），但 `fetch_data` 會再次使用 `settings.MARKET_TYPE`（可能為 "TW"）處理，造成不一致。
- 當 `settings.MARKET_TYPE="TW"` 時，`_format_ticker("NVDA")` 不會改變（US 邏輯直接返回），但日誌與邏輯可能混淆。

#### 修正方案

1. **`data_loader.py`**
   - `fetch_data()` 新增可選參數 `market_type: Optional[str] = None`。
   - 若未提供，使用 `settings.MARKET_TYPE`（向後相容）。
   - 將 `market_type` 傳給 `_format_ticker()`，確保格式化邏輯一致。
   - 日誌訊息改為顯示實際使用的 `market_type`。

2. **`dashboard.py`**
   - 側邊欄新增 `st.selectbox` 市場類型選擇器（TW / US / CRYPTO），預設 TW。
   - 根據使用者選擇的市場類型動態格式化 ticker：
     * TW：純數字補 `.TW`（如 `2330` → `2330.TW`）
     * US：直接使用（如 `NVDA`）
     * CRYPTO：補 `-USD`（如 `BTC` → `BTC-USD`）
   - 預設 ticker 依市場類型自動調整（TW→2330, US→NVDA, CRYPTO→BTC）。
   - 呼叫 `fetch_data()` 時傳入 `market_type=market_type` 參數。
   - 更新說明文字，反映支援三種市場類型。

#### 測試結果

- ✅ US 市場：`fetch_data(tickers=['NVDA'], market_type='US')` → 成功下載 1522 筆資料。
- ✅ CRYPTO 市場：`fetch_data(tickers=['BTC'], market_type='CRYPTO')` → 成功下載 2213 筆資料（自動轉為 `BTC-USD`）。
- ✅ Dashboard 介面：使用者可切換市場類型，ticker 自動格式化，資料讀取正常。

#### 向後相容性

- `fetch_data()` 的 `market_type` 參數為可選，未提供時使用 `settings.MARKET_TYPE`。
- 現有 CLI 工具（`oracle_chat.py`, `model_lstm.py`, `backtester.py`, `main.py`）無需修改，仍使用 `settings.MARKET_TYPE`。

#### 後續修正：oracle.ask() 也需要 market_type

**問題**：Dashboard 中呼叫 `oracle.ask()` 時，`ask()` 內部會再次呼叫 `_get_market_hexagram()`，而該方法會呼叫 `fetch_data()` 但未傳入 `market_type`，導致使用 `settings.MARKET_TYPE`（預設 "TW"），US/CRYPTO 市場無法正確讀取。

**修正**：
- `Oracle._get_market_hexagram()` 新增 `market_type: Optional[str] = None` 參數。
- `Oracle.ask()` 新增 `market_type: Optional[str] = None` 參數，並傳給 `_get_market_hexagram()`。
- `dashboard.py` 呼叫 `oracle.ask()` 時傳入 `market_type=market_type`。

**測試**：`_get_market_hexagram('NVDA', market_type='US')` → 成功取得卦象（夬 / Guai）。

#### 後續修正：卦象一致性確保

**問題**：Dashboard 中上方顯示的卦象與下方 `oracle.ask()` 解讀可能不一致，因為：
1. 上方：從 `raw_df` → `encoded_df` → `ritual_sequence` → `interpret_sequence()` 計算卦象。
2. 下方：`oracle.ask()` → `_get_market_hexagram()` → 再次下載資料並計算卦象。

即使使用相同的 ticker 和 market_type，兩次計算可能因時間差或資料更新導致卦象不同。

**修正**：
- `Oracle.ask()` 新增 `hexagram_info: Optional[dict] = None` 參數，可接受已計算的卦象資訊（包含 `hexagram_name`, `chinese_name`, `hexagram_id`, `ritual_sequence`）。
- 若提供 `hexagram_info`，`ask()` 跳過 `_get_market_hexagram()`，直接使用提供的資訊。
- `dashboard.py` 中，將已計算好的卦象資訊（`hexagram_name_full`, `chinese_name`, `hexagram_id`, `ritual_sequence`）傳給 `oracle.ask()`。
- 確保 `hexagram_name` 處理一致（移除括號），與 `_get_market_hexagram()` 邏輯相同。

**效果**：
- ✅ 上方顯示與下方解讀使用**完全相同**的卦象（同一個 `ritual_sequence`）。
- ✅ 避免重複下載資料與計算，提升效能。
- ✅ Terminal 測試：`ask('NVDA', 'Should I buy?', market_type='US', hexagram_info=h)` → 成功。

---

## 2025-01-23 | Dashboard 本卦與之卦並排顯示功能

### 步驟：卦象視覺化增強 - 顯示本卦與之卦

**日期**: 2025-01-23  
**狀態**: ✅ 完成

#### 設計目標

當卦象中有動爻（Old Yin 6 或 Old Yang 9）時，Dashboard 應同時顯示本卦（Current Hexagram）和之卦（Future Hexagram），讓使用者清楚看到變動的方向。

#### 實作細節

1. **新增 CSS 樣式**
   - `.hex-line.moving`：動爻高亮樣式，橙色邊框（`#ff9800`）與脈衝動畫效果。
   - `.hexagram-container`：卦象容器樣式，用於並排顯示時的佈局。
   - `.hexagram-arrow`：箭頭樣式，用於顯示本卦 → 之卦的轉換方向。

2. **增強 `draw_hexagram()` 函數**
   - 新增 `moving_lines: list[int] | None` 參數：標記動爻位置（1-based，例如 [1, 3] 表示初爻和三爻）。
   - 新增 `show_title: bool` 參數：控制是否顯示標題和元資料（並排顯示時隱藏）。
   - 動爻會自動添加 `moving` CSS 類別，顯示橙色邊框與脈衝動畫。

3. **新增 `calculate_future_binary()` 函數**
   - 計算之卦的二進制編碼。
   - 規則：
     * 6 (老陰) → 1 (陽)
     * 9 (老陽) → 0 (陰)
     * 7 (少陽) → 1 (陽，不變)
     * 8 (少陰) → 0 (陰，不變)

4. **更新卦象顯示邏輯（`dashboard.py`）**
   - **無動爻**：僅顯示本卦（保持原有行為）。
   - **有動爻**：並排顯示本卦 → 之卦：
     * 使用 `st.columns([1, 0.2, 1])` 三欄佈局。
     * 左欄：本卦（顯示動爻高亮）。
     * 中欄：➡️ 箭頭指示變動方向。
     * 右欄：之卦（顯示變動後的卦象）。
     * 底部：動爻說明（例如「動爻：初爻、三爻 (2 個)」）。

5. **之卦名稱獲取**
   - 使用 `oracle.core.get_hexagram_name(future_binary)` 取得之卦的中文名稱和英文名稱。
   - 處理名稱格式（移除括號）與本卦一致。

#### 視覺效果

- **無動爻**：單一卦象顯示（與之前相同）。
- **有動爻**：
  - 左側：本卦（動爻有橙色高亮與脈衝動畫）。
  - 中間：➡️ 箭頭（垂直置中）。
  - 右側：之卦（變動後的結果）。
  - 底部：動爻說明文字。

#### 技術決策

- **為什麼動爻使用橙色高亮？**
  - 橙色（`#ff9800`）在淺色主題中清晰可見，且與金融警示色調一致。
  - 脈衝動畫（`pulse-moving`）能吸引使用者注意變動的爻位。

- **為什麼使用三欄佈局？**
  - `[1, 0.2, 1]` 比例確保本卦和之卦等寬，箭頭欄位較窄，整體平衡。
  - 箭頭垂直置中，視覺上連接兩個卦象。

- **為什麼之卦不顯示動爻標記？**
  - 之卦是變動後的結果，動爻標記僅用於本卦，標示哪些爻位正在變動。

#### 測試結果

- ✅ 無動爻：單一卦象顯示正常。
- ✅ 有動爻：本卦與之卦並排顯示，動爻高亮正常。
- ✅ 動爻說明：正確顯示動爻位置（如「動爻：初爻、三爻 (2 個)」）。
- ✅ 之卦名稱：正確取得並顯示中文和英文名稱。

#### 向後相容性

- 無動爻時保持原有顯示方式，不影響現有功能。
- 所有 CSS 樣式與現有淺色主題一致。

---

## 2025-01-23 | 卦象「只算一次」與易經原文一致性修正

### 步驟：Calculate Once, Use Everywhere（Dashboard ↔ Oracle）

**日期**: 2025-01-23  
**狀態**: ✅ 完成

#### 問題描述

1. Dashboard 前端會自己算一次卦象（`MarketEncoder.generate_hexagrams`），用來畫本卦／之卦視覺化。
2. `Oracle.ask()` 內部又會重新抓一次 yfinance 資料、重新跑 encoder 再算一次卦象。
3. 若資料更新點落在兩次計算之間，或設定略有差異，可能導致：
   - 上方圖形顯示的是某一卦（如「中孚 → 頤」），
   - 下方 AI 解讀卻是另一卦或對應錯誤（例如易經原文出現「師卦」）。

#### 設計目標

- **卦象只算一次**：由 Dashboard 取得完整市場資料並產生卦象，再把同一份狀態物件傳給 `Oracle`。
- **前後端卦象完全一致**：畫面上的本卦／之卦與 `Oracle.ask()` 內部使用的 `ritual_sequence`、本卦 ID、之卦 ID 完全相同。
- **易經原文嚴格對應**：所有引用的經文（本卦＋之卦＋動爻）必須 100% 來自與畫面相同的卦與爻位。

#### 實作細節

1. **`oracle_chat.py` – `Oracle.ask()` 接受預計算市場狀態**
   - 函式簽名改為：
     ```python
     def ask(
         self,
         ticker: str,
         question: str,
         market_type: Optional[str] = None,
         precomputed_data: Optional[dict] = None,
         hexagram_info: Optional[dict] = None,
     ) -> str:
     ```
   - 參數優先順序：
     1. `precomputed_data`（Dashboard 單一來源）
     2. `hexagram_info`（舊版相容）
     3. `_get_market_hexagram()`（完全由 Oracle 端重新計算）
   - `precomputed_data` 結構範例：
     ```python
     {
       "ticker": backend_ticker,
       "market_type": market_type,
       "raw_df": raw_df,
       "encoded_df": encoded_df,
       "latest_row_index": latest_row.name,
       "ritual_sequence": ritual_sequence,
       "ritual_sequence_str": ritual_sequence_str,
       "binary_code": binary_code,
       "hexagram_id": hexagram_id,
       "hex_name": hexagram_name_full,
       "hex_name_stripped": hexagram_name,
       "chinese_name": chinese_name,
       "future_binary": future_binary,
       "future_hex_name": future_hex_name_full,
       "future_hex_name_stripped": future_hex_name,
       "future_chinese_name": future_chinese_name,
     }
     ```
   - `ask()` 內部僅做型別整理（`ritual_sequence` → `List[int]`、卦名去括號），**不再重抓 yfinance／不再重跑 encoder**。

2. **`dashboard.py` – 建立 `current_market_state` 作為單一真實來源**
   - 在取得 `raw_df`、`encoded_df`、`latest_row`、`ritual_sequence`、`Hexagram_Binary`、本卦資訊後，建立：
     ```python
     current_market_state = {
         "ticker": backend_ticker,
         "market_type": market_type,
         "raw_df": raw_df,
         "encoded_df": encoded_df,
         "latest_row_index": latest_row.name,
         "ritual_sequence": ritual_sequence,
         "ritual_sequence_str": ritual_sequence_str,
         "binary_code": binary_code,
         "hexagram_id": hexagram_id,
         "hex_name": hexagram_name_full,
         "hex_name_stripped": hexagram_name,
         "chinese_name": chinese_name,
         "future_binary": future_binary,
         "future_hex_name": future_hex_name_full,
         "future_hex_name_stripped": future_hex_name,
         "future_chinese_name": future_chinese_name,
     }
     ```
   - K 線圖、本卦／之卦視覺化全部直接使用這個 `current_market_state`。
   - 呼叫 `Oracle` 時改為：
     ```python
     ai_answer = oracle.ask(
         backend_ticker,
         question or "Should I buy now?",
         market_type=market_type,
         precomputed_data=current_market_state,
     )
     ```

3. **`Oracle` 端易經原文來源修正**
   - 新增 `_iching_raw` 與 `_name_to_number` 快取：
     - `_iching_raw`: `number -> iching_complete.json` 條目
     - `_name_to_number`: 中文卦名 → number（用於 HEXAGRAM_MAP 不完整時補齊 ID）
   - `_resolve_strategy()` 若從 `HEXAGRAM_MAP` 得到的 `current_hex_id` 或 `future_hex_id` 為 0，會改用 `_name_to_number` 以卦名查找正確的 `number`。
   - `_get_iching_wisdom()` 不再依賴向量庫決定「是哪一卦」，而是根據查詢計畫中給定的 `hex_id`、`type`、`line_numbers`，**直接從 `iching_complete.json` 抽出：
     - 本卦主辭：`judgment` + `image`
     - 本卦動爻：對應 `lines[position]` 的 `meaning` + `xiang`
     - 之卦主辭：`future_hex_id` 對應條目的 `judgment` + `image`
   - 產生帶有標籤的原文片段，例如：
     - `【本卦（中孚）：61. 中孚卦】卦辭：…`
     - `【本卦動爻（中孚 第 5 爻，悔）：61. 中孚卦】第 5 爻：… 小象：…`
     - `【之卦（頤）：27. 頤卦】卦辭：…`

4. **LLM 提示調整**
   - 強制要求 Gemini 在「📜 易經原文 (The Source)」段落**完整列出** `_get_iching_wisdom` 提供的所有段落，不得自行刪減或只選部分，並保留標籤，如 `【本卦…】`、`【之卦…】`。

#### 效果

- ✅ Dashboard 顯示的本卦／之卦與 `Oracle.ask()` 使用的卦象完全同步，卦象只算一次。
- ✅ 易經原文只會出現正確的卦（例如中孚／頤），不再混入「師卦」等不相關內容。
- ✅ 貞／悔 分析與卦象視覺（本卦／之卦／動爻）保持 100% 對應。

#### 技術決策

- **為什麼改用 `precomputed_data`？**
  - 明確劃分責任：Dashboard 負責「算卦＋畫圖」，`Oracle` 負責「在既有卦象上解讀」。
  - 減少 I/O 與重複計算，避免因資料時間點不同導致卦象不一致。

- **為什麼直接從 `iching_complete.json` 抽原文？**
  - 相對於語義搜尋，直接以 `hex_id + line_number` 抽取可以完全保證與畫面一致。
  - 便於未來做更細緻的標註（如顯示「本卦／之卦／貞／悔」在文本中的精確來源）。

---
