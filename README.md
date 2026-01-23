# Quantum I-Ching 股市卜卦系統

結合量化市場資料、易經六十四卦與 Google Gemini 的 **AI 金融卜卦系統**。  
專案目標是將股票價格結構轉換為卦象，並透過 RAG + LLM 產生「看盤＋卜卦」式的現代金融解讀。

> **重要聲明**：本專案所有內容僅供研究與教育參考，  
> 不構成任何投資建議、買賣邀約或報酬保證，實際投資決策請自行審慎評估風險。

---

## 功能總覽

- **資料層 (Phase 1 & 3)**
  - `data_loader.py`：從 Yahoo Finance 下載歷史股價（支援美股 / 台股 / 加密貨幣）。
  - `market_encoder.py`：將價格與成交量轉成四象（6/7/8/9），再組成六爻卦象。
  - `iching_core.py`：由「儀式數字序列」解碼出當前卦／未來卦與動爻。
  - `data_processor.py`：將含卦象的時間序列轉換為 LSTM 可用的訓練資料。
  - `model_lstm.py` + `backtester.py`：Hybrid LSTM 模型與簡易回測框架。

- **知識庫與 RAG (Phase 2)**
  - `config.py` + `HEXAGRAM_MAP`：完整 64 卦對照表（ID、英文名、繁中卦名）。
  - `setup_iching_db.py`：從 [john-walks-slow/open-iching](https://github.com/john-walks-slow/open-iching) 下載完整易經資料（64 卦含六爻），轉為統一格式並儲存至 `data/iching_complete.json`。
  - `convert_iching_s2t.py`：將 `iching_complete.json` 內簡體中文轉為繁體，並重建 ChromaDB 向量庫。
  - `knowledge_loader.py`：將 JSON 轉成可嵌入文件物件（主卦 + 六爻，共約 450 份文件）。
  - `vector_store.py`：使用 ChromaDB + SentenceTransformers 建立本地向量資料庫。
  - `oracle_chat.py` (`Oracle` 類別)：
    - 實作傳統「之卦 (Zhi Gua)」策略：依動爻數量（0-6）動態選擇查詢策略。
    - 引入 **貞 (Zhen) / 悔 (Hui)** 架構：貞=主體/支撐/長期/進場，悔=客體/阻力/短期/出場。
    - 讀取市場資料 → 生成卦象 → 依策略查詢易經文本 → 呼叫 Gemini。
    - 輸出結構化、繁體中文的投資解讀（Executive Summary / 易經原文 / 現代解讀 / 操作建議）。

- **前端介面 (Phase 4)**
  - `dashboard.py`：Streamlit Web 儀表板：
    - 左側：輸入股票代號（支援台股純數字，例如 `2330`）、提問。
    - 中間：最近 60 日 K 線圖（Plotly White 主題）。
    - 右側：I-Ching 市場卦象視覺化（六爻、陽實線／陰斷線）。
    - 下方：`Oracle` 產生的卜卦解讀，以 `st.info` 卡片顯示。

詳細技術設計與歷次修正，請見 `DEV_LOG.md`。

---

## 安裝與環境設定

### 1. 建立虛擬環境（建議）

```bash
cd "I-Ching AI"
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3. 設定環境變數（Google Gemini）

在專案根目錄建立 `.env` 檔案（或使用系統環境變數）：

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. 初始化易經知識庫

首次使用前，需要下載並建立易經向量資料庫：

```bash
# 步驟 1: 下載易經資料（從 john-walks-slow/open-iching）
python setup_iching_db.py

# 步驟 2: 簡體轉繁體並重建向量庫（約 450 份文件：64 主卦 + 386 爻）
python convert_iching_s2t.py
```

**說明**：
- `setup_iching_db.py` 會從 jsDelivr CDN 下載 open-iching 的 `iching/iching.json`，驗證後轉為統一格式儲存。
- `convert_iching_s2t.py` 使用 OpenCC 將簡體轉為繁體，並重建 ChromaDB 向量庫（若未安裝 `opencc-python-reimplemented` 會自動安裝）。
- 之後執行 `oracle_chat.py` 或 `dashboard.py` 時，向量庫會自動載入。

---

## 市場設定與台股支援

在 `config.py` 中：

```python
settings = Settings(
    # ...
    MARKET_TYPE="TW",  # 'US', 'TW', 'CRYPTO'
)
```

- **台股 (`MARKET_TYPE="TW"`)**
  - `MarketDataLoader` 會自動將 ticker 補成 `2330.TW`、`2317.TW` 等格式。
  - Streamlit 預設輸入欄可直接輸入 **純數字代碼**：`2330`、`2317`……
  - 前端會自動補 `.TW` 作為實際查詢用代碼。

- **美股 (`MARKET_TYPE="US"`)**
  - 直接使用原始代碼，例如：`NVDA`、`AAPL`。

- **加密貨幣 (`MARKET_TYPE="CRYPTO"`)**
  - 自動補 `-USD`，例如輸入 `BTC` 會查詢 `BTC-USD`。

---

## 執行神諭後端 (CLI)

單純測試 `Oracle` 的問答能力，可以直接執行：

```bash
python oracle_chat.py
```

程式會：

- 初始化 `Oracle`（載入 ChromaDB、SentenceTransformers、Gemini 模型）。
- 以預設示範（例如 `NVDA`, "Should I buy now?"）產生完整卜卦解讀，輸出至終端機。

---

## 執行 Streamlit 儀表板 (`dashboard.py`)

### 啟動方式

```bash
streamlit run dashboard.py
```

啟動後瀏覽器會自動打開（預設 `http://localhost:8501`）：

1. 在左側輸入：
   - 台股代碼（例如 `2330` 或 `2330.TW`）。
   - 問題文字（中英皆可，如「我現在該買嗎？」）。
2. 按下 `Consult the Oracle (卜卦)`。
3. 會看到：
   - 中間：最近 60 日 K 線圖，標題會顯示「代碼（公司名稱）＋卦名」。
   - 右側：六爻卦象（陽實線／陰斷線）與卦名／卦號。
   - 下方：`🧠 Oracle's Advice / 卜卦解讀`（整段在單一 info 卡片內，含 Markdown 標題與條列式建議）。

---

## 檔案導覽 (簡版)

- `config.py`：全域設定（日期、目標股票清單、MARKET_TYPE、HEXAGRAM_MAP）。
- `data_loader.py`：抓取 Yahoo Finance 歷史資料，支援 TW / US / CRYPTO。
- `market_encoder.py`：Whale Volume 四象邏輯 + 六爻卦象生成。
- `iching_core.py`：當前卦／未來卦／動爻計算與卦名查詢。
- `setup_iching_db.py`：下載 open-iching 易經資料並轉為統一格式。
- `convert_iching_s2t.py`：簡體轉繁體並重建向量庫。
- `knowledge_loader.py`：載入 `data/iching_complete.json` 為 RAG 文件（主卦 + 六爻）。
- `vector_store.py`：Chroma 向量資料庫（語義檢索易經文本）。
- `oracle_chat.py`：Quantum I-Ching 神諭（整合市場資料、卦象、RAG、Gemini），實作之卦策略與貞／悔架構。
- `dashboard.py`：Streamlit 前端儀表板（台股優先、K 線＋卦象＋解讀）。
- `model_lstm.py`：LSTM 模型與訓練流程。
- `backtester.py`：策略回測引擎。
- `DEV_LOG.md`：完整開發日誌與除錯紀錄（推薦先閱讀）。

---

## 注意事項與建議

- **API 成本與頻率**
  - Gemini 呼叫會產生費用，建議開發／測試時控制詢問次數。
  - Yahoo Finance 資料抓取頻繁時可能觸發速率限制，可適度快取本地資料。

- **法律與風險**
  - 本專案僅供學術與技術研究之用。
  - 任何基於本系統產生之內容進行的交易或投資行為，風險自負。

---

## 核心功能：之卦 (Zhi Gua) 策略

`Oracle` 類別實作傳統易經「之卦」解法，依動爻（6、9）數量動態選擇查詢策略：

- **0 動爻**：Total Acceptance，查本卦卦辭／象辭。
- **1 動爻**：Specific Focus，僅查該動爻文本（如「乾卦 初九」）。
- **2 動爻**：Primary vs Secondary，下爻為貞（進場／支撐）、上爻為悔（出場／阻力）。
- **3 動爻**：Hedging Moment，本卦為貞（持有）、之卦為悔（風險）。
- **4 或 5 動爻**：Trend Reversal，之卦為貞（主趨勢）、本卦為悔（歷史）。
- **6 動爻**：Extreme Reversal，乾卦用「用九」、坤卦用「用六」，其餘用之卦卦辭。

系統會依策略自動查詢對應的易經文本，並在 Gemini 提示中注入 **貞 (Zhen) / 悔 (Hui)** 架構，產出具主客、支撐／阻力、進出場意涵的金融建議。

---

若要了解所有細部設計決策（含 yfinance 相容性、Windows 編碼問題、Gemini 模型選擇、Streamlit UI 除錯歷程、之卦策略實作、open-iching 資料來源等），請參考 `DEV_LOG.md`。  
未來如要擴充新的前端（例如 FastAPI / React），建議沿用 `Oracle` 作為單一後端入口。 

