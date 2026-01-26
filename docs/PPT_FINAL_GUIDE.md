# PowerPoint 架構圖最終生成指南

## ✅ 已完成的優化

### 1. 內容人性化
所有圖表已重新設計，使用**人性化描述**：
- ✅ **主要文字**：使用清晰的中文描述（如「市場資料載入器」）
- ✅ **程式碼名稱**：放在括號內，字體較小（如 `(MarketDataLoader)`）
- ✅ **技術術語**：轉換為易於理解的形式（如「OHLCV」→「開高低收成交量」）

### 2. 字體大小優化
- 節點文字：32px（主要描述）
- 群組標題：36px
- 邊緣標籤：28px
- 程式碼名稱（括號內）：較小字體

### 3. 佈局優化
- 橫向佈局（LR）：充分利用 16:9 寬螢幕
- 更大的節點間距：120px
- 更大的層級間距：150px
- 邊框加粗：5px，更清晰

## 📝 圖表內容對照

### 完整系統架構圖

**優化前**：
- `MarketDataLoader`
- `data_loader.py`
- `Settings & HEXAGRAM_MAP`

**優化後**：
- **市場資料載入器** `(MarketDataLoader)`
- **Yahoo Finance 金融資料來源**
- **系統設定與六十四卦對照表** `(Settings & HEXAGRAM_MAP)`

### 詳細資料流程圖

**優化前**：
- `MarketDataLoader`
- `raw_df OHLCV`
- `encoded_df`

**優化後**：
- **市場資料載入器** `(MarketDataLoader)`
- **原始市場資料** `(raw_df OHLCV)`
- **編碼後的卦象資料** `(encoded_df)`

### 大衍之數映射流程

**優化前**：
- `OHLCV`
- `RVOL`
- `Ritual_Sequence`

**優化後**：
- **市場歷史資料 開高低收成交量** `(OHLCV)`
- **相對成交量** `(RVOL)`
- **儀式序列** `(Ritual_Sequence)`

## 🎨 生成高品質圖片（推薦方法）

### 方法 1: 使用 Mermaid Live Editor（最簡單、最可靠）

1. **開啟**: https://mermaid.live/

2. **複製程式碼**: 
   - 開啟 `ARCHITECTURE_DIAGRAM_PPT.md`
   - 複製對應的 Mermaid 程式碼塊（在 ```mermaid 和 ``` 之間）

3. **貼上並調整**:
   - 貼到編輯器左側
   - 點擊右上角 "Settings"
   - 調整以下設定：
     - **Font Size**: 28 或更大（建議 32）
     - **Theme**: 選擇適合簡報的風格
     - **Background**: 白色

4. **下載**:
   - 點擊 "Actions" → "Download PNG"
   - 選擇高解析度選項（如果有的話）

5. **後處理**（如果需要）:
   - 使用圖片編輯軟體調整為 1920x1080
   - 確保文字清晰可讀

### 方法 2: 安裝 Playwright（最佳品質）

```bash
# 安裝
pip install playwright
playwright install chromium

# 生成
python generate_ppt_large_font.py
```

這會生成：
- 1920x1080 高解析度圖片
- 字體大小 28-36px
- 充分利用 16:9 空間
- 人性化描述，程式碼名稱在括號內

### 方法 3: 使用 Mermaid CLI（需要 Node.js）

```bash
# 安裝
npm install -g @mermaid-js/mermaid-cli

# 將 Mermaid 程式碼儲存為 .mmd 檔案
# 然後執行
mmdc -i diagram.mmd -o diagram.png -w 1920 -H 1080 -b white -s 2
```

## 📋 檢查清單

生成圖片前，確認：
- [ ] 所有文字都是人性化描述
- [ ] 程式碼名稱都在括號內
- [ ] 字體大小至少 28px
- [ ] 圖片尺寸為 1920x1080 (16:9)
- [ ] 文字清晰可讀
- [ ] 充分利用寬螢幕空間

## 💡 提示

1. **如果字體仍然太小**：
   - 在 Mermaid Live Editor 中進一步調整字體大小
   - 或使用圖片編輯軟體放大文字

2. **如果內容太密集**：
   - 可以進一步簡化某些節點
   - 或將相關功能合併

3. **最佳實踐**：
   - 使用 Mermaid Live Editor 進行最終調整
   - 可以即時看到效果
   - 確保所有文字都清晰可讀

## 📁 檔案位置

- **Mermaid 程式碼**: `ARCHITECTURE_DIAGRAM_PPT.md`
- **生成腳本**: `generate_ppt_large_font.py`
- **輸出目錄**: `docs/ppt_images/`

## 🎯 最終目標

生成適合 PowerPoint 簡報的架構圖：
- ✅ 字體清晰可讀（28+ 號）
- ✅ 內容人性化（非技術人員也能理解）
- ✅ 充分利用 16:9 空間
- ✅ 程式碼名稱在括號內（較小字體）
