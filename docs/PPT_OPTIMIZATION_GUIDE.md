# PowerPoint 架構圖優化指南

## 目標

生成適合 PowerPoint 的架構圖：
- ✅ **尺寸**: 1920x1080 (16:9)
- ✅ **字體**: 最小 28 號字體
- ✅ **內容**: 簡化，只保留核心資訊
- ✅ **佈局**: 充分利用 16:9 寬螢幕空間

## 已完成的優化

### 1. 內容簡化

**原始版本**包含：
- 詳細的檔案路徑（如 `data_loader.py`）
- 功能列表（如 `• 支援 TW/US/CRYPTO`）
- 詳細說明文字

**簡化版本**只保留：
- 核心模組名稱（如 `MarketDataLoader`）
- 主要功能（如 `計算技術指標`）
- 關鍵資料流

### 2. 佈局優化

- **橫向佈局 (LR)**: 充分利用 16:9 寬螢幕
- **更大的節點間距**: 讓內容更清晰
- **簡化的標題**: 避免被遮擋

### 3. 字體大小優化

Mermaid 的字體大小取決於節點大小，透過：
- 簡化文字內容 → 節點變大 → 字體變大
- 增加節點間距 → 整體佈局更寬鬆
- 使用更大的邊框 (4px) → 視覺更清晰

## 生成高品質圖片的方法

### 方法 1: 使用 Mermaid Live Editor（推薦，最簡單）

1. **開啟**: https://mermaid.live/
2. **複製程式碼**: 從 `ARCHITECTURE_DIAGRAM_PPT.md` 複製 Mermaid 程式碼
3. **貼上**: 貼到編輯器左側
4. **調整設定**:
   - 點擊右上角 "Settings"
   - 調整 "Font Size" 為 28 或更大
   - 調整 "Theme" 為適合簡報的風格
5. **下載**: 點擊 "Actions" → "Download PNG"
6. **後處理**: 使用圖片編輯軟體調整為 1920x1080（如果需要）

### 方法 2: 使用 Playwright（需要安裝）

```bash
# 安裝
pip install playwright
playwright install chromium

# 生成
python generate_ppt_final.py
```

這會生成：
- 1920x1080 高解析度圖片
- 字體大小優化（28-32 號）
- 充分利用 16:9 空間

### 方法 3: 使用 Mermaid CLI（需要 Node.js）

```bash
# 安裝
npm install -g @mermaid-js/mermaid-cli

# 將 Mermaid 程式碼儲存為 .mmd 檔案
# 然後執行
mmdc -i diagram.mmd -o diagram.png -w 1920 -H 1080 -b white
```

## 當前生成的圖片

`docs/ppt_images/` 目錄中已有：
- `ppt_01_完整系統架構簡報優化版_極簡風格.png`
- `ppt_02_詳細資料流程圖簡報優化版_橫向流程.png`
- `ppt_03_大衍之數映射流程簡報優化版.png`

這些圖片已經：
- ✅ 簡化內容
- ✅ 使用橫向佈局
- ✅ 移除不必要的細節

## 進一步優化建議

如果字體仍然不夠大，可以：

1. **進一步簡化文字**:
   - 只保留模組名稱，移除所有說明
   - 例如：`MarketDataLoader` 而不是 `MarketDataLoader<br/>data_loader.py`

2. **使用 Mermaid Live Editor 手動調整**:
   - 在編輯器中可以即時看到效果
   - 可以調整字體大小和佈局
   - 可以下載高解析度圖片

3. **後處理**:
   - 使用圖片編輯軟體（如 Photoshop、GIMP）
   - 調整對比度和銳化
   - 確保文字清晰可讀

## 在 PowerPoint 中使用

1. **插入圖片**: 插入 → 圖片 → 此裝置
2. **選擇檔案**: 選擇 `docs/ppt_images/` 中的圖片
3. **調整大小**: 
   - 圖片已經是 16:9 比例
   - 可以直接填滿投影片
   - 建議使用「圖片填滿」模式

## 注意事項

- Mermaid 的字體大小主要取決於節點大小
- 簡化文字是讓字體變大的最有效方法
- 如果使用 Playwright，可以透過 HTML/CSS 進一步控制字體大小
- 建議使用 Mermaid Live Editor 進行最終調整，確保字體清晰可讀
