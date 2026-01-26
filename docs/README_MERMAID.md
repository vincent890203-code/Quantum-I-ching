# Mermaid 圖表轉換說明

## 為什麼 Mermaid 顯示為程式碼而不是圖片？

Mermaid 是一種**文字格式的圖表語言**，需要**渲染器**才能顯示為圖片。就像 Markdown 需要渲染器才能顯示為格式化的文字一樣。

## 解決方案

### 方案 1: 使用支援 Mermaid 的 Markdown 檢視器

以下工具/平台支援直接渲染 Mermaid：

1. **GitHub** - 在 `.md` 檔案中直接支援 Mermaid
2. **GitLab** - 同樣支援 Mermaid
3. **VS Code** - 安裝 "Markdown Preview Mermaid Support" 擴充功能
4. **Obsidian** - 內建支援 Mermaid
5. **Typora** - 付費 Markdown 編輯器，支援 Mermaid
6. **Notion** - 支援 Mermaid 程式碼塊

### 方案 2: 使用線上工具轉換為圖片

1. **Mermaid Live Editor**
   - 網址: https://mermaid.live/
   - 步驟:
     1. 複製 Mermaid 程式碼
     2. 貼上到編輯器
     3. 點擊 "Actions" → "Download PNG" 或 "Download SVG"

2. **Mermaid.ink** (API)
   - 網址: https://mermaid.ink/
   - 可以透過 URL 直接生成圖片

### 方案 3: 使用本專案提供的轉換工具

執行以下命令生成 HTML 檔案（可在瀏覽器中查看）：

```bash
python convert_mermaid_to_image.py
```

這會：
1. 讀取 `ARCHITECTURE_DIAGRAM.md`
2. 提取所有 Mermaid 程式碼塊
3. 生成 HTML 檔案到 `docs/architecture_diagrams/`
4. 在瀏覽器中開啟 HTML 即可看到渲染後的圖表

### 方案 4: 使用 Mermaid CLI 轉換為圖片

安裝 Mermaid CLI：

```bash
npm install -g @mermaid-js/mermaid-cli
```

轉換單一檔案：

```bash
# 將 Mermaid 程式碼儲存為 .mmd 檔案
mmdc -i diagram.mmd -o diagram.png
mmdc -i diagram.mmd -o diagram.svg
```

### 方案 5: 使用 Python 自動化轉換

如果需要批量轉換，可以使用以下 Python 套件：

```bash
pip install playwright mermaid
playwright install chromium
```

然後使用 `convert_mermaid_to_image.py` 腳本。

## 推薦做法

1. **開發時**: 使用 VS Code + Mermaid 擴充功能，即時預覽
2. **文件展示**: 將 Markdown 推送到 GitHub，自動渲染
3. **簡報使用**: 使用 Mermaid Live Editor 轉換為 PNG，插入簡報
4. **本地查看**: 使用 `convert_mermaid_to_image.py` 生成 HTML 檔案

## 快速轉換步驟（使用 Mermaid Live Editor）

1. 開啟 https://mermaid.live/
2. 從 `ARCHITECTURE_DIAGRAM.md` 複製 Mermaid 程式碼塊（包含 ```mermaid 和 ``` 之間的部分）
3. 貼上到編輯器左側
4. 右側會即時顯示渲染後的圖表
5. 點擊右上角 "Actions" → "Download PNG" 或 "Download SVG"
6. 儲存圖片檔案

## 注意事項

- Mermaid 圖表在不同渲染器中可能顯示略有不同
- 複雜圖表可能需要調整樣式以適應不同平台
- 建議使用 SVG 格式以獲得最佳品質和可縮放性
