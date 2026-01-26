# 將 Mermaid 圖表匯出為圖片指南

## 快速開始

執行以下命令即可將所有 Mermaid 圖表匯出為 PNG 圖片：

```bash
python export_mermaid_to_image.py
```

## 安裝方法

腳本會自動嘗試多種方法，您只需要安裝其中一種：

### 方法 1: Playwright (推薦，最可靠)

```bash
pip install playwright
playwright install chromium
```

### 方法 2: Mermaid CLI (需要 Node.js)

```bash
npm install -g @mermaid-js/mermaid-cli
```

### 方法 3: 使用線上 API (無需安裝，但需要網路)

如果前兩種方法都不可用，腳本會自動使用 Mermaid.ink API（需要網路連線）。

## 輸出位置

所有圖片會儲存在：
```
docs/architecture_images/
```

檔案命名格式：
- `diagram_01_完整系統架構.png`
- `diagram_02_詳細資料流程圖.png`
- 等等...

## 手動轉換單一圖表

### 使用 Mermaid Live Editor (最簡單)

1. 開啟 https://mermaid.live/
2. 複製 Mermaid 程式碼
3. 貼上到編輯器
4. 點擊 "Actions" → "Download PNG" 或 "Download SVG"

### 使用 Mermaid CLI

1. 將 Mermaid 程式碼儲存為 `diagram.mmd`
2. 執行：
   ```bash
   mmdc -i diagram.mmd -o diagram.png
   ```

## 故障排除

### Playwright 安裝失敗

如果 `playwright install chromium` 失敗，可能是網路問題。可以：
1. 使用代理
2. 或使用 Mermaid CLI 方法
3. 或使用線上 API 方法

### Mermaid CLI 找不到

確保 Node.js 已安裝：
```bash
node --version
npm --version
```

然後安裝：
```bash
npm install -g @mermaid-js/mermaid-cli
```

### 所有方法都失敗

如果所有方法都失敗，建議：
1. 使用 Mermaid Live Editor 手動轉換
2. 或將 Markdown 推送到 GitHub，GitHub 會自動渲染

## 批次轉換

腳本會自動處理 `ARCHITECTURE_DIAGRAM.md` 中的所有 Mermaid 圖表，無需手動操作。
