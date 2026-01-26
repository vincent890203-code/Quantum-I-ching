# 快速匯出 Mermaid 為圖片指南

## ✅ 最簡單的方法：使用 Mermaid Live Editor

**無需安裝任何軟體，最可靠！**

### 步驟：

1. **開啟 Mermaid Live Editor**
   - 網址：https://mermaid.live/

2. **複製 Mermaid 程式碼**
   - 開啟 `ARCHITECTURE_DIAGRAM.md`
   - 找到要轉換的圖表（在 ```mermaid 和 ``` 之間）
   - 複製整個程式碼塊（**不包含** ```mermaid 和 ```）

3. **貼上並下載**
   - 將程式碼貼上到編輯器左側
   - 右側會即時顯示渲染後的圖表
   - 點擊右上角 **"Actions"** → **"Download PNG"** 或 **"Download SVG"**
   - 儲存圖片檔案

### 範例：

從 `ARCHITECTURE_DIAGRAM.md` 複製：
```mermaid
flowchart TB
    subgraph DataLayer["📊 資料獲取層"]
    ...
```

貼到 https://mermaid.live/ 即可！

---

## 🔧 自動化方法（需要安裝）

### 方法 1: 使用 Playwright (推薦)

```bash
# 安裝
pip install playwright
playwright install chromium

# 執行
python export_mermaid_to_image.py
```

### 方法 2: 使用 Mermaid CLI

```bash
# 安裝 (需要 Node.js)
npm install -g @mermaid-js/mermaid-cli

# 執行
python export_mermaid_to_image.py
```

---

## 📁 輸出位置

所有圖片會儲存在：
```
docs/architecture_images/
```

---

## 💡 提示

- **PNG**: 適合簡報、文件插入
- **SVG**: 適合需要縮放的場合，品質最佳
- **批次轉換**: 使用自動化腳本一次處理所有圖表
- **單一轉換**: 使用 Mermaid Live Editor 手動轉換

---

## 🎯 推薦流程

1. **快速單一轉換**: 使用 Mermaid Live Editor
2. **批次轉換**: 安裝 Playwright 後使用腳本
3. **文件展示**: 推送到 GitHub，自動渲染
