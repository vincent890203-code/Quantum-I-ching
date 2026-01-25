# Streamlit Cloud 部署指南

## 📋 回答你的問題

**Q: 如何讓 https://quantum-i-ching.streamlit.app/ 網站上也跑好 `python save_model_c.py`？我只要把 `data/volatility_model.json` 上傳上去即可嗎？**

**A: 是的！** 你只需要：

1. ✅ **上傳 `data/volatility_model.json` 到 GitHub**（不需要在 Streamlit Cloud 上運行訓練腳本）
2. ✅ **確保 `.gitignore` 不忽略這個文件**
3. ✅ **Streamlit Cloud 會自動從 GitHub 讀取模型文件**

## 🚀 完整部署步驟

### 步驟 1: 確認模型文件存在

```bash
# 在本機先訓練模型（如果還沒有的話）
python save_model_c.py

# 確認文件存在
ls data/volatility_model.json
```

### 步驟 2: 更新 .gitignore（如果需要）

如果 `data/volatility_model.json` 被 `.gitignore` 忽略，需要允許它：

```bash
# 檢查是否被忽略
git check-ignore -v data/volatility_model.json
```

如果被忽略，編輯 `.gitignore`，移除或註解掉 `data/volatility_model.json` 這一行。

### 步驟 3: 將模型文件加入 Git 並上傳

```bash
cd "c:\Users\USER\Desktop\I-Ching AI"

# 添加模型文件
git add data/volatility_model.json

# 提交
git commit -m "Add volatility model for Streamlit Cloud deployment"

# 推送到 GitHub
git push origin main
```

### 步驟 4: 在 Streamlit Cloud 上部署

1. **登入 Streamlit Cloud**: https://share.streamlit.io/
2. **連接 GitHub Repository**:
   - 點擊 "New app"
   - 選擇你的 GitHub 帳號和 `I-Ching AI` repository
   - 選擇 `main` 分支
3. **設置應用配置**:
   - **Main file path**: `dashboard.py`
   - **Python version**: `3.11` 或 `3.12`（根據你的環境）
4. **設置環境變數**（在 Advanced settings）:
   - `GOOGLE_API_KEY`: 你的 Google Gemini API Key
5. **點擊 Deploy**

### 步驟 5: 驗證部署

部署完成後，訪問你的 Streamlit Cloud URL（例如 `https://quantum-i-ching.streamlit.app/`），測試：

1. 輸入股票代號（例如 `NVDA`）
2. 點擊 "Consult the Oracle"
3. 檢查「🌊 波動率爆發機率 (Volatility Radar)」是否正常顯示

## ⚠️ 重要注意事項

### 1. 模型文件大小

- `data/volatility_model.json` 通常只有幾百 KB 到幾 MB
- GitHub 單文件限制是 100 MB，所以沒問題
- 如果模型文件很大（>50MB），考慮使用 Git LFS

### 2. 不需要在 Streamlit Cloud 上訓練

**為什麼不需要運行 `save_model_c.py`？**

- ✅ **速度**: 訓練需要時間（幾分鐘），會讓應用啟動變慢
- ✅ **資源**: Streamlit Cloud 免費版有資源限制，不適合訓練
- ✅ **穩定性**: 預訓練的模型更穩定，不會因為每次部署而改變
- ✅ **效率**: 直接讀取模型文件更快

**什麼時候需要重新訓練？**

- 當你更新了特徵工程邏輯
- 當你獲得了更多訓練數據
- 當你想要更新模型性能

**如何更新模型？**

1. 在本機運行 `python save_model_c.py`
2. 將新的 `data/volatility_model.json` 上傳到 GitHub
3. Streamlit Cloud 會自動重新部署並使用新模型

### 3. 其他需要上傳的文件

除了 `data/volatility_model.json`，還需要確保以下文件也在 GitHub 上：

- ✅ `data/iching_book.json` - 易經書籍數據（源數據）
- ✅ `data/iching_complete.json` - 完整易經數據（源數據）
- ✅ `requirements.txt` - 依賴套件清單
- ✅ 所有 `.py` 源代碼文件

### 4. 不需要上傳的文件

以下文件**不應該**上傳（已在 `.gitignore` 中）：

- ❌ `data/chroma_db/` - 向量資料庫（會在 Streamlit Cloud 上自動重建）
- ❌ `data/*.png` - 生成的圖片
- ❌ `.env` - 環境變數（包含 API keys）
- ❌ `__pycache__/` - Python 緩存

### 5. ChromaDB 向量資料庫

**重要**: `data/chroma_db/` 不應該上傳，因為：

- 文件很大（可能幾百 MB）
- Streamlit Cloud 會自動運行初始化腳本重建資料庫

**如何確保 ChromaDB 在 Streamlit Cloud 上可用？**

在 `dashboard.py` 或初始化腳本中，確保有自動重建邏輯：

```python
# 在 Oracle 初始化時檢查並重建向量資料庫
if not os.path.exists("data/chroma_db/iching_knowledge"):
    from setup_iching_db import main as setup_db
    setup_db()
```

或者，在 Streamlit Cloud 的部署配置中添加一個啟動腳本。

## 🔍 故障排除

### 問題 1: 模型文件找不到

**症狀**: Dashboard 顯示 "⚠️ 波動性模型尚未訓練"

**解決方案**:
1. 確認 `data/volatility_model.json` 在 GitHub 上
2. 檢查文件路徑是否正確（`data/volatility_model.json`）
3. 確認 Streamlit Cloud 已成功部署最新版本

### 問題 2: 模型載入失敗

**症狀**: 錯誤訊息 "無法載入波動性模型"

**解決方案**:
1. 確認模型文件格式正確（JSON 格式）
2. 檢查 XGBoost 版本是否相容
3. 查看 Streamlit Cloud 的日誌（Logs 標籤）

### 問題 3: 預測結果異常

**症狀**: 波動率機率始終為 0 或 100%

**解決方案**:
1. 確認特徵順序與訓練時一致
2. 檢查 `dashboard.py` 中的特徵提取邏輯
3. 確認模型文件是正確的版本

## 📝 檢查清單

在上傳到 GitHub 前，確認：

- [ ] `data/volatility_model.json` 存在且已訓練
- [ ] `.gitignore` 不忽略 `data/volatility_model.json`
- [ ] `requirements.txt` 包含所有必要套件
- [ ] `GOOGLE_API_KEY` 已設置在 Streamlit Cloud 環境變數中
- [ ] 所有源代碼文件都已上傳
- [ ] `data/iching_book.json` 和 `data/iching_complete.json` 已上傳

## 🎯 總結

**簡短答案**: 是的，你只需要上傳 `data/volatility_model.json` 到 GitHub，Streamlit Cloud 會自動讀取並使用它。不需要在 Streamlit Cloud 上運行 `save_model_c.py`。

**完整流程**:
1. 本機訓練模型 → `python save_model_c.py`
2. 上傳模型文件 → `git add data/volatility_model.json && git commit && git push`
3. Streamlit Cloud 自動部署 → 模型自動可用

就是這麼簡單！🚀
