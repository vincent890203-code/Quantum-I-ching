# GitHub 上傳指南

## ✅ 已更新的 .gitignore

`.gitignore` 已更新，包含以下規則：

### 已忽略的文件類型

1. **Python 相關**
   - `__pycache__/` - Python 緩存文件
   - `*.pyc`, `*.pyo`, `*.pyd` - 編譯的 Python 文件
   - `venv/`, `env/`, `.venv` - 虛擬環境

2. **模型文件（不應該上傳）**
   - `data/*.pth` - PyTorch 模型
   - `data/*.pt` - PyTorch 模型
   - `data/*.pkl` - Pickle 文件
   - `data/volatility_model.json` - XGBoost 模型
   - `data/best_model.pth` - 最佳模型

3. **向量資料庫（不應該上傳）**
   - `data/chroma_db/` - ChromaDB 向量資料庫（包含所有子目錄）

4. **生成的圖片和圖表（可選）**
   - `data/*.png` - SHAP 圖表、特徵重要性圖等
   - `data/*.jpg`, `data/*.jpeg` - 圖片文件

5. **配置和優化結果（可選）**
   - `config/best_params.json` - Optuna 優化結果

6. **環境變數**
   - `.env`, `*.env` - 環境變數文件（包含 API keys）

7. **IDE 和作業系統**
   - `.vscode/`, `.idea/` - IDE 配置
   - `.DS_Store`, `Thumbs.db` - 作業系統文件

### ⚠️ 需要手動處理的文件

以下文件已經被 Git 追蹤，需要手動從 Git 中移除（但保留本地文件）：

```bash
# 移除已追蹤的向量資料庫文件
git rm --cached data/chroma_db/chroma.sqlite3

# 移除已追蹤的模型文件（如果有的話）
git rm --cached data/best_model.pth

# 移除已追蹤的圖片文件（如果有的話）
git rm --cached data/*.png
```

## 📋 應該上傳的文件

### 核心代碼文件
- ✅ `*.py` - 所有 Python 源代碼
- ✅ `requirements.txt` - 依賴套件清單
- ✅ `README.md` - 專案說明
- ✅ `DEV_LOG.md` - 開發日誌
- ✅ `PPT_架構.md` - 架構文檔

### 配置文件
- ✅ `config.py` - 配置模組
- ✅ `.gitignore` - Git 忽略規則

### 數據文件（源數據，應該保留）
- ✅ `data/iching_book.json` - 易經書籍數據
- ✅ `data/iching_complete.json` - 完整易經數據

### 腳本文件
- ✅ `experiment_*.py` - 實驗腳本
- ✅ `save_model_c.py` - 模型保存腳本
- ✅ `tune_hyperparameters.py` - 超參數優化腳本
- ✅ `reset_data.py` - 數據重置腳本

## 🚀 上傳步驟

1. **檢查 .gitignore 是否生效**：
   ```bash
   git status
   ```
   確認以下文件不會出現在未追蹤列表中：
   - `data/volatility_model.json`
   - `data/*.png`
   - `data/chroma_db/`
   - `config/best_params.json`

2. **移除已追蹤但應該忽略的文件**：
   ```bash
   # 如果這些文件已經被追蹤，需要移除
   git rm --cached data/chroma_db/chroma.sqlite3
   git rm --cached data/best_model.pth
   git rm --cached data/*.png
   ```

3. **添加新文件**：
   ```bash
   git add .gitignore
   git add *.py
   git add requirements.txt
   git add README.md
   git add DEV_LOG.md
   git add PPT_架構.md
   git add data/iching_book.json
   git add data/iching_complete.json
   ```

4. **提交更改**：
   ```bash
   git commit -m "Add volatility prediction model deployment and update .gitignore"
   ```

5. **推送到 GitHub**：
   ```bash
   git push origin main
   ```

## 📝 注意事項

1. **模型文件**：
   - 訓練好的模型文件（`.pth`, `.json`）不應該上傳
   - 用戶需要自己運行 `python save_model_c.py` 來生成模型

2. **向量資料庫**：
   - ChromaDB 資料庫文件很大，不應該上傳
   - 用戶需要運行 `python setup_iching_db.py` 來初始化資料庫

3. **環境變數**：
   - `.env` 文件包含 API keys，絕對不能上傳
   - 應該在 `README.md` 中說明需要設置哪些環境變數

4. **生成的圖片**：
   - SHAP 圖表、特徵重要性圖等是實驗結果
   - 可以選擇上傳（用於文檔）或忽略（讓用戶自己生成）

5. **Optuna 優化結果**：
   - `config/best_params.json` 是優化結果
   - 可以選擇上傳（作為參考）或忽略（讓用戶自己優化）

## 🔍 檢查清單

在上傳前，確認：

- [ ] `.gitignore` 已更新
- [ ] 所有模型文件（`.pth`, `.json`）都被忽略
- [ ] 向量資料庫（`data/chroma_db/`）被忽略
- [ ] 環境變數文件（`.env`）被忽略
- [ ] `__pycache__/` 被忽略
- [ ] 所有源代碼文件（`.py`）都包含
- [ ] `requirements.txt` 已更新
- [ ] `README.md` 包含使用說明
