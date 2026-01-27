# 專案重組計畫（Project Reorganization Plan）

**狀態：僅規劃，尚未執行任何檔案搬移。**

---

## 一、建議目錄結構樹狀圖

```
I-Ching AI/                                         # 專案根目錄
├── .devcontainer/                                  # Dev Container 設定
│   └── devcontainer.json                           # 容器環境與開發設定
├── .gitignore                                      # Git 版控忽略規則
├── README.md                                       # 專案總覽與使用說明
├── requirements.txt                                # Python 套件依賴列表
│
├── config/                                         # 超參數優化結果（Optuna）
│   └── best_params.json                            # Optuna 優化後最佳參數
│
├── data/                                           # 原始與衍生資料
│   ├── iching_book.json                            # 易經書籍原始內容
│   ├── iching_complete.json                        # 64 卦完整知識庫（setup 產生）
│   ├── chroma_db/                                  # ChromaDB 向量庫（執行時產生）
│   └── *.csv, *.pkl                                # 預處理快取（價格特徵、訓練資料等）
│
├── docs/                                           # 文件與架構圖（既有＋根目錄 *.md 移入）
│   ├── architecture_diagrams/                      # 架構圖 HTML（由 Mermaid 匯出）
│   ├── architecture_images/                        # 架構圖 PNG 原圖
│   ├── ppt_images/                                 # PPT 最終用圖（16:9、高解析度等）
│   ├── ARCHITECTURE.md                             # 系統架構總覽說明
│   ├── ARCHITECTURE_DIAGRAM.md                     # 架構圖逐張說明
│   ├── ARCHITECTURE_DIAGRAM_PPT.md                 # 架構圖轉 PPT 的說明
│   ├── ARCHITECTURE_DIAGRAM.html                   # 架構圖整合 HTML 頁面
│   ├── PPT_架構.md                                 # 投影片架構與分頁規劃
│   ├── DEV_LOG.md                                  # 開發紀錄與變更日誌
│   ├── FEATURE_FUNNEL_AUDIT.md                     # 特徵漏斗設計與審核紀錄
│   ├── GITHUB_UPLOAD.md                            # 推送至 GitHub 的操作說明
│   ├── LOOKAHEAD_BIAS_AUDIT.md                     # Look-ahead 偏誤檢查說明
│   ├── STREAMLIT_CLOUD_DEBUG.md                    # Streamlit Cloud 除錯筆記
│   ├── STREAMLIT_CLOUD_DEPLOY.md                   # Streamlit Cloud 部署步驟
│   └── UPDATE_SUMMARY.md                           # 版本更新摘要
│
├── models/                                         # 模型產物（*.json, *.pth）
│   ├── model_d_pure_quantum.json                   # Model D（Pure Quantum）XGBoost 權重
│   ├── volatility_model.json                       # Model C 波動性模型權重
│   └── best_model.pth                              # LSTM 最佳權重檔（若存在）
│
├── notebooks/                                      # Jupyter 實驗區（目前無 .ipynb，預留）
│
├── output/                                         # 所有圖表與報告產出
│   ├── ablation_de_dashboard.png                   # Model D vs E 消融 Dashboard
│   ├── feature_funnel.png                          # 特徵漏斗示意圖（Model A–E）
│   ├── model_d_equity_curve.png                    # Model D vs Buy & Hold 權益曲線
│   ├── model_d_roc_curve.png                       # Model D ROC 曲線
│   ├── model_d_underwater.png                      # Model D 回撤（Underwater）圖
│   ├── performance_dashboard_model_d_vs_bh.png     # 綜合績效儀表板（D vs B&H）
│   ├── debug_anomaly_metrics.png                   # 回測異常診斷圖（部位、波動、PnL）
│   └── …                                           # 其他 visualize_* / quant_report_* 產出
│
├── scripts/                                        # 可執行腳本與工具
│   ├── main.py                                     # CLI 主流程：載入→編碼→解讀→輸出
│   ├── dashboard.py                                # Streamlit 儀表板入口
│   ├── quant_report_model_d.py                     # Model D 完整量化報告（ROC/Equity/DD）
│   ├── quant_report_ablation_de.py                 # Model D vs E 消融比較報告
│   ├── multi_asset_validation.py                   # 多資產泛化驗證（高 Beta/低波動/指數）
│   ├── screen_candidates.py                        # 0050 成分股高 Beta 候選篩選
│   ├── diagnose_anomaly.py                         # Model D 回測異常診斷（部位與風控）
│   ├── setup_iching_db.py                          # 下載並整理 64 卦 JSON 至 data/
│   ├── reset_data.py                               # 清除 data 快取與舊模型
│   ├── convert_iching_s2t.py                       # 易經資料簡體→繁體並重建向量庫
│   ├── seed_data.py                                # 以 HEXAGRAM_MAP 產生種子 JSON
│   ├── model_d_pure_quantum.py                     # 訓練與儲存 Model D；提供推論函式
│   ├── save_model_c.py                             # 訓練 Model C 並輸出波動性模型
│   ├── tune_hyperparameters.py                     # LSTM 超參數 Optuna 優化
│   ├── feature_importance_threshold_tuning.py      # 特徵重要度與閾值掃描＋PR 曲線
│   ├── inspect_model_features.py                   # 掃描 XGBoost 模型特徵並稽核 A–E
│   ├── pure_quantum_walkforward.py                 # Pure Quantum Walk-Forward 回測
│   ├── experiment_baseline.py                      # LSTM Baseline vs Quantum 比較實驗
│   ├── experiment_xgboost.py                       # XGBoost 易經 vs 純數值實驗
│   ├── experiment_shap.py                          # SHAP 解釋與依賴圖生成
│   ├── visualize_feature_funnel.py                 # 產生特徵漏斗圖 PNG
│   ├── visualize_performance_dashboard.py          # 產生 Model D Dashboard 圖
│   ├── visualize_final_roc.py                      # 終版 ROC 視覺化
│   ├── visualize_metrics.py                        # 混淆矩陣與 ROC 等評估圖
│   ├── visualize_pnl.py                            # PnL 與走勢比較圖
│   ├── plot_pr_curve.py                            # Precision-Recall 曲線比較
│   ├── export_mermaid_to_image.py                  # Mermaid 圖轉 PNG/JPG（Playwright/Mermaid CLI）
│   ├── convert_mermaid_to_image.py                 # 使用 Mermaid CLI 匯出單一圖檔
│   ├── convert_to_ppt_format.py                    # 調整架構圖為 16:9 PPT 格式
│   ├── export_ppt_diagrams.py                      # 匯出多張架構圖供簡報使用
│   ├── generate_ppt_diagrams_highres.py            # 產出高解析度架構圖
│   ├── generate_ppt_final.py                       # 產出最終版簡報配圖
│   ├── generate_ppt_large_font.py                  # 產出大字體版簡報配圖
│   ├── fix_install.ps1                             # 安裝／環境問題修復腳本
│   └── install_dependencies.ps1                    # 一次性安裝專案依賴腳本
│
└── src/                                            # 核心邏輯模組（可被 scripts 匯入）
    ├── config.py                                   # 全域設定與六十四卦對照
    ├── data_loader.py                              # 市場資料下載與前處理
    ├── data_processor.py                           # 特徵工程與資料張量化
    ├── iching_core.py                              # 易經卦象與之卦運算核心
    ├── market_encoder.py                           # 價格資料→大衍之數→卦象編碼
    ├── model_lstm.py                               # QuantumLSTM 模型與訓練邏輯
    ├── model_definitions.py                        # Model A–E 特徵組定義
    ├── knowledge_loader.py                         # 讀取易經 JSON 並拆分文件
    ├── vector_store.py                             # Chroma 向量庫封裝與查詢
    ├── backtester.py                               # LSTM 策略回測與績效計算
    └── oracle_chat.py                              # 神諭對話邏輯（Gemini + RAG）
```

**說明：** `ARCHITECTURE_DIAGRAM_files/` 若為 ARCHITECTURE_DIAGRAM.html 的靜態資源，可一併移入 `docs/` 或保留於根目錄並在樹狀圖中標註。

---

## 二、檔案分類對照表

| 檔案名稱 | 建議位置 | 說明／角色 |
|----------|----------|------------|
| **核心邏輯（src/）** |
| config.py | src/ | 全域設定、六十四卦對照、超參預設 |
| data_loader.py | src/ | 從 Yahoo Finance 載入市場資料 |
| data_processor.py | src/ | 特徵工程、序列生成、StandardScaler、QuantumDataset |
| iching_core.py | src/ | 卦象查詢、之卦計算、易經核心邏輯 |
| market_encoder.py | src/ | 大衍之數編碼、RVOL→卦象轉換 |
| model_lstm.py | src/ | QuantumLSTM 定義與訓練流程 |
| model_definitions.py | src/ | Model A–E 特徵定義（Single Source of Truth） |
| knowledge_loader.py | src/ | 從 iching_complete.json 載入知識、產生 IChingDocument |
| vector_store.py | src/ | ChromaDB 向量庫、語義檢索 |
| backtester.py | src/ | 回測流程（載入模型、計算績效、繪圖） |
| oracle_chat.py | src/ | 神諭對話、Gemini API、市場＋易經＋RAG 整合 |
| **可執行腳本（scripts/）** |
| main.py | scripts/ | CLI 易經分析流程：資料載入→編碼→解碼→視覺化 |
| dashboard.py | scripts/ | Streamlit 儀表板入口 |
| quant_report_model_d.py | scripts/ | Model D 量化報告：ROC、Equity、Underwater、CAGR/Sharpe 等 |
| quant_report_ablation_de.py | scripts/ | Model D vs E 消融實驗、Dashboard 圖 |
| multi_asset_validation.py | scripts/ | 多資產驗證（高 Beta／低波動／指數） |
| screen_candidates.py | scripts/ | 0050 成分股篩選、Model D 候選名單 |
| diagnose_anomaly.py | scripts/ | Model D 回測異常診斷（部位、波動、單期報酬） |
| setup_iching_db.py | scripts/ | 下載 open-iching → data/iching_complete.json |
| reset_data.py | scripts/ | 清除 data 快取（CSV/PKL/chroma_db）、保留原始 JSON |
| convert_iching_s2t.py | scripts/ | 簡繁轉換 iching_complete.json、重建 ChromaDB |
| seed_data.py | scripts/ | 產生 64 卦種子 JSON（scripts/ 內既有） |
| model_d_pure_quantum.py | scripts/ | Model D 訓練、儲存 model_d_pure_quantum.json、推論介面 |
| save_model_c.py | scripts/ | 訓練 Model C、儲存 volatility_model.json |
| tune_hyperparameters.py | scripts/ | Optuna 超參優化、寫入 config/best_params.json |
| feature_importance_threshold_tuning.py | scripts/ | 特徵重要度、閾值掃描、PR 曲線 |
| inspect_model_features.py | scripts/ | 掃描 XGBoost 模型、對照 Model A–E 特徵 |
| pure_quantum_walkforward.py | scripts/ | Pure Quantum Walk-Forward 回測 |
| experiment_baseline.py | scripts/ | LSTM Baseline vs Quantum 對比實驗 |
| experiment_xgboost.py | scripts/ | XGBoost 易經 vs 純數值、特徵重要性 |
| experiment_shap.py | scripts/ | SHAP 解釋、特徵依賴圖 |
| visualize_feature_funnel.py | scripts/ | Model A–E 特徵漏斗圖 → output/feature_funnel.png |
| visualize_performance_dashboard.py | scripts/ | Model D vs B&H 績效儀表板 → output/ |
| visualize_final_roc.py | scripts/ | ROC 曲線 → data/（建議改為 output/） |
| visualize_metrics.py | scripts/ | Confusion Matrix、ROC → data/（建議改為 output/） |
| visualize_pnl.py | scripts/ | PnL 比較圖 → data/（建議改為 output/） |
| plot_pr_curve.py | scripts/ | PR 曲線比較 → data/（建議改為 output/） |
| export_mermaid_to_image.py | scripts/ | Mermaid → PNG/JPG |
| convert_mermaid_to_image.py | scripts/ | Mermaid CLI 轉圖 |
| convert_to_ppt_format.py | scripts/ | 架構圖 → 16:9 PPT 格式 |
| export_ppt_diagrams.py | scripts/ | 匯出 PPT 用架構圖 |
| generate_ppt_diagrams_highres.py | scripts/ | 高解析度 PPT 圖 |
| generate_ppt_final.py | scripts/ | 最終版 PPT 圖 |
| generate_ppt_large_font.py | scripts/ | 大字版 PPT 圖 |
| fix_install.ps1 | scripts/ | 依賴修復（PowerShell） |
| install_dependencies.ps1 | scripts/ | 依賴安裝（PowerShell） |
| **資料（data/）** |
| iching_book.json | data/ | 易經書籍原始資料 |
| iching_complete.json | data/ | 64 卦知識庫（setup 產生） |
| volatility_model.json | data/ → models/ | Model C 權重；建議移至 models/ |
| **模型產物（models/）** |
| model_d_pure_quantum.json | 根目錄 → models/ | Model D XGBoost 模型 |
| **配置（config/）** |
| best_params.json | config/ | Optuna 最佳超參（tune 產生） |
| **文件（docs/）** |
| ARCHITECTURE.md | docs/ | 架構說明 |
| ARCHITECTURE_DIAGRAM.md | docs/ | 架構圖說明 |
| ARCHITECTURE_DIAGRAM_PPT.md | docs/ | 架構圖 PPT 版說明 |
| ARCHITECTURE_DIAGRAM.html | docs/ | 架構圖 HTML |
| PPT_架構.md | docs/ | PPT 架構說明 |
| DEV_LOG.md | docs/ | 開發日誌 |
| FEATURE_FUNNEL_AUDIT.md | docs/ | 特徵漏斗審計 |
| GITHUB_UPLOAD.md | docs/ | 上傳指南 |
| LOOKAHEAD_BIAS_AUDIT.md | docs/ | Look-ahead 偏誤審計 |
| STREAMLIT_CLOUD_DEBUG.md | docs/ | Streamlit Cloud 除錯 |
| STREAMLIT_CLOUD_DEPLOY.md | docs/ | Streamlit Cloud 部署 |
| UPDATE_SUMMARY.md | docs/ | 更新摘要 |
| **產出（output/）** |
| ablation_de_dashboard.png | output/ | 已存在 |
| feature_funnel.png | output/ | 已存在；根目錄副本可刪 |
| model_d_equity_curve.png | output/ | 已存在 |
| model_d_roc_curve.png | output/ | 已存在 |
| model_d_underwater.png | output/ | 已存在 |
| performance_dashboard_model_d_vs_bh.png | 根目錄 → output/ | 移入 output/ |
| debug_anomaly_metrics.png | 根目錄 → output/ | 移入 output/ |
| feature_funnel.png（根目錄） | 刪除或合併 | 與 output/ 重複，建議只保留 output/ |
| **根目錄保留** |
| .gitignore | 根目錄 | 版控忽略規則 |
| README.md | 根目錄 | 專案說明（可擇一保留於根或 docs/） |
| requirements.txt | 根目錄 | 依賴列表 |
| .devcontainer/ | 根目錄 | 開發容器設定 |
| **預留** |
| notebooks/ | 預留 | 目前無 .ipynb，供日後實驗使用 |

---

## 三、後續須配合的程式調整（搬移後）

1. **匯入路徑**
   - 若將核心模組放入 `src/`，所有 `from config import ...`、`from data_loader import ...` 等須改為 `from src.config import ...`、`from src.data_loader import ...`，或將專案根目錄加入 `PYTHONPATH` 並把 `src` 設為 package（`__init__.py`），再依專案慣例統一 `from src.xxx` 或 `from config`（若 config 保留在根目錄）。
2. **模型與資料路徑**
   - `model_d_pure_quantum.py`、`save_model_c`、`dashboard`、`backtester` 等所參考的 `model_d_pure_quantum.json`、`data/volatility_model.json`、`data/best_model.pth` 須改為 `models/` 對應路徑。
   - `knowledge_loader`、`vector_store` 的 `data/iching_complete.json`、`data/chroma_db` 維持不變；若 `volatility_model` 移至 `models/`，需更新 `dashboard`／`save_model_c` 的讀寫路徑。
3. **圖表輸出路徑**
   - `visualize_*`、`plot_pr_curve`、`diagnose_anomaly`、`quant_report_*` 等目前寫入 `data/` 或根目錄的 PNG，建議統一改為 `output/`。
   - 針對「單一實驗生圖 Scripts」與其對應輸出圖表，建議路徑調整如下（僅為規劃，尚未改動程式碼）：  

     | Script | 目前輸出路徑 | 建議統一路徑 | 備註 |
     |--------|-------------|-------------|------|
     | `quant_report_model_d.py` | `output/model_d_roc_curve.png`、`output/model_d_equity_curve.png`、`output/model_d_underwater.png` | 維持 `output/…` | 已符合規劃，僅確認路徑存在 |
     | `quant_report_ablation_de.py` | `output/ablation_de_dashboard.png` | 維持 `output/ablation_de_dashboard.png` | 已符合規劃 |
     | `visualize_performance_dashboard.py` | `output/model_d_equity_curve.png`（讀取）、`performance_dashboard_model_d_vs_bh.png`（寫入根目錄） | 推薦改為讀寫皆在 `output/`，例如 `output/performance_dashboard_model_d_vs_bh.png` | 並將根目錄 PNG 移入 `output/` |
     | `visualize_feature_funnel.py` | `output/feature_funnel.png`、`feature_funnel.png`（根目錄複本） | 僅保留 `output/feature_funnel.png`，必要時由程式明確指定複製位置 | 根目錄副本可刪除／改為選用 |
     | `diagnose_anomaly.py` | `debug_anomaly_metrics.png`（根目錄） | 改為 `output/debug_anomaly_metrics.png` | 對應 PNG 已規劃移入 `output/` |
     | `pure_quantum_walkforward.py` | `data/pure_quantum_equity_curve.png` | 改為 `output/pure_quantum_equity_curve.png` | 與其他績效圖一致放在 `output/` |
     | `feature_importance_threshold_tuning.py` | `data/pr_curve_feature_tuning.png` | 改為 `output/pr_curve_feature_tuning.png` | PR 曲線類圖表集中於 `output/` |
     | `plot_pr_curve.py` | `data/pr_curve_comparison.png`、`data/pr_curve_comparison_oos.png` | 改為 `output/pr_curve_comparison.png`、`output/pr_curve_comparison_oos.png` | OOS／整體 PR 圖皆放 `output/` |
     | `visualize_pnl.py` | `data/pnl_comparison.png`、`data/pnl_comparison_gross_net.png`、`data/pnl_walk_forward_vs_single.png` | 改為 `output/pnl_comparison.png`、`output/pnl_comparison_gross_net.png`、`output/pnl_walk_forward_vs_single.png` | PnL 圖全部集中至 `output/` |
     | `visualize_final_roc.py` | `data/final_roc_curves.png` | 改為 `output/final_roc_curves.png` | 最終 ROC 圖屬報告產物 |
     | `visualize_metrics.py` | `data/confusion_matrix_2330.png`、`data/roc_curve_2330.png` | 改為 `output/confusion_matrix_2330.png`、`output/roc_curve_2330.png` | Confusion Matrix／ROC 圖放 `output/` |
     | `experiment_shap.py` | `data/shap_summary.png`、`data/shap_dependence_moving_lines.png`、`data/shap_dependence_energy_delta.png` | 改為 `output/shap_summary.png`、`output/shap_dependence_moving_lines.png`、`output/shap_dependence_energy_delta.png` | SHAP 可視化結果歸類為實驗圖表 |
     | `experiment_xgboost.py` | `data/feature_importance.png`、`data/feature_importance_with_iching.png`、`data/feature_importance_baseline.png`、`data/feature_importance_lean.png` | 改為 `output/feature_importance.png`、`output/feature_importance_with_iching.png`、`output/feature_importance_baseline.png`、`output/feature_importance_lean.png` | 特徵重要度圖集中於 `output/` |
     | `backtester.py` | `data/backtest_result.png` | 改為 `output/backtest_result.png` | 回測結果圖視為報告輸出 |

4. **執行入口**
   - `main.py`、`dashboard.py` 移入 `scripts/` 後，執行方式改為例如：
     - `python scripts/main.py`
     - `streamlit run scripts/dashboard.py`
   - 若希望維持 `python main.py`，可保留 `main.py` 在根目錄作為薄包裝，僅呼叫 `scripts` 內邏輯。

---

## 四、前後端串聯影響與注意事項

### 4.1 本專案的「前後端」為何？

- **前端**：`dashboard.py`（Streamlit 儀表板），使用者透過瀏覽器操作。
- **後端**：`oracle_chat`、`data_processor`、`model_d_pure_quantum`、`data_loader`、`market_encoder`、`vector_store`、`knowledge_loader` 等 Python 模組。  
- **串聯方式**：同一個 Python 行程；儀表板直接 `import` 上述模組並呼叫，**沒有獨立的 HTTP API 或前後端分離**。

### 4.2 只搬檔案、不改程式會不會導致出錯？

**會。** 若僅依計畫搬移目錄與檔案，但**未**同步修改匯入路徑與檔案路徑，會出現：

| 狀況 | 結果 |
|------|------|
| 核心模組移至 `src/`，匯入仍為 `from config`、`from data_loader` 等 | `ModuleNotFoundError`，**Dashboard 與所有腳本無法啟動** |
| `model_d_pure_quantum.json` 移至 `models/`，但 `load_model_d` 仍讀根目錄 | `FileNotFoundError`，**儀表板無法載入 Model D** |
| `volatility_model.json` 移至 `models/`，但 `save_model_c`／相關讀取仍用 `data/` | 寫入／讀取路徑錯誤，**訓練或載入失敗** |
| `dashboard.py` 移至 `scripts/`，但 Streamlit Cloud 仍跑 `streamlit run dashboard.py` | 部署**找不到入口**，上線失敗 |

因此，**必須**依「三、後續須配合的程式調整」完成所有修改，前後端串聯才會恢復正常。

### 4.3 關鍵依賴一覽（搬移時必查）

- **Dashboard → 後端**：`oracle_chat`、`data_processor`、`model_d_pure_quantum.load_model_d`。  
  - `load_model_d` 讀取 `model_d_pure_quantum.json`（目前根目錄）。
- **Oracle / 知識庫**：`data/iching_complete.json`、`data/chroma_db`；`knowledge_loader`、`vector_store` 路徑保持不變即可。
- **模型檔**：`model_d_pure_quantum.json`、`data/volatility_model.json`、`data/best_model.pth`；若移至 `models/`，**所有讀寫處**都要改為 `models/` 對應路徑。
- **匯入**：約 90+ 處 `from config`、`from data_loader` 等；搬至 `src/` 後須改為 `from src.xxx` 或透過 `PYTHONPATH`／package 設定讓 `import` 可解析。

### 4.4 正確完成調整後，會不會影響串聯？

**若依照計畫完成下列事項，前後端串聯不應受影響：**

1. 依「三」全面更新 **匯入路徑**（含 `src` 目錄與 `scripts` 內互參）。
2. 全面更新 **模型與資料路徑**（含 `models/`、`config/`、`data/` 等）。
3. 圖表與報告產出統一改寫入 **`output/`**，且讀取端若有參考則一併更新。
4. **執行方式**改為 `python scripts/main.py`、`streamlit run scripts/dashboard.py`；若有 **Streamlit Cloud／Docker**，同步修改啟動指令與工作目錄。

建議**分階段搬移與修改**，每階段後跑一次：

- `streamlit run scripts/dashboard.py`（或 `dashboard.py` 當時路徑）
- `python scripts/main.py`（或 `main.py` 當時路徑）
- `python scripts/quant_report_model_d.py`（確保 Model D 報告可產出）

通過後再進行下一階段，可降低重組過程導致整機出錯的風險。

---

## 五、總結

- **src/**：共 11 個核心模組，負責配置、資料、易經、模型、回測、神諭。
- **scripts/**：共 35+ 個可執行腳本，包含主程式、報告、實驗、視覺化、PPT/圖表、設置與工具。
- **data/**：保留原始與衍生資料；`volatility_model.json` 建議遷至 **models/**。
- **models/**：集中存放 `model_d_pure_quantum.json`、`volatility_model.json` 及任意 `*.pth`。
- **output/**：集中所有圖表與報告產出；根目錄散落的 PNG 移入並刪除重複。
- **config/**：存放 `best_params.json`。
- **docs/**：集中專案與架構說明、審計與部署文件；既有 `docs/` 子目錄結構可保留。
- **notebooks/**：預留，目前無 `.ipynb`。

若同意此計畫，可再進行實際搬移與上述程式修改；若有特定檔案希望維持原位置，可標註後一併調整計畫。
