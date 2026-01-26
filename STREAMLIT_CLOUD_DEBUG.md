# Streamlit Cloud 除錯指南：修復「象曰:」和「小象:」顯示問題

## 問題描述
本地端已經修復了無意義的「象曰:」和「小象:」標籤，但 Streamlit Cloud 上仍然顯示這些標籤。

## 原因分析
Streamlit Cloud 可能因為以下原因沒有更新：
1. **緩存問題**：Streamlit 的 `@st.cache_resource` 緩存了舊版本的 Oracle 實例
2. **部署延遲**：GitHub 推送後，Streamlit Cloud 需要時間自動重新部署
3. **版本不同步**：本地和雲端的代碼版本不一致

## 解決方案

### 步驟 1: 確認代碼已推送到 GitHub
```bash
# 檢查最新提交
git log --oneline -3

# 確認 oracle_chat.py 的修改已包含
git show HEAD:oracle_chat.py | grep -A 2 "if image:"
```

### 步驟 2: 更新緩存版本號（已完成）
在 `dashboard.py` 中，我已經將 `_ORACLE_VERSION` 從 `"2.0"` 更新為 `"2.1"`，這會強制 Streamlit 清除舊緩存。

### 步驟 3: 在 Streamlit Cloud 控制台操作

1. **登入 Streamlit Cloud**
   - 訪問：https://share.streamlit.io/
   - 登入你的帳號

2. **找到你的應用**
   - 找到 "quantum-i-ching" 應用

3. **強制重新部署**
   - 點擊應用右上角的 "⋮"（三個點）菜單
   - 選擇 **"Reboot app"** 或 **"Redeploy"**
   - 這會強制 Streamlit Cloud 重新拉取最新代碼並清除所有緩存

4. **檢查部署日誌**
   - 在應用頁面，點擊 **"Manage app"**
   - 查看 **"Logs"** 標籤
   - 確認部署成功，沒有錯誤訊息

### 步驟 4: 清除瀏覽器緩存（可選）
如果重新部署後仍然有問題：
1. 在瀏覽器中按 `Ctrl + Shift + R`（Windows）或 `Cmd + Shift + R`（Mac）強制刷新
2. 或清除瀏覽器緩存後重新訪問

### 步驟 5: 驗證修復
訪問 https://quantum-i-ching.streamlit.app/ 並測試：
1. 輸入股票代號（例如 `2330`）
2. 點擊 "Consult the Oracle"
3. 展開「易經原文」部分
4. 確認不再出現無意義的「象曰:」或「小象:」標籤

## 技術細節

### 修改內容
在 `oracle_chat.py` 的 `_get_iching_wisdom()` 方法中：

**主卦部分**（第 498-505 行）：
```python
# 修改前
text = f"{prefix}\n卦辭：{judgment}\n象曰：{image}".strip()

# 修改後
parts_list = [prefix]
if judgment:
    parts_list.append(f"卦辭：{judgment}")
if image:  # 只在有內容時才添加
    parts_list.append(f"象曰：{image}")
text = "\n".join(parts_list).strip()
```

**動爻部分**（第 522-526 行）：
```python
# 修改前
text = f"{prefix} 第 {ln} 爻：{meaning}\n小象：{xiang}".strip()

# 修改後
parts_list = [f"{prefix} 第 {ln} 爻：{meaning}"]
if xiang:  # 只在有內容時才添加
    parts_list.append(f"小象：{xiang}")
text = "\n".join(parts_list).strip()
```

### 緩存版本號更新
在 `dashboard.py` 中：
```python
_ORACLE_VERSION = "2.1"  # 從 "2.0" 更新為 "2.1" 以強制清除緩存
```

## 如果問題仍然存在

### 檢查項目
1. **確認 GitHub 上的代碼是最新的**
   ```bash
   git push origin main
   ```

2. **檢查 Streamlit Cloud 部署狀態**
   - 在 Streamlit Cloud 控制台查看部署日誌
   - 確認沒有部署錯誤

3. **手動觸發重新部署**
   - 在 Streamlit Cloud 控制台點擊 "Reboot app"

4. **檢查環境變數**
   - 確認 `GOOGLE_API_KEY` 已正確設置
   - 確認所有必要的環境變數都已配置

5. **查看 Streamlit Cloud 日誌**
   - 在 "Manage app" > "Logs" 中查看錯誤訊息
   - 如果有 Python 錯誤，會顯示在日誌中

## 常見問題

**Q: 為什麼本地端已經修復，但雲端沒有？**
A: Streamlit Cloud 使用緩存機制，需要重新部署才能清除緩存。

**Q: 需要等待多久才能看到更新？**
A: 通常幾分鐘內，但如果需要重新部署，可能需要 5-10 分鐘。

**Q: 如何確認代碼已更新？**
A: 在 Streamlit Cloud 的 "Manage app" > "Logs" 中查看部署時間，應該是最新的。
