"""Quantum I-Ching Streamlit 儀表板介面.

此模組提供使用者透過瀏覽器與 Quantum I-Ching 神諭互動的前端介面。
"""

from __future__ import annotations

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import xgboost as xgb
import numpy as np

from oracle_chat import Oracle
from data_processor import DataProcessor


# 常用台股公司名稱對應表（可依需求擴充）
TW_COMPANY_NAME_TO_TICKER: dict[str, str] = {
    "台積電": "2330",
    "臺積電": "2330",
    "台灣積體電路": "2330",
    "台灣積體電路製造": "2330",
    "鴻海": "2317",
    "鴻海精密": "2317",
    "鴻海精密工業": "2317",
    "聯發科": "2454",
    "聯發科技": "2454",
    "中鋼": "2002",
    "台達電": "2308",
}

# 反向映射：從股票代號到中文名稱（用於圖表標題顯示）
TW_TICKER_TO_CHINESE_NAME: dict[str, str] = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2002": "中鋼",
    "2308": "台達電",
}


def _normalize_tw_name(name: str) -> str:
    """簡單正規化台股公司名稱，去除空白與常見尾詞."""
    s = name.strip()
    for suffix in ["股份有限公司", "公司", "股份有限"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.replace(" ", "")


# ===== Streamlit 基本設定 =====
st.set_page_config(
    layout="wide",
    page_title="量子易經",
)


# ===== 全局樣式（支援深色/淺色模式自動切換） =====
_CUSTOM_CSS = """
<style>
/* 整體背景與字體（主區域改為淡灰色） */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f0f2f6;
    color: #333333;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stSidebar"] {
    color: #222222;
}

/* 深色模式支援 */
@media (prefers-color-scheme: dark) {
    /* 整體背景與字體（深色模式） */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    /* 側邊欄內所有文字元素 */
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }
    
    /* Streamlit 預設元素的深色模式調整 */
    .stMarkdown, .stText, .stTitle, h1, h2, h3, h4, h5, h6, p, span, div {
        color: #e2e8f0 !important;
    }
    
    /* 側邊欄輸入框 */
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea,
    [data-testid="stSidebar"] .stSelectbox > div > div > select {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }
    
    /* 側邊欄輸入框標籤 */
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #e2e8f0 !important;
    }
    
    /* 側邊欄輸入框內的文字 */
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    /* 輸入框和按鈕（主區域） */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }
    
    /* 側邊欄按鈕 */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    
    /* 側邊欄表單按鈕 */
    [data-testid="stSidebar"] button[type="submit"],
    [data-testid="stSidebar"] .stForm button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }
    
    /* 按鈕（主區域） */
    .stButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }
    
    .stButton > button:hover {
        background-color: #2563eb !important;
    }
    
    /* 側邊欄說明文字 */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-baseweb="tooltip"] {
        color: #94a3b8 !important;
    }
    
    /* 側邊欄分隔線 */
    [data-testid="stSidebar"] hr,
    [data-testid="stSidebar"] .stDivider {
        border-color: #475569 !important;
    }
    
    /* 側邊欄所有可能的文字元素（更全面的覆蓋） */
    [data-testid="stSidebar"] [class*="st"],
    [data-testid="stSidebar"] [class*="element"],
    [data-testid="stSidebar"] [class*="widget"],
    [data-testid="stSidebar"] [class*="css"] {
        color: #e2e8f0 !important;
    }
    
    /* 強制覆蓋 Streamlit 內建樣式 */
    [data-testid="stSidebar"] [style*="color"] {
        color: #e2e8f0 !important;
    }
    
    /* 側邊欄下拉選單選項 */
    [data-testid="stSidebar"] [role="listbox"],
    [data-testid="stSidebar"] [role="option"] {
        background-color: #334155 !important;
        color: #e2e8f0 !important;
    }
    
    /* 側邊欄下拉選單選項懸停 */
    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: #475569 !important;
        color: #ffffff !important;
    }
    
    /* 卡片和容器 */
    .stCard,
    [data-testid="stExpander"],
    [data-testid="stContainer"] {
        background-color: #1e293b !important;
        border-color: #475569 !important;
        color: #e2e8f0 !important;
    }
    
    /* 表格 */
    .stDataFrame,
    table {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    /* 標籤和徽章 */
    .ticker-badge {
        background-color: #334155 !important;
        border-color: #475569 !important;
        color: #e2e8f0 !important;
    }
    
    .ticker-badge span.symbol {
        color: #60a5fa !important;
    }
    
    .ticker-badge span.label {
        color: #94a3b8 !important;
    }
    
    /* Oracle Advice 容器 */
    .oracle-advice {
        background-color: #1e293b !important;
        border-color: #475569 !important;
        color: #e2e8f0 !important;
    }
    
    .oracle-advice-title {
        color: #e2e8f0 !important;
    }
    
    .oracle-disclaimer {
        color: #94a3b8 !important;
        border-top-color: #475569 !important;
    }
    
    /* 卦象標題 */
    .hexagram-title {
        color: #e2e8f0 !important;
    }
    
    .hex-label {
        color: #94a3b8 !important;
    }
    
    .hex-meta {
        color: #94a3b8 !important;
    }
    
    /* 卦象箭頭 */
    .hexagram-arrow {
        color: #60a5fa !important;
    }
    
    /* 卡片標題 */
    .stCard-header {
        color: #e2e8f0 !important;
    }
    
    /* 成功/錯誤/資訊訊息框 */
    .stSuccess {
        background-color: #065f46 !important;
        color: #d1fae5 !important;
    }
    
    .stError {
        background-color: #991b1b !important;
        color: #fee2e2 !important;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        color: #dbeafe !important;
    }
    
    .stWarning {
        background-color: #78350f !important;
        color: #fef3c7 !important;
    }
    
    /* 分隔線 */
    hr, .stDivider {
        border-color: #475569 !important;
    }
    
    /* 說明文字 */
    .stCaption {
        color: #94a3b8 !important;
    }
    
    /* 展開器標題 */
    [data-testid="stExpander"] summary {
        color: #e2e8f0 !important;
    }
    
    /* Plotly 圖表容器在深色模式下的調整 */
    .js-plotly-plot,
    .plotly {
        background-color: #1e293b !important;
    }
    
    /* Plotly 圖表文字顏色（深色模式下） */
    .js-plotly-plot .xtick text,
    .js-plotly-plot .ytick text,
    .js-plotly-plot .gtitle,
    .js-plotly-plot .g-xtitle,
    .js-plotly-plot .g-ytitle {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
}

/* 強制 Plotly 圖表文字為黑色（在所有模式下） */
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .gtitle,
.js-plotly-plot .g-xtitle,
.js-plotly-plot .g-ytitle,
.js-plotly-plot text {
    fill: #000000 !important;
    color: #000000 !important;
    stroke: none !important;  /* 移除描邊，避免文字變粗 */
    stroke-width: 0 !important;
    text-shadow: none !important;  /* 移除陰影 */
    font-weight: normal !important;  /* 確保不是粗體 */
}

/* 確保 Plotly 圖表線條為黑色 */
.js-plotly-plot .gridlayer path,
.js-plotly-plot .xlines path,
.js-plotly-plot .ylines path {
    stroke: #000000 !important;
}

/* 隱藏 Streamlit 自動生成的標題錨點連結（無意義的連結圖標） */
[data-testid="stHeaderActionElements"],
.st-emotion-cache-gi0tri,
.st-emotion-cache-kwyva7,
a[aria-label="Link to heading"] {
    display: none !important;
    visibility: hidden !important;
}

/* 主要內容卡片 */
.stCard {
    background-color: #ffffff;
    padding: 20px 22px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
    border: 1px solid #e3e6ec;
}

.stCard-header {
    font-size: 1rem;
    font-weight: 600;
    color: #1f2933;
    margin-bottom: 12px;
}

/* 卦象顯示容器（置中排列，留足夠空白） */
.hexagram-wrapper {
    display: flex;
    flex-direction: column-reverse; /* 由下往上排列，符合易經爻位 */
    gap: 8px;
    padding: 12px 4px 4px 4px;
}

.hex-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.hex-label {
    width: 40px;
    font-size: 0.78rem;
    color: #6b7280;
    text-align: right;
}

.hex-line {
    flex: 1;
    height: 14px;
    border-radius: 999px;
    position: relative;
    overflow: hidden;
    background-color: transparent;
}

/* 陽爻：實線（深藍色） */
.hex-line.yang {
    background-color: #004e92;
}

/* 陰爻：兩端紅橘色，中間留白 */
.hex-line.yin::before,
.hex-line.yin::after {
    content: "";
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    height: 100%;
    width: 38%;
    border-radius: 999px;
    background-color: #d9534f;
}

/* 深色模式下的卦象線條 */
@media (prefers-color-scheme: dark) {
    /* 陽爻：淺藍色（深色模式下更明顯） */
    .hex-line.yang {
        background-color: #60a5fa !important;
    }
    
    /* 陰爻：淺橘色（深色模式下更明顯） */
    .hex-line.yin::before,
    .hex-line.yin::after {
        background-color: #fb923c !important;
    }
    
    /* 動爻高亮（深色模式） */
    .hex-line.moving {
        box-shadow: 0 0 0 2px #fbbf24 !important;
    }
    
    @keyframes pulse-moving-dark {
        0%, 100% {
            box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.6);
        }
        50% {
            box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.9);
        }
    }
    
    .hex-line.moving {
        animation: pulse-moving-dark 2s ease-in-out infinite !important;
    }
}

.hex-line.yin::before {
    left: 0;
}

.hex-line.yin::after {
    right: 0;
}

/* 動爻高亮樣式（6=老陰，9=老陽） */
.hex-line.moving {
    box-shadow: 0 0 0 2px #ff9800;
    animation: pulse-moving 2s ease-in-out infinite;
}

@keyframes pulse-moving {
    0%, 100% {
        box-shadow: 0 0 0 2px rgba(255, 152, 0, 0.5);
    }
    50% {
        box-shadow: 0 0 0 3px rgba(255, 152, 0, 0.8);
    }
}

/* 卦象容器（用於並排顯示） */
.hexagram-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

.hexagram-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #1f2933;
    margin-bottom: 4px;
}

.hexagram-arrow {
    font-size: 2rem;
    color: #004e92;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px 0;
}

.hex-meta {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 8px;
}

.ticker-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid #d0d7e2;
    background-color: #ffffff;
    font-size: 0.8rem;
    color: #374151;
}

.ticker-badge span.symbol {
    font-weight: 700;
    color: #004e92;
}

.ticker-badge span.label {
    font-size: 0.75rem;
    color: #6b7280;
}

.oracle-advice {
    background-color: #ffffff;
    border-radius: 10px;
    border: 1px solid #d0d7e2;
    padding: 18px 20px;
}

.oracle-advice-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1f2933;
    margin-bottom: 8px;
}

.oracle-disclaimer {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 12px;
    border-top: 1px dashed #e5e7eb;
    padding-top: 8px;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# 注入 JavaScript 來動態調整 Plotly 圖表以適應深色模式
_DARK_MODE_SCRIPT = """
<script>
(function() {
    function adjustPlotlyForDarkMode() {
        // 始終使用白色背景和黑色文字/線條，確保可讀性（不管深色模式如何）
        function tryAdjust() {
            const plotlyDivs = document.querySelectorAll('.js-plotly-plot');
            if (plotlyDivs.length === 0) {
                return false;
            }
            
            plotlyDivs.forEach(function(plotDiv) {
                if (window.Plotly) {
                    try {
                        // 強制使用白色背景和黑色文字/線條，確保在所有模式下都可讀
                        // 使用 update 方法來更強制地設定所有屬性
                        window.Plotly.relayout(plotDiv, {
                            'paper_bgcolor': '#ffffff',
                            'plot_bgcolor': '#ffffff',
                            'font': {'color': '#000000', 'family': 'Arial, sans-serif'},
                            'title': {'font': {'color': '#000000', 'size': 14, 'family': 'Arial, sans-serif'}},
                            'xaxis': {
                                'gridcolor': '#000000',
                                'linecolor': '#000000',
                                'zerolinecolor': '#000000',
                                'showgrid': true,
                                'gridwidth': 1,
                                'showline': true,
                                'linewidth': 2,
                                'title': {'font': {'color': '#000000', 'size': 12, 'family': 'Arial, sans-serif'}},
                                'tickfont': {'color': '#000000', 'size': 11, 'family': 'Arial, sans-serif'},
                                'tickcolor': '#000000'
                            },
                            'yaxis': {
                                'gridcolor': '#000000',
                                'linecolor': '#000000',
                                'zerolinecolor': '#000000',
                                'showgrid': true,
                                'gridwidth': 1,
                                'showline': true,
                                'linewidth': 2,
                                'title': {'font': {'color': '#000000', 'size': 12, 'family': 'Arial, sans-serif'}},
                                'tickfont': {'color': '#000000', 'size': 11, 'family': 'Arial, sans-serif'},
                                'tickcolor': '#000000'
                            }
                        });
                        
                        // 使用 CSS 強制覆蓋 Plotly 的文字顏色
                        const plotlyContainer = plotDiv.closest('.js-plotly-plot') || plotDiv;
                        if (plotlyContainer) {
                            const style = document.createElement('style');
                            style.textContent = `
                                .js-plotly-plot .xtick text,
                                .js-plotly-plot .ytick text,
                                .js-plotly-plot .gtitle,
                                .js-plotly-plot .g-xtitle,
                                .js-plotly-plot .g-ytitle {
                                    fill: #000000 !important;
                                    color: #000000 !important;
                                    stroke: none !important;
                                    stroke-width: 0 !important;
                                    text-shadow: none !important;
                                    font-weight: normal !important;
                                }
                            `;
                            if (!document.head.querySelector('style[data-plotly-fix]')) {
                                style.setAttribute('data-plotly-fix', 'true');
                                document.head.appendChild(style);
                            }
                        }
                    } catch(e) {
                        console.log('Plotly adjustment error:', e);
                    }
                }
            });
            return true;
        }
        
        // 多次嘗試，確保圖表渲染後應用
        setTimeout(tryAdjust, 500);
        setTimeout(tryAdjust, 1000);
        setTimeout(tryAdjust, 2000);
        setTimeout(tryAdjust, 3000);
    }
    
    // 初始調整（多次嘗試以確保圖表已渲染）
    adjustPlotlyForDarkMode();
    setTimeout(adjustPlotlyForDarkMode, 500);
    setTimeout(adjustPlotlyForDarkMode, 1500);
    setTimeout(adjustPlotlyForDarkMode, 3000);
    
    // 監聽深色模式變化
    if (window.matchMedia) {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        mediaQuery.addEventListener('change', function() {
            setTimeout(adjustPlotlyForDarkMode, 100);
            setTimeout(adjustPlotlyForDarkMode, 500);
        });
    }
    
    // 監聽新圖表添加（使用更積極的策略）
    const observer = new MutationObserver(function() {
        setTimeout(adjustPlotlyForDarkMode, 100);
        setTimeout(adjustPlotlyForDarkMode, 500);
        setTimeout(adjustPlotlyForDarkMode, 1000);
    });
    observer.observe(document.body, {childList: true, subtree: true});
    
    // 監聽 Plotly 圖表渲染完成事件
    if (window.Plotly) {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(adjustPlotlyForDarkMode, 500);
            setTimeout(adjustPlotlyForDarkMode, 1500);
        });
    }
})();
</script>
"""

# 使用 components.html 來注入 JavaScript（這會在每次頁面載入時執行）
try:
    import streamlit.components.v1 as components
    components.html(_DARK_MODE_SCRIPT, height=0, width=0)
except:
    # 如果 components 不可用，使用 markdown（可能不會執行，但至少不會報錯）
    st.markdown(_DARK_MODE_SCRIPT, unsafe_allow_html=True)


# ===== Oracle 初始化（資源快取） =====
# 添加版本號以強制清除舊緩存
_ORACLE_VERSION = "2.1"  # 當 Oracle 類簽名改變時，更新此版本號以清除緩存（更新：修復象曰/小象顯示問題）

@st.cache_resource(show_spinner="正在加載中")
def get_oracle(_version: str = _ORACLE_VERSION) -> Oracle:
    """以資源快取方式初始化 Oracle，避免重複載入模型與向量資料庫.
    
    Args:
        _version: 版本號，用於強制清除緩存（當 Oracle 類簽名改變時更新）
    """
    return Oracle()


@st.cache_resource(show_spinner="正在載入波動性模型...")
def load_volatility_model(model_path: str = "data/volatility_model.json") -> xgb.XGBClassifier | None:
    """載入波動性預測模型.
    
    Args:
        model_path: 模型檔案路徑。
    
    Returns:
        載入的 XGBoost 模型，如果檔案不存在則返回 None。
    """
    if not os.path.exists(model_path):
        return None
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"載入波動性模型時發生錯誤: {e}")
        return None


def draw_hexagram(
    ritual_seq: str | None,
    binary_string: str | None,
    name: str | None,
    moving_lines: list[int] | None = None,
    show_title: bool = True,
) -> None:
    """繪製卦象圖形（淺色金融風格）.

    Args:
        ritual_seq: 儀式數字序列字串（如 "987896"）
        binary_string: 六位二進制字串（1=陽爻、0=陰爻）
        name: 卦名（中英文說明）
        moving_lines: 動爻位置列表（1-based，例如 [1, 3] 表示初爻和三爻是動爻）
        show_title: 是否顯示標題和元資料
    """
    if not binary_string or len(binary_string) != 6:
        st.warning("卦象二進制字串格式不正確，無法顯示卦象。")
        return

    # 從頂爻到初爻排列（binary_string[0] = 底爻，因此需要反轉）
    bits = list(binary_string)
    labels = ["上爻", "五爻", "四爻", "三爻", "二爻", "初爻"]
    moving_set = set(moving_lines) if moving_lines else set()

    st.markdown('<div class="hexagram-wrapper">', unsafe_allow_html=True)
    for idx, bit in enumerate(reversed(bits)):
        css_class = "yang" if bit == "1" else "yin"
        # 檢查是否為動爻（idx 從 0 開始，對應 6-idx 爻位，1-based）
        line_position = 6 - idx  # 1-based position (初爻=1, 上爻=6)
        if line_position in moving_set:
            css_class += " moving"
        label = labels[idx] if idx < len(labels) else ""
        st.markdown(
            f'<div class="hex-row">'
            f'<div class="hex-label">{label}</div>'
            f'<div class="hex-line {css_class}"></div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 額外文字說明（僅在 show_title=True 時顯示）
    if show_title and (ritual_seq or name):
        meta_parts: list[str] = []
        if ritual_seq:
            meta_parts.append(f"Ritual：{ritual_seq}")
        if binary_string:
            meta_parts.append(f"Binary：{binary_string}")
        if name:
            meta_parts.append(f"Hexagram：{name}")
        meta_text = " | ".join(meta_parts)
        st.markdown(
            f'<div class="hex-meta">{meta_text}</div>',
            unsafe_allow_html=True,
        )


def calculate_future_binary(ritual_sequence: list[int]) -> str:
    """計算之卦的二進制編碼.

    Args:
        ritual_sequence: 儀式數字序列（例如 [7, 9, 8, 8, 9, 7]）

    Returns:
        之卦的六位二進制字串（1=陽爻、0=陰爻）
    """
    # 6 (老陰) -> 1 (陽), 9 (老陽) -> 0 (陰)
    # 7 (少陽) -> 1 (陽), 8 (少陰) -> 0 (陰)
    future_bits = []
    for n in ritual_sequence:
        if n == 6:  # 老陰變少陽
            future_bits.append("1")
        elif n == 9:  # 老陽變少陰
            future_bits.append("0")
        elif n == 7:  # 少陽不變
            future_bits.append("1")
        elif n == 8:  # 少陰不變
            future_bits.append("0")
        else:
            # 預設處理
            future_bits.append("1" if n % 2 == 1 else "0")
    return "".join(future_bits)


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """簡單切割 Markdown，依標題（# / ## / ###）分段."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for line in lines:
        if line.lstrip().startswith("#"):
            # 儲存上一段
            if current_title is not None:
                sections.append((current_title, current_body))
            # 新標題
            title = line.lstrip("#").strip()
            current_title = title
            current_body = []
        else:
            current_body.append(line)

    if current_title is not None:
        sections.append((current_title, current_body))

    # 轉成 (title, content_str)
    return [(t, "\n".join(b).strip()) for t, b in sections]


def plot_volatility_gauge(probability: float) -> go.Figure:
    """創建半圓形儀表板風格的波動率 Gauge Chart（帶漸層效果和中心指針）.
    
    Args:
        probability: 波動性爆發機率（0-100）。
    
    Returns:
        Plotly Figure 物件。
    """
    # 決定狀態標籤和數字顏色
    if probability < 50:
        status_label = "Stable"
        number_color = "#2ECC71"  # 綠色
    else:
        status_label = "Risk"
        number_color = "#E74C3C"  # 紅色
    
    # 創建從綠色到紅色的漸層（通過多個 steps 模擬）
    # 從 0% (綠色，安全) 到 100% (紅色，危險) 的漸層
    def rgb_to_hex(r, g, b):
        """將 RGB 轉換為十六進制顏色."""
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    
    # 創建漸層 steps（從綠色 #2ECC71 到紅色 #E74C3C）
    # 綠色: (46, 204, 113) -> 紅色: (231, 76, 60)
    # 確保靠近 0 的部分是清楚的綠色
    gradient_steps = []
    num_steps = 25  # 增加 steps 數量以獲得更平滑的漸層
    
    for i in range(num_steps):
        # 計算當前 step 的範圍
        start_val = (i / num_steps) * 100
        end_val = ((i + 1) / num_steps) * 100
        
        # 計算漸層顏色（從綠色到紅色）
        ratio = i / (num_steps - 1)  # 0 到 1
        
        # 確保前 20% 保持清楚的綠色
        if ratio < 0.2:
            # 0-20% 保持純綠色
            r, g, b = 46, 204, 113
        else:
            # 20-100% 漸層到紅色
            adjusted_ratio = (ratio - 0.2) / 0.8  # 重新映射到 0-1
            r = 46 + (231 - 46) * adjusted_ratio
            g = 204 - (204 - 76) * adjusted_ratio
            b = 113 - (113 - 60) * adjusted_ratio
        
        color = rgb_to_hex(r, g, b)
        gradient_steps.append({
            'range': [start_val, end_val],
            'color': color,
            'thickness': 0.25  # 大幅加粗弧線
        })
    
    # 創建半圓形儀表板 Gauge Chart（使用中心指針）
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{status_label}</b>",
            'font': {'size': 22, 'family': "Arial, sans-serif", 'color': "#333333", 'weight': 'bold'}
        },
        number={
            'font': {'size': 80, 'color': number_color, 'family': "Arial, sans-serif", 'weight': 'bold'},
            'suffix': '%',
            'valueformat': '.1f'
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 3,
                'tickcolor': "#333333",
                'tickmode': 'linear',
                'tick0': 0,
                'dtick': 10,
                'tickfont': {'size': 18, 'color': "#333333", 'family': "Arial, sans-serif", 'weight': 'bold'},  # 大幅增大刻度標籤
                'ticklen': 12,
                'ticklabelstep': 1
            },
            'bar': {'color': "#000000", 'thickness': 0.2},  # 更粗的指針條
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cccccc",
            'steps': gradient_steps,  # 使用漸層 steps
            'threshold': {
                'line': {'color': "#000000", 'width': 5},  # 更粗的指針線
                'thickness': 0.95,
                'value': probability  # 指針指向當前值（從中心延伸）
            }
        }
    ))
    
    # 更新佈局（白色背景，專業風格，響應式設計）
    fig.update_layout(
        height=450,  # 進一步增加高度以容納更大的字體
        margin=dict(l=60, r=60, t=90, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#333333", 'family': "Arial, sans-serif"},
        # 確保圖表自動適應容器寬度，數字自動居中
        autosize=True,
        # 使用響應式布局
        template="plotly_white"
    )
    
    return fig


def _render_quantitative_bridge(
    raw_df: pd.DataFrame,
    ritual_sequence: list[int],
    moving_lines: list[int],
    latest_row: pd.Series | None = None,
) -> None:
    """在圖表與文字解讀之間插入「量化橋接」指標列.

    - 價格：當日收盤與昨日比較
    - RVOL：當日量 / 20 日平均量
    - 系統狀態：依動爻數量評估穩定度
    - 趨勢強度：基於 Energy_Delta 或 RVOL
    """
    if raw_df is None or raw_df.empty:
        return

    # 確保有足夠資料計算漲跌與均量
    if "Close" not in raw_df.columns:
        return

    latest_close = float(raw_df["Close"].iloc[-1])
    prev_close = float(raw_df["Close"].iloc[-2]) if len(raw_df) > 1 else latest_close
    price_delta = latest_close - prev_close
    price_delta_pct = (price_delta / prev_close * 100) if prev_close != 0 else 0.0

    volume_available = "Volume" in raw_df.columns and not raw_df["Volume"].isna().all()
    if volume_available:
        vol_series = raw_df["Volume"].astype(float)
        current_vol = float(vol_series.iloc[-1])
        avg_vol_20 = float(vol_series.tail(20).mean())
        rvol = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
    else:
        current_vol = 0.0
        avg_vol_20 = 0.0
        rvol = 1.0

    # 系統狀態：依動爻數量判斷（使用 Moving_Lines_Count）
    moving_count = len(moving_lines)
    if moving_count == 0:
        system_state = "🔒 Locked"
        system_desc = "Energy Squeeze"
    elif moving_count <= 2:
        system_state = "🌊 Flowing"
        system_desc = f"{moving_count} moving lines"
    else:
        system_state = "🔥 Chaotic"
        system_desc = f"{moving_count} moving lines"

    # 趨勢強度：基於 Energy_Delta 或 RVOL
    # 優先使用 Energy_Delta（如果 latest_row 可用）
    if latest_row is not None:
        try:
            # 嘗試從 latest_row 提取 Energy_Delta
            processor = DataProcessor()
            ritual_seq_str = "".join(str(n) for n in ritual_sequence)
            iching_features = processor.extract_iching_features(ritual_seq_str)
            energy_delta = iching_features[3]  # Energy_Delta
            
            if energy_delta > 0:
                trend_label = "Bullish"
                trend_desc = f"Energy +{energy_delta:.1f}"
            elif energy_delta < 0:
                trend_label = "Bearish"
                trend_desc = f"Energy {energy_delta:.1f}"
            else:
                trend_label = "Neutral"
                trend_desc = "Energy balanced"
        except Exception:
            # Fallback: 使用 RVOL
            if rvol > 1.5:
                trend_label = "Bullish"
                trend_desc = f"High volume (RVOL {rvol:.2f}x)"
            elif rvol < 0.8:
                trend_label = "Bearish"
                trend_desc = f"Low volume (RVOL {rvol:.2f}x)"
            else:
                trend_label = "Neutral"
                trend_desc = f"Normal volume (RVOL {rvol:.2f}x)"
    else:
        # Fallback: 使用 RVOL
        if rvol > 1.5:
            trend_label = "Bullish"
            trend_desc = f"High volume (RVOL {rvol:.2f}x)"
        elif rvol < 0.8:
            trend_label = "Bearish"
            trend_desc = f"Low volume (RVOL {rvol:.2f}x)"
        else:
            trend_label = "Neutral"
            trend_desc = f"Normal volume (RVOL {rvol:.2f}x)"

    # Top Row: Key Metrics
    st.markdown("### 📊 量化橋接 (Quantitative Bridge)")
    col_close, col_vol, col_rvol = st.columns(3)

    # 收盤價指標
    with col_close:
        delta_str = f"{price_delta:+.2f} ({price_delta_pct:+.2f}%)"
        st.metric(
            label="收盤價 (Close Price)",
            value=f"{latest_close:,.2f}",
            delta=delta_str,
            delta_color="normal" if price_delta >= 0 else "inverse",
            help="當日股票交易結束時的最後一筆成交價格。",
        )

    # 成交量指標
    with col_vol:
        if volume_available:
            st.metric(
                label="成交量 (Volume)",
                value=f"{current_vol:,.0f}",
                delta=f"20日均量: {avg_vol_20:,.0f}",
                help="當日該股票交易的總股數。反映市場的活躍程度。",
            )
        else:
            st.metric(
                label="成交量 (Volume)",
                value="N/A",
                delta="資料不足",
                help="當日該股票交易的總股數。反映市場的活躍程度。",
            )

    # RVOL 指標
    with col_rvol:
        if volume_available and avg_vol_20 > 0:
            rvol_str = f"{rvol:.2f}x"
            st.metric(
                label="RVOL (相對成交量)",
                value=rvol_str,
                delta="高於 20 日均量" if rvol > 1 else "低於 / 接近 20 日均量",
                delta_color="inverse" if rvol > 1.5 else "normal",
                help="Relative Volume。當日成交量與過去一段時間平均成交量的比值。RVOL > 1 代表今日成交量放大。計算方式：今日成交量 / 過去 20 日平均成交量。",
            )
        else:
            st.metric(
                label="RVOL (相對成交量)",
                value="N/A",
                delta="資料不足",
                help="Relative Volume。當日成交量與過去一段時間平均成交量的比值。RVOL > 1 代表今日成交量放大。計算方式：今日成交量 / 過去 20 日平均成交量。",
            )

    # Middle Row: System State & Trend Strength
    col_state, col_trend = st.columns(2)

    # 系統狀態
    with col_state:
        st.metric(
            label="系統狀態 (System State)",
            value=system_state,
            delta=system_desc,
            help="對應易經的『動爻』數量。動爻越多，代表市場內部能量越不穩定，變盤機率越高。0 動爻：能量擠壓，結構穩定。1-2 動爻：能量流動，趨勢醞釀。3+ 動爻：能量混亂，變盤機率高。",
        )

    # 趨勢強度
    with col_trend:
        # 添加熊/牛圖標
        trend_display = f"{trend_label} {'🐂' if trend_label == 'Bullish' else ('🐻' if trend_label == 'Bearish' else '➖')}"
        st.metric(
            label="趨勢強度 (Trend Strength)",
            value=trend_display,
            delta=trend_desc,
            delta_color="normal" if trend_label == "Bullish" else ("inverse" if trend_label == "Bearish" else "off"),
            help="基於能量變化（Energy_Delta）或相對成交量（RVOL）判斷。正值表示能量增強，負值表示能量減弱。",
        )


def _classify_action_tone(text: str) -> str:
    """根據文字內容推斷操作建議色彩：buy / sell / neutral."""
    t = text.lower()
    # 偏多 / 買進
    buy_keywords = [
        "買進",
        "加碼",
        "佈局",
        "偏多",
        "看多",
        "buy",
        "long",
    ]
    sell_keywords = [
        "賣出",
        "減碼",
        "停損",
        "風險",
        "觀望",
        "看空",
        "sell",
        "short",
    ]
    if any(k in text for k in buy_keywords) or any(k in t for k in buy_keywords):
        return "buy"
    if any(k in text for k in sell_keywords) or any(k in t for k in sell_keywords):
        return "sell"
    return "neutral"


def _parse_ai_response(ai_answer: str) -> dict | None:
    """解析 AI 回應但不顯示，只返回解析後的數據."""
    if not ai_answer:
        return None

    # --- 優先嘗試：依 Markdown 標題分段 ---
    sections = _split_markdown_sections(ai_answer)
    summary_text: str | None = None
    action_text: str | None = None
    source_text: str | None = None
    decoding_text: str | None = None

    if sections:
        for title, body in sections:
            lower_title = title.lower()
            if ("投資快訊" in title or "executive" in lower_title) and not summary_text:
                summary_text = body.strip()
            elif (
                "操作建議" in title
                or "action plan" in lower_title
                or "操作策略" in title
            ) and not action_text:
                action_text = body.strip()
            elif (
                "易經原文" in title
                or "經文" in title
                or "the source" in lower_title
            ) and not source_text:
                source_text = body.strip()
            elif (
                "現代解讀" in title
                or "deep dive" in lower_title
                or "解析" in title
            ) and not decoding_text:
                decoding_text = body.strip()

        # 若仍有缺漏，嘗試以剩餘段落補齊
        if summary_text is None and sections:
            summary_text = sections[0][1].strip()
        if decoding_text is None and sections:
            used_bodies = {summary_text, action_text, source_text}
            remain_parts = [
                body.strip()
                for _, body in sections
                if body.strip() and body.strip() not in used_bodies
            ]
            decoding_text = "\n\n".join(remain_parts).strip() if remain_parts else None

    # --- Fallback：純文字斷行解析 ---
    if summary_text is None or action_text is None:
        paragraphs = [p.strip() for p in ai_answer.split("\n\n") if p.strip()]
        if summary_text is None:
            summary_text = paragraphs[0] if paragraphs else ai_answer

        if action_text is None:
            action_candidates = [
                p
                for p in paragraphs
                if ("操作建議" in p or "建議" in p or "策略" in p)
            ]
            action_text = action_candidates[0] if action_candidates else summary_text

        if source_text is None:
            source_lines: list[str] = []
            for line in ai_answer.splitlines():
                if (
                    "《" in line
                    or "卦辭" in line
                    or "彖傳" in line
                    or "象傳" in line
                    or "爻辭" in line
                ):
                    source_lines.append(line)
            source_text = "\n".join(source_lines).strip() or None

        if decoding_text is None:
            remaining_text = (
                "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
            )
            decoding_text = remaining_text or None

    # 最終 fallback：全部使用原文
    if summary_text is None:
        summary_text = ai_answer
    if action_text is None:
        action_text = ai_answer
    
    # 返回解析後的內容（不顯示）
    return {
        "summary": summary_text,
        "action": action_text,
        "source": source_text,
        "decoding": decoding_text,
        "full_answer": ai_answer
    }


def render_ai_response(ai_answer: str) -> dict | None:
    """依資訊層級呈現 AI 回應，避免重複段落."""
    if not ai_answer:
        st.info("目前尚未取得 Oracle 回應。")
        return None

    # 先解析數據
    response_data = _parse_ai_response(ai_answer)
    if not response_data:
        return None
    
    summary_text = response_data.get("summary", "")
    action_text = response_data.get("action", "")
    source_text = response_data.get("source", "")
    decoding_text = response_data.get("decoding", "")

    # --- 呈現層級 ---
    st.markdown("## 🔮 Oracle's Advice / 卜卦解讀")

    # 1. Executive Summary
    st.markdown("### 🚀 投資快訊 (Executive Summary)")
    st.markdown(summary_text)

    # 2. Action Plan（永遠顯示，且僅顯示一次）
    st.markdown("### 🎯 關鍵操作建議 (Action Plan)")
    tone = _classify_action_tone(action_text)
    if tone == "buy":
        st.success(action_text)
    elif tone == "sell":
        st.error(action_text)
    else:
        st.info(action_text)

    # 3. 詳細內容（易經原文 + 現代解讀）置於單一 expander
    with st.expander("📜 點擊查看：易經原文與詳細現代解讀", expanded=False):
        st.markdown("#### 📖 易經原文 (The Source)")
        if source_text:
            st.markdown(source_text)
        else:
            st.markdown("_目前回應中未偵測到明確的易經原文段落。_")

        st.divider()

        st.markdown("#### 💡 現代解讀 (Deep Dive)")
        if decoding_text:
            st.markdown(decoding_text)
        else:
            st.markdown("_目前回應中未偵測到額外的現代金融解讀內容。_")

    st.caption(
        "以上內容僅供研究與教育參考，不構成任何投資建議或買賣邀約，實際投資決策請自行評估風險。"
    )
    
    # 返回解析後的內容，供追問系統使用
    return response_data


def render_followup_system(
    oracle: Oracle,
    response_data: dict,
    full_answer: str,
    ticker: str,
    market_state: dict,
    market_type: str
) -> None:
    """渲染追問系統，讓用戶可以針對已生成的 Oracle 回答進行追問."""
    # 初始化 session_state 中的追問歷史
    if 'followup_history' not in st.session_state:
        st.session_state.followup_history = []
    
    # 獨立容器顯示追問系統
    with st.container(border=True):
        st.markdown("## 💬 AI 小幫手 / Follow-up Questions")
        st.caption("您可以針對上述卜卦解讀進行追問，AI 會根據原始回答為您進一步說明。")
        
        # 顯示所有追問歷史（討論串形式）- 先顯示已生成的內容，避免變淡
        if st.session_state.followup_history:
            st.divider()
            st.markdown("### 💬 討論串")
            # 從舊到新顯示所有追問（已生成的內容持久顯示）
            for idx, item in enumerate(st.session_state.followup_history, 1):
                with st.container():
                    st.markdown(f"**Q{idx}: {item['question']}**")
                    st.markdown(item['answer'])
                    if idx < len(st.session_state.followup_history):
                        st.divider()
        
        # 處理待處理的追問（在顯示歷史後處理，確保新回答在底部）
        # 使用獨立的容器來隔離加載狀態，確保不影響其他區塊
        loading_container = st.empty()
        if 'pending_followup_question' in st.session_state and st.session_state.pending_followup_question:
            followup_question = st.session_state.pending_followup_question
            # 清除待處理標記
            del st.session_state.pending_followup_question
            
            if followup_question.strip():
                # 在獨立的容器中顯示加載提示（只在 AI 小幫手區塊內）
                with loading_container.container():
                    st.info("🤔 正在思考中...")
                
                try:
                    # 構建追問的提示詞
                    followup_prompt = f"""你是一位專業的 AI 金融顧問，正在協助用戶理解 Quantum I-Ching 的卜卦解讀。

**原始卜卦解讀內容：**
{full_answer}

**股票資訊：**
- 股票代號：{ticker}
- 市場類型：{market_type}

**用戶的追問：**
{followup_question.strip()}

請根據上述原始卜卦解讀內容，針對用戶的追問提供詳細、專業且易懂的回答。回答時：
1. 直接回應用戶的問題，不要重複原始回答的內容
2. 如果問題涉及原始回答中的特定部分，請引用並詳細說明
3. 使用繁體中文回答
4. 保持專業但易懂的語氣
5. 如果問題超出原始回答的範圍，可以基於易經原理和金融知識進行合理推論

請開始回答："""

                    # 使用 Gemini 模型生成回答
                    response = oracle.model.generate_content(followup_prompt)
                    
                    # 清除加載提示
                    loading_container.empty()
                    
                    if response and hasattr(response, 'text'):
                        followup_answer = response.text
                        
                        # 保存到歷史記錄
                        st.session_state.followup_history.append({
                            "question": followup_question.strip(),
                            "answer": followup_answer
                        })
                        
                        # 重新運行以顯示新回答
                        st.rerun()
                    else:
                        loading_container.empty()
                        st.error("無法生成回答，請稍後再試。")
                        
                except Exception as e:
                    loading_container.empty()
                    st.error(f"發生錯誤：{str(e)}")
        
        # 使用表單來輸入新問題
        with st.form(key="followup_form", clear_on_submit=True):
            followup_question = st.text_input(
                "請輸入您的問題",
                placeholder="例如：為什麼建議持有？風險在哪裡？這個卦象的具體含義是什麼？",
                key="followup_input",
                label_visibility="visible"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submitted = st.form_submit_button("發送", use_container_width=True, type="primary")
        
        # 處理表單提交（保存問題到 session_state，然後 rerun）
        if submitted:
            if followup_question and followup_question.strip():
                # 保存問題到 session_state，因為表單提交後輸入框會被清除
                st.session_state['pending_followup_question'] = followup_question.strip()
                st.rerun()
            else:
                st.warning("請輸入問題後再發送。")
        
        # 清除追問歷史按鈕（放在最後）
        if st.session_state.followup_history:
            if st.button("清除所有追問", use_container_width=True, key="clear_followup_btn"):
                st.session_state.followup_history = []
                st.rerun()


def render_volatility_gauge(
    raw_df: pd.DataFrame,
    ritual_sequence: list[int],
    latest_row: pd.Series
) -> None:
    """顯示波動率 Gauge Chart（簡約風格）.
    
    使用精簡版 XGBoost 模型預測波動性爆發機率，並以簡約的 Gauge Chart 視覺化。
    
    Args:
        raw_df: 原始市場資料 DataFrame。
        ritual_sequence: 儀式數字序列。
        latest_row: 最新一筆編碼資料（包含 Close, Volume, RVOL, Daily_Return）。
    """
    # 載入模型
    model = load_volatility_model()
    if model is None:
        st.warning("⚠️ 波動性模型尚未訓練，請先執行 `python save_model_c.py`")
        return
    
    try:
        # 提取易經特徵
        processor = DataProcessor()
        ritual_seq_str = "".join(str(n) for n in ritual_sequence)
        iching_features = processor.extract_iching_features(ritual_seq_str)
        
        # 提取數值特徵（從最新一筆資料）
        try:
            close_val = float(latest_row['Close'])
            volume_val = float(latest_row.get('Volume', 0))
            rvol_val = float(latest_row.get('RVOL', 1.0))
            daily_return_val = float(latest_row.get('Daily_Return', 0))
        except (KeyError, ValueError) as e:
            st.warning(f"無法提取數值特徵: {e}")
            return
        
        # 只使用精簡特徵：Moving_Lines_Count 和 Energy_Delta
        moving_lines_count = iching_features[2]  # Moving_Lines_Count
        energy_delta = iching_features[3]  # Energy_Delta
        
        # 組合特徵向量（順序必須與訓練時一致）
        feature_vector = np.array([
            close_val,              # Close
            volume_val,             # Volume
            rvol_val,               # RVOL
            daily_return_val,       # Daily_Return
            moving_lines_count,     # Moving_Lines_Count
            energy_delta            # Energy_Delta
        ]).reshape(1, -1)
        
        # 預測波動性爆發機率
        prob_breakout = model.predict_proba(feature_vector)[0, 1]
        prob_percent = prob_breakout * 100
        
        # 顯示標題（使用原生 help 參數）
        st.subheader("波動率爆發機率 (Volatility Probability)", help="基於易經動爻與能量差計算的波動率擠壓指標。使用 XGBoost Model C 預測未來 5 天內波動性爆發（|Return_5d| > 3%）的機率。")
        
        # 使用新的簡約 Gauge Chart 函數
        fig = plot_volatility_gauge(prob_percent)
        
        # 顯示 Gauge Chart
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示簡潔的解釋性資訊
        st.caption(
            f"動爻數量: {int(moving_lines_count)} | 能量變化: {energy_delta:.2f} | 預測機率: {prob_percent:.1f}%"
        )
        
        # 顯示特徵值（用於調試，可選）
        with st.expander("🔍 查看特徵值（用於調試）", expanded=False):
            st.markdown(f"**數值特徵：**")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text("Close:")
            with col2:
                st.text(f"{close_val:.2f}")
            st.caption("當日股票交易結束時的最後一筆成交價格。")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text("Volume:")
            with col2:
                st.text(f"{volume_val:,.0f}")
            st.caption("當日該股票交易的總股數。反映市場的活躍程度。")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text("RVOL:")
            with col2:
                st.text(f"{rvol_val:.2f}")
            st.caption("Relative Volume。當日成交量與過去一段時間平均成交量的比值。RVOL > 1 代表今日成交量放大。")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text("Daily Return:")
            with col2:
                st.text(f"{daily_return_val:.4f}")
            st.caption("今日收盤價與昨日收盤價的變化百分比。計算公式：(今收 - 昨收) / 昨收 * 100%。")
            
            st.markdown(f"**易經特徵：**")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text("Moving_Lines_Count:")
            with col2:
                st.text(f"{moving_lines_count:.0f}")
            st.caption("易經卦象中發生變化的爻的數量。動爻越多，代表市場內部能量越不穩定，變盤機率越高。")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.text("Energy_Delta:")
            with col2:
                st.text(f"{energy_delta:.2f}")
            st.caption("能量變化指標。計算方式：未來卦陽爻數量 - 主卦陽爻數量。正值表示能量增強，負值表示能量減弱。")
            
            st.markdown(f"**預測結果：**")
            st.markdown(f"- 波動性爆發機率: {prob_percent:.2f}%")
            
    except Exception as e:
        st.error(f"計算波動性預測時發生錯誤: {e}")


def render_sentiment_gauge(binary_string: str | None) -> None:
    """根據卦象二進制字串顯示多空情緒儀表（自訂 HTML/CSS 樣式）."""
    if not isinstance(binary_string, str) or len(binary_string) != 6:
        return
    yang_count = binary_string.count("1")
    yang_score = int(yang_count / 6 * 100)

    # 顏色邏輯：>50% 紅色（多頭），<=50% 綠色（空頭）
    bar_color = "#ff4b4b" if yang_score > 50 else "#00c853"
    emoji = "🐂" if yang_score > 50 else "🐻"
    sentiment_label = "多方氣勢強" if yang_score > 50 else "空方壓力重"

    st.markdown("### 🔮 多方能量 (Bullish Probability)")

    # 自訂 HTML/CSS 進度條（含 tooltip）
    tooltip_text = "基於『之卦（未來）』的陽爻比例計算。陽爻越多，代表多方氣勢越強；陰爻越多，代表空方壓力越重。"
    st.markdown(
        f"""
    <div style="position: relative;" title="{tooltip_text}">
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 3px; margin-bottom: 8px; cursor: help;" title="{tooltip_text}">
            <div style="width: {yang_score}%; background-color: {bar_color}; height: 24px; border-radius: 8px; transition: width 0.5s ease-in-out; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <span style="color: white; font-weight: 600; font-size: 0.85rem;">{yang_score}%</span>
            </div>
        </div>
    </div>
    <p style="font-size: 0.9rem; color: #374151; margin-top: 4px; margin-bottom: 0;" title="{tooltip_text}">
        {emoji} <strong>{sentiment_label}</strong> - 多方能量約為 {yang_score}%（以陽爻比例估算）
        <span style="font-size: 0.75rem; color: #6b7280; margin-left: 4px;">(懸停查看說明)</span>
    </p>
    """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Streamlit 入口主程式."""
    # ===== 側邊欄設定 =====
    with st.sidebar:
        st.title("🔮 設定 (Settings)")

        market_type = st.selectbox(
            "市場類型 / Market Type",
            options=["TW", "US", "CRYPTO"],
            index=0,  # 預設台股
            help="TW: 台灣股市（如 2330、2317）\nUS: 美股（如 NVDA、AAPL）\nCRYPTO: 加密貨幣（如 BTC、ETH）"
        )

        user_ticker = st.text_input(
            "股票代號 / Ticker Symbol",
            value="2330" if market_type == "TW" else ("NVDA" if market_type == "US" else "BTC"),
            max_chars=20,
            help="台股可直接輸入數字（如 2330），美股輸入代碼（如 NVDA），加密貨幣輸入代碼（如 BTC）"
        ).strip().upper()

        # 固定問題為「目前趨勢」
        question = "目前趨勢"

        consult = st.button("Consult the Oracle (卜卦)", use_container_width=True)

        st.markdown("---")
        
        # 清除緩存按鈕（用於解決緩存問題）
        if st.button("🔄 清除緩存並重新載入", use_container_width=True, help="如果遇到 TypeError，請點擊此按鈕清除緩存"):
            get_oracle.clear()
            st.success("緩存已清除！請重新點擊「Consult the Oracle」按鈕。")
            st.rerun()
        
        st.markdown("---")
        st.caption(
            "說明：系統會根據近期股價結構（支援台股、美股、加密貨幣）生成卦象，"
            "並透過 Gemini 模型給出結合易經與現代金融的解讀。"
            "所有內容僅供研究與教育參考，並不構成任何投資建議或買賣指示。"
        )

    # ===== 主畫面標題 =====
    st.title("量子易經 股市卜卦系統")
    st.markdown(
        "結合 **量化價格結構** 與 **易經六十四卦** 的 AI 金融解讀介面。"
    )

    # 主要佈局：左側 K 線圖（2/3），右側卦象卡片（1/3）
    col_chart, col_hex = st.columns([2, 1])

    # 保存 consult 狀態到 session_state（用於追問系統保持狀態）
    if consult:
        st.session_state['last_consult_ticker'] = user_ticker
        st.session_state['last_consult_market_type'] = market_type
        # 清除之前的追問歷史
        if 'followup_history' in st.session_state:
            st.session_state.followup_history = []
    
    # 檢查是否應該顯示結果（包括追問後的狀態）
    should_show_results = consult or (
        st.session_state.get('last_consult_ticker') == user_ticker and
        st.session_state.get('last_consult_market_type') == market_type and
        st.session_state.get('last_consult_ticker') is not None
    )
    
    if should_show_results:
        if not user_ticker:
            st.error("請輸入有效的股票代號（Ticker）。")
            return

        # 根據使用者選擇的市場類型格式化 ticker
        original_input = user_ticker
        display_name_override: str | None = None
        resolved_code: str | None = None  # 提升到外層作用域，供後續使用

        if market_type == "TW":
            # 台股：支援「公司名稱」或「股票代號」
            norm = _normalize_tw_name(user_ticker)

            if user_ticker.isdigit():
                resolved_code = user_ticker
            elif user_ticker.endswith(".TW") and user_ticker[:-3].isdigit():
                resolved_code = user_ticker[:-3]
            elif norm in TW_COMPANY_NAME_TO_TICKER:
                resolved_code = TW_COMPANY_NAME_TO_TICKER[norm]
                display_name_override = original_input  # 優先顯示使用者輸入的中文名稱

            if resolved_code is None:
                st.error(
                    "台股目前僅支援「股票代號」或已知公司名稱。"
                    "請輸入正確的台股代號（如 2330），或將公司名稱加入程式中的對應表。"
                )
                return

            backend_ticker = f"{resolved_code}.TW"

        elif market_type == "CRYPTO":
            # 加密貨幣：補 -USD，已有 -USD 則直接使用
            if user_ticker.endswith("-USD"):
                backend_ticker = user_ticker
            else:
                backend_ticker = f"{user_ticker}-USD"
        else:  # US
            # 美股：直接使用，不補後綴
            backend_ticker = user_ticker
        
        # 在確定 backend_ticker 後，清除對應的 Oracle's Advice 緩存（新的 consult 需要重新生成）
        if consult:
            oracle_cache_key = f"oracle_answer_{backend_ticker}_{market_type}"
            if oracle_cache_key in st.session_state:
                del st.session_state[oracle_cache_key]

        try:
            oracle = get_oracle()
        except Exception as e:  # pragma: no cover - 主要是環境設定錯誤
            st.error(
                "無法初始化 Quantum I-Ching Oracle，請確認 GOOGLE_API_KEY "
                "與向量資料庫設定是否正確。\n\n"
                f"詳細錯誤：{e}"
            )
            return

        # ===== Step 1: 取得市場資料與卦象 =====
        try:
            raw_df = oracle.data_loader.fetch_data(tickers=[backend_ticker], market_type=market_type)
        except Exception as e:
            st.error(f"下載市場資料時發生錯誤：{e}")
            return

        if raw_df is None or raw_df.empty:
            st.error(
                f"無法取得 `{user_ticker}` 的市場資料，"
                "請確認代號是否正確或日期區間內是否有交易資料。"
            )
            return

        try:
            encoded_df = oracle.encoder.generate_hexagrams(raw_df)
        except Exception as e:
            st.error(f"將市場資料轉換為易經卦象時發生錯誤：{e}")
            return

        if (
            encoded_df is None
            or encoded_df.empty
            or "Ritual_Sequence" not in encoded_df.columns
        ):
            st.error(
                "資料不足以生成卦象（需要至少 26 天以上的有效價格資料）。"
            )
            return

        # 過濾掉 Ritual_Sequence 或 Hexagram_Binary 為空的列
        valid_rows = encoded_df.dropna(
            subset=["Ritual_Sequence", "Hexagram_Binary"]
        )
        if valid_rows.empty:
            st.error(
                "雖然成功下載價格資料，但尚未累積足夠的技術指標樣本以生成完整卦象。"
            )
            return

        latest_row = valid_rows.iloc[-1]

        ritual_sequence_str = str(latest_row["Ritual_Sequence"])
        try:
            ritual_sequence = [int(ch) for ch in ritual_sequence_str]
        except ValueError:
            st.error("儀式數字序列格式錯誤，無法解析。")
            return

        if len(ritual_sequence) != 6:
            st.error(
                f"儀式數字序列長度不正確（期望 6 位，實際為 {len(ritual_sequence)}）。"
            )
            return

        binary_code = str(latest_row["Hexagram_Binary"])
        if not binary_code or len(binary_code) != 6:
            st.error("卦象二進制編碼缺失或格式錯誤，無法顯示卦象。")
            return

        # 使用 IChingCore 取得卦象名稱（本卦）
        try:
            interpretation = oracle.core.interpret_sequence(ritual_sequence)
            current_hex = interpretation.get("current_hex", {}) or {}
            hexagram_name_full = current_hex.get("name", "Unknown")
            chinese_name = current_hex.get("nature", "?")
            hexagram_id = current_hex.get("id", 0)
        except Exception as e:
            st.error(f"解析卦象資訊時發生錯誤：{e}")
            return

        # 英文名稱可能含括號，取主要名稱
        if "(" in hexagram_name_full:
            hexagram_name = hexagram_name_full.split("(", 1)[0].strip()
        else:
            hexagram_name = hexagram_name_full

        # 構造單一來源的市場狀態（Calculate Once, Use Everywhere）
        current_market_state: dict = {
            "ticker": backend_ticker,
            "market_type": market_type,
            "raw_df": raw_df,
            "encoded_df": encoded_df,
            "latest_row_index": latest_row.name,
            "ritual_sequence": ritual_sequence,
            "ritual_sequence_str": ritual_sequence_str,
            "binary_code": binary_code,
            "hexagram_id": hexagram_id,
            "hex_name": hexagram_name_full,
            "hex_name_stripped": hexagram_name,
            "chinese_name": chinese_name,
        }

        # ===== Step 2: 市場 K 線圖（左側） =====
        stock_name: str | None = None
        # 嘗試從 yfinance 取得標的名稱（台股 / 美股皆適用）
        try:
            formatted_ticker = oracle.data_loader._format_ticker(backend_ticker)  # type: ignore[attr-defined]
            info = yf.Ticker(formatted_ticker).info or {}
            stock_name = info.get("shortName") or info.get("longName")
        except Exception:
            stock_name = None

        # 決定顯示用代號與名稱（確保圖表標題清楚標示「代號 + 名稱」）
        display_code = backend_ticker
        
        # 優先順序：display_name_override > 台股中文名稱 > yfinance 英文名稱 > 原始輸入
        if display_name_override:
            display_name = display_name_override
        elif market_type == "TW" and resolved_code:
            # 台股：嘗試從反向映射取得中文名稱
            chinese_name_from_map = TW_TICKER_TO_CHINESE_NAME.get(resolved_code)
            display_name = chinese_name_from_map or stock_name or original_input
        else:
            display_name = stock_name or original_input

        with col_chart:
                chart_df = raw_df.tail(60).copy()
                if chart_df.empty:
                    st.warning("近期 60 日內資料不足，無法繪製 K 線圖。")
                else:
                    # 確保索引為 DatetimeIndex 以利圖表顯示
                    chart_df = chart_df.reset_index().rename(
                        columns={"index": "Date"}
                    )

                    date_col = (
                        "Date" if "Date" in chart_df.columns else chart_df.columns[0]
                    )

                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=chart_df[date_col],
                                open=chart_df["Open"],
                                high=chart_df["High"],
                                low=chart_df["Low"],
                                close=chart_df["Close"],
                                increasing_line_color="#22c55e",
                                decreasing_line_color="#ef4444",
                                name="Price",
                            )
                        ]
                    )

                    # 使用黑色文字和線條，確保在白色背景上清晰可見
                    fig.update_layout(
                        title=dict(
                            text=(
                                f"{display_code} ({display_name})"
                                + f" - {chinese_name} / {hexagram_name} "
                                f"(最近 60 日價格走勢)"
                            ),
                            font=dict(
                                color="#000000",
                                size=14,
                                family="Arial, sans-serif",
                            ),
                            x=0.02,  # 左對齊
                            xanchor="left",
                        ),
                        template="plotly_white",
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        margin=dict(l=10, r=10, t=40, b=10),
                        # 全域字體設定（黑色，確保所有文字都可讀）
                        font=dict(color="#000000", size=12, family="Arial, sans-serif"),
                    )
                    
                    # 添加縱向交替色塊（對齊 X 軸主要時間刻度）
                    # 使用兩種低對比度的深灰色交替排列
                    dates = pd.to_datetime(chart_df[date_col])
                    min_date = dates.min()
                    max_date = dates.max()
                    
                    # 淺灰色與白色交替排列
                    color1 = "#e8e8e8"  # 淺灰色
                    color2 = "#ffffff"  # 白色
                    
                    # 根據日期範圍計算色塊數量（約每 2 週一個色塊，對齊主要時間刻度）
                    date_range_days = (max_date - min_date).days
                    # 計算合理的色塊數量（每 14 天一個，但至少 4 個，最多 12 個）
                    num_bands = max(4, min(12, int(date_range_days / 14)))
                    
                    # 創建交替色塊，對齊日期邊界
                    shapes = []
                    band_width_days = date_range_days / num_bands
                    
                    for i in range(num_bands):
                        # 計算每個色塊的起始和結束日期
                        start_offset = i * band_width_days
                        end_offset = (i + 1) * band_width_days
                        
                        start_date = min_date + pd.Timedelta(days=start_offset)
                        end_date = min_date + pd.Timedelta(days=end_offset)
                        
                        # 最後一個色塊延伸到最大日期
                        if i == num_bands - 1:
                            end_date = max_date
                        
                        # 交替使用兩種顏色
                        band_color = color1 if i % 2 == 0 else color2
                        
                        shapes.append(
                            dict(
                                type="rect",
                                xref="x",
                                yref="paper",  # 使用 paper 參考以覆蓋整個 Y 軸範圍
                                x0=start_date,
                                y0=0,
                                x1=end_date,
                                y1=1,
                                fillcolor=band_color,
                                opacity=1.0,  # 完全不透明，確保色塊清晰可見
                                layer="below",  # 放在 K 線下方
                                line_width=0,  # 無邊框
                            )
                        )
                    
                    # 使用 update_xaxes 和 update_yaxes 強制設定所有屬性
                    # X軸不使用網格線，Y軸使用深灰色網格線
                    fig.update_xaxes(
                        title="Date",
                        title_font=dict(color="#000000", size=12, family="Arial, sans-serif"),
                        tickfont=dict(color="#000000", size=11, family="Arial, sans-serif"),
                        # X軸標籤置中對齊（使用 period 模式讓標籤在刻度區間中間）
                        ticklabelmode='period',
                        # 隱藏軸線（完全透明）
                        linecolor="rgba(0,0,0,0)",  # 完全透明
                        zeroline=False,  # 隱藏零線
                        showgrid=False,  # X軸不使用網格線
                        showline=False,  # 隱藏軸線邊框
                        rangeslider=dict(visible=False),
                    )
                    
                    fig.update_yaxes(
                        title="Price",
                        title_font=dict(color="#000000", size=12, family="Arial, sans-serif"),
                        tickfont=dict(color="#000000", size=11, family="Arial, sans-serif"),
                        gridcolor="#808080",  # 深灰色網格線（不是黑色）
                        # 隱藏軸線（完全透明）
                        linecolor="rgba(0,0,0,0)",  # 完全透明
                        zeroline=False,  # 隱藏零線
                        showgrid=True,  # Y軸使用網格線
                        gridwidth=1,
                        showline=False,  # 隱藏軸線邊框
                    )
                    
                    # 將色塊添加到圖表
                    fig.update_layout(shapes=shapes)

                    st.plotly_chart(fig, use_container_width=True)

        # ===== Step 3: 卦象視覺化卡片（右側） =====
        with col_hex:
                # 直接使用簡潔佈局，不額外加外框
                st.markdown("#### I-Ching 市場卦象")
                st.markdown(
                    f'<div class="ticker-badge">'
                    f'<span class="symbol">{display_code}</span>'
                    f'<span class="label"> / {display_name} / 市場結構卦象</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # 檢查是否有動爻（6 或 9）
                moving_lines = [i + 1 for i, n in enumerate(ritual_sequence) if n in (6, 9)]
                has_moving_lines = len(moving_lines) > 0

                if has_moving_lines:
                    # 有動爻：顯示本卦 -> 之卦
                    # 計算之卦資訊
                    future_binary = calculate_future_binary(ritual_sequence)
                    try:
                        # 使用 IChingCore 取得之卦名稱
                        future_hex_info = oracle.core.get_hexagram_name(future_binary)
                        future_chinese_name = future_hex_info.get("nature", "?")
                        future_hex_name_full = future_hex_info.get("name", "Unknown")
                        if "(" in future_hex_name_full:
                            future_hex_name = future_hex_name_full.split("(", 1)[0].strip()
                        else:
                            future_hex_name = future_hex_name_full
                    except Exception as e:
                        future_chinese_name = "?"
                        future_hex_name = "Unknown"

                    # 使用三欄佈局：本卦 | 箭頭 | 之卦
                    col_main, col_arrow, col_future = st.columns([1, 0.2, 1])

                    with col_main:
                        st.markdown('<div class="hexagram-container">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="hexagram-title">本卦 (Current)</div>',
                            unsafe_allow_html=True,
                        )
                        draw_hexagram(
                            ritual_seq=ritual_sequence_str,
                            binary_string=binary_code,
                            name=f"{chinese_name} / {hexagram_name}",
                            moving_lines=moving_lines,
                            show_title=False,
                        )
                        st.markdown(
                            f'<div class="hex-meta" style="margin-top: 8px;">{chinese_name} ({hexagram_name})</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_arrow:
                        st.markdown(
                            '<div class="hexagram-arrow">➡️</div>',
                            unsafe_allow_html=True,
                        )

                    # 將之卦資訊存入 current_market_state，供 Oracle 使用（例如標題／説明）
                    current_market_state["future_binary"] = future_binary
                    current_market_state["future_hex_name"] = future_hex_name_full
                    current_market_state["future_hex_name_stripped"] = future_hex_name
                    current_market_state["future_chinese_name"] = future_chinese_name

                    with col_future:
                        st.markdown('<div class="hexagram-container">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="hexagram-title">之卦 (Future)</div>',
                            unsafe_allow_html=True,
                        )
                        draw_hexagram(
                            ritual_seq=None,  # 之卦不需要顯示 ritual sequence
                            binary_string=future_binary,
                            name=f"{future_chinese_name} / {future_hex_name}",
                            moving_lines=None,  # 之卦不顯示動爻標記
                            show_title=False,
                        )
                        st.markdown(
                            f'<div class="hex-meta" style="margin-top: 8px;">{future_chinese_name} ({future_hex_name})</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    # 顯示動爻說明
                    moving_lines_str = "、".join(
                        ["初", "二", "三", "四", "五", "上"][line - 1] + "爻"
                        for line in sorted(moving_lines)
                    )
                    st.caption(f"動爻：{moving_lines_str} ({len(moving_lines)} 個)")

                    # 依之卦顯示 Sentiment Gauge
                    render_sentiment_gauge(current_market_state.get("future_binary"))

                else:
                    # 無動爻：只顯示本卦
                    st.markdown(
                        f"**卦名：** {chinese_name} "
                        f"({hexagram_name}, ID: {hexagram_id})"
                    )

                    draw_hexagram(
                        ritual_seq=ritual_sequence_str,
                        binary_string=binary_code,
                        name=f"{chinese_name} / {hexagram_name}",
                        moving_lines=None,
                        show_title=True,
                    )

                    # 若無之卦，使用本卦陽爻比例顯示情緒儀表
                    render_sentiment_gauge(binary_code)

        # ===== Step 4: 量化橋接指標列（連結價格與卦象） =====
        moving_lines_for_state = [
            i + 1 for i, n in enumerate(ritual_sequence) if n in (6, 9)
        ]
        _render_quantitative_bridge(
            raw_df=raw_df,
            ritual_sequence=ritual_sequence,
            moving_lines=moving_lines_for_state,
            latest_row=latest_row,
        )

        # ===== Step 4.5: 波動率 Gauge Chart（簡約風格） =====
        render_volatility_gauge(
            raw_df=raw_df,
            ritual_sequence=ritual_sequence,
            latest_row=latest_row
        )

        # ===== Step 5: AI 易經解讀（依資訊層級呈現） =====
        # 檢查是否已經生成過 Oracle's Advice（避免追問時重新生成）
        oracle_cache_key = f"oracle_answer_{backend_ticker}_{market_type}"
        
        # 只在第一次 consult 時生成，追問時使用緩存
        if consult and (oracle_cache_key not in st.session_state):
            # 只在真正點擊 consult 按鈕且沒有緩存時生成
            with st.spinner("Analyzing Market Structure & Consulting Spirits..."):
                ai_answer = oracle.ask(
                    backend_ticker,
                    question,  # 固定為「目前趨勢」
                    precomputed_data=current_market_state,
                    market_type=market_type,
                )
            
            # 解析但不顯示（只保存到 session_state，統一在下面顯示）
            response_data = _parse_ai_response(ai_answer)
            if response_data:
                st.session_state[oracle_cache_key] = {
                    'ai_answer': ai_answer,
                    'response_data': response_data,
                    'ticker': backend_ticker,
                    'market_state': current_market_state,
                    'market_type': market_type
                }
        elif oracle_cache_key in st.session_state:
            # 使用緩存的結果（追問時不重新生成，直接顯示）
            cached = st.session_state[oracle_cache_key]
            ai_answer = cached['ai_answer']
            response_data = cached['response_data']
        else:
            # 沒有緩存且沒有點擊 consult，不顯示
            response_data = None
            ai_answer = None
        
        # 始終顯示 Oracle's Advice（不受追問加載狀態影響）
        if response_data:
            # 直接顯示已緩存的內容（不重新解析，避免變淡）
            with st.container(border=True):
                # 直接顯示已解析的內容，不重新調用 render_ai_response
                st.markdown("## 🔮 Oracle's Advice / 卜卦解讀")
                st.markdown("### 🚀 投資快訊 (Executive Summary)")
                st.markdown(response_data.get('summary', ''))
                st.markdown("### 🎯 關鍵操作建議 (Action Plan)")
                action_text = response_data.get('action', '')
                tone = _classify_action_tone(action_text)
                if tone == "buy":
                    st.success(action_text)
                elif tone == "sell":
                    st.error(action_text)
                else:
                    st.info(action_text)
                with st.expander("📜 點擊查看：易經原文與詳細現代解讀", expanded=False):
                    st.markdown("#### 📖 易經原文 (The Source)")
                    source_text = response_data.get('source', '')
                    if source_text:
                        st.markdown(source_text)
                    else:
                        st.markdown("_目前回應中未偵測到明確的易經原文段落。_")
                    st.divider()
                    st.markdown("#### 💡 現代解讀 (Deep Dive)")
                    decoding_text = response_data.get('decoding', '')
                    if decoding_text:
                        st.markdown(decoding_text)
                    else:
                        st.markdown("_目前回應中未偵測到額外的現代金融解讀內容。_")
                st.caption(
                    "以上內容僅供研究與教育參考，不構成任何投資建議或買賣邀約，實際投資決策請自行評估風險。"
                )
        
        # ===== 追問系統（獨立容器，不在 Oracle's Advice 框內） =====
        if response_data:
            render_followup_system(oracle, response_data, ai_answer, backend_ticker, current_market_state, market_type)

    else:
        # 尚未按下按鈕時，給予簡短提示
        with col_chart:
            st.markdown(
                "在左側輸入股票代號與問題，按下 **Consult the Oracle (卜卦)** "
                "即可生成對應的卦象與 AI 解讀。"
            )


if __name__ == "__main__":
    main()

