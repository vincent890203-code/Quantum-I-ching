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
    page_title="Quantum I-Ching",
)


# ===== 全局樣式（淺色金融風格） =====
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


# ===== Oracle 初始化（資源快取） =====
# 添加版本號以強制清除舊緩存
_ORACLE_VERSION = "2.0"  # 當 Oracle 類簽名改變時，更新此版本號以清除緩存

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


def _render_quantitative_bridge(
    raw_df: pd.DataFrame,
    ritual_sequence: list[int],
    moving_lines: list[int],
) -> None:
    """在圖表與文字解讀之間插入「量化橋接」指標列.

    - 價格：當日收盤與昨日比較
    - RVOL：當日量 / 20 日平均量
    - 系統狀態：依動爻數量評估穩定度
    - 趨勢強度：Price vs MA20 粗略判斷多空
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

    # 系統狀態：依動爻數量判斷
    moving_count = len(moving_lines)
    if moving_count == 0:
        system_state = "Stable"
        system_desc = "0 動爻：結構相對穩定"
    elif moving_count <= 2:
        system_state = "Active"
        system_desc = f"{moving_count} 動爻：結構開始活躍"
    else:
        system_state = "Volatile"
        system_desc = f"{moving_count} 動爻：結構高度波動"

    # 趨勢強度：以 Price > MA20 粗略判斷
    ma20 = float(raw_df["Close"].tail(20).mean()) if len(raw_df) >= 20 else latest_close
    if latest_close >= ma20:
        trend_label = "Bullish"
        trend_desc = "價格高於 20 日均線"
    else:
        trend_label = "Bearish"
        trend_desc = "價格低於 20 日均線"

    st.markdown("### 📊 量化橋接 (Quantitative Bridge)")
    col_p, col_rvol, col_state, col_trend = st.columns(4)

    # 價格指標
    with col_p:
        delta_str = f"{price_delta:+.2f} ({price_delta_pct:+.2f}%)"
        st.metric(
            label="收盤價 (Close Price)",
            value=f"{latest_close:,.2f}",
            delta=delta_str,
            delta_color="normal" if price_delta >= 0 else "inverse",
            help="目前的收盤價。括號內為與前一日的漲跌幅。",
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
                help="相對成交量 (Relative Volume)。\n計算方式：今日成交量 / 過去 20 日平均成交量。\n數值 > 1.0 代表爆量，易經中常對應『變爻』的產生。",
            )
        else:
            st.metric(
                label="RVOL (相對成交量)",
                value="N/A",
                delta="資料不足",
                help="相對成交量 (Relative Volume)。\n計算方式：今日成交量 / 過去 20 日平均成交量。\n數值 > 1.0 代表爆量，易經中常對應『變爻』的產生。",
            )

    # 系統狀態
    with col_state:
        st.metric(
            label="系統狀態 (System State)",
            value=system_state,
            delta=system_desc,
            help="對應易經的『動爻』數量。\n- Stable (0 動爻): 局勢穩定，看本卦。\n- Active (1-2 動爻): 趨勢醞釀中，關注變爻。\n- Volatile (3+ 動爻): 局勢混亂，變盤機率高，參考之卦。",
        )

    # 趨勢強度
    with col_trend:
        # 加上 🐂/🐻 圖示
        trend_display = f"{trend_label} {'🐂' if trend_label == 'Bullish' else '🐻'}"
        st.metric(
            label="趨勢強度 (Trend Strength)",
            value=trend_display,
            delta=trend_desc,
            delta_color="normal" if trend_label == "Bullish" else "inverse",
            help="基於股價與 20 日均線 (月線) 的乖離判斷。\n- 牛市 🐂: 股價在均線之上，支撐強。\n- 熊市 🐻: 股價在均線之下，壓力大。",
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


def render_ai_response(ai_answer: str) -> None:
    """依資訊層級呈現 AI 回應，避免重複段落."""
    if not ai_answer:
        st.info("目前尚未取得 Oracle 回應。")
        return

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


def render_volatility_radar(
    raw_df: pd.DataFrame,
    ritual_sequence: list[int],
    latest_row: pd.Series
) -> None:
    """顯示波動率雷達（Volatility Radar）.
    
    使用精簡版 XGBoost 模型預測波動性爆發機率。
    
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
        # latest_row 是 pandas Series，可以直接使用索引訪問
        try:
            close_val = float(latest_row['Close'])
            volume_val = float(latest_row.get('Volume', 0))
            rvol_val = float(latest_row.get('RVOL', 1.0))
            daily_return_val = float(latest_row.get('Daily_Return', 0))
        except (KeyError, ValueError) as e:
            st.warning(f"無法提取數值特徵: {e}")
            return
        
        numerical_features = np.array([
            close_val,
            volume_val,
            rvol_val,
            daily_return_val
        ])
        
        # 只使用精簡特徵：Moving_Lines_Count 和 Energy_Delta
        moving_lines_count = iching_features[2]  # Moving_Lines_Count
        energy_delta = iching_features[3]  # Energy_Delta
        
        # 組合特徵向量（順序必須與訓練時一致）
        feature_vector = np.array([
            numerical_features[0],  # Close
            numerical_features[1],  # Volume
            numerical_features[2],  # RVOL
            numerical_features[3],  # Daily_Return
            moving_lines_count,     # Moving_Lines_Count
            energy_delta            # Energy_Delta
        ]).reshape(1, -1)
        
        # 預測波動性爆發機率
        prob_breakout = model.predict_proba(feature_vector)[0, 1]
        prob_percent = prob_breakout * 100
        
        # 顯示波動率雷達
        st.markdown("### 🌊 波動率爆發機率 (Volatility Radar)")
        
        # 根據機率決定警告級別
        if prob_percent > 70:
            status_emoji = "🔴"
            status_text = "極度危險 (Extreme Risk)"
            status_color = "#dc2626"  # 紅色
            pulse_style = "animation: pulse-danger 2s ease-in-out infinite;"
        elif prob_percent > 50:
            status_emoji = "🟠"
            status_text = "警戒 (Warning)"
            status_color = "#f59e0b"  # 橙色
            pulse_style = ""
        else:
            status_emoji = "🟢"
            status_text = "平穩 (Stable)"
            status_color = "#10b981"  # 綠色
            pulse_style = ""
        
        # 添加 CSS 動畫（如果需要）
        if pulse_style:
            st.markdown(
                f"""
                <style>
                @keyframes pulse-danger {{
                    0%, 100% {{
                        box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4);
                    }}
                    50% {{
                        box-shadow: 0 0 0 8px rgba(220, 38, 38, 0);
                    }}
                }}
                .volatility-radar-container {{
                    {pulse_style}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        
        # 顯示機率儀表
        st.markdown(
            f"""
            <div class="volatility-radar-container" style="background-color: #ffffff; border-radius: 12px; padding: 20px; border: 2px solid {status_color}; margin-bottom: 12px;">
                <div style="text-align: center; margin-bottom: 16px;">
                    <div style="font-size: 3rem; font-weight: 700; color: {status_color}; margin-bottom: 8px;">
                        {prob_percent:.1f}%
                    </div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #374151;">
                        {status_emoji} {status_text}
                    </div>
                </div>
                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 3px; margin-bottom: 12px;">
                    <div style="width: {prob_percent}%; background-color: {status_color}; height: 28px; border-radius: 8px; transition: width 0.5s ease-in-out; display: flex; align-items: center; justify-content: flex-end; padding-right: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <span style="color: white; font-weight: 600; font-size: 0.9rem;">{prob_percent:.1f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 顯示解釋性資訊
        tooltip_text = (
            "AI 偵測到「暴風雨前的寧靜」。"
            "當動爻少（Moving_Lines_Count 低）且能量增強（Energy_Delta 高）時，變盤機率大增。"
            "\n\n"
            f"當前狀態：\n"
            f"- 動爻數量: {int(moving_lines_count)}\n"
            f"- 能量變化: {energy_delta:.2f}\n"
            f"- 預測機率: {prob_percent:.1f}%"
        )
        
        st.markdown(
            f"""
            <div style="background-color: #f9fafb; border-radius: 8px; padding: 12px; border-left: 4px solid {status_color};">
                <p style="font-size: 0.9rem; color: #374151; margin: 0;" title="{tooltip_text}">
                    <strong>📊 解釋：</strong> {tooltip_text.split('\\n\\n')[0]}
                    <span style="font-size: 0.75rem; color: #6b7280; margin-left: 4px;">(懸停查看詳細資訊)</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 顯示特徵值（用於調試）
        with st.expander("🔍 查看特徵值（用於調試）", expanded=False):
            st.markdown(f"**數值特徵：**")
            st.markdown(f"- Close: {numerical_features[0]:.2f}")
            st.markdown(f"- Volume: {numerical_features[1]:,.0f}")
            st.markdown(f"- RVOL: {numerical_features[2]:.2f}")
            st.markdown(f"- Daily_Return: {numerical_features[3]:.4f}")
            st.markdown(f"**易經特徵：**")
            st.markdown(f"- Moving_Lines_Count: {moving_lines_count:.0f}")
            st.markdown(f"- Energy_Delta: {energy_delta:.2f}")
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

        question = st.text_area(
            "問題 / Question",
            value="Should I buy now? / 我現在該買嗎？",
            height=100,
        ).strip()

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
    st.title("Quantum I-Ching 股市卜卦系統")
    st.markdown(
        "結合 **量化價格結構** 與 **易經六十四卦** 的 AI 金融解讀介面。"
    )

    # 主要佈局：左側 K 線圖（2/3），右側卦象卡片（1/3）
    col_chart, col_hex = st.columns([2, 1])

    if consult:
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

        try:
            oracle = get_oracle()
        except Exception as e:  # pragma: no cover - 主要是環境設定錯誤
            st.error(
                "無法初始化 Quantum I-Ching Oracle，請確認 GOOGLE_API_KEY "
                "與向量資料庫設定是否正確。\n\n"
                f"詳細錯誤：{e}"
            )
            return

        with st.spinner("Analyzing Market Structure & Consulting Spirits..."):
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

                    # 計算 MA20 / MA60 作為技術參考線
                    if "Close" in chart_df.columns:
                        chart_df["MA20"] = (
                            chart_df["Close"].rolling(window=20).mean()
                        )
                        chart_df["MA60"] = (
                            chart_df["Close"].rolling(window=60).mean()
                        )
                    else:
                        chart_df["MA20"] = None
                        chart_df["MA60"] = None

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

                    # 加入 MA20 / MA60 線條
                    fig.add_trace(
                        go.Scatter(
                            x=chart_df[date_col],
                            y=chart_df["MA20"],
                            mode="lines",
                            line=dict(color="#facc15", width=1.5),
                            name="MA20 (貞/Support)",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=chart_df[date_col],
                            y=chart_df["MA60"],
                            mode="lines",
                            line=dict(color="#a855f7", width=1.5),
                            name="MA60 (悔/Resistance)",
                        )
                    )

                    fig.update_layout(
                        title=(
                            f"{display_code} ({display_name})"
                            + f" - {chinese_name} / {hexagram_name} "
                            f"(最近 60 日價格走勢)"
                        ),
                        template="plotly_white",
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_rangeslider_visible=False,
                        font=dict(color="#333333"),
                    )

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
            )

            # ===== Step 4.5: 波動率雷達（Volatility Radar） =====
            render_volatility_radar(
                raw_df=raw_df,
                ritual_sequence=ritual_sequence,
                latest_row=latest_row
            )

            # ===== Step 5: AI 易經解讀（依資訊層級呈現） =====
            # 使用單一來源的市場狀態，確保上方顯示與下方解讀使用完全相同的卦象
            ai_answer = oracle.ask(
                backend_ticker,
                question or "Should I buy now?",
                precomputed_data=current_market_state,
                market_type=market_type,
            )

            # 以帶邊框容器包覆整體文字解讀區，與上方圖表區隔
            with st.container(border=True):
                render_ai_response(ai_answer)

    else:
        # 尚未按下按鈕時，給予簡短提示
        with col_chart:
            st.markdown(
                "在左側輸入股票代號與問題，按下 **Consult the Oracle (卜卦)** "
                "即可生成對應的卦象與 AI 解讀。"
            )


if __name__ == "__main__":
    main()

