"""Quantum I-Ching Streamlit 儀表板介面.

此模組提供使用者透過瀏覽器與 Quantum I-Ching 神諭互動的前端介面。
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from oracle_chat import Oracle


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
        if market_type == "TW":
            # 台股：純數字補 .TW，已有 .TW 則直接使用
            if user_ticker.isdigit():
                backend_ticker = f"{user_ticker}.TW"
            elif user_ticker.endswith(".TW"):
                backend_ticker = user_ticker
            else:
                backend_ticker = f"{user_ticker}.TW"
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

                    fig.update_layout(
                        title=(
                            (
                                f"{user_ticker} ({stock_name})"
                                if stock_name
                                else user_ticker
                            )
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
                    f'<span class="symbol">{user_ticker}</span>'
                    f'<span class="label"> / 市場結構卦象</span>'
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

            # ===== Step 4: AI 易經解讀（置於折線圖下方，使用 Streamlit 內建框線） =====
            # 使用單一來源的市場狀態，確保上方顯示與下方解讀使用完全相同的卦象
            ai_answer = oracle.ask(
                backend_ticker,
                question or "Should I buy now?",
                precomputed_data=current_market_state,
                market_type=market_type,
            )

            st.markdown("### 🧠 Oracle's Advice / 卜卦解讀")
            # 使用 st.info 提供完整包覆的卡片樣式，並保留 Markdown 格式
            st.info(ai_answer)
            st.caption(
                "以上內容僅供研究與教育參考，"
                "不構成任何投資建議、買賣邀約或報酬保證，"
                "實際投資決策請自行審慎評估風險。"
            )

    else:
        # 尚未按下按鈕時，給予簡短提示
        with col_chart:
            st.markdown(
                "在左側輸入股票代號與問題，按下 **Consult the Oracle (卜卦)** "
                "即可生成對應的卦象與 AI 解讀。"
            )


if __name__ == "__main__":
    main()

