"""Quantum I-Ching 專案神諭對話模組.

此模組整合市場資料分析、易經卦象解讀和知識庫檢索，
使用 Google Gemini API 提供智慧化的金融建議。
"""

import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from config import settings
from data_loader import MarketDataLoader
from iching_core import IChingCore
from market_encoder import MarketEncoder
from vector_store import IChingVectorStore


# 載入環境變數
load_dotenv()


class Oracle:
    """易經神諭類別.

    整合市場資料分析、易經卦象解讀和知識庫檢索，
    使用 Google Gemini API 提供智慧化的金融建議。
    """

    def __init__(self) -> None:
        """初始化神諭系統.

        Raises:
            ValueError: 如果 GOOGLE_API_KEY 未設定
        """
        # 檢查 API Key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 未設定。\n"
                "請在 .env 檔案中設定 GOOGLE_API_KEY，或使用環境變數。\n"
                "例如：GOOGLE_API_KEY=your_api_key_here"
            )

        # 設定 Gemini API
        genai.configure(api_key=api_key)

        # 初始化市場資料處理組件
        self.data_loader = MarketDataLoader()
        self.encoder = MarketEncoder()
        self.core = IChingCore()

        # 初始化向量資料庫（載入 ChromaDB）
        self.vector_store = IChingVectorStore()

        # 初始化 Gemini 模型
        # 嘗試多個模型，按優先順序：gemini-2.5-flash > gemini-pro-latest > gemini-2.5-pro
        model_names = [
            "gemini-2.5-flash",      # 最新、最快、最便宜
            "gemini-pro-latest",      # 通用版本
            "gemini-2.5-pro",         # 更強大的版本
            "gemini-2.0-flash",      # 備用選項
        ]
        
        self.model = None
        self.model_name = None
        
        # 嘗試初始化模型（不進行實際 API 調用）
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.model_name = model_name
                print(f"[INFO] Initialized Gemini model: {model_name}")
                break
            except Exception as e:
                # 如果模型不可用，嘗試下一個
                print(f"[DEBUG] Model {model_name} not available: {str(e)[:50]}")
                continue
        
        if self.model is None:
            raise ValueError(
                "無法初始化任何 Gemini 模型。\n"
                "請檢查 API Key 是否正確，或稍後再試。\n"
                "嘗試的模型: " + ", ".join(model_names)
            )

    def _get_market_hexagram(self, ticker: str) -> dict:
        """獲取市場卦象.

        從股票資料中提取最新的易經卦象資訊。

        Args:
            ticker: 股票代號（例如 "NVDA"）

        Returns:
            包含卦象資訊的字典：
            - `hexagram_name`: 卦象英文名稱（例如 "The Well"）
            - `chinese_name`: 卦象繁體中文名稱（例如 "井"）
            - `hexagram_id`: 卦象編號（1-64）
            - `ritual_sequence`: 儀式數字序列（例如 [9, 8, 7, 8, 9, 6]）
            - `binary_code`: 二進制編碼（例如 "101010"）

        Raises:
            ValueError: 如果無法獲取或處理資料
        """
        # 步驟 1: 載入市場資料
        raw_data = self.data_loader.fetch_data(tickers=[ticker])
        if raw_data.empty:
            raise ValueError(f"無法獲取 {ticker} 的市場資料")

        # 步驟 2: 編碼為易經卦象
        encoded_data = self.encoder.generate_hexagrams(raw_data)
        if encoded_data.empty:
            raise ValueError(
                f"{ticker} 的資料不足，需要至少 26 天才能生成完整卦象"
            )

        # 步驟 3: 獲取最新的卦象資料
        latest_row = encoded_data.iloc[-1]
        latest_index = encoded_data.index[-1]

        # 提取儀式序列和二進制編碼
        ritual_sequence_str = None
        if 'Ritual_Sequence' in latest_row.index:
            ritual_sequence_str = latest_row['Ritual_Sequence']
        elif hasattr(latest_row, 'get'):
            ritual_sequence_str = latest_row.get('Ritual_Sequence', None)

        if ritual_sequence_str is None or ritual_sequence_str == '':
            raise ValueError(
                f"無法取得 {ticker} 的儀式數字序列（可能資料不足）"
            )

        # 轉換為整數列表
        ritual_sequence = [int(char) for char in str(ritual_sequence_str)]
        if len(ritual_sequence) != 6:
            raise ValueError(
                f"儀式數字序列長度不正確（應為 6，實際為 {len(ritual_sequence)}）"
            )

        # 提取二進制編碼
        binary_code = None
        if 'Hexagram_Binary' in latest_row.index:
            binary_code = str(latest_row['Hexagram_Binary'])
        elif hasattr(latest_row, 'get'):
            binary_code = str(latest_row.get('Hexagram_Binary', ''))

        # 步驟 4: 解碼卦象
        interpretation = self.core.interpret_sequence(ritual_sequence)
        current_hex = interpretation['current_hex']

        # 提取卦象名稱
        # 注意：name 可能包含括號（例如 "Qian (The Creative)"），提取主要名稱
        name_full = current_hex.get('name', 'Unknown')
        if '(' in name_full:
            hexagram_name = name_full.split('(')[0].strip()
        else:
            hexagram_name = name_full
        chinese_name = current_hex.get('nature', '?')
        hexagram_id = current_hex.get('id', 0)

        return {
            'hexagram_name': hexagram_name,
            'chinese_name': chinese_name,
            'hexagram_id': hexagram_id,
            'ritual_sequence': ritual_sequence,
            'binary_code': binary_code
        }

    def _get_iching_wisdom(
        self,
        hexagram_name: str,
        user_question: str
    ) -> str:
        """從向量資料庫檢索易經智慧.

        使用語義搜尋找出與卦象和問題相關的易經文本。

        Args:
            hexagram_name: 卦象英文名稱（例如 "The Well"）
            user_question: 用戶問題

        Returns:
            檢索到的易經文本內容（如果找不到則返回空字串）
        """
        # 構造查詢文字
        # 結合卦象名稱和用戶問題，提高檢索相關性
        query_text = f"{hexagram_name} meaning {user_question}"

        # 查詢向量資料庫（返回最相關的 3 個結果）
        try:
            results = self.vector_store.query(query_text, n_results=3)
            if results:
                # 合併所有檢索結果
                context = "\n\n".join(results)
                return context
            else:
                return ""
        except Exception as e:
            print(f"向量資料庫查詢錯誤: {e}")
            return ""

    def ask(self, ticker: str, question: str) -> str:
        """詢問神諭.

        整合市場資料分析、易經卦象解讀和知識庫檢索，
        使用 Gemini API 生成智慧化的金融建議。

        Args:
            ticker: 股票代號（例如 "NVDA"）
            question: 用戶問題（例如 "Should I buy now?"）

        Returns:
            Gemini 生成的回答文字

        Raises:
            ValueError: 如果無法獲取市場資料或處理卦象
            Exception: 如果 Gemini API 調用失敗
        """
        try:
            # 步驟 1: 獲取市場卦象
            hexagram_info = self._get_market_hexagram(ticker)
            hexagram_name = hexagram_info['hexagram_name']
            chinese_name = hexagram_info['chinese_name']
            hexagram_id = hexagram_info['hexagram_id']

            # 步驟 2: 檢索易經智慧
            retrieved_context = self._get_iching_wisdom(hexagram_name, question)

            # 步驟 3: 構造系統提示
            system_prompt = f"""You are a sophisticated AI Financial Advisor named 'Quantum I-Ching'.
Your goal is to interpret ancient I-Ching hexagrams into **actionable modern stock market insights**.

**Context:**
* Stock: {ticker}
* Hexagram: {hexagram_name} ({chinese_name}, ID: {hexagram_id})
* I-Ching Text: {retrieved_context if retrieved_context else "No specific scripture found, use general I-Ching principles"}
* User Question: {question}

**Response Guidelines:**
1. **Tone**: Professional, crisp, and modern. Like a Bloomberg analyst who happens to be an I-Ching scholar. Avoid overly flowery or archaic language (do NOT use '吾', '汝', '此乃'). Use standard modern Traditional Chinese (繁體中文).

2. **Structure** (Use Markdown headers and bullet points):
    * **🚀 投資快訊 (Executive Summary)**: A 1-sentence bottom line (e.g., "短期整理，長期看多" or "建議等待更好的進場時機").
    * **📜 易經原文 (The Source)**: Quote the most relevant 1-2 sentences from the provided I-Ching Text (Judgement or Image). If no specific text is provided, use general I-Ching principles related to this hexagram.
    * **💡 現代解讀 (Modern Decoding)**: Translate the metaphor into financial terms.
        * *Example:* If 'The Well' (井) -> Mention 'Infrastructure', 'Deep Value', 'Dividends', or 'Accumulation'.
        * *Example:* If 'The Creative' (乾) -> Mention 'High Momentum', 'Breakout', or 'Overbought'.
        * *Example:* If 'Waiting' (需) -> Mention 'Consolidation', 'Patience', or 'Wait for Catalyst'.
    * **📈 操作建議 (Action Plan)**: Give concrete advice based on the hexagram (e.g., '建議設定止損於 X', '採用定期定額策略', '等待成交量放大').

**Strict Output Format**: 
- Use Markdown headers (##) for each section
- Use bullet points for details
- Keep the tone professional and modern
- All output must be in Traditional Chinese (繁體中文)
- Do NOT use ancient Chinese style or archaic expressions"""

            # 步驟 4: 生成回答
            try:
                response = self.model.generate_content(system_prompt)
                
                # 步驟 5: 提取文字回應
                if response and hasattr(response, 'text'):
                    return response.text
                else:
                    return "無法生成回答，請稍後再試。"
            except Exception as api_error:
                # 處理 API 調用錯誤
                error_msg = str(api_error)
                if "404" in error_msg or "not found" in error_msg.lower():
                    return (
                        f"模型錯誤: 當前使用的模型可能不可用。\n"
                        f"使用的模型: {self.model_name}\n"
                        f"請檢查 API Key 權限或稍後再試。\n"
                        f"錯誤詳情: {error_msg[:200]}"
                    )
                else:
                    return (
                        f"API 調用錯誤: {error_msg[:200]}\n"
                        f"請檢查 API Key 是否正確設定，或稍後再試。"
                    )

        except ValueError as e:
            # 處理資料獲取錯誤
            return f"錯誤: {str(e)}"
        except Exception as e:
            # 處理其他錯誤
            return f"發生錯誤: {str(e)}\n請檢查設定或稍後再試。"


if __name__ == "__main__":
    # 測試執行
    try:
        oracle = Oracle()
        answer = oracle.ask("NVDA", "Should I buy now?")
        print("\n" + "=" * 60)
        print("  Quantum I-Ching Oracle Response")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")
    except ValueError as e:
        print(f"\n[錯誤] {e}\n")
    except Exception as e:
        print(f"\n[錯誤] 發生未預期的錯誤: {e}\n")