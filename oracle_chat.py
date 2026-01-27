"""Quantum I-Ching 專案神諭對話模組.

此模組整合市場資料分析、易經卦象解讀和知識庫檢索，
使用 Google Gemini API 提供智慧化的金融建議。
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

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

        # 快取易經原始 JSON（用於依 hex_id／爻位嚴格抓原文）
        self._iching_raw: Dict[int, Dict[str, Any]] | None = None
        # 快取「中文卦名 → number」映射（用於當 HEXAGRAM_MAP 不完整時查找）
        self._name_to_number: Dict[str, int] | None = None

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

    def _get_market_hexagram(self, ticker: str, market_type: Optional[str] = None) -> dict:
        """獲取市場卦象.

        從股票資料中提取最新的易經卦象資訊。

        Args:
            ticker: 股票代號（例如 "NVDA"）
            market_type: 市場類型（'US', 'TW', 'CRYPTO'），若為 None 則使用 settings.MARKET_TYPE

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
        raw_data = self.data_loader.fetch_data(tickers=[ticker], market_type=market_type)
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

    def _load_iching_raw(self) -> None:
        """載入 iching_complete.json 並以 hex_id 為 key 快取，同時建立卦名映射."""
        if self._iching_raw is not None:
            return
        path = Path("data/iching_complete.json")
        if not path.exists():
            self._iching_raw = {}
            self._name_to_number = {}
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._iching_raw = {}
            self._name_to_number = {}
            return
        mapping: Dict[int, Dict[str, Any]] = {}
        name_map: Dict[str, int] = {}
        if isinstance(data, list):
            for item in data:
                try:
                    num = int(item.get("number", 0))
                    name = item.get("name", "")
                except Exception:
                    continue
                if num > 0:
                    mapping[num] = item
                    if name:
                        name_map[name] = num
        self._iching_raw = mapping
        self._name_to_number = name_map

    def _get_future_hexagram_name(self, ritual_sequence: List[int]) -> str:
        """取得之卦（變爻後）的卦名.

        依傳統規則：6→7（老陰→少陽）、9→8（老陽→少陰），
        再以奇=1、偶=0 轉二進位，查 HEXAGRAM_MAP 得之卦名。

        Args:
            ritual_sequence: 儀式數字序列，例如 [8, 7, 9, 6, 8, 8]

        Returns:
            之卦的卦名字串（例如 "Ji Ji (After Completion)"）
        """
        # 6→7, 9→8；7、8 不變
        transformed = [
            7 if n == 6 else (8 if n == 9 else n)
            for n in ritual_sequence
        ]
        # 奇=1（陽），偶=0（陰）
        binary = "".join("1" if n % 2 == 1 else "0" for n in transformed)
        info = self.core.get_hexagram_name(binary)
        name_full = info.get("name", "Unknown")
        return name_full

    def _resolve_strategy(
        self, current_hex_name: str, ritual_sequence: List[int]
    ) -> Tuple[str, List[dict], str]:
        """依動爻數量決定之卦策略：情境、查詢計畫、之卦名.

        動爻為 6 或 9。

        Returns:
            strategy_context: 文字說明（給 Gemini 用的策略敘述）
            search_plan: 查詢計畫列表，每一筆包含：
                - query: 查詢文字（給向量模型做語義補強）
                - hex_id: 卦象編號（嚴格限制檢索範圍）
                - type: "main" 或 "line"
                - line_numbers: 需要的爻位列表（僅 type="line" 時使用）
            future_hex_name: 之卦名稱（供系統提示顯示）
        """
        # 從 ritual_sequence 推算本卦與動爻
        current_binary = "".join(
            "1" if n in (9, 7) else "0" for n in ritual_sequence
        )
        current_hex = self.core.get_hexagram_name(current_binary)
        current_hex_id = current_hex.get("id", 0)
        current_nature = current_hex.get("nature", "?")
        
        # 如果 HEXAGRAM_MAP 不完整導致 current_hex_id = 0，從 iching_complete.json 查找
        if current_hex_id == 0 and current_nature != "?":
            self._load_iching_raw()
            if self._name_to_number and current_nature in self._name_to_number:
                current_hex_id = self._name_to_number[current_nature]

        moving = [i + 1 for i, n in enumerate(ritual_sequence) if n in (6, 9)]
        count = len(moving)
        future_hex_name = self._get_future_hexagram_name(ritual_sequence)
        
        # 取得之卦的中文名稱（用於查詢）
        future_binary = "".join(
            "1" if (7 if n == 6 else (8 if n == 9 else n)) % 2 == 1 else "0"
            for n in ritual_sequence
        )
        future_hex = self.core.get_hexagram_name(future_binary)
        future_nature = future_hex.get("nature", "?")
        future_hex_id = future_hex.get("id", 0)
        
        # 如果 HEXAGRAM_MAP 不完整導致 future_hex_id = 0，從 iching_complete.json 查找
        if future_hex_id == 0 and future_nature != "?":
            self._load_iching_raw()
            if self._name_to_number and future_nature in self._name_to_number:
                future_hex_id = self._name_to_number[future_nature]

        # 查詢用：中文關鍵詞 + 嚴格 metadata，確保只在本卦／之卦中檢索
        # 文件格式：主卦 = "【{number}. {name}卦】\n卦辭：{judgment}\n象曰：{image}"
        #           爻 = "【{name}卦】 {meaning}\n象曰：{xiang}"
        q_main = f"{current_nature}卦 卦辭 象曰"
        q_future = f"{future_nature}卦 卦辭 象曰"

        if count == 0:
            # 0 動爻：全盤接受，市場穩定
            ctx = "Total Acceptance. 市場穩定，以本卦卦辭／象辭為主。"
            search_plan = [
                {
                    "query": q_main,
                    "hex_id": current_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"本卦（{current_nature}）",
                }
            ]
            return (ctx, search_plan, future_hex_name)

        if count == 1:
            # 1 動爻：焦點在單一動爻
            line = moving[0]
            # 轉換爻位為中文（1=初, 2=二, 3=三, 4=四, 5=五, 6=上）
            line_names = ["初", "二", "三", "四", "五", "上"]
            line_name = line_names[line - 1] if 1 <= line <= 6 else str(line)
            ctx = "Specific Focus. 注意單一動爻所指的層級或事件。"
            # 本卦指定動爻 + 之卦主卦，一起作為經文來源
            search_plan = [
                {
                    "query": f"{current_nature}卦 {line_name}爻",
                    "hex_id": current_hex_id,
                    "type": "line",
                    "line_numbers": [line],
                    "label": f"本卦動爻（{current_nature} 第 {line} 爻）",
                },
                {
                    "query": q_future,
                    "hex_id": future_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"之卦（{future_nature}）",
                },
            ]
            return (ctx, search_plan, future_hex_name)

        if count == 2:
            # 2 動爻：主客對照，下爻貞（進場/支撐），上爻悔（出場/阻力）
            lo, hi = sorted(moving)[0], sorted(moving)[1]
            # 轉換爻位為中文
            line_names = ["初", "二", "三", "四", "五", "上"]
            lo_name = line_names[lo - 1] if 1 <= lo <= 6 else str(lo)
            hi_name = line_names[hi - 1] if 1 <= hi <= 6 else str(hi)
            ctx = (
                "Primary vs Secondary. 下爻為貞（進場／支撐），"
                "上爻為悔（出場／阻力），需兼看兩爻。"
            )
            search_plan = [
                {
                    "query": f"{current_nature}卦 {lo_name}爻",
                    "hex_id": current_hex_id,
                    "type": "line",
                    "line_numbers": [lo],
                    "label": f"本卦動爻（{current_nature} 第 {lo} 爻，貞）",
                },
                {
                    "query": f"{current_nature}卦 {hi_name}爻",
                    "hex_id": current_hex_id,
                    "type": "line",
                    "line_numbers": [hi],
                    "label": f"本卦動爻（{current_nature} 第 {hi} 爻，悔）",
                },
                {
                    # 額外加入之卦主卦作為輔助參考，讓易經原文同時包含本卦與之卦
                    "query": q_future,
                    "hex_id": future_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"之卦（{future_nature}）",
                },
            ]
            return (ctx, search_plan, future_hex_name)

        if count == 3:
            # 3 動爻：對沖時刻，本卦貞（持有），之卦悔（風險）
            ctx = (
                "Hedging Moment. 本卦為貞（持有），之卦為悔（風險），"
                "需權衡本卦卦辭與之卦卦辭。"
            )
            search_plan = [
                {
                    "query": q_main,
                    "hex_id": current_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"本卦（{current_nature}）",
                },
                {
                    "query": q_future,
                    "hex_id": future_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"之卦（{future_nature}）",
                },
            ]
            return (ctx, search_plan, future_hex_name)

        if count in (4, 5):
            # 4 或 5 動爻：趨勢反轉，之卦貞（主趨勢），本卦悔（歷史）
            ctx = (
                "Trend Reversal. 之卦為貞（主趨勢），本卦為悔（歷史），"
                "以之卦卦辭為主、本卦卦辭為輔。"
            )
            search_plan = [
                {
                    "query": q_future,
                    "hex_id": future_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"之卦（{future_nature}，主趨勢）",
                },
                {
                    "query": q_main,
                    "hex_id": current_hex_id,
                    "type": "main",
                    "line_numbers": [],
                    "label": f"本卦（{current_nature}，歷史）",
                },
            ]
            return (ctx, search_plan, future_hex_name)

        # 6 動爻：極端反轉
        if current_nature == "乾":
            ctx = "Extreme Reversal. 乾卦六爻全變，用「用九」為準。"
            search_plan = [
                {
                    "query": "乾卦 用九",
                    "hex_id": current_hex_id,
                    "type": "line",
                    # 一般 open-iching 會把「用九」放在 position=7
                    "line_numbers": [7],
                    "label": "本卦（乾卦 用九）",
                }
            ]
            return (ctx, search_plan, future_hex_name)
        if current_nature == "坤":
            ctx = "Extreme Reversal. 坤卦六爻全變，用「用六」為準。"
            search_plan = [
                {
                    "query": "坤卦 用六",
                    "hex_id": current_hex_id,
                    "type": "line",
                    "line_numbers": [7],
                    "label": "本卦（坤卦 用六）",
                }
            ]
            return (ctx, search_plan, future_hex_name)
        ctx = "Extreme Reversal. 六爻全變，以之卦卦辭為準。"
        search_plan = [
            {
                "query": q_future,
                "hex_id": future_hex_id,
                "type": "main",
                "line_numbers": [],
                "label": f"之卦（{future_nature}）",
            }
        ]
        return (ctx, search_plan, future_hex_name)

    def _get_iching_wisdom(
        self,
        search_queries: List[dict],
        user_question: str
    ) -> str:
        """依本卦／之卦＋爻位，直接從原始 JSON 抽取精確經文.

        不再依賴語義搜尋決定「是哪一卦」，而是嚴格根據
        `hex_id`、`type`（main/line）、`line_numbers` 直接從
        `data/iching_complete.json` 取出對應經文，確保與市場卦象
        （本卦／之卦）完全一致。

        Args:
            search_queries: 查詢計畫列表（每筆包含 hex_id, type, line_numbers, label 等）
            user_question: 用戶問題（目前僅供未來擴充使用）

        Returns:
            合併後的易經文本；若無結果則回傳空字串。
        """
        if not search_queries:
            return ""

        self._load_iching_raw()
        if not self._iching_raw:
            return ""

        parts: List[str] = []
        seen: set = set()

        for spec in search_queries:
            hex_id = spec.get("hex_id")
            doc_type = spec.get("type")
            line_numbers = spec.get("line_numbers") or []
            label = spec.get("label")  # 如「本卦」「之卦」「本卦動爻」等

            if not hex_id or hex_id not in self._iching_raw:
                continue

            hex_obj = self._iching_raw.get(hex_id) or {}
            name = hex_obj.get("name", "?")
            number = hex_obj.get("number", hex_id)
            judgment = hex_obj.get("judgment") or hex_obj.get("judgement") or ""
            image = hex_obj.get("image") or ""

            prefix = f"【{label}：{number}. {name}卦】" if label else f"【{number}. {name}卦】"

            if doc_type == "main":
                # 只在有內容時才添加標籤
                parts_list = [prefix]
                if judgment:
                    parts_list.append(f"卦辭：{judgment}")
                if image:
                    parts_list.append(f"象曰：{image}")
                text = "\n".join(parts_list).strip()
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
            elif doc_type == "line":
                lines = hex_obj.get("lines") or []
                for ln in line_numbers:
                    for idx, line in enumerate(lines):
                        pos = line.get("position", idx + 1)
                        try:
                            pos = int(pos)
                        except Exception:
                            pos = idx + 1
                        if pos != ln:
                            continue
                        meaning = line.get("meaning") or line.get("text") or ""
                        xiang = line.get("xiang") or ""
                        # 只在有內容時才添加標籤
                        parts_list = [f"{prefix} 第 {ln} 爻：{meaning}"]
                        if xiang:
                            parts_list.append(f"小象：{xiang}")
                        text = "\n".join(parts_list).strip()
                        if text and text not in seen:
                            seen.add(text)
                            parts.append(text)

        return "\n\n".join(parts) if parts else ""

    def ask(
        self,
        ticker: str,
        question: str,
        market_type: Optional[str] = None,
        precomputed_data: Optional[dict] = None,
        hexagram_info: Optional[dict] = None,
    ) -> str:
        """詢問神諭.

        整合市場資料分析、易經卦象解讀和知識庫檢索，
        使用 Gemini API 生成智慧化的金融建議。

        Args:
            ticker: 股票代號（例如 "NVDA"）
            question: 用戶問題（例如 "Should I buy now?"）
            market_type: 市場類型（'US', 'TW', 'CRYPTO'），若為 None 則使用 settings.MARKET_TYPE
            precomputed_data: 由前端或其他模組預先計算好的市場卦象資訊，
                建議結構包含：
                - ritual_sequence: List[int]
                - hex_name: 完整卦名（可含括號）
                - chinese_name: 卦名（中文）
                - hexagram_id: 卦序號
                以及可選的：
                - future_hex_name / future_binary 等（目前僅用於顯示）
            hexagram_info: 舊版相容參數；可選的已計算卦象資訊

        Returns:
            Gemini 生成的回答文字

        Raises:
            ValueError: 如果無法獲取市場資料或處理卦象
            Exception: 如果 Gemini API 調用失敗
        """
        try:
            print(f"[Oracle] Consulting for {ticker}...")

            # 步驟 1: 獲取市場卦象（含 ritual_sequence）
            # 優先使用 precomputed_data，其次 hexagram_info，最後才重新計算
            if precomputed_data is not None:
                print("[Oracle] Using pre-computed market data from Dashboard.")
                # 由前端／呼叫端預先計算的市場狀態（Calculate Once, Use Everywhere）
                ritual_sequence = precomputed_data.get("ritual_sequence", [])
                hexagram_name_full = (
                    precomputed_data.get("hex_name")
                    or precomputed_data.get("hexagram_name")
                    or "Unknown"
                )
                chinese_name = precomputed_data.get("chinese_name", "?")
                hexagram_id = precomputed_data.get("hexagram_id", 0)

                # 確保 ritual_sequence 為 List[int]
                if isinstance(ritual_sequence, str):
                    ritual_sequence = [int(ch) for ch in str(ritual_sequence)]
                elif not isinstance(ritual_sequence, list):
                    ritual_sequence = list(ritual_sequence) if ritual_sequence else []

                # 處理 hexagram_name（移除括號，與 _get_market_hexagram 一致）
                if "(" in hexagram_name_full:
                    hexagram_name = hexagram_name_full.split("(", 1)[0].strip()
                else:
                    hexagram_name = hexagram_name_full

            elif hexagram_info is not None:
                # 舊版相容邏輯：由呼叫端傳入 hexagram_info
                hexagram_name_full = hexagram_info.get("hexagram_name", "Unknown")
                chinese_name = hexagram_info.get("chinese_name", "?")
                hexagram_id = hexagram_info.get("hexagram_id", 0)
                ritual_sequence = hexagram_info.get("ritual_sequence", [])
                # 確保 ritual_sequence 是列表格式
                if isinstance(ritual_sequence, str):
                    ritual_sequence = [int(ch) for ch in str(ritual_sequence)]
                elif not isinstance(ritual_sequence, list):
                    ritual_sequence = list(ritual_sequence) if ritual_sequence else []
                # 處理 hexagram_name（移除括號，與 _get_market_hexagram 一致）
                if "(" in hexagram_name_full:
                    hexagram_name = hexagram_name_full.split("(", 1)[0].strip()
                else:
                    hexagram_name = hexagram_name_full
            else:
                # 完全由 Oracle 端重新計算（向後相容）
                hexagram_info = self._get_market_hexagram(ticker, market_type=market_type)
                hexagram_name = hexagram_info["hexagram_name"]
                chinese_name = hexagram_info["chinese_name"]
                hexagram_id = hexagram_info["hexagram_id"]
                ritual_sequence = hexagram_info["ritual_sequence"]

            # 步驟 2: 依之卦法解析策略（情境、查詢列表、之卦名）
            strategy_context, search_queries, future_hex_name = self._resolve_strategy(
                hexagram_name, ritual_sequence
            )

            # 步驟 3: 依查詢列表檢索易經智慧（不再只查本卦名）
            retrieved_context = self._get_iching_wisdom(search_queries, question)

            # 步驟 4: 構造系統提示（注入策略情境與貞／悔框架）
            system_prompt = f"""You are a sophisticated AI Financial Advisor named 'Quantum I-Ching'.
Your goal is to interpret ancient I-Ching hexagrams into **actionable modern stock market insights** using the traditional **Zhi Gua (之卦)** method and the **Zhen (貞) / Hui (悔)** framework.

**Zhen (貞) vs Hui (悔) — 必須遵守的解釋框架：**
* **貞 (Zhen)**: 主體、支撐、長期、進場、持有。在操作上對應：趨勢支撐、主要方向、可倚賴的層級。
* **悔 (Hui)**: 客體、阻力、短期、出場、風險。在操作上對應：風險管理、壓力位、需警惕的層級。
請依當前之卦策略，在「投資快訊」「現代解讀」「操作建議」中，明確標示哪些建議屬貞（主／支撐／長期）、哪些屬悔（客／阻力／短期），例如：貞—持有、逢回加碼；悔—遇壓減碼、嚴格止損。

**之卦策略 (Zhi Gua Strategy):**
{strategy_context}

**Context:**
* Stock: {ticker}
* 本卦 (Current Hexagram): {hexagram_name} ({chinese_name}, ID: {hexagram_id})
* 之卦 (Future Hexagram): {future_hex_name}
* I-Ching Text (依策略檢索): {retrieved_context if retrieved_context else "No specific scripture found, use general I-Ching principles"}
* User Question: {question}

**Response Guidelines:**
1. **Tone**: Professional, crisp, and modern. Like a Bloomberg analyst who happens to be an I-Ching scholar. Avoid overly flowery or archaic language (do NOT use '吾', '汝', '此乃'). Use standard modern Traditional Chinese (繁體中文).

2. **Structure** (Use Markdown headers and bullet points):
    * **🚀 投資快訊 (Executive Summary)**: A 1-sentence bottom line. Where applicable, state which aspect is 貞 (main/support) and which is 悔 (risk/resistance).
    * **📜 易經原文 (The Source)**: **MUST quote ALL provided I-Ching Text sections exactly as given, preserving the labels like 【本卦...】【之卦...】. Do NOT summarize, omit, or select only parts. If multiple sections are provided (e.g., both 本卦 and 之卦), display ALL of them.**
    * **💡 現代解讀 (Modern Decoding)**: Translate the metaphor into financial terms. Map 貞 to trend/support and 悔 to risk/exit levels when the strategy involves both.
    * **📈 操作建議 (Action Plan)**: Give concrete advice. Use 貞 for entries, hold zones, and support; use 悔 for exits, stop-loss, and resistance. Example: 「貞：XX 以下視為支撐，可持有」；「悔：YY 以上注意風險，考慮減碼」.

**Strict Output Format**: 
- Use Markdown headers (##) for each section
- Use bullet points for details
- Keep the tone professional and modern
- All output must be in Traditional Chinese (繁體中文)
- Do NOT use ancient Chinese style or archaic expressions
- Always apply the Zhen/Hui framework when the strategy indicates Primary vs Secondary, Hedging, Trend Reversal, or Extreme Reversal."""

            # 步驟 5: 生成回答
            try:
                response = self.model.generate_content(system_prompt)

                # 步驟 6: 提取文字回應
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
    # 本機測試（不經前端）：NVDA 為美股，須傳 market_type="US"
    try:
        oracle = Oracle()
        # 與前端 dashboard.py 一致，統一使用「目前趨勢」作為預設提問
        answer = oracle.ask("2330", "目前趨勢", market_type="TW")
        print("\n" + "=" * 60)
        print("  Quantum I-Ching Oracle Response")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")
    except ValueError as e:
        print(f"\n[錯誤] {e}\n")
    except Exception as e:
        print(f"\n[錯誤] 發生未預期的錯誤: {e}\n")