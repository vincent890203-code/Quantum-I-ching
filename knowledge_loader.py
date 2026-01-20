"""Quantum I-Ching 專案知識庫載入模組.

此模組負責載入易經知識庫 JSON 檔案，並將其轉換為適合嵌入的文件物件。
用於 RAG（檢索增強生成）系統的資料準備。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class IChingDocument:
    """易經文件資料類別.

    用於表示單一易經卦象的完整資訊，適合用於向量嵌入和檢索。

    Attributes:
        id: 卦象編號（1-64）
        content: 完整的文字內容，用於語義嵌入
            - 格式：包含卦象名稱、卦辭、象辭等完整資訊
        metadata: 結構化元資料，用於檢索和 UI 顯示
            - 包含 "name"（英文名稱）
            - 包含 "chinese_name"（繁體中文名稱）
            - 可擴充其他欄位
    """
    id: int
    content: str
    metadata: Dict[str, str]


class IChingKnowledgeLoader:
    """易經知識庫載入器類別.

    從 JSON 檔案載入易經資料，並轉換為適合嵌入的文件物件。
    支援 RAG 系統的資料準備流程。
    """

    def __init__(self, file_path: str = "data/iching_book.json") -> None:
        """初始化知識庫載入器.

        Args:
            file_path: JSON 檔案路徑，預設為 "data/iching_book.json"
        """
        self.file_path = Path(file_path)

    def load_documents(self) -> List[IChingDocument]:
        """載入並轉換易經文件.

        從 JSON 檔案讀取資料，將每個卦象轉換為 IChingDocument 物件。
        構造完整的文字內容用於語義嵌入。

        Returns:
            包含所有 64 卦的 IChingDocument 列表

        Raises:
            FileNotFoundError: 如果指定的 JSON 檔案不存在
            json.JSONDecodeError: 如果 JSON 檔案格式錯誤
            ValueError: 如果資料結構不符合預期

        Note:
            - 使用 UTF-8 編碼讀取檔案以正確處理中文字元
            - content 欄位包含完整的文字資訊，用於語義搜尋
            - metadata 欄位保留結構化資料，用於 UI 顯示和過濾
        """
        # 檢查檔案是否存在
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"知識庫檔案不存在: {self.file_path}\n"
                f"請先執行 scripts/seed_data.py 生成資料檔案"
            )

        # 讀取 JSON 檔案
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                hexagrams_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON 檔案格式錯誤: {self.file_path}\n"
                f"錯誤詳情: {e}"
            )

        # 驗證資料結構
        if not isinstance(hexagrams_data, list):
            raise ValueError(
                f"JSON 檔案應包含一個列表，實際類型: {type(hexagrams_data)}"
            )

        documents = []

        # 遍歷每個卦象
        for hex_data in hexagrams_data:
            # 驗證必要欄位
            required_fields = ["hexagram_id", "name", "chinese_name", "judgement", "image"]
            missing_fields = [field for field in required_fields if field not in hex_data]
            if missing_fields:
                raise ValueError(
                    f"卦象資料缺少必要欄位: {missing_fields}\n"
                    f"資料: {hex_data}"
                )

            hex_id = hex_data["hexagram_id"]
            name = hex_data["name"]
            chinese_name = hex_data["chinese_name"]
            judgement = hex_data["judgement"]
            image = hex_data["image"]

            # 構造完整的文字內容
            # 格式：Hexagram [ID]: [Chinese Name] [Name]. Judgement: [Text]. Image: [Text]. Key Lines: [Lines Text]
            # 注意：目前 JSON 中沒有 Key Lines 欄位，使用空字串或可選文字
            key_lines = hex_data.get("key_lines", "")  # 如果未來有 Key Lines 欄位
            
            if key_lines:
                content = (
                    f"Hexagram {hex_id}: {chinese_name} {name}. "
                    f"Judgement: {judgement} "
                    f"Image: {image} "
                    f"Key Lines: {key_lines}"
                )
            else:
                # 如果沒有 Key Lines，省略該部分
                content = (
                    f"Hexagram {hex_id}: {chinese_name} {name}. "
                    f"Judgement: {judgement} "
                    f"Image: {image}"
                )

            # 構造元資料
            metadata = {
                "name": name,
                "chinese_name": chinese_name,
                "hexagram_id": str(hex_id),  # 轉為字串以便 JSON 序列化
            }

            # 建立文件物件
            document = IChingDocument(
                id=hex_id,
                content=content,
                metadata=metadata
            )

            documents.append(document)

        return documents