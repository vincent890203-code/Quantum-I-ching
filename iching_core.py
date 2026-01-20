"""Quantum I-Ching 專案易經核心邏輯模組.

此模組提供易經卦象解釋和未來卦（之卦/Zhi Gua）計算的核心功能。
"""

from typing import Dict, List, Optional, Tuple

from config import HEXAGRAM_MAP, settings


class IChingCore:
    """易經核心邏輯類別.

    提供卦象查詢、未來卦計算和序列解釋功能。
    
    核心概念：
    - 當前卦（本卦）：由儀式數字序列直接轉換而成的卦象
    - 未來卦（之卦）：根據變爻規則計算出的變動後卦象
    - 動爻：值為 6 或 9 的爻位，代表正在變動的狀態
    """

    def get_hexagram_name(self, binary_string: str) -> Dict[str, any]:
        """查詢卦象名稱.

        根據二進制字串查詢 HEXAGRAM_MAP，取得卦象資訊。

        Args:
            binary_string: 六位二進制字串（例如 "111111"），代表六爻卦象。
                - "1" 代表陽爻
                - "0" 代表陰爻

        Returns:
            包含卦象資訊的字典：
            - `id`: 卦象編號（1-64）
            - `name`: 卦象英文/拼音名稱
            - `nature`: 卦象繁體中文名稱
            如果查不到，返回預設值：`{"id": 0, "name": "Unknown", "nature": "?"}`

        Note:
            由於 HEXAGRAM_MAP 目前只包含前 8 卦，
            查不到的卦會返回預設值而非拋出異常。
        """
        hexagram = HEXAGRAM_MAP.get(
            binary_string,
            {"id": 0, "name": "Unknown", "nature": "?"}
        )
        return hexagram

    def calculate_future_hexagram(
        self,
        ritual_sequence: List[int]
    ) -> str:
        """計算未來卦（之卦）.

        根據易經變爻規則，將儀式數字序列轉換為未來卦的二進制字串。
        
        變爻規則：
        - 9 (老陽) -> 變為 0 (陰爻)
        - 6 (老陰) -> 變為 1 (陽爻)
        - 7 (少陽) -> 保持 1 (陽爻)
        - 8 (少陰) -> 保持 0 (陰爻)

        Args:
            ritual_sequence: 包含 6 個整數的列表，代表六爻的儀式數字。
                例如：`[9, 8, 7, 8, 9, 6]`
                順序：索引 0 為底部第1爻，索引 5 為頂部第6爻

        Returns:
            六位二進制字串，代表未來卦。
            例如：`[9, 8, 7, 8, 9, 6]` -> `"001001"`

        Raises:
            ValueError: 如果 `ritual_sequence` 長度不等於 6
        """
        if len(ritual_sequence) != 6:
            raise ValueError(
                f"儀式數字序列必須包含恰好 6 個元素，實際得到 {len(ritual_sequence)} 個"
            )

        future_binary = ""
        for num in ritual_sequence:
            if num == 9:  # 老陽 -> 變為陰爻
                future_binary += "0"
            elif num == 6:  # 老陰 -> 變為陽爻
                future_binary += "1"
            elif num == 7:  # 少陽 -> 保持陽爻
                future_binary += "1"
            elif num == 8:  # 少陰 -> 保持陰爻
                future_binary += "0"
            else:
                # 處理異常值（理論上不應該發生）
                raise ValueError(
                    f"儀式數字必須為 6, 7, 8, 9 之一，實際得到 {num}"
                )

        return future_binary

    def interpret_sequence(
        self,
        ritual_sequence: List[int]
    ) -> Dict[str, any]:
        """解釋儀式數字序列.

        高階方法，整合當前卦、未來卦和動爻的計算。

        Args:
            ritual_sequence: 包含 6 個整數的列表，代表六爻的儀式數字。
                例如：`[9, 8, 7, 8, 9, 6]`
                順序：索引 0 為底部第1爻，索引 5 為頂部第6爻

        Returns:
            包含完整解釋結果的字典：
            - `current_hex`: 當前卦資訊（來自 `get_hexagram_name`）
                - `id`: 卦象編號
                - `name`: 卦象名稱
                - `nature`: 卦象繁體中文名稱
            - `future_hex`: 未來卦資訊（來自 `get_hexagram_name`）
                - 結構與 `current_hex` 相同
            - `moving_lines`: 動爻列表（1-based index）
                - 值為 6 或 9 的爻位（1 到 6）
                - 例如：`[1, 5]` 表示第1爻和第5爻是動爻

        Raises:
            ValueError: 如果 `ritual_sequence` 長度不等於 6

        Example:
            >>> core = IChingCore()
            >>> result = core.interpret_sequence([9, 8, 7, 8, 9, 6])
            >>> result['current_hex']['name']
            'Qian'
            >>> result['moving_lines']
            [1, 5, 6]  # 第1、5、6爻是動爻（值為9, 9, 6）
        """
        if len(ritual_sequence) != 6:
            raise ValueError(
                f"儀式數字序列必須包含恰好 6 個元素，實際得到 {len(ritual_sequence)} 個"
            )

        # Step 1: 計算當前卦的二進制字串
        # 9 和 7 -> "1"（陽爻），6 和 8 -> "0"（陰爻）
        current_binary = "".join(
            "1" if num in [9, 7] else "0"
            for num in ritual_sequence
        )

        # Step 2: 計算未來卦的二進制字串
        future_binary = self.calculate_future_hexagram(ritual_sequence)

        # Step 3: 識別動爻（1-based index）
        # 值為 6 或 9 的爻位是動爻
        # 列表中索引 i 對應第 (i+1) 爻（因為 1-based indexing）
        moving_lines = [
            i + 1  # 轉換為 1-based index
            for i, num in enumerate(ritual_sequence)
            if num in [6, 9]
        ]

        # 查詢卦象名稱
        current_hex = self.get_hexagram_name(current_binary)
        future_hex = self.get_hexagram_name(future_binary)

        return {
            "current_hex": current_hex,
            "future_hex": future_hex,
            "moving_lines": moving_lines,
        }