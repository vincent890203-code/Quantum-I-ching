"""Quantum I-Ching 專案市場資料編碼模組.

此模組將股票市場資料轉換為易經六十四卦的表示方式。
使用「Whale Volume Weighting」邏輯將價格變動映射到四象（6, 7, 8, 9）。
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import settings


class MarketEncoder:
    """市場資料編碼器類別.

    將股票價格資料轉換為易經卦象表示。
    使用相對成交量（RVOL）和價格變動率來判斷四象（老陽、少陽、老陰、少陰）。
    
    四象對應：
    - 9 (老陽 / 變動之陽): 上漲且成交量異常放大（RVOL > 2.0）
    - 7 (少陽 / 靜止之陽): 上漲但成交量正常（RVOL <= 2.0）
    - 8 (少陰 / 靜止之陰): 下跌但成交量正常（RVOL <= 2.0）
    - 6 (老陰 / 變動之陰): 下跌且成交量異常放大（RVOL > 2.0）
    """

    def _calculate_technical_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """計算技術指標.

        計算日報酬率、20日均量，以及相對成交量（RVOL）。

        Args:
            df: 包含 Open, High, Low, Close, Volume 欄位的 DataFrame。
                可以是標準 DataFrame 或 MultiIndex DataFrame。

        Returns:
            新增了以下欄位的 DataFrame：
            - Daily_Return: 日報酬率（Close 的百分比變動）
            - Volume_MA20: 20日成交量移動平均
            - RVOL: 相對成交量（Volume / Volume_MA20）

        Note:
            - 前20天的 Volume_MA20 和 RVOL 會是 NaN
            - 除以零的情況會被處理（RVOL 設為 NaN）
        """
        df = df.copy()

        # 確保有 Close 和 Volume 欄位
        # 檢查欄位是否存在（支援標準和 MultiIndex DataFrame）
        missing_cols = []
        if 'Close' not in df.columns:
            missing_cols.append('Close')
        if 'Volume' not in df.columns:
            missing_cols.append('Volume')
        
        if missing_cols:
            raise ValueError(
                f"DataFrame 必須包含 'Close' 和 'Volume' 欄位。"
                f"缺少欄位: {missing_cols}。"
                f"實際欄位: {list(df.columns)}"
            )

        # 計算日報酬率（向量化）
        df['Daily_Return'] = df['Close'].pct_change()

        # 計算20日成交量移動平均（向量化）
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()

        # 計算相對成交量 RVOL（向量化）
        # 處理除以零的情況：當 Volume_MA20 為 0 時，RVOL 設為 NaN
        df['RVOL'] = np.where(
            df['Volume_MA20'] != 0,
            df['Volume'] / df['Volume_MA20'],
            np.nan
        )

        return df

    def _get_ritual_number(
        self,
        return_val: float,
        rvol_val: float
    ) -> int:
        """根據報酬率和相對成交量決定儀式數字.

        實作「Whale Volume Weighting」邏輯：
        - 方向（Direction）: 由 return_val 與 YIN_YANG_THRESHOLD 比較決定
        - 能量（Energy）: 由 rvol_val 決定是否為「Whale」級成交量

        Args:
            return_val: 日報酬率
            rvol_val: 相對成交量（RVOL）

        Returns:
            儀式數字：6（老陰）、7（少陽）、8（少陰）、9（老陽）

        Note:
            - RVOL > 2.0 被視為「Whale」級（高能量，導致變動）
            - RVOL <= 2.0 被視為正常或弱勢（低能量，靜止狀態）
        """
        # 判斷方向：上漲（陽）或下跌（陰）
        is_up = return_val > settings.YIN_YANG_THRESHOLD

        # 判斷能量：是否為 Whale 級成交量
        is_whale = rvol_val > 2.0

        # 映射到四象
        if is_up and is_whale:
            return 9  # 老陽（變動之陽）
        elif not is_up and is_whale:
            return 6  # 老陰（變動之陰）
        elif is_up and not is_whale:
            return 7  # 少陽（靜止之陽）
        else:  # not is_up and not is_whale
            return 8  # 少陰（靜止之陰）

    def generate_hexagrams(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """生成易經六十四卦.

        從股票價格資料生成六爻卦象。
        使用滾動窗口（6天）構造每個卦象。

        Args:
            df: 原始價格資料 DataFrame。
                - 多檔股票：MultiIndex DataFrame（columns: [Ticker, OHLCV]）
                - 單檔股票：標準 DataFrame（columns: OHLCV）

        Returns:
            新增了以下欄位的 DataFrame：
            - Ritual_Num: 儀式數字（6, 7, 8, 9）
            - Ritual_Sequence: 連續6天的儀式數字序列（字串，例如 "987896"）
                - 順序：最舊的在底部（第1位），最新的在頂部（第6位）
            - Hexagram_Binary: 二進制字串（用於查詢 HEXAGRAM_MAP）
                - 9 和 7 -> "1"（陽爻）
                - 6 和 8 -> "0"（陰爻）
                - 例如："987896" -> "101010"

        Note:
            - 會自動丟棄 Volume_MA20 為 NaN 的列（前20天）
            - 每個 ticker 分別處理
            - 需要至少26天資料才能生成完整卦象（20天MA + 6天窗口）
        """
        if df.empty:
            return df.copy()

        # 判斷是否為 MultiIndex（多檔股票）
        is_multiindex = isinstance(df.columns, pd.MultiIndex)

        if is_multiindex:
            # 多檔股票：分別處理每個 ticker
            result_dfs = []

            for ticker in df.columns.get_level_values(0).unique():
                ticker_df = df[ticker].copy()

                # 計算技術指標
                ticker_df = self._calculate_technical_indicators(ticker_df)

                # 丟棄 Volume_MA20 為 NaN 的列（前20天）
                ticker_df = ticker_df.dropna(subset=['Volume_MA20'])

                if ticker_df.empty:
                    continue

                # 計算儀式數字（向量化）
                ticker_df['Ritual_Num'] = np.select(
                    condlist=[
                        (ticker_df['Daily_Return'] > settings.YIN_YANG_THRESHOLD) &
                        (ticker_df['RVOL'] > 2.0),  # is_up and is_whale
                        (ticker_df['Daily_Return'] <= settings.YIN_YANG_THRESHOLD) &
                        (ticker_df['RVOL'] > 2.0),  # not is_up and is_whale
                        (ticker_df['Daily_Return'] > settings.YIN_YANG_THRESHOLD) &
                        (ticker_df['RVOL'] <= 2.0),  # is_up and not is_whale
                    ],
                    choicelist=[9, 6, 7],
                    default=8  # not is_up and not is_whale
                )

                # 處理 NaN 值（RVOL 為 NaN 時設為 8）
                ticker_df['Ritual_Num'] = ticker_df['Ritual_Num'].fillna(8)

                # 使用滾動窗口生成六爻序列
                ritual_sequences = []
                hexagram_binaries = []

                for i in range(len(ticker_df)):
                    if i < 5:  # 前5天無法形成完整卦象
                        ritual_sequences.append(None)
                        hexagram_binaries.append(None)
                    else:
                        window = ticker_df['Ritual_Num'].iloc[i-5:i+1]
                        if window.isna().any():
                            ritual_sequences.append(None)
                            hexagram_binaries.append(None)
                        else:
                            ritual_nums = window.astype(int).tolist()
                            ritual_sequence = ''.join(map(str, ritual_nums))
                            binary_sequence = ''.join(
                                '1' if num in [9, 7] else '0'
                                for num in ritual_nums
                            )
                            ritual_sequences.append(ritual_sequence)
                            hexagram_binaries.append(binary_sequence)

                ticker_df['Ritual_Sequence'] = ritual_sequences
                ticker_df['Hexagram_Binary'] = hexagram_binaries

                # 使用 MultiIndex 保持結構
                ticker_df.columns = pd.MultiIndex.from_product(
                    [[ticker], ticker_df.columns]
                )
                result_dfs.append(ticker_df)

            if result_dfs:
                result = pd.concat(result_dfs, axis=1)
            else:
                result = pd.DataFrame()
        else:
            # 單檔股票：直接處理
            result = self._calculate_technical_indicators(df.copy())

            # 丟棄 Volume_MA20 為 NaN 的列
            result = result.dropna(subset=['Volume_MA20'])

            if not result.empty:
                # 計算儀式數字（向量化）
                result['Ritual_Num'] = np.select(
                    condlist=[
                        (result['Daily_Return'] > settings.YIN_YANG_THRESHOLD) &
                        (result['RVOL'] > 2.0),
                        (result['Daily_Return'] <= settings.YIN_YANG_THRESHOLD) &
                        (result['RVOL'] > 2.0),
                        (result['Daily_Return'] > settings.YIN_YANG_THRESHOLD) &
                        (result['RVOL'] <= 2.0),
                    ],
                    choicelist=[9, 6, 7],
                    default=8
                )

                # 處理 NaN
                result['Ritual_Num'] = result['Ritual_Num'].fillna(8)

                # 生成六爻序列
                ritual_sequences = []
                hexagram_binaries = []

                for i in range(len(result)):
                    if i < 5:
                        ritual_sequences.append(None)
                        hexagram_binaries.append(None)
                    else:
                        window = result['Ritual_Num'].iloc[i-5:i+1]
                        if window.isna().any():
                            ritual_sequences.append(None)
                            hexagram_binaries.append(None)
                        else:
                            ritual_nums = window.astype(int).tolist()
                            ritual_sequence = ''.join(map(str, ritual_nums))
                            binary_sequence = ''.join(
                                '1' if num in [9, 7] else '0'
                                for num in ritual_nums
                            )
                            ritual_sequences.append(ritual_sequence)
                            hexagram_binaries.append(binary_sequence)

                result['Ritual_Sequence'] = ritual_sequences
                result['Hexagram_Binary'] = hexagram_binaries

        return result