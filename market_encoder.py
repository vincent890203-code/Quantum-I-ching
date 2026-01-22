"""Quantum I-Ching 專案市場資料編碼模組.

此模組將股票市場資料轉換為易經六十四卦的表示方式。
使用「大衍之數」(Da Yan Zhi Shu) 機率分布，以 RVOL 百分位數作為能量代理變數。
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import settings


class MarketEncoder:
    """市場資料編碼器類別.

    將股票價格資料轉換為易經卦象表示。
    使用「大衍之數」傳統機率分布，以 RVOL 百分位數映射到四象（老陽、少陽、老陰、少陰）。
    
    大衍之數機率分布：
    - 6 (老陰 / 變動之陰): 1/16 (6.25%) - 極低能量，往往預示突破
    - 8 (少陰 / 靜止之陰): 7/16 (43.75%) - 低-中能量，靜態市場
    - 7 (少陽 / 靜止之陽): 5/16 (31.25%) - 中-高能量，活躍但靜態市場
    - 9 (老陽 / 變動之陽): 3/16 (18.75%) - 極高能量，突破
    """

    def _calculate_technical_indicators(
        self,
        df: pd.DataFrame,
        rvol_window: int = 120
    ) -> pd.DataFrame:
        """計算技術指標.

        計算日報酬率、20日均量、相對成交量（RVOL），以及 RVOL 百分位數。

        Args:
            df: 包含 Open, High, Low, Close, Volume 欄位的 DataFrame。
                可以是標準 DataFrame 或 MultiIndex DataFrame。
            rvol_window: 用於計算 RVOL 百分位數的滾動窗口大小（預設 120 天）。

        Returns:
            新增了以下欄位的 DataFrame：
            - Daily_Return: 日報酬率（Close 的百分比變動）
            - Volume_MA20: 20日成交量移動平均
            - RVOL: 相對成交量（Volume / Volume_MA20）
            - RVOL_Percentile: RVOL 的百分位數排名（0.0-1.0）

        Note:
            - 前20天的 Volume_MA20 和 RVOL 會是 NaN
            - 前 rvol_window 天的 RVOL_Percentile 會是 NaN（需要足夠歷史資料）
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

        # 計算 RVOL 百分位數（使用滾動窗口）
        # 對於每個時間點，計算當前 RVOL 在過去 rvol_window 天內的百分位數排名
        def calculate_percentile(series):
            """計算序列最後一個值在整個序列中的百分位數排名。"""
            if len(series) == 0:
                return np.nan
            current_val = series.iloc[-1]
            if pd.isna(current_val):
                return np.nan
            # 計算當前值在窗口內的排名（百分位數）
            # rank(pct=True) 會返回 0.0-1.0 之間的百分位數
            # method='average' 處理相同值的情況
            ranked = pd.Series(series).rank(pct=True, method='average', na_option='keep')
            return ranked.iloc[-1]
        
        df['RVOL_Percentile'] = df['RVOL'].rolling(
            window=rvol_window,
            min_periods=min(10, rvol_window // 4)  # 至少需要一些資料點
        ).apply(calculate_percentile, raw=False)

        return df

    def _get_dayan_yao(self, rvol_percentile: float) -> int:
        """根據 RVOL 百分位數決定爻（使用大衍之數機率分布）.

        實作傳統「大衍之數」機率分布：
        - 6 (老陰): 1/16 (0.0625) - 極低能量
        - 8 (少陰): 7/16 (0.4375) - 低-中能量
        - 7 (少陽): 5/16 (0.3125) - 中-高能量
        - 9 (老陽): 3/16 (0.1875) - 極高能量

        Args:
            rvol_percentile: RVOL 百分位數（0.0-1.0）

        Returns:
            儀式數字：6（老陰）、7（少陽）、8（少陰）、9（老陽）

        Note:
            - 如果 rvol_percentile 為 NaN，返回 8（少陰，預設靜態狀態）
            - 使用精確的累積機率閾值：1/16, 8/16, 13/16, 16/16
        """
        # 處理 NaN 值
        if pd.isna(rvol_percentile):
            return 8  # 預設為少陰（靜態狀態）

        # 大衍之數累積機率閾值
        threshold_6 = 1 / 16  # 0.0625
        threshold_8 = 8 / 16  # 0.5000 (1/16 + 7/16)
        threshold_7 = 13 / 16  # 0.8125 (8/16 + 5/16)

        # 根據百分位數映射到四象
        if rvol_percentile < threshold_6:
            return 6  # 老陰（變動之陰）- 極低能量
        elif rvol_percentile < threshold_8:
            return 8  # 少陰（靜止之陰）- 低-中能量
        elif rvol_percentile < threshold_7:
            return 7  # 少陽（靜止之陽）- 中-高能量
        else:
            return 9  # 老陽（變動之陽）- 極高能量

    def generate_hexagrams(
        self,
        df: pd.DataFrame,
        rvol_window: int = 120
    ) -> pd.DataFrame:
        """生成易經六十四卦.

        從股票價格資料生成六爻卦象。
        使用滾動窗口（6天）構造每個卦象。
        使用「大衍之數」機率分布，以 RVOL 百分位數決定爻。

        Args:
            df: 原始價格資料 DataFrame。
                - 多檔股票：MultiIndex DataFrame（columns: [Ticker, OHLCV]）
                - 單檔股票：標準 DataFrame（columns: OHLCV）
            rvol_window: 用於計算 RVOL 百分位數的滾動窗口大小（預設 120 天）。

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
            - 需要至少 max(26, rvol_window) 天資料才能生成完整卦象
            - 使用大衍之數機率分布映射 RVOL 百分位數到爻
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

                # 計算技術指標（包含 RVOL 百分位數）
                ticker_df = self._calculate_technical_indicators(ticker_df, rvol_window)

                # 丟棄 Volume_MA20 為 NaN 的列（前20天）
                ticker_df = ticker_df.dropna(subset=['Volume_MA20'])

                if ticker_df.empty:
                    continue

                # 使用大衍之數方法計算儀式數字（向量化）
                ticker_df['Ritual_Num'] = ticker_df['RVOL_Percentile'].apply(
                    lambda x: self._get_dayan_yao(x)
                )

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
            result = self._calculate_technical_indicators(df.copy(), rvol_window)

            # 丟棄 Volume_MA20 為 NaN 的列
            result = result.dropna(subset=['Volume_MA20'])

            if not result.empty:
                # 使用大衍之數方法計算儀式數字（向量化）
                result['Ritual_Num'] = result['RVOL_Percentile'].apply(
                    lambda x: self._get_dayan_yao(x)
                )

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