"""Quantum I-Ching 專案配置模組.

此模組提供全專案共用的配置設定，包括：
- 全局設定（日期範圍、標的股票、閾值等）
- 易經六十四卦對照表
"""

import dataclasses
from datetime import date
from typing import TypedDict


class HexagramDict(TypedDict):
    """六十四卦字典結構定義."""
    id: int
    name: str
    nature: str  # 繁體中文卦名


@dataclasses.dataclass(frozen=True)
class Settings:
    """全局配置設定類別.

    使用 frozen dataclass 確保設定不可變，符合函數式程式設計原則。
    
    Attributes:
        START_DATE: 資料起始日期（ISO 格式字串）
        END_DATE: 資料結束日期（ISO 格式字串，預設為今日）
        TARGET_TICKERS: 目標股票代號清單
        YIN_YANG_THRESHOLD: 判斷陰陽的價格變動閾值（預設 0.0）
        MARKET_TYPE: 市場類型，選項：'US'（美股）、'TW'（台股）、'CRYPTO'（加密貨幣），預設為 'US'
    """
    START_DATE: str = "2020-01-01"
    END_DATE: str = dataclasses.field(
        default_factory=lambda: date.today().isoformat()
    )
    TARGET_TICKERS: list[str] = dataclasses.field(
        # 預設改為台股常見標的，方便本系統以台灣市場為主
        default_factory=lambda: ["2330.TW", "2317.TW", "2454.TW"]
        # 根據 MARKET_TYPE 可以設定不同的範例：
        # US: ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMD"]
        # TW: ["2330.TW", "2317.TW", "2454.TW"]
        # CRYPTO: ["BTC-USD", "ETH-USD", "SOL-USD"]
    )
    YIN_YANG_THRESHOLD: float = 0.0
    MARKET_TYPE: str = "TW"  # 選項：'US', 'TW', 'CRYPTO'
    
    # 模型超參數（來自 Optuna 優化結果）
    SEQUENCE_LENGTH: int = 30  # 最佳值：30（月週期）
    HIDDEN_DIM: int = 256  # 最佳值：256
    NUM_LAYERS: int = 1  # 最佳值：1
    DROPOUT: float = 0.35  # 最佳值：0.35
    LEARNING_RATE: float = 0.001  # 最佳值：0.001
    PREDICTION_WINDOW: int = 5  # T+5 波動性預測


# 六十四卦對照表
# 鍵值為二進制字串（1=陽爻，0=陰爻），值為卦象資訊字典
HEXAGRAM_MAP: dict[str, HexagramDict] = {
    "111111": {"id": 1, "name": "Qian (The Creative)", "nature": "乾"},
    "000000": {"id": 2, "name": "Kun (The Receptive)", "nature": "坤"},
    "100010": {"id": 3, "name": "Chun (Difficulty at the Beginning)", "nature": "屯"},
    "010001": {"id": 4, "name": "Meng (Youthful Folly)", "nature": "蒙"},
    "111010": {"id": 5, "name": "Xu (Waiting)", "nature": "需"},
    "010111": {"id": 6, "name": "Song (Conflict)", "nature": "訟"},
    "010000": {"id": 7, "name": "Shi (The Army)", "nature": "師"},
    "000010": {"id": 8, "name": "Bi (Holding Together)", "nature": "比"},
    "111011": {"id": 9, "name": "Xiao Chu (The Taming Power of the Small)", "nature": "小畜"},
    "110111": {"id": 10, "name": "Lv (Treading)", "nature": "履"},
    "111000": {"id": 11, "name": "Tai (Peace)", "nature": "泰"},
    "000111": {"id": 12, "name": "Pi (Standstill)", "nature": "否"},
    "101111": {"id": 13, "name": "Tong Ren (Fellowship with Men)", "nature": "同人"},
    "111101": {"id": 14, "name": "Da You (Possession in Great Measure)", "nature": "大有"},
    "001000": {"id": 15, "name": "Qian (Modesty)", "nature": "謙"},
    "000100": {"id": 16, "name": "Yu (Enthusiasm)", "nature": "豫"},
    "100110": {"id": 17, "name": "Sui (Following)", "nature": "隨"},
    "011001": {"id": 18, "name": "Gu (Work on What Has Been Spoiled)", "nature": "蠱"},
    "110000": {"id": 19, "name": "Lin (Approach)", "nature": "臨"},
    "000011": {"id": 20, "name": "Guan (Contemplation)", "nature": "觀"},
    "100101": {"id": 21, "name": "Shi He (Biting Through)", "nature": "噬嗑"},
    "101001": {"id": 22, "name": "Bi (Grace)", "nature": "賁"},
    "000001": {"id": 23, "name": "Bo (Splitting Apart)", "nature": "剝"},
    "100000": {"id": 24, "name": "Fu (Return)", "nature": "復"},
    "100111": {"id": 25, "name": "Wu Wang (Innocence)", "nature": "無妄"},
    "111001": {"id": 26, "name": "Da Chu (The Taming Power of the Great)", "nature": "大畜"},
    "100001": {"id": 27, "name": "Yi (The Corners of the Mouth)", "nature": "頤"},
    "011110": {"id": 28, "name": "Da Guo (Preponderance of the Great)", "nature": "大過"},
    "010010": {"id": 29, "name": "Kan (The Abysmal)", "nature": "坎"},
    "101101": {"id": 30, "name": "Li (The Clinging, Fire)", "nature": "離"},
    "001110": {"id": 31, "name": "Xian (Influence)", "nature": "咸"},
    "011100": {"id": 32, "name": "Heng (Duration)", "nature": "恆"},
    "001111": {"id": 33, "name": "Dun (Retreat)", "nature": "遯"},
    "111100": {"id": 34, "name": "Da Zhuang (The Power of the Great)", "nature": "大壯"},
    "000101": {"id": 35, "name": "Jin (Progress)", "nature": "晉"},
    "101000": {"id": 36, "name": "Ming Yi (Darkening of the Light)", "nature": "明夷"},
    "101011": {"id": 37, "name": "Jia Ren (The Family)", "nature": "家人"},
    "110101": {"id": 38, "name": "Kui (Opposition)", "nature": "睽"},
    "001010": {"id": 39, "name": "Jian (Obstruction)", "nature": "蹇"},
    "010100": {"id": 40, "name": "Xie (Deliverance)", "nature": "解"},
    "110001": {"id": 41, "name": "Sun (Decrease)", "nature": "損"},
    "100011": {"id": 42, "name": "Yi (Increase)", "nature": "益"},
    "111110": {"id": 43, "name": "Guai (Break-through)", "nature": "夬"},
    "011111": {"id": 44, "name": "Gou (Coming to Meet)", "nature": "姤"},
    "000110": {"id": 45, "name": "Cui (Gathering Together)", "nature": "萃"},
    "011000": {"id": 46, "name": "Sheng (Pushing Upward)", "nature": "升"},
    "010110": {"id": 47, "name": "Kun (Oppression)", "nature": "困"},
    "011010": {"id": 48, "name": "Jing (The Well)", "nature": "井"},
    "101110": {"id": 49, "name": "Ge (Revolution)", "nature": "革"},
    "011101": {"id": 50, "name": "Ding (The Cauldron)", "nature": "鼎"},
    "100100": {"id": 51, "name": "Zhen (The Arousing, Shock)", "nature": "震"},
    "001001": {"id": 52, "name": "Gen (Keeping Still, Mountain)", "nature": "艮"},
    "001011": {"id": 53, "name": "Jian (Development)", "nature": "漸"},
    "110100": {"id": 54, "name": "Gui Mei (The Marrying Maiden)", "nature": "歸妹"},
    "101100": {"id": 55, "name": "Feng (Abundance)", "nature": "豐"},
    "001101": {"id": 56, "name": "Lv (The Wanderer)", "nature": "旅"},
    "011011": {"id": 57, "name": "Sun (The Gentle, Wind)", "nature": "巽"},
    "110110": {"id": 58, "name": "Dui (The Joyous, Lake)", "nature": "兌"},
    "010011": {"id": 59, "name": "Huan (Dispersion)", "nature": "渙"},
    "110010": {"id": 60, "name": "Jie (Limitation)", "nature": "節"},
    "110011": {"id": 61, "name": "Zhong Fu (Inner Truth)", "nature": "中孚"},
    "001100": {"id": 62, "name": "Xiao Guo (Preponderance of the Small)", "nature": "小過"},
    "101010": {"id": 63, "name": "Ji Ji (After Completion)", "nature": "既濟"},
    "010101": {"id": 64, "name": "Wei Ji (Before Completion)", "nature": "未濟"},
}


# 全局設定實例
settings = Settings()