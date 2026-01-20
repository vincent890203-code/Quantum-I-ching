"""Quantum I-Ching 專案主程式.

整合所有模組，執行完整的易經分析流程：
1. 資料載入
2. 資料編碼
3. 卦象解碼
4. 結果視覺化
"""

import sys
from typing import List, Optional

# 設定輸出編碼為 UTF-8（處理 Windows 終端編碼問題）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Python < 3.7 或 reconfigure 不可用時，使用環境變數設定
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'

# 檢查必要依賴是否已安裝
# 使用英文錯誤訊息確保在所有終端環境下都能正常顯示
try:
    import pandas as pd
except ImportError:
    print("[ERROR] Missing required package: 'pandas'")
    print("  Please run: pip install pandas")
    print("  Or run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] Missing required package: 'numpy'")
    print("  Please run: pip install numpy")
    print("  Or run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] Missing required package: 'yfinance'")
    print("  Please run: pip install yfinance")
    print("  Or run: pip install -r requirements.txt")
    sys.exit(1)

from config import settings
from data_loader import MarketDataLoader
from iching_core import IChingCore
from market_encoder import MarketEncoder


def print_hexagram_visual(ritual_sequence: List[int]) -> None:
    """以 ASCII 藝術顯示六爻卦象.

    從頂部（第6爻）到底部（第1爻）顯示卦象。

    Args:
        ritual_sequence: 儀式數字序列，從底部到頂部（索引 0 到 5）。
            - 9/7 (陽爻): 顯示為 `---------`
            - 6/8 (陰爻): 顯示為 `---   ---`
    """
    print("\n卦象視覺化（從上到下）：")
    print("─" * 50)
    
    # 從頂部到底部（反向遍歷）
    for i in range(5, -1, -1):
        line_num = i + 1  # 1-based index
        ritual_num = ritual_sequence[i]
        
        # 判斷是陽爻還是陰爻
        is_yang = ritual_num in [9, 7]
        
        if is_yang:
            line = "─────────"
            label = "陽"
        else:
            line = "───   ───"
            label = "陰"
        
        # 標記動爻（9 或 6）
        if ritual_num in [9, 6]:
            marker = " * (變)"
        else:
            marker = ""
        
        print(f"第 {line_num} 爻 {label}: {line}{marker}")
    
    print("─" * 50)
    print("(* 變 = 動爻，會變動到相反的狀態)\n")


def format_moving_lines(moving_lines: List[int]) -> str:
    """格式化動爻列表為可讀字串.

    Args:
        moving_lines: 動爻列表（1-based index）

    Returns:
        格式化的字串，例如 "1, 5, 6" 或 "無"
    """
    if not moving_lines:
        return "無"
    return ", ".join(map(str, moving_lines))


def main(ticker: str = "NVDA") -> None:
    """主程式執行函數.

    Args:
        ticker: 要分析的股票代號，預設為 "NVDA"

    Raises:
        SystemExit: 如果發生嚴重錯誤，會以錯誤碼退出
    """
    try:
        print(f"\n{'='*60}")
        print(f"  Quantum I-Ching 分析系統")
        print(f"{'='*60}\n")

        # ========== 步驟 1: 資料載入 ==========
        print("[步驟 1] 載入市場資料...")
        loader = MarketDataLoader()
        raw_data = loader.fetch_data(tickers=[ticker])

        if raw_data.empty:
            print(f"[錯誤] 無法獲取 {ticker} 的資料")
            sys.exit(1)

        num_rows = len(raw_data)
        print(f"[成功] 獲取 {num_rows} 筆 {ticker} 資料記錄\n")

        # ========== 步驟 2: 資料編碼 ==========
        print("[步驟 2] 將資料編碼為易經卦象...")
        encoder = MarketEncoder()
        encoded_data = encoder.generate_hexagrams(raw_data)

        if encoded_data.empty:
            print("[錯誤] 編碼後的資料為空（可能資料不足，需要至少 26 天）")
            sys.exit(1)

        # 取得最新一筆記錄
        latest_row = encoded_data.iloc[-1]
        latest_index = encoded_data.index[-1]
        
        # 格式化日期字串
        if hasattr(latest_index, 'strftime'):
            date_str = latest_index.strftime('%Y-%m-%d')
        else:
            date_str = str(latest_index)

        # 提取 Ritual_Sequence（字串，需轉換為列表）
        # 使用 try-except 處理可能的 KeyError 或 AttributeError
        try:
            # 嘗試使用直接索引
            if 'Ritual_Sequence' in latest_row.index:
                ritual_sequence_str = latest_row['Ritual_Sequence']
            else:
                # 如果沒有找到，嘗試使用 get 方法
                ritual_sequence_str = latest_row.get('Ritual_Sequence', None)
        except (KeyError, AttributeError, TypeError) as e:
            print(f"[錯誤] 無法存取 Ritual_Sequence 欄位: {e}")
            print(f"  可用欄位: {list(encoded_data.columns)}")
            sys.exit(1)
        
        if ritual_sequence_str is None or ritual_sequence_str == '' or pd.isna(ritual_sequence_str):
            print("[錯誤] 無法取得儀式數字序列（可能資料不足，需要至少 26 天）")
            print(f"  Ritual_Sequence 值: {ritual_sequence_str}")
            sys.exit(1)

        # 將字串轉換為整數列表
        # 例如: "987896" -> [9, 8, 7, 8, 9, 6]
        try:
            ritual_sequence = [int(char) for char in str(ritual_sequence_str)]
        except (ValueError, TypeError) as e:
            print(f"[錯誤] 無法解析儀式數字序列: {ritual_sequence_str}")
            print(f"  錯誤詳情: {e}")
            sys.exit(1)
        
        # 驗證序列長度
        if len(ritual_sequence) != 6:
            print(f"[錯誤] 儀式數字序列長度不正確（應為 6，實際為 {len(ritual_sequence)}）")
            print(f"  序列內容: {ritual_sequence}")
            sys.exit(1)
        
        # 提取二進制字串
        try:
            if 'Hexagram_Binary' in latest_row.index:
                binary_code = str(latest_row['Hexagram_Binary'])
            else:
                binary_code = str(latest_row.get('Hexagram_Binary', ''))
            if binary_code == 'nan' or binary_code == '':
                binary_code = ''
        except (KeyError, AttributeError, TypeError):
            binary_code = ''
        
        print(f"[成功] 生成卦象資料\n")
        print(f"  日期: {date_str}")
        print(f"  儀式序列: {ritual_sequence}")
        print(f"  二進制碼: {binary_code}\n")

        # ========== 步驟 3: 卦象解碼 ==========
        print("[步驟 3] 解釋卦象（當前卦 -> 未來卦）...")
        core = IChingCore()
        interpretation = core.interpret_sequence(ritual_sequence)

        current_hex = interpretation['current_hex']
        future_hex = interpretation['future_hex']
        moving_lines = interpretation['moving_lines']

        print(f"[成功] 完成卦象解釋\n")

        # ========== 步驟 4: 結果視覺化 ==========
        print("=" * 60)
        print(f"  === Quantum I-Ching 分析報告: {ticker} ===")
        print("=" * 60)
        print(f"\n[分析日期] {date_str}")
        print(f"[資料筆數] {num_rows} 筆")

        print(f"\n[儀式數字序列]（由下至上）:")
        print(f"   {ritual_sequence}")
        print(f"   [第1爻] <- 底部")
        print(f"   [第6爻] <- 頂部")

        print(f"\n[二進制編碼] {binary_code}")
        print(f"   (1 = 陽爻, 0 = 陰爻)")

        print(f"\n[當前卦（本卦）]")
        print(f"   編號: {current_hex['id']}")
        print(f"   名稱: {current_hex['name']} ({current_hex['nature']})")
        if current_hex['id'] == 0:
            print(f"   [注意] 此卦尚未在 HEXAGRAM_MAP 中定義")

        print(f"\n[未來卦（之卦）]")
        print(f"   編號: {future_hex['id']}")
        print(f"   名稱: {future_hex['name']} ({future_hex['nature']})")
        if future_hex['id'] == 0:
            print(f"   [注意] 此卦尚未在 HEXAGRAM_MAP 中定義")

        print(f"\n[變動]")
        print(f"   當前: {current_hex['name']} ({current_hex['nature']}) → 未來: {future_hex['name']} ({future_hex['nature']})")
        print(f"   動爻: {format_moving_lines(moving_lines)}")

        # ASCII 藝術顯示
        print_hexagram_visual(ritual_sequence)

        print("=" * 60)
        print("分析完成！")
        print("=" * 60 + "\n")

    except ValueError as e:
        print(f"\n[錯誤] 驗證錯誤: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[錯誤] 發生未預期的錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 可以修改這裡的 ticker 來分析不同的股票
    main(ticker="NVDA")