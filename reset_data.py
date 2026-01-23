"""Quantum I-Ching 專案資料重置工具.

此腳本用於清除已處理的資料快取，強制重新生成包含新欄位（Future_Hex_ID, Num_Moving_Lines）的資料。
保留原始資料檔案（iching_complete.json）。
"""

import os
import shutil
from pathlib import Path


def clear_data_cache() -> None:
    """清除資料快取，但保留原始資料檔案.
    
    刪除的檔案/目錄：
    - data/*.csv（已處理的 CSV 檔案）
    - data/*.pkl（已處理的 PKL 檔案）
    - data/chroma_db/（向量資料庫，會自動重建）
    - data/best_model.pth（舊模型，需要重新訓練）
    
    保留的檔案：
    - data/iching_complete.json（原始易經資料）
    - data/iching_book.json（原始易經書籍資料）
    """
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("[INFO] data/ 目錄不存在，無需清除。")
        return
    
    deleted_items = []
    
    # 刪除 CSV 檔案
    for csv_file in data_dir.glob("*.csv"):
        csv_file.unlink()
        deleted_items.append(f"  - {csv_file.name}")
    
    # 刪除 PKL 檔案
    for pkl_file in data_dir.glob("*.pkl"):
        pkl_file.unlink()
        deleted_items.append(f"  - {pkl_file.name}")
    
    # 刪除舊模型（架構已變更）
    model_file = data_dir / "best_model.pth"
    if model_file.exists():
        model_file.unlink()
        deleted_items.append(f"  - {model_file.name} (舊模型，需要重新訓練)")
    
    # 刪除向量資料庫（會自動重建）
    chroma_dir = data_dir / "chroma_db"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        deleted_items.append(f"  - chroma_db/ (向量資料庫，會自動重建)")
    
    if deleted_items:
        print("[INFO] 已清除以下快取檔案/目錄：")
        for item in deleted_items:
            print(item)
    else:
        print("[INFO] 沒有找到需要清除的快取檔案。")
    
    # 列出保留的檔案
    preserved_files = []
    for json_file in data_dir.glob("*.json"):
        preserved_files.append(f"  - {json_file.name}")
    
    if preserved_files:
        print("\n[INFO] 保留的原始資料檔案：")
        for item in preserved_files:
            print(item)


def main() -> None:
    """主函數."""
    print("=" * 60)
    print("Quantum I-Ching 資料重置工具")
    print("=" * 60)
    print("\n此工具將清除已處理的資料快取，強制重新生成包含新欄位的資料。")
    print("新欄位：Future_Hex_ID（變卦 ID）、Num_Moving_Lines（動爻數量）\n")
    
    response = input("確定要清除快取嗎？(y/N): ").strip().lower()
    
    if response == 'y':
        clear_data_cache()
        print("\n[SUCCESS] 資料快取已清除。")
        print("\n下一步：")
        print("  1. 執行 `python model_lstm.py` 重新訓練模型（會自動重新生成資料）")
        print("  2. 或執行 `python backtester.py` 進行回測（會自動重新生成資料）")
    else:
        print("\n[INFO] 操作已取消。")


if __name__ == "__main__":
    main()
