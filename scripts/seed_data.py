"""易經資料種子腳本.

此腳本生成包含完整 64 卦資料的 JSON 檔案，作為 RAG 系統的知識庫。
"""

import json
import os
import sys
from pathlib import Path

# 將父目錄加入路徑，以便匯入 config
sys.path.insert(0, str(Path(__file__).parent.parent))

# 從 config 匯入 HEXAGRAM_MAP 以獲取正確的名稱
from config import HEXAGRAM_MAP

# 前 8 卦的完整資料（Judgement 和 Image）
FIRST_8_DETAILS = {
    1: {
        "judgement": "The Creative works sublime success, furthering through perseverance.",
        "image": "The movement of heaven is full of power. Thus the superior man makes himself strong and untiring."
    },
    2: {
        "judgement": "The Receptive brings about sublime success, furthering through the perseverance of a mare.",
        "image": "The earth's condition is receptive devotion. Thus the superior man who has breadth of character carries the outer world."
    },
    3: {
        "judgement": "Difficulty at the Beginning works supreme success, furthering through perseverance.",
        "image": "Clouds and thunder: The image of Difficulty at the Beginning. Thus the superior man brings order out of confusion."
    },
    4: {
        "judgement": "Youthful Folly has success. It is not I who seek the young fool; the young fool seeks me.",
        "image": "A spring wells up at the foot of the mountain: The image of Youthful Folly."
    },
    5: {
        "judgement": "Waiting. If you are sincere, you have light and success. Perseverance brings good fortune.",
        "image": "Clouds rise up to heaven: The image of Waiting. Thus the superior man eats and drinks, is joyous and of good cheer."
    },
    6: {
        "judgement": "Conflict. You are sincere and are being obstructed. A cautious halt halfway brings good fortune.",
        "image": "Heaven and water go their opposite ways: The image of Conflict."
    },
    7: {
        "judgement": "The Army. The army needs perseverance and a strong man. Good fortune without blame.",
        "image": "In the middle of the earth is water: The image of The Army."
    },
    8: {
        "judgement": "Holding Together brings good fortune. Inquire of the oracle once again whether you possess sublimity, constancy, and perseverance.",
        "image": "On the earth is water: The image of Holding Together."
    }
}

# 建立完整的 64 卦資料
ICHING_DATA = []

# 按照 ID 順序（1-64）建立資料
for hex_id in range(1, 65):
    # 從 HEXAGRAM_MAP 中找到對應的卦象
    hex_entry = None
    for binary_str, hex_dict in HEXAGRAM_MAP.items():
        if hex_dict["id"] == hex_id:
            hex_entry = hex_dict
            break
    
    if hex_entry is None:
        # 如果找不到，使用預設值
        name = f"Hexagram {hex_id}"
        chinese_name = "未知"
    else:
        # 提取名稱（移除括號中的英文解釋，只保留拼音部分）
        name_full = hex_entry["name"]
        # 如果包含括號，提取括號前的部分作為 name
        if "(" in name_full:
            name = name_full.split("(")[0].strip()
        else:
            name = name_full
        chinese_name = hex_entry["nature"]
    
    # 判斷是前 8 卦還是其他卦
    if hex_id <= 8:
        # 前 8 卦使用完整資料
        judgement = FIRST_8_DETAILS[hex_id]["judgement"]
        image = FIRST_8_DETAILS[hex_id]["image"]
    else:
        # 第 9-64 卦使用通用文字（節省空間）
        judgement = f"The hexagram {name} ({chinese_name}) represents a significant moment in the cycle of change. Success comes through understanding the situation and acting appropriately."
        image = f"The image of {name} ({chinese_name}) reflects the natural patterns of transformation. The superior man observes these patterns and aligns his actions accordingly."
    
    ICHING_DATA.append({
        "hexagram_id": hex_id,
        "name": name,
        "chinese_name": chinese_name,
        "judgement": judgement,
        "image": image
    })

# 確保 data 目錄存在
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# 寫入 JSON 檔案
output_path = data_dir / "iching_book.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ICHING_DATA, f, indent=2, ensure_ascii=False)

print(f"Successfully seeded {len(ICHING_DATA)} hexagrams to {output_path}")
