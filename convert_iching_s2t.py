"""將 data/iching_complete.json 內簡體中文轉為繁體，並重建 ChromaDB 向量庫。"""

import json
import sys
from pathlib import Path

# 簡體→繁體：使用 OpenCC（若未安裝會自動 pip install）
try:
    from opencc import OpenCC
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencc-python-reimplemented", "-q"])
    from opencc import OpenCC

PATH = Path("data/iching_complete.json")


def main() -> None:
    if not PATH.exists():
        print(f"[ERROR] 找不到 {PATH}，請先執行 python setup_iching_db.py")
        sys.exit(1)

    cc = OpenCC("s2tw")  # 簡體 → 臺灣繁體

    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERROR] 預期 JSON 為 list")
        sys.exit(1)

    for h in data:
        for key in ("name", "judgment", "image"):
            if isinstance(h.get(key), str) and h[key]:
                h[key] = cc.convert(h[key])
        for ln in h.get("lines") or []:
            if not isinstance(ln, dict):
                continue
            for k in ("meaning", "xiang"):
                if isinstance(ln.get(k), str) and ln[k]:
                    ln[k] = cc.convert(ln[k])

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("[OK] data/iching_complete.json 已轉為繁體。")

    # 重建向量庫
    from knowledge_loader import IChingKnowledgeLoader
    loader = IChingKnowledgeLoader()
    loader.build_vector_db()


if __name__ == "__main__":
    main()
