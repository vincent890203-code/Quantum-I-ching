"""Quantum I-Ching：從驗證來源下載完整易經資料（64 卦含六爻）.

使用 john-walks-slow/open-iching（iching/iching.json），
經嚴格驗證後轉為統一格式並儲存至 data/iching_complete.json。
可選：合併 ichuan/xiang.json 之象傳。僅從指定 URL 取得，不進行 AI 生成。
"""

import json
import sys
from pathlib import Path

import requests

# john-walks-slow/open-iching 結構：id, name, scripture, lines[{id, type, name, scripture}]
URL_ICHING = "https://cdn.jsdelivr.net/gh/john-walks-slow/open-iching@main/iching/iching.json"
URL_XIANG = "https://cdn.jsdelivr.net/gh/john-walks-slow/open-iching@main/ichuan/xiang.json"
OUT_PATH = Path("data/iching_complete.json")


def main() -> None:
    # 1. 下載 iching.json
    try:
        resp = requests.get(URL_ICHING, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] 無法下載易經資料：{e}")
        print("請確認網路連線，或稍後再試。")
        sys.exit(1)

    try:
        raw = resp.json()
    except json.JSONDecodeError as e:
        print(f"[ERROR] 下載內容不是有效 JSON：{e}")
        sys.exit(1)

    # 2. 嚴格驗證（john-walks-slow 格式：id, name, scripture, lines）
    try:
        if not isinstance(raw, list):
            raise ValueError(f"預期頂層為 list，實際為 {type(raw)}")
        if len(raw) != 64:
            raise ValueError(f"驗證失敗：卦數應為 64，實際為 {len(raw)}")
        h1 = next((h for h in raw if h.get("id") == 1), raw[0])
        name = (h1.get("name") or "").strip()
        if name != "乾":
            raise ValueError(f"驗證失敗：第 1 卦應為「乾」，實際為「{name}」")
        if "lines" not in h1:
            raise ValueError("驗證失敗：第 1 卦缺少 'lines' 欄位")
        lines = h1["lines"]
        if not lines or not isinstance(lines, list):
            raise ValueError("驗證失敗：第 1 卦的 'lines' 必須為非空 list")
        has = any(
            isinstance(ln, dict) and (ln.get("scripture") or ln.get("name"))
            for ln in lines
        )
        if not has:
            raise ValueError("驗證失敗：第 1 卦的 lines 中至少需包含 scripture 或 name")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 3. 可選：下載象傳（象曰）用以填 image / 小象
    xiang_lookup: dict = {}
    try:
        r2 = requests.get(URL_XIANG, timeout=15)
        if r2.status_code == 200:
            xiang_lookup = r2.json() or {}
    except Exception:
        pass

    # 4. 轉為統一格式（number, name, judgment, image, lines[{position, meaning, xiang}]）
    out = []
    for h in raw:
        hid = h.get("id")
        hname = h.get("name") or "?"
        scripture = h.get("scripture") or ""
        image = xiang_lookup.get(f"#{hid}", "")

        line_list = []
        for ln in h.get("lines") or []:
            if not isinstance(ln, dict):
                continue
            pos = ln.get("id")
            if pos is None:
                pos = len(line_list) + 1
            try:
                pos = int(pos)
            except (TypeError, ValueError):
                pos = len(line_list) + 1
            yaoming = ln.get("name") or ""
            yao_text = ln.get("scripture") or ""
            meaning = f"{yaoming}：{yao_text}" if yaoming and yao_text else (yao_text or yaoming)
            xiang = xiang_lookup.get(f"#{hid}.{ln.get('id', pos)}", "")
            line_list.append({"position": pos, "meaning": meaning, "xiang": xiang})

        out.append({
            "number": hid,
            "name": hname,
            "judgment": scripture,
            "image": image,
            "lines": line_list,
        })

    # 5. 儲存（utf-8）
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[OK] Verified I-Ching Data (64 Hexagrams + Lines) saved.")


if __name__ == "__main__":
    main()
