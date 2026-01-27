"""模型特徵審核腳本（XGBoost）.

此腳本會：
1. 掃描當前目錄及 output/、models/ 等常見模型資料夾中的 XGBoost 模型檔案。
2. 使用 xgboost.Booster 載入每一個模型並讀取 feature_names。
3. 依 Model A / B / C / D / E（固定定義見 model_definitions.py）整理成表。
4. 對 Model D (Pure Quantum) 進行特別驗證：
   - 確認「Close, Volume, RVOL」沒有出現在特徵中。
   - 確認「Moving_Lines_Count, Energy_Delta, Daily_Return」必須出現在特徵中。
"""

from __future__ import annotations

import ast
import os
from typing import Dict, Iterable, List, Optional, Tuple

import xgboost as xgb

from model_definitions import MODEL_ORDER, get_display_name, get_features

# ANSI 顏色常數（Windows 10+ 終端通常支援；若不支援則只會顯示原始文字）
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# 路徑匹配關鍵字（model_definitions 固定特徵；此處固定檔名對應）
PATH_PATTERNS: Dict[str, List[str]] = {
    "A": ["model_a", "model-a"],
    "B": ["model_b", "model-b"],
    "C": ["volatility_model"],
    "D": ["model_d_pure_quantum", "model_d", "pure_quantum"],
    "E": ["model_e", "model-e"],
}
EXPECTED_MODELS = [(mid, get_display_name(mid), PATH_PATTERNS[mid]) for mid in MODEL_ORDER]


def is_model_file(filename: str) -> bool:
    """判斷檔名是否為可能的 XGBoost 模型檔."""
    lower = filename.lower()
    return lower.endswith((".json", ".model", ".pkl"))


def iter_model_files(root: str = ".") -> Iterable[str]:
    """遞迴掃描專案下的模型檔案，略過明顯不相關的目錄（如 venv、__pycache__ 等）."""
    skip_dirs = {
        ".git",
        ".cursor",
        "__pycache__",
        "venv",
        ".venv",
        ".idea",
        ".vscode",
    }

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if is_model_file(fname):
                yield os.path.join(dirpath, fname)


def normalize_path_for_match(path: str) -> str:
    """正規化路徑供匹配使用（統一分隔符、小寫）。"""
    return os.path.normpath(path).replace("\\", "/").lower()


def match_path_to_model(path: str) -> Optional[str]:
    """依路徑匹配到 Model A/B/C/D/E，回傳 model_id；若無則回傳 None."""
    norm = normalize_path_for_match(path)
    for model_id, _name, patterns in EXPECTED_MODELS:
        for p in patterns:
            if p.lower() in norm:
                return model_id
    return None


def load_booster(model_path: str) -> Optional[xgb.Booster]:
    """嘗試以 xgboost.Booster 載入模型，若失敗則回傳 None."""
    try:
        booster = xgb.Booster(model_file=model_path)
        return booster
    except Exception as e:
        print(
            f"{YELLOW}[WARNING]{RESET} 無法以 xgboost.Booster 載入模型: "
            f"{model_path}，錯誤: {e}"
        )
        return None


def extract_feature_names(booster: xgb.Booster) -> List[str]:
    """從 Booster 物件中取得特徵名稱列表，若無則嘗試從 attributes 回溯."""
    if booster.feature_names is not None:
        return list(booster.feature_names)

    attrs = booster.attributes() or {}
    raw = attrs.get("feature_names")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []


def is_model_d(path: str) -> bool:
    """判斷檔案是否對應到 Model D (Pure Quantum)."""
    norm = normalize_path_for_match(path)
    return "model_d" in norm or "model-d" in norm or "pure_quantum" in norm


def verify_model_d_features(features: Iterable[str]) -> Tuple[bool, List[str]]:
    """對 Model D 進行特定特徵驗證（依 model_definitions），回傳 (是否通過, 訊息列表)."""
    feat_set = set(features)
    forbidden = {"Close", "Volume", "RVOL"}
    required = set(get_features("D"))

    errors: List[str] = []

    # 檢查不應出現的特徵
    present_forbidden = sorted(forbidden & feat_set)
    if present_forbidden:
        errors.append(
            f"禁止特徵存在於模型中: {present_forbidden} "
            "(預期 Pure Quantum 模型不應依賴原始價格/成交量/RVOL)"
        )

    # 檢查必須存在的特徵（依 model_definitions.Model D）
    missing_required = sorted(required - feat_set)
    if missing_required:
        errors.append(
            f"缺少必要特徵: {missing_required} "
            f"(model_definitions.Model D 應包含: {sorted(required)})"
        )

    return (len(errors) == 0, errors)


def format_model_table(
    model_info: Dict[str, Tuple[str, List[str]]],
    other_rows: List[Tuple[str, int, str]],
) -> str:
    """將 Model A~E 與其他模型整理成表格輸出."""
    header = ("Model", "Status", "Path", "Feature Count", "List of Features")
    mw = max(len(header[0]), max(len(get_display_name(mid)) for mid in MODEL_ORDER))
    pw = len(header[2])
    cw = max(len(header[3]), 3)
    col_status = 6
    for _id, (path, feats) in model_info.items():
        pw = max(pw, len(path))
        cw = max(cw, len(str(len(feats))))
    for path, cnt, _ in other_rows:
        pw = max(pw, len(path))
        cw = max(cw, len(str(cnt)))
    col_feat = len(header[4])

    lines: List[str] = []
    sep = f"{'-' * mw}  {'-' * col_status}  {'-' * pw}  {'-' * cw}  {'-' * col_feat}"
    lines.append(f"{header[0]:<{mw}}  {header[1]:<{col_status}}  {header[2]:<{pw}}  {header[3]:>{cw}}  {header[4]}")
    lines.append(sep)

    for model_id in MODEL_ORDER:
        display_name = get_display_name(model_id)
        if model_id in model_info:
            path, feats = model_info[model_id]
            cnt = len(feats)
            feat_str = ", ".join(feats)
            lines.append(f"{display_name:<{mw}}  {'找到':<{col_status}}  {path:<{pw}}  {cnt:>{cw}}  {feat_str}")
        else:
            lines.append(f"{display_name:<{mw}}  {'未找到':<{col_status}}  {'-':<{pw}}  {'-':>{cw}}  -")

    for path, cnt, feat_str in other_rows:
        lines.append(f"{'<其他>':<{mw}}  {'找到':<{col_status}}  {path:<{pw}}  {cnt:>{cw}}  {feat_str}")

    return "\n".join(lines)


def main() -> None:
    """主程式入口：執行模型掃描與特徵審核."""
    print("=" * 80)
    print("XGBoost 模型特徵審核（Truth Source）— Model A / B / C / D / E")
    print("=" * 80)
    print("[INFO] 掃描專案中的 XGBoost 模型檔案 (*.json / *.model / *.pkl)...\n")

    model_paths = sorted(set(iter_model_files(".")))
    # 僅篩選可能為 XGBoost 的檔，略過明顯的 config 等
    skip_substrings = ["config", "iching_book", "iching_complete", "devcontainer", "best_params"]
    def should_skip(p: str) -> bool:
        n = normalize_path_for_match(p)
        return any(s in n for s in skip_substrings)
    model_paths = [p for p in model_paths if not should_skip(p)]

    model_info: Dict[str, Tuple[str, List[str]]] = {}
    other_rows: List[Tuple[str, int, str]] = []
    any_model_d_checked = False

    for path in model_paths:
        rel_path = os.path.relpath(path, ".")
        print(f"[INFO] 解析模型: {rel_path}")

        booster = load_booster(path)
        if booster is None:
            continue

        features = extract_feature_names(booster)
        feat_count = len(features)
        feat_str = ", ".join(features)
        model_id = match_path_to_model(path)

        if model_id is not None:
            if model_id not in model_info:
                model_info[model_id] = (rel_path, features)
            else:
                other_rows.append((rel_path, feat_count, feat_str))
        else:
            other_rows.append((rel_path, feat_count, feat_str))

        if is_model_d(path):
            any_model_d_checked = True
            passed, messages = verify_model_d_features(features)
            if passed:
                print(f"{GREEN}[OK]{RESET} Model D 特徵驗證通過：未使用 Close/Volume/RVOL，且包含指定純量子特徵。")
            else:
                print(f"{RED}[WARNING]{RESET} Model D 特徵驗證未通過，請檢查特徵工程流程：")
                for msg in messages:
                    print(f"  - {RED}{msg}{RESET}")
        print("")

    print("=" * 80)
    print("Model A / B / C / D / E 特徵總覽")
    print("=" * 80)
    print(format_model_table(model_info, other_rows))
    print("=" * 80)

    print("\n" + "=" * 80)
    print("固定定義 (model_definitions.py) — Model A/B/C/D/E 特徵")
    print("=" * 80)
    for mid in MODEL_ORDER:
        name = get_display_name(mid)
        feats = get_features(mid)
        print(f"  {name}: {len(feats)} 個 — {', '.join(feats)}")
    print("=" * 80)

    missing = [mid for mid in MODEL_ORDER if mid not in model_info]
    if missing:
        names = [get_display_name(mid) for mid in missing]
        print(f"\n{YELLOW}[INFO]{RESET} 未找到對應模型檔: {', '.join(names)}")
        print("  - Model A/B: 由 experiment_xgboost.py、visualize_final_roc.py 訓練，目前腳本未儲存。")
        print("  - Model C: 執行 save_model_c.py 後會產生 data/volatility_model.json。")
        print("  - Model D: 執行 model_d_pure_quantum.py 後會產生 model_d_pure_quantum.json。")
        print("  - Model E: 由 quant_report_ablation_de.py 訓練，目前腳本未儲存。")
    if not any_model_d_checked and "D" not in model_info:
        print(f"\n{YELLOW}[INFO]{RESET} 未找到 Model D 模型檔，無法執行 Pure Quantum 特徵驗證。")


if __name__ == "__main__":
    main()

