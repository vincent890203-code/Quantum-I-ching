"""將 Mermaid 圖表匯出為 PNG/JPG 圖片檔案的腳本.

支援多種方法：
1. 使用 Playwright (推薦，需要安裝)
2. 使用 Mermaid CLI (需要 Node.js)
3. 使用線上 API (無需安裝)

安裝 Playwright:
    pip install playwright
    playwright install chromium

或安裝 Mermaid CLI:
    npm install -g @mermaid-js/mermaid-cli

使用方法:
    python export_mermaid_to_image.py
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import json
import base64

# 嘗試匯入 playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# 檢查 mermaid-cli 是否可用
def check_mermaid_cli() -> bool:
    """檢查 mermaid-cli 是否已安裝."""
    try:
        result = subprocess.run(
            ["mmdc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


MERMAID_CLI_AVAILABLE = check_mermaid_cli()


def extract_mermaid_blocks(content: str) -> List[Tuple[str, str]]:
    """從 Markdown 內容中提取所有 Mermaid 程式碼塊."""
    pattern = r'##\s+(.+?)\n\n```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def export_with_playwright(
    mermaid_code: str,
    output_path: Path,
    title: str = "Diagram"
) -> bool:
    """使用 Playwright 匯出 Mermaid 圖表."""
    if not PLAYWRIGHT_AVAILABLE:
        return False
    
    try:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: white; }}
        .mermaid {{ background: white; }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
        
        temp_html = Path("temp_mermaid.html")
        temp_html.write_text(html_content, encoding="utf-8")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(f"file://{temp_html.absolute()}")
            page.wait_for_timeout(3000)
            page.wait_for_selector("svg", timeout=10000)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(output_path), type="png", full_page=True)
            browser.close()
        
        temp_html.unlink()
        return True
    except Exception as e:
        print(f"  Playwright error: {e}")
        return False


def export_with_mermaid_cli(
    mermaid_code: str,
    output_path: Path,
    title: str = "Diagram"
) -> bool:
    """使用 Mermaid CLI 匯出圖表."""
    if not MERMAID_CLI_AVAILABLE:
        return False
    
    try:
        # 建立臨時 .mmd 檔案
        temp_mmd = Path("temp_diagram.mmd")
        temp_mmd.write_text(mermaid_code, encoding="utf-8")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 執行 mmdc
        result = subprocess.run(
            [
                "mmdc",
                "-i", str(temp_mmd),
                "-o", str(output_path),
                "-w", "1920",
                "-H", "1080",
                "-b", "white"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        temp_mmd.unlink()
        
        if result.returncode == 0 and output_path.exists():
            return True
        else:
            if result.stderr:
                print(f"  Mermaid CLI error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Mermaid CLI error: {e}")
        return False


def export_with_api(
    mermaid_code: str,
    output_path: Path,
    title: str = "Diagram"
) -> bool:
    """使用 Mermaid.ink API 匯出圖表（需要網路連線）."""
    try:
        import urllib.request
        import urllib.parse
        
        # 將 Mermaid 程式碼編碼為 base64
        encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        
        # Mermaid.ink API
        url = f"https://mermaid.ink/img/{encoded}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下載圖片
        urllib.request.urlretrieve(url, output_path)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"  API error: {e}")
        return False


def export_mermaid_to_image(
    mermaid_code: str,
    output_path: Path,
    title: str = "Diagram",
    format: str = "png"
) -> bool:
    """將 Mermaid 程式碼匯出為圖片（嘗試多種方法）."""
    # 嘗試方法 1: Playwright
    if PLAYWRIGHT_AVAILABLE:
        print(f"  Trying Playwright...")
        if export_with_playwright(mermaid_code, output_path, title):
            return True
    
    # 嘗試方法 2: Mermaid CLI
    if MERMAID_CLI_AVAILABLE:
        print(f"  Trying Mermaid CLI...")
        if export_with_mermaid_cli(mermaid_code, output_path, title):
            return True
    
    # 嘗試方法 3: API (需要網路)
    print(f"  Trying Mermaid.ink API...")
    if export_with_api(mermaid_code, output_path, title):
        return True
    
    return False


def main():
    """主函數."""
    print("=" * 60)
    print("Mermaid to Image Exporter")
    print("=" * 60)
    
    # 檢查可用方法
    methods = []
    if PLAYWRIGHT_AVAILABLE:
        methods.append("Playwright")
    if MERMAID_CLI_AVAILABLE:
        methods.append("Mermaid CLI")
    methods.append("Mermaid.ink API")
    
    print(f"Available methods: {', '.join(methods)}")
    print()
    
    # 讀取檔案
    mermaid_file = Path("ARCHITECTURE_DIAGRAM.md")
    if not mermaid_file.exists():
        print(f"Error: File not found: {mermaid_file}")
        return
    
    print(f"Reading: {mermaid_file}")
    content = mermaid_file.read_text(encoding="utf-8")
    
    diagrams = extract_mermaid_blocks(content)
    if not diagrams:
        print("No Mermaid diagrams found")
        return
    
    print(f"Found {len(diagrams)} diagrams")
    print("=" * 60)
    
    # 建立輸出目錄
    output_dir = Path("docs/architecture_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 匯出每個圖表
    success_count = 0
    for i, (title, mermaid_code) in enumerate(diagrams, 1):
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        png_path = output_dir / f"diagram_{i:02d}_{safe_title}.png"
        
        print(f"\n[{i}/{len(diagrams)}] {title}")
        print(f"  Output: {png_path.name}")
        
        if export_mermaid_to_image(mermaid_code, png_path, title):
            print(f"  [OK] Exported successfully")
            success_count += 1
        else:
            print(f"  [FAILED] All methods failed")
            print(f"  Please install one of:")
            print(f"    - pip install playwright && playwright install chromium")
            print(f"    - npm install -g @mermaid-js/mermaid-cli")
    
    print("\n" + "=" * 60)
    print(f"Export complete: {success_count}/{len(diagrams)} successful")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
