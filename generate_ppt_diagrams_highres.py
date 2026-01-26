"""生成高解析度 PowerPoint 架構圖 (1920x1080).

使用 Playwright 渲染 Mermaid 圖表，確保字體清晰可讀（最小 28 號字體）。

安裝依賴:
    pip install playwright
    playwright install chromium

使用方法:
    python generate_ppt_diagrams_highres.py
"""

import re
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("警告: Playwright 未安裝，將使用 API 方法（品質較低）")
    print("建議安裝: pip install playwright && playwright install chromium")


def extract_mermaid_blocks(content: str) -> list[tuple[str, str]]:
    """提取 Mermaid 程式碼塊."""
    pattern = r'##\s+(.+?)\n\n```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def create_ppt_html(mermaid_code: str, title: str) -> str:
    """建立適合 PowerPoint 的 HTML（優化字體大小和佈局）."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            width: 1920px;
            height: 1080px;
            background: #ffffff;
            font-family: 'Microsoft JhengHei', 'Arial', sans-serif;
            overflow: hidden;
            padding: 30px;
        }}
        .mermaid {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .mermaid svg {{
            max-width: 100%;
            max-height: 100%;
        }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                padding: 30,
                nodeSpacing: 80,
                rankSpacing: 100
            }},
            themeVariables: {{
                fontSize: '32px',
                fontFamily: 'Microsoft JhengHei, Arial, sans-serif',
                primaryColor: '#ffffff',
                primaryTextColor: '#000000',
                primaryBorderColor: '#000000',
                lineColor: '#000000',
                secondaryColor: '#f0f0f0',
                tertiaryColor: '#e0e0e0',
                clusterBkg: '#ffffff',
                clusterBorder: '#000000',
                defaultLinkColor: '#000000',
                titleColor: '#000000',
                edgeLabelBackground: '#ffffff',
                mainBkg: '#ffffff',
                secondBkg: '#f0f0f0',
                tertiaryBkg: '#e0e0e0',
                clusterTextColor: '#000000',
                clusterBorder: '#333333',
                defaultLinkColor: '#333333',
                titleColor: '#000000'
            }}
        }});
    </script>
</body>
</html>"""


def export_with_playwright_highres(
    mermaid_code: str,
    output_path: Path,
    title: str
) -> bool:
    """使用 Playwright 生成高解析度圖片."""
    if not PLAYWRIGHT_AVAILABLE:
        return False
    
    try:
        temp_html = Path("temp_ppt_highres.html")
        html_content = create_ppt_html(mermaid_code, title)
        temp_html.write_text(html_content, encoding="utf-8")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            
            page.goto(f"file://{temp_html.absolute()}")
            page.wait_for_timeout(6000)  # 等待更長時間確保完整渲染
            
            try:
                page.wait_for_selector("svg", timeout=20000)
            except Exception:
                print(f"  Warning: SVG not found")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 截圖整個頁面
            page.screenshot(
                path=str(output_path),
                type="png",
                full_page=False,
                clip={"x": 0, "y": 0, "width": 1920, "height": 1080}
            )
            
            browser.close()
        
        temp_html.unlink()
        return True
    except Exception as e:
        print(f"  Error: {e}")
        if temp_html.exists():
            temp_html.unlink()
        return False


def main():
    """主函數."""
    print("=" * 60)
    print("生成高解析度 PowerPoint 架構圖 (1920x1080)")
    print("=" * 60)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("\n警告: Playwright 未安裝")
        print("請執行: pip install playwright && playwright install chromium")
        print("將嘗試使用 API 方法（品質較低）\n")
    
    ppt_file = Path("ARCHITECTURE_DIAGRAM_PPT.md")
    if not ppt_file.exists():
        print(f"錯誤: 找不到 {ppt_file}")
        return
    
    print(f"讀取: {ppt_file}")
    content = ppt_file.read_text(encoding="utf-8")
    
    diagrams = extract_mermaid_blocks(content)
    if not diagrams:
        print("未找到 Mermaid 圖表")
        return
    
    print(f"找到 {len(diagrams)} 個圖表\n")
    
    output_dir = Path("docs/ppt_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, (title, mermaid_code) in enumerate(diagrams, 1):
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        output_name = f"ppt_highres_{i:02d}_{safe_title}.png"
        output_path = output_dir / output_name
        
        print(f"[{i}/{len(diagrams)}] {title}")
        print(f"  輸出: {output_path.name}")
        
        if PLAYWRIGHT_AVAILABLE:
            print(f"  使用 Playwright 高解析度渲染...")
            if export_with_playwright_highres(mermaid_code, output_path, title):
                print(f"  [OK] 成功生成 1920x1080 圖片")
                success_count += 1
            else:
                print(f"  [失敗]")
        else:
            print(f"  [跳過] 需要 Playwright")
    
    print("\n" + "=" * 60)
    print(f"完成: {success_count}/{len(diagrams)} 成功")
    print(f"輸出目錄: {output_dir}")
    if success_count > 0:
        print("\n生成的圖片已優化為:")
        print("  - 尺寸: 1920x1080 (16:9)")
        print("  - 字體: 最小 28-32 號")
        print("  - 佈局: 充分利用寬螢幕空間")
        print("  - 內容: 簡化，只保留核心資訊")
    print("=" * 60)


if __name__ == "__main__":
    main()
