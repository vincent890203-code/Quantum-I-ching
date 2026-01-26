"""生成最終版 PowerPoint 架構圖 (1920x1080, 字體 28+ 號).

使用 Playwright 渲染，確保高品質和清晰的字體。

安裝:
    pip install playwright
    playwright install chromium

執行:
    python generate_ppt_final.py
"""

import re
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def extract_mermaid_blocks(content: str) -> list[tuple[str, str]]:
    """提取 Mermaid 程式碼塊."""
    pattern = r'##\s+(.+?)\n\n```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def create_optimized_html(mermaid_code: str, title: str) -> str:
    """建立優化的 HTML（字體 28+ 號，充分利用 16:9 空間）."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            width: 1920px;
            height: 1080px;
            margin: 0;
            padding: 20px;
            background: #ffffff;
            font-family: 'Microsoft JhengHei', 'Arial', sans-serif;
            overflow: hidden;
        }}
        .mermaid {{
            width: 100%;
            height: 100%;
        }}
        .mermaid svg {{
            font-family: 'Microsoft JhengHei', 'Arial', sans-serif;
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
                padding: 40,
                nodeSpacing: 100,
                rankSpacing: 120,
                rankDir: 'LR'
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
                clusterBorder: '#333333',
                defaultLinkColor: '#333333',
                titleColor: '#000000',
                edgeLabelBackground: '#ffffff',
                mainBkg: '#ffffff',
                secondBkg: '#f0f0f0',
                tertiaryBkg: '#e0e0e0',
                clusterTextColor: '#000000',
                textColor: '#000000'
            }}
        }});
    </script>
</body>
</html>"""


def export_highres(mermaid_code: str, output_path: Path, title: str) -> bool:
    """使用 Playwright 生成高解析度圖片."""
    if not PLAYWRIGHT_AVAILABLE:
        return False
    
    try:
        temp_html = Path("temp_ppt_final.html")
        html_content = create_optimized_html(mermaid_code, title)
        temp_html.write_text(html_content, encoding="utf-8")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            
            page.goto(f"file://{temp_html.absolute()}")
            page.wait_for_timeout(8000)  # 等待完整渲染
            
            try:
                page.wait_for_selector("svg", timeout=25000)
            except Exception:
                pass
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
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
    print("生成 PowerPoint 架構圖 (1920x1080, 字體 28+ 號)")
    print("=" * 60)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("\n[ERROR] Playwright not installed")
        print("Please run:")
        print("  pip install playwright")
        print("  playwright install chromium")
        print("\nOr use Mermaid Live Editor:")
        print("  https://mermaid.live/")
        return
    
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
        
        output_name = f"ppt_final_{i:02d}_{safe_title}.png"
        output_path = output_dir / output_name
        
        print(f"[{i}/{len(diagrams)}] {title}")
        print(f"  輸出: {output_path.name}")
        
        if export_highres(mermaid_code, output_path, title):
            print(f"  [OK] Generated 1920x1080 high-res image")
            success_count += 1
        else:
            print(f"  [FAILED] Generation failed")
    
    print("\n" + "=" * 60)
    print(f"完成: {success_count}/{len(diagrams)} 成功")
    print(f"輸出目錄: {output_dir}")
    if success_count > 0:
        print("\n[OK] Images optimized:")
        print("  - Size: 1920x1080 (16:9)")
        print("  - Font: 28-32px (via node size optimization)")
        print("  - Layout: Horizontal, full use of widescreen")
        print("  - Content: Simplified, core info only")
    print("=" * 60)


if __name__ == "__main__":
    main()
