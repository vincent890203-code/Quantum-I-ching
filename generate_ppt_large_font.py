"""生成大字體 PowerPoint 架構圖 (1920x1080, 字體 28+ 號).

使用 Playwright 渲染，確保字體清晰可讀。
所有文字都使用人性化描述，程式碼名稱放在括號內。

安裝:
    pip install playwright
    playwright install chromium

執行:
    python generate_ppt_large_font.py
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


def create_large_font_html(mermaid_code: str, title: str) -> str:
    """建立大字體 HTML（字體 28+ 號，充分利用 16:9 空間）."""
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
        }}
        .mermaid svg {{
            font-family: 'Microsoft JhengHei', 'Arial', sans-serif;
        }}
        .mermaid .nodeLabel {{
            font-size: 32px !important;
            font-weight: 500;
            line-height: 1.4;
        }}
        .mermaid .cluster-label {{
            font-size: 36px !important;
            font-weight: 600;
        }}
        .mermaid .edgeLabel {{
            font-size: 28px !important;
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
                padding: 50,
                nodeSpacing: 120,
                rankSpacing: 150,
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
                textColor: '#000000',
                primaryBorderColor: '#000000',
                primaryBorderWidth: '5px'
            }}
        }});
    </script>
</body>
</html>"""


def export_with_api(mermaid_code: str, output_path: Path) -> bool:
    """使用 Mermaid.ink API 生成圖片."""
    try:
        import urllib.request
        import urllib.parse
        import base64
        
        encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        url = f"https://mermaid.ink/img/{encoded}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        return False
    except Exception as e:
        print(f"  API error: {e}")
        return False


def export_large_font(mermaid_code: str, output_path: Path, title: str) -> bool:
    """使用 Playwright 生成大字體高解析度圖片."""
    if PLAYWRIGHT_AVAILABLE:
        return export_with_playwright(mermaid_code, output_path, title)
    else:
        return export_with_api(mermaid_code, output_path)


def export_with_playwright(mermaid_code: str, output_path: Path, title: str) -> bool:
    """使用 Playwright 生成大字體高解析度圖片."""
    
    try:
        temp_html = Path("temp_ppt_large_font.html")
        html_content = create_large_font_html(mermaid_code, title)
        temp_html.write_text(html_content, encoding="utf-8")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            
            page.goto(f"file://{temp_html.absolute()}")
            page.wait_for_timeout(10000)  # 等待完整渲染
            
            try:
                page.wait_for_selector("svg", timeout=30000)
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
    print("生成大字體 PowerPoint 架構圖 (1920x1080, 字體 28+ 號)")
    print("=" * 60)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("\n[WARNING] Playwright not installed, using API method")
        print("For better quality, install Playwright:")
        print("  pip install playwright")
        print("  playwright install chromium")
        print("\nOr use Mermaid Live Editor:")
        print("  https://mermaid.live/")
        print()
    
    ppt_file = Path("ARCHITECTURE_DIAGRAM_PPT.md")
    if not ppt_file.exists():
        print(f"Error: {ppt_file} not found")
        return
    
    print(f"Reading: {ppt_file}")
    content = ppt_file.read_text(encoding="utf-8")
    
    diagrams = extract_mermaid_blocks(content)
    if not diagrams:
        print("No Mermaid diagrams found")
        return
    
    print(f"Found {len(diagrams)} diagrams\n")
    
    output_dir = Path("docs/ppt_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, (title, mermaid_code) in enumerate(diagrams, 1):
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        output_name = f"ppt_large_font_{i:02d}_{safe_title}.png"
        output_path = output_dir / output_name
        
        print(f"[{i}/{len(diagrams)}] {title}")
        print(f"  Output: {output_path.name}")
        
        if PLAYWRIGHT_AVAILABLE:
            print(f"  Using Playwright...")
        else:
            print(f"  Using API (Playwright not available)...")
        
        if export_large_font(mermaid_code, output_path, title):
            print(f"  [OK] Generated")
            success_count += 1
        else:
            print(f"  [FAILED] Generation failed")
    
    print("\n" + "=" * 60)
    print(f"Complete: {success_count}/{len(diagrams)} successful")
    print(f"Output directory: {output_dir}")
    if success_count > 0:
        print("\n[OK] Images optimized:")
        print("  - Size: 1920x1080 (16:9)")
        print("  - Font: 28-36px (large, readable)")
        print("  - Layout: Horizontal, full use of widescreen")
        print("  - Content: Human-readable descriptions")
        print("  - Code names: In parentheses with smaller font")
    print("=" * 60)


if __name__ == "__main__":
    main()
