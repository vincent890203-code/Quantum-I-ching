"""為 PowerPoint 生成優化的架構圖.

此腳本會：
1. 讀取 ARCHITECTURE_DIAGRAM_PPT.md
2. 使用更大的渲染尺寸（適合 16:9）
3. 生成高解析度圖片 (1920x1080)
4. 確保字體清晰可讀

使用方法:
    python export_ppt_diagrams.py
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def extract_mermaid_blocks(content: str) -> List[Tuple[str, str]]:
    """從 Markdown 內容中提取所有 Mermaid 程式碼塊."""
    pattern = r'##\s+(.+?)\n\n```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def create_html_for_ppt(mermaid_code: str, title: str = "Diagram") -> str:
    """建立適合 PowerPoint 的 HTML（更大的渲染尺寸，優化字體大小）."""
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background-color: #ffffff;
            font-family: 'Microsoft JhengHei', 'Arial', sans-serif;
            width: 1920px;
            height: 1080px;
            overflow: hidden;
        }}
        .mermaid {{
            background-color: white;
            padding: 20px;
            width: 100%;
            height: 100%;
        }}
        .mermaid svg {{
            max-width: 100%;
            height: auto;
        }}
        .mermaid .nodeLabel {{
            font-size: 28px !important;
            font-weight: 500;
        }}
        .mermaid .cluster-label {{
            font-size: 32px !important;
            font-weight: 600;
        }}
        .mermaid .edgeLabel {{
            font-size: 24px !important;
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
                padding: 20
            }},
            themeVariables: {{
                fontSize: '28px',
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
                tertiaryBkg: '#e0e0e0'
            }}
        }});
    </script>
</body>
</html>"""
    return html_template


def export_with_playwright_ppt(
    mermaid_code: str,
    output_path: Path,
    title: str = "Diagram"
) -> bool:
    """使用 Playwright 匯出為 PowerPoint 格式（1920x1080）."""
    if not PLAYWRIGHT_AVAILABLE:
        return False
    
    try:
        temp_html = Path("temp_ppt_mermaid.html")
        html_content = create_html_for_ppt(mermaid_code, title)
        temp_html.write_text(html_content, encoding="utf-8")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # 使用更大的視窗尺寸以確保高品質
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            
            page.goto(f"file://{temp_html.absolute()}")
            page.wait_for_timeout(5000)  # 等待更長時間確保渲染完成
            
            try:
                page.wait_for_selector("svg", timeout=15000)
            except Exception:
                print(f"  Warning: SVG not found for {title}")
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(output_path),
                type="png",
                full_page=True,
                clip={"x": 0, "y": 0, "width": 1920, "height": 1080}
            )
            
            browser.close()
        
        temp_html.unlink()
        return True
    except Exception as e:
        print(f"  Playwright error: {e}")
        if temp_html.exists():
            temp_html.unlink()
        return False


def export_with_api_ppt(
    mermaid_code: str,
    output_path: Path,
    title: str = "Diagram"
) -> bool:
    """使用 Mermaid.ink API 匯出（需要網路）."""
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


def main():
    """主函數."""
    print("=" * 60)
    print("PowerPoint 架構圖生成器 (16:9 優化)")
    print("=" * 60)
    
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
        
        output_name = f"ppt_{i:02d}_{safe_title}.png"
        output_path = output_dir / output_name
        
        print(f"[{i}/{len(diagrams)}] {title}")
        print(f"  輸出: {output_path.name}")
        
        if PLAYWRIGHT_AVAILABLE:
            print(f"  使用 Playwright 渲染...")
            if export_with_playwright_ppt(mermaid_code, output_path, title):
                print(f"  [OK] 成功生成")
                success_count += 1
                continue
        
        print(f"  使用 API 渲染...")
        if export_with_api_ppt(mermaid_code, output_path, title):
            print(f"  [OK] 成功生成")
            success_count += 1
        else:
            print(f"  [失敗] 請安裝 Playwright 以獲得最佳效果")
    
    print("\n" + "=" * 60)
    print(f"完成: {success_count}/{len(diagrams)} 成功")
    print(f"輸出目錄: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
