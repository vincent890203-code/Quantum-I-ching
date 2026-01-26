"""將 Mermaid 圖表轉換為圖片檔案的工具腳本.

此腳本可以將 ARCHITECTURE_DIAGRAM.md 中的 Mermaid 程式碼轉換為 PNG/SVG 圖片。

使用方法:
    python convert_mermaid_to_image.py

需要安裝:
    pip install playwright mermaid
    playwright install chromium
"""

import re
import os
from pathlib import Path
from typing import List, Tuple


def extract_mermaid_blocks(content: str) -> List[Tuple[str, str]]:
    """從 Markdown 內容中提取所有 Mermaid 程式碼塊.
    
    Args:
        content: Markdown 檔案內容
        
    Returns:
        List of (title, mermaid_code) tuples
    """
    pattern = r'##\s+(.+?)\n\n```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def create_html_renderer(mermaid_code: str, title: str = "Diagram") -> str:
    """建立包含 Mermaid.js 的 HTML 檔案內容.
    
    Args:
        mermaid_code: Mermaid 程式碼
        title: 圖表標題
        
    Returns:
        HTML 字串
    """
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .mermaid {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
    return html_template


def main():
    """主函數：讀取 ARCHITECTURE_DIAGRAM.md 並生成 HTML 檔案."""
    # 讀取 Mermaid 檔案
    mermaid_file = Path("ARCHITECTURE_DIAGRAM.md")
    if not mermaid_file.exists():
        print(f"Error: File not found: {mermaid_file}")
        return
    
    print(f"Reading file: {mermaid_file}")
    content = mermaid_file.read_text(encoding="utf-8")
    
    # 提取所有 Mermaid 程式碼塊
    diagrams = extract_mermaid_blocks(content)
    
    if not diagrams:
        print("No Mermaid code blocks found")
        return
    
    print(f"Found {len(diagrams)} Mermaid diagrams")
    
    # 建立輸出目錄
    output_dir = Path("docs/architecture_diagrams")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 為每個圖表生成 HTML 檔案
    html_files = []
    for i, (title, mermaid_code) in enumerate(diagrams, 1):
        # 清理標題作為檔案名
        safe_title = re.sub(r'[^\w\s-]', '', title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        filename = f"diagram_{i:02d}_{safe_title}.html"
        filepath = output_dir / filename
        
        # 生成 HTML
        html_content = create_html_renderer(mermaid_code, title)
        filepath.write_text(html_content, encoding="utf-8")
        html_files.append((title, filepath))
        
        print(f"  [OK] Generated: {filename} - {title}")
    
    # 生成索引頁面
    index_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Quantum I-Ching 架構圖索引</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        ul {
            list-style: none;
            padding: 0;
        }
        li {
            margin: 15px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
            border-left: 4px solid #0277bd;
        }
        a {
            color: #0277bd;
            text-decoration: none;
            font-weight: 500;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Quantum I-Ching 系統架構圖</h1>
        <ul>
"""
    
    for title, filepath in html_files:
        relative_path = filepath.relative_to(output_dir.parent)
        index_html += f'            <li><a href="{relative_path}">{title}</a></li>\n'
    
    index_html += """        </ul>
        <hr style="margin-top: 30px; border: none; border-top: 1px solid #ddd;">
        <p style="color: #666; font-size: 0.9em; margin-top: 20px;">
            <strong>說明:</strong> 這些 HTML 檔案包含 Mermaid 圖表，可以在瀏覽器中開啟查看。
            <br>要將圖表轉換為 PNG/SVG 圖片，可以使用瀏覽器的截圖功能或使用 Mermaid CLI 工具。
        </p>
    </div>
</body>
</html>"""
    
    index_path = output_dir.parent / "architecture_index.html"
    index_path.write_text(index_html, encoding="utf-8")
    
    print(f"\n[OK] All diagrams generated to: {output_dir}")
    print(f"[OK] Index page: {index_path}")
    print("\nUsage instructions:")
    print("1. Open HTML files in browser to view diagrams")
    print("2. Use browser screenshot or developer tools to export as images")
    print("3. Or use Mermaid CLI to convert to PNG/SVG:")
    print("   npm install -g @mermaid-js/mermaid-cli")
    print("   mmdc -i diagram.mmd -o diagram.png")


if __name__ == "__main__":
    main()
