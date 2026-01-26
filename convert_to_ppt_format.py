"""將架構圖轉換為 PowerPoint 16:9 格式 (1920x1080).

此腳本會：
1. 讀取現有的架構圖 PNG
2. 調整為 16:9 比例 (1920x1080)
3. 保持內容不變，添加適當的背景
4. 輸出為適合 PowerPoint 的圖片

使用方法:
    python convert_to_ppt_format.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

# PowerPoint 16:9 標準尺寸
PPT_WIDTH = 1920
PPT_HEIGHT = 1080
PPT_RATIO = 16 / 9


def resize_for_ppt(input_path: Path, output_path: Path, background_color: str = "#FFFFFF") -> None:
    """將圖片調整為 PowerPoint 16:9 格式.
    
    Args:
        input_path: 輸入圖片路徑
        output_path: 輸出圖片路徑
        background_color: 背景顏色（十六進制）
    """
    # 讀取原始圖片
    img = Image.open(input_path)
    original_width, original_height = img.size
    
    # 計算縮放比例（保持寬高比）
    # 目標是填滿 1920x1080，但保持原始比例
    scale_w = PPT_WIDTH / original_width
    scale_h = PPT_HEIGHT / original_height
    scale = min(scale_w, scale_h)  # 使用較小的比例，確保圖片完全顯示
    
    # 計算新尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 調整圖片大小（使用高品質重採樣）
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 建立 1920x1080 的背景
    ppt_img = Image.new('RGB', (PPT_WIDTH, PPT_HEIGHT), background_color)
    
    # 計算居中位置
    x_offset = (PPT_WIDTH - new_width) // 2
    y_offset = (PPT_HEIGHT - new_height) // 2
    
    # 將調整後的圖片貼到背景上（居中）
    ppt_img.paste(resized_img, (x_offset, y_offset), resized_img if resized_img.mode == 'RGBA' else None)
    
    # 儲存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ppt_img.save(output_path, 'PNG', quality=95, optimize=True)
    
    print(f"  [OK] {input_path.name} -> {output_path.name}")
    print(f"       原始: {original_width}x{original_height}, 調整後: {new_width}x{new_height}, 居中於 {PPT_WIDTH}x{PPT_HEIGHT}")


def main():
    """主函數：處理所有架構圖."""
    print("=" * 60)
    print("將架構圖轉換為 PowerPoint 16:9 格式")
    print("=" * 60)
    
    # 輸入和輸出目錄
    input_dir = Path("docs/architecture_images")
    output_dir = Path("docs/ppt_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 要處理的圖片列表
    images_to_process = [
        "diagram_01_完整系統架構類似_RAG_架構圖風格.png",
        "diagram_02_詳細資料流程圖_主要流程使用者查詢_Oracle_解讀.png",
        "diagram_04_大衍之數映射流程.png"
    ]
    
    print(f"\n處理圖片...")
    print(f"輸入目錄: {input_dir}")
    print(f"輸出目錄: {output_dir}")
    print(f"目標尺寸: {PPT_WIDTH}x{PPT_HEIGHT} (16:9)\n")
    
    success_count = 0
    for img_name in images_to_process:
        input_path = input_dir / img_name
        if not input_path.exists():
            print(f"  [SKIP] {img_name} 不存在")
            continue
        
        # 生成輸出檔名（添加 _ppt 後綴）
        base_name = img_name.replace('.png', '')
        output_name = f"{base_name}_ppt.png"
        output_path = output_dir / output_name
        
        try:
            resize_for_ppt(input_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"  [FAILED] {img_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"轉換完成: {success_count}/{len(images_to_process)} 成功")
    print(f"輸出位置: {output_dir}")
    print("=" * 60)
    print("\n使用說明:")
    print("1. 這些圖片已調整為 16:9 比例 (1920x1080)")
    print("2. 可以直接插入 PowerPoint 簡報")
    print("3. 圖片內容保持不變，已居中顯示在白色背景上")


if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        print("錯誤: 需要安裝 Pillow 套件")
        print("請執行: pip install Pillow")
        exit(1)
    
    main()
