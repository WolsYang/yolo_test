import cv2
import os
import random
import glob
from pathlib import Path

# 設定
SELECTE_IMG_NUM = 40
DATASET_DIR = Path("/home/wols/yolo_test/dataset_cut")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("/home/wols/yolo_test/verification_output")

# 類別名稱 (來自 pcb_data.yaml)
CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (0, 255, 255), (255, 0, 255)
]

def draw_yolo_box(img, class_id, x_center, y_center, w, h):
    height, width, _ = img.shape
    
    # 轉換 YOLO 格式到像素座標
    x_center *= width
    y_center *= height
    w *= width
    h *= height
    
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    
    color = COLORS[class_id % len(COLORS)]
    label = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def verify_single_image(img_path):
    img_path = Path(img_path)
    
    # 在 dataset_cut 中，images 和 labels 是平行的資料夾
    # 結構可能是 images/train/xxx.jpg -> labels/train/xxx.txt
    
    # 取得相對於 images 的路徑 (例如 train/xxx.jpg)
    try:
        rel_path = img_path.relative_to(IMAGES_DIR)
        label_path = LABELS_DIR / rel_path.with_suffix(".txt")
    except ValueError:
        print(f"路徑錯誤: {img_path} 不在 {IMAGES_DIR} 內")
        return
    
    if not label_path.exists():
        # 有些圖片可能沒有標註 (背景圖)，這是正常的
        pass
        
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image {img_path}")
        return
        
    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            
            draw_yolo_box(
                img, class_id, 
                x_center, y_center, 
                w, h)
        
    output_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(output_path), img)

def main():
    # 清空並建立輸出目錄
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 收集所有圖片 (遞迴搜尋)
    all_images = list(IMAGES_DIR.glob("**/*.jpg"))
    
    if not all_images:
        print(f"No images found in {IMAGES_DIR}!")
        return

    # 隨機選取
    selected_images = random.sample(
        all_images, 
        min(SELECTE_IMG_NUM, len(all_images)))
    
    print(f"正在驗證 {len(selected_images)} 張圖片，輸出至 {OUTPUT_DIR}...")
    
    for img_path in selected_images:
        verify_single_image(img_path)
        
    print("驗證完成。")

if __name__ == "__main__":
    main()
