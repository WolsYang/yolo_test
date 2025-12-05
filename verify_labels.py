import cv2
import os
import random
import glob
from pathlib import Path
# 隨機抽取20張圖片
# 依照標注資料畫到圖片上，並保存到verification_output文件夾下
# 驗證資料是否正確

# Paths
SELECTE_IMG_NUM = 20
DATASET_DIR = Path("datasets/pcb_dataset")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = Path("verification_output")

# Classes (from pcb_data.yaml)
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
    
    # Convert YOLO format to pixel coordinates
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
    try:
        relative_path = img_path.relative_to(IMAGES_DIR)
    except ValueError:
        print(f"Skipping {img_path}: Not in {IMAGES_DIR}")
        return

    label_path = LABELS_DIR / relative_path.with_suffix(".txt")
    
    if not label_path.exists():
        print(f"Label not found for {img_path}")
        return
        
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image {img_path}")
        return
        
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
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
    print(f"Saved {output_path}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Gather all images
    all_images = list(
        IMAGES_DIR.glob("**/*.jpg"))
    
    if not all_images:
        print("No images found!")
        return

    # Select 20 random images
    selected_images = random.sample(
        all_images, 
        min(SELECTE_IMG_NUM, len(all_images)))
    
    print(f"Verifying {len(selected_images)} images...")
    
    for img_path in selected_images:
        verify_single_image(img_path)

if __name__ == "__main__":
    main()
