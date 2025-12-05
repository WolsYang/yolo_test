import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
# 把data文件夹下的Annotations和images文件夹下的文件
# 移动到datasets/pcb_dataset文件夹下
# 並且把格式從Pascal VOC 格式 (XML)轉換成yolo格式 (txt)
# 然后按照80%训练集，20%验证集的比例
# 将数据集分成训练集和验证集

# Define paths
DATA_ROOT = Path("data")
ANNOTATIONS_DIR = DATA_ROOT / "Annotations"
IMAGES_DIR = DATA_ROOT / "images"
OUTPUT_DIR = Path("datasets/pcb_dataset")

# Define classes
CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

CLASS_MAP = {cls: i for i, cls in enumerate(CLASSES)}

def convert_bbox(size, box):
    # 正規劃係數
    # YOLO 模型要求的座標必須是 0 到 1 之間的小數，而不是原本的像素值
    dw = 1. / size[0]
    dh = 1. / size[1]
    # 計算中心點
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    # 計算長寬
    w = box[1] - box[0]
    h = box[3] - box[2]
    # 將中心點和長寬轉換為 YOLO 格式
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_file(
    xml_file, image_file, 
    output_images_dir, output_labels_dir, stats):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    # '/'是 pathlib 模組專門設計的路徑連接運算子，
    label_file = output_labels_dir / xml_file.with_suffix('.txt').name
    
    with open(label_file, 'w') as f:
        # 遍歷xml文件中的所有object
        for obj in root.iter('object'):
            # 取出object的class
            cls = obj.find('name').text
            cls_key = cls
            if cls not in CLASS_MAP:
                # 嘗試忽略大小寫比對
                for k in CLASS_MAP:
                    if k.lower() == cls.lower():
                        cls_key = k
                        break
            
            if cls_key not in CLASS_MAP:
                print(f"Warning: Class '{cls}' not found in CLASS_MAP. Skipping.")
                stats['unknown_classes'] += 1
                continue
                
            cls_id = CLASS_MAP[cls_key]
            # 從 XML 讀取矩形的四個頂點。
            xmlbox = obj.find('bndbox')
            # 把四個角的值讀出來，轉成浮點數 (float)，存成一個 Tuple
            voc_coord = (
                float(xmlbox.find('xmin').text), 
                float(xmlbox.find('xmax').text), 
                float(xmlbox.find('ymin').text), 
                float(xmlbox.find('ymax').text))
            # Pascal VOC 格式 (xmin, xmax, ymin, ymax) 
            # 轉換成 YOLO 格式 (x_center, y_center, width, height)
            yolo_coord = convert_bbox((w, h), voc_coord)
            f.write(
                f"{cls_id} {yolo_coord[0]:.6f} {yolo_coord[1]:.6f} {yolo_coord[2]:.6f} {yolo_coord[3]:.6f}\n")
            
    # Copy image
    shutil.copy(image_file, output_images_dir / image_file.name)

def main():
    # 統計數據
    stats = {
        'missing_images': 0,
        'missing_dirs': 0,
        'unknown_classes': 0,
        'processed_files': 0
    }

    # 建立資料夾
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    # 收集所有配對
    pairs = []
    # 遍歷 Annotations 目錄下的所有類別目錄
    for cls_dir in ANNOTATIONS_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
            
        # 對應的圖片目錄
        img_cls_dir = IMAGES_DIR / cls_dir.name
        if not img_cls_dir.exists():
            print(f"Warning: Image directory {img_cls_dir} does not exist. Skipping.")
            stats['missing_dirs'] += 1
            continue
            
        for xml_file in cls_dir.glob("*.xml"):
            # 找到對應的圖片 (jpg)
            img_file = img_cls_dir / xml_file.with_suffix('.jpg').name
            if img_file.exists():
                pairs.append((xml_file, img_file))
            else:
                print(f"Warning: Image for {xml_file} not found.")
                stats['missing_images'] += 1

    print(f"Found {len(pairs)} image-annotation pairs.")
    
    # 隨機打亂並分割
    random.seed(42)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Processing {len(train_pairs)} training pairs...")
    for xml, img in train_pairs:
        process_file(xml, img, OUTPUT_DIR / 'images' / 'train', OUTPUT_DIR / 'labels' / 'train', stats)
        stats['processed_files'] += 1
        
    print(f"Processing {len(val_pairs)} validation pairs...")
    for xml, img in val_pairs:
        process_file(xml, img, OUTPUT_DIR / 'images' / 'val', OUTPUT_DIR / 'labels' / 'val', stats)
        stats['processed_files'] += 1
        
    print("\n" + "="*30)
    print("Data Preparation Statistics")
    print("="*30)
    print(f"Total processed pairs(images and labels): {stats['processed_files']}")
    print(f"Missing images: {stats['missing_images']}")
    print(f"Missing directories: {stats['missing_dirs']}")
    print(f"Unknown classes objects: {stats['unknown_classes']}")
    print("="*30)
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
