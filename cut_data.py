import os
import cv2
import shutil
import numpy as np
import random
from pathlib import Path

class ImageSlicer:
    def __init__(self, 
        source_dir, output_dir, 
        slice_size=640, stride=600,
        split_ratio=0.9):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.slice_size = slice_size
        self.stride = stride
        self.split_ratio = split_ratio
        
        # 設定目錄結構
        self.images_train = self.output_dir / 'images' / 'train'
        self.images_val = self.output_dir / 'images' / 'val'
        self.labels_train = self.output_dir / 'labels' / 'train'
        self.labels_val = self.output_dir / 'labels' / 'val'
        
        # 清除並建立輸出目錄
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            
        for d in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            d.mkdir(parents=True, exist_ok=True)

    def process(self):
        # 尋找所有圖片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        # 遍歷來源目錄尋找圖片
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(Path(root) / file)
        
        print(f"找到 {len(image_files)} 張原始圖片。")
        
        # 隨機打亂並分割
        random.shuffle(image_files)
        split_idx = int(len(image_files) * self.split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"訓練集: {len(train_files)} 張, 驗證集: {len(val_files)} 張")
        
        # 處理訓練集
        for img_path in train_files:
            self.process_single_image(img_path, 'train')
            
        # 處理驗證集
        for img_path in val_files:
            self.process_single_image(img_path, 'val')

    def process_single_image(self, img_path, subset):
        # 尋找對應的標註檔
        label_path = None
        
        # 1. 檢查同目錄
        potential_label = img_path.with_suffix('.txt')
        if potential_label.exists():
            label_path = potential_label
        else:
            # 2. 檢查標準 YOLO 結構
            parts = list(img_path.parts)
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'labels'
                potential_label = Path(*parts).with_suffix('.txt')
                if potential_label.exists():
                    label_path = potential_label
        
        if label_path and label_path.exists():
            self.slice_image(img_path, label_path, subset)
        else:
            print(f"警告: 找不到 {img_path} 的標註檔，跳過。")

    def slice_image(self, image_path, label_path, subset):
        # 根據 subset 決定輸出目錄
        if subset == 'train':
            img_out_dir = self.images_train
            lbl_out_dir = self.labels_train
        else:
            img_out_dir = self.images_val
            lbl_out_dir = self.labels_val

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"讀取錯誤 {image_path}")
            return
            
        h, w = img.shape[:2]
        
        # 讀取標註
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    w_abs = bw * w
                    h_abs = bh * h
                    x_center_abs = cx * w
                    y_center_abs = cy * h
                    
                    x1 = x_center_abs - w_abs / 2
                    y1 = y_center_abs - h_abs / 2
                    x2 = x_center_abs + w_abs / 2
                    y2 = y_center_abs + h_abs / 2
                    
                    labels.append({
                        'class': cls,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'w': w_abs, 'h': h_abs,
                        'area': w_abs * h_abs,
                        'original_line': line.strip()
                    })
        
        # 產生切片座標
        x_steps = list(range(0, w - self.slice_size + 1, self.stride))
        if w > self.slice_size and (w - self.slice_size) % self.stride != 0:
             x_steps.append(w - self.slice_size)
        
        y_steps = list(range(0, h - self.slice_size + 1, self.stride))
        if h > self.slice_size and (h - self.slice_size) % self.stride != 0:
            y_steps.append(h - self.slice_size)

        count = 0
        used_labels = set()
        
        for y in y_steps:
            for x in x_steps:
                slice_rect = (x, y, x + self.slice_size, y + self.slice_size)
                
                slice_labels = []
                
                for i, label in enumerate(labels):
                    ix1 = max(label['x1'], x)
                    iy1 = max(label['y1'], y)
                    ix2 = min(label['x2'], x + self.slice_size)
                    iy2 = min(label['y2'], y + self.slice_size)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        intersection_area = (ix2 - ix1) * (iy2 - iy1)
                        
                        if intersection_area < (label['area'] / 3):
                            pass
                        else:
                            used_labels.add(i)
                            
                            nx1 = ix1 - x
                            ny1 = iy1 - y
                            nx2 = ix2 - x
                            ny2 = iy2 - y
                            
                            ncx = (nx1 + nx2) / 2.0 / self.slice_size
                            ncy = (ny1 + ny2) / 2.0 / self.slice_size
                            nw = (nx2 - nx1) / self.slice_size
                            nh = (ny2 - ny1) / self.slice_size
                            
                            ncx = max(0.0, min(1.0, ncx))
                            ncy = max(0.0, min(1.0, ncy))
                            nw = max(0.0, min(1.0, nw))
                            nh = max(0.0, min(1.0, nh))
                            
                            slice_labels.append(f"{label['class']} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                
                stem = image_path.stem
                slice_name = f"{stem}_{x}_{y}_{count}"
                
                save_img_path = img_out_dir / f"{slice_name}.jpg"
                save_lbl_path = lbl_out_dir / f"{slice_name}.txt"
                
                crop_img = img[y:y+self.slice_size, x:x+self.slice_size]
                cv2.imwrite(str(save_img_path), crop_img)
                
                if slice_labels:
                    with open(save_lbl_path, 'w') as f:
                        f.write('\n'.join(slice_labels))
                else:
                    Path(save_lbl_path).touch()
                
                count += 1
        
        if len(used_labels) < len(labels):
            unused = set(range(len(labels))) - used_labels
            # print(f"警告: {image_path.name} 中有 {len(unused)} 個標註未被完整保留。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/wols/yolo_test/datasets/pcb_dataset', help='來源資料集目錄')
    parser.add_argument('--output', type=str, default='/home/wols/yolo_test/dataset_cut', help='輸出目錄')
    parser.add_argument('--stride', type=int, default=600, help='滑動視窗步長')
    parser.add_argument('--split', type=float, default=0.9, help='訓練集比例 (0.0-1.0)')
    args = parser.parse_args()
    
    slicer = ImageSlicer(args.source, args.output, stride=args.stride, split_ratio=args.split)
    slicer.process()
