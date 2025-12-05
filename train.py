from ultralytics import YOLO
from clearml import Task
import sys
task = Task.init(
    project_name="pcb_yolo_training_img_size", 
    task_name="e50_p30_i640_b8_cm")
    
def main():
    # 訓練一個乾淨的模型 (Train from scratch)
    # 使用 .yaml 檔案來建立一個全新的模型結構，而不載入預訓練權重
    # 這樣可以避免遷移學習帶來的影響
    # Ultralytics 會先檢查你當前目錄下有沒有這個檔案。
    # 如果沒有它會去它自己的安裝目錄下找內建的配置文件。
    # model = YOLO("yolo12n.yaml")
    # task.set_parameter("model_variant", "yolo12n")
    model = YOLO("yolo12m.yaml")
    task.set_parameter("model_variant", "yolo12m")
    args = dict(
        data="pcb_data_cut.yaml", 
        epochs=50,
        patience=30,
        imgsz=640,
        batch=8,
        # project="pcb_yolo_training_50",
        # name="yolov12_pcb_scratch_50"
        )
    task.set_parameters(args)
    results = model.train(**args)
    # 訓練
    # results = model.train(
    #     data="pcb_data.yaml", 
    #     # 從頭訓練通常需要更多的 epochs
    #     # epochs=100,
    #     # 嘗試不同的epochs
    #     epochs= 50, 
    #     # 如果 30 個 epochs 內 validation loss 沒下降就停止
    #     patience=30,
    #     # 訓練時會將所有圖片縮放 (Resize) 到的尺寸
    #     # 因為pcb fail 很小，可能要指定大一點
    #     imgsz=1280, 
    #     # 指定訓練一次要讀入的圖片數量
    #     # 取決於VRAM，因為imgsz變大，batch要變小
    #     batch=4,
    #     # 訓練結果會存放在這個主資料夾下
    #     project="pcb_yolo_training_50",
    #     # 訓練的結果會存放在 project/name 裡面
    #     name="yolov12_pcb_scratch_50"
    # )

if __name__ == "__main__":
    main()
