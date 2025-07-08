from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

model.train(data="park.yaml", epochs=10, imgsz=640, batch=16, name="parking", project="runs/train")
