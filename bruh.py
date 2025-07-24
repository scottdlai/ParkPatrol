from ultralytics import YOLO
model = YOLO("yolov8n.pt")  
model.train(data="park.yaml", epochs=1, imgsz=640, batch=4, name="parking", project="runs/train")