from ultralytics import YOLO
import torch


model = YOLO('yolov8n.yaml')  # must contain correct nc and names
model.model.load_state_dict(torch.load('runs/parking_trainer_20250723_172420/model_best.pt'))
# test images are currently the same as validation images, will add dedicated testing images once more data is created 
# conf = minimum confidence for model to classify an object 
results = model.predict(source="data/images/val", save=True, conf=0.00)
