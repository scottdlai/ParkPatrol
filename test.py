from ultralytics import YOLO

model = YOLO("runs/train/parking/weights/best.pt")

# test images are currently the same as validation images, will add dedicated testing images once more data is created 
# conf = minimum confidence for model to classify an object 
results = model.predict(source="data/images/test", save=True, conf=0.25)
