from ultralytics import YOLO


model = YOLO('model_v2.pt')

# test images are currently the same as validation images, will add dedicated testing images once more data is created
# conf = minimum confidence for model to classify an object
results = model.predict(source="data/images/test", save=True, show_labels=False, show_conf=False, conf=0.60)

print(results)
