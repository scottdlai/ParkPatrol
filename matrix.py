from ultralytics import YOLO

model = YOLO('models_trained/modelv8m.pt')
results = model.val(data="park.yaml", conf=0.60)
cm = results.confusion_matrix


