from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T
from dataset import ParkingDataset
import os 

#should probably switch to more advanced model once we get the training loop working
model = YOLO('YOLOv8n.pt')

transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#initialize training and validations sets 
training_set = ParkingDataset('data/images/train', 'data/labels/train', transform)
validation_set = ParkingDataset('data/images/valid', 'data/labels/valid', transform)
classes = ('Occupied', 'Vacant')

#stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
