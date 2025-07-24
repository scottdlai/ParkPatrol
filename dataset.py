import torch
from torch.utils.data import Dataset
from PIL import Image
import os 

class ParkingDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.labels = sorted(os.listdir(label_dir))

        self.class_names = ['Occupied', 'Vacant']
        self.num_classes = len(self.class_names)
        self.class_dict = {i: name for i, name in enumerate(self.class_names)}

    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        #open and resize images 
        img= Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        #convert labels to tensors 
        labels = []
        with open(label_path, 'r') as x:
            for line in x:
                parts = list(map(float, line.strip().split()))
                labels.append([idx] + parts)
        labels = torch.tensor(labels, dtype=torch.float32)

        return img, labels