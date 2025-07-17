from ultralytics import YOLO
from ultralytics.nn.tasks import v8DetectionLoss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as T
from dataset import ParkingDataset
import datetime

#should probably switch to more advanced model once we get the training loop working
model = YOLO('YOLOv8n.pt')

transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#initialize training and validations sets 
training_set = ParkingDataset('data/images/train', 'data/labels/train', transform)
validation_set = ParkingDataset('data/images/valid', 'data/labels/valid', transform)

#helps determien how many data samples to combine into batch 
def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, labels


train_dataloader = DataLoader(training_set, batch_size = 64, shuffle=True, collate_fn=collate)
valid_dataloader = DataLoader(validation_set, batch_size = 64, shuffle = True)
classes = ('Occupied', 'Vacant')



#stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_fn = v8DetectionLoss()

def train_one_epoch(epoch_index, training_loader, model, optimizer, loss_fn, tb_writer=None):
    model.train() 
    running_loss = 0
    last_loss = 0


    for i, data in enumerate(training_loader):
        img, labels = data
        optimizer.zero_grad()

        outputs = model.model(img)

        loss, loss_items = loss_fn({'preds': outputs, 'targets': labels})
        loss.backward()
       
        optimizer.step()
        
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

        print(f"box_loss: {loss_items[0]:.4f}, cls_loss: {loss_items[1]:.4f}, dfl_loss: {loss_items[2]:.4f}")

    return last_loss


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/parking_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS): 



