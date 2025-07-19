from ultralytics import YOLO
from ultralytics.nn.tasks import v8DetectionLoss
from ultralytics.utils import SETTINGS
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as T
from dataset import ParkingDataset
import datetime as dt
import os

#stops ultralytics from downloading the cifar10 dataset
SETTINGS['datasets_dir'] = './data'

#should probably switch to more advanced model once we get the training loop working
model = YOLO('yolov8n.pt')

transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#initialize training and validations sets 
training_set = ParkingDataset('data/images/train', 'data/labels/train', transform)
validation_set = ParkingDataset('data/images/val', 'data/labels/val', transform)

#helps determien how many data samples to combine into batch 
def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, labels


train_dataloader = DataLoader(training_set, batch_size = 64, shuffle=True, collate_fn=collate)
valid_dataloader = DataLoader(validation_set, batch_size = 64, shuffle = True, collate_fn = collate)
classes = ('Occupied', 'Vacant')



#stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_fn = v8DetectionLoss(model.model)

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


timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp_initial = timestamp
writer = SummaryWriter('runs/parking_trainer_{}'.format(timestamp))
epoch_number = 0
epoch_best = 0

EPOCHS = 5
best_valid_loss = float('inf')

#training loop 
for epoch in range(EPOCHS): 
    print(f'EPOCH {epoch_number + 1}:')

    model.model.train(True)
    avg_loss = train_one_epoch(epoch_number, train_dataloader, model, optimizer, loss_fn, writer)

    print(f"training loss: {avg_loss:.4f}")

    running_valid_loss = 0.0

    model.model.eval()

    with torch.no_grad(): 
        for i, data in enumerate(valid_dataloader):
            valid_inputs, valid_labels = data 

            valid_outputs = model.model(valid_inputs)
            valid_loss, valid_loss_items = loss_fn({'preds': valid_outputs, 'targets': valid_labels})
            running_valid_loss += valid_loss.item()
        
        avg_valid_loss = running_valid_loss / (i + 1)

        print(f"validation loss: {valid_loss:.4f}")
        print(f"box_loss: {valid_loss_items[0]:.4f}, cls_loss: {valid_loss_items[1]:.4f}, dfl_loss: {valid_loss_items[2]:.4f}")
        print(f"average validation loss: {avg_valid_loss:.4f}")



        writer.add_scalars('Training vs. Validation Loss',
            { 'Training' : avg_loss, 'Validation' : avg_valid_loss },
            epoch_number + 1)
        writer.flush()

        # save the best reults 
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epoch_best = epoch_number
            model_path = "runs/parking_trainer_{}/model_best.pt".format(timestamp_initial)
            torch.save(model.model.state_dict(), model_path)

        epoch_number += 1


print("training and validation done, model saved to runs/parking_trainer_{}/model_best.pt".format(timestamp_initial))


