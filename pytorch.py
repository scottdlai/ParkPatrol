from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import SETTINGS
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from dataset import ParkingDataset
import datetime as dt
from types import SimpleNamespace

#stops ultralytics from downloading the cifar10 dataset
SETTINGS['datasets_dir'] = './data'

#should probably switch to more advanced model once we get the training loop working
model = YOLO('yolov8n.pt').model.train()
# weights for the three losses that make up v8detectionloss 
model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#initialize training and validations sets 
training_set = ParkingDataset('data/images/train', 'data/labels/train', transform)
validation_set = ParkingDataset('data/images/val', 'data/labels/val', transform)

#helps determien how many data samples to combine into batch 
def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return images, labels


train_dataloader = DataLoader(training_set, batch_size = 4, shuffle=True, collate_fn=collate)
valid_dataloader = DataLoader(validation_set, batch_size = 4, shuffle = True, collate_fn = collate)
classes = ('Occupied', 'Vacant')



#stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


loss_fn = v8DetectionLoss(model)

def train_one_epoch(epoch_index, training_loader, model, optimizer, loss_fn, tb_writer=None):
    box_loss = 0.0
    cls_loss = 0.0
    dfl_loss = 0.0


    print("beginning training")
    for i, data in enumerate(training_loader):
        img, targets = data
        optimizer.zero_grad()

        outputs = model(img)

        # dict format for v8detecionloss input 
        batch = {
            "batch_idx": targets[:, 0].long(),
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:6]
        }


        print("calculating loss and optimizing")
        loss, loss_items = loss_fn(outputs, batch)
        
        print("calculation complete")
        loss.requires_grad = True
        loss.sum().backward()
        print("backward calculation complete")
        print("doing fancy optimizer stuff")
        optimizer.step()
        print("fancy optimizer stuff done")
        
        

        print(f"batch {i+1} loss:")
        #tb_x = epoch_index * len(training_loader) + i + 1
        #tb_writer.add_scalar('Loss/train', running_loss, tb_x)
        box_loss += loss_items[0]
        cls_loss += loss_items[1]
        dfl_loss += loss_items[2]

        print(f"box_loss: {loss_items[0]:.4f}, cls_loss: {loss_items[1]:.4f}, dfl_loss: {loss_items[2]:.4f}")
        print("training done")

    box_loss = box_loss / 4
    cls_loss = cls_loss / 4
    dfl_loss = dfl_loss / 4
    print(f"final loss: box_loss: {box_loss:.4f}, cls_loss: {cls_loss:.4f}, dfl_loss: {dfl_loss:.4f}")

    return box_loss, cls_loss, dfl_loss



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
    box_loss, cls_loss, dfl_loss = train_one_epoch(epoch_number, train_dataloader, model, optimizer, loss_fn, writer)
    print("train_one_epoch complete")

    print(f"box_loss: {box_loss:.4f}, cls_loss: {cls_loss:.4f}, dfl_loss: {dfl_loss:.4f}")

    running_valid_loss = 0.0

    # note: validation does not work right now 
    model.model.eval()

    with torch.no_grad(): 
        print("beginning validation")
        for i, data in enumerate(valid_dataloader):
            imgs, targets = data
            outputs = model(imgs)
            batch = {
            "batch_idx": targets[:, 0].long(),
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:6]
            }
            valid_loss, valid_loss_items = loss_fn(outputs, batch)
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
            model_path = "best/parking_trainer_{}/model_best.pt".format(timestamp_initial)
            torch.save(model.model.state_dict(), model_path)

        epoch_number += 1


print("training and validation done, model saved to best/parking_trainer_{}/model_best.pt".format(timestamp_initial))


