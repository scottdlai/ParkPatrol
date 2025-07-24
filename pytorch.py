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

#originally created with model.train() from ultralytics
yolo = YOLO('parking.pt')
model = yolo.model.train()

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

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 3

train_dataloader = DataLoader(training_set, TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
valid_dataloader = DataLoader(validation_set, VAL_BATCH_SIZE, shuffle = True, collate_fn = collate)




# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)


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
        loss.requires_grad = True
        print("calculation complete")
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

    box_loss = box_loss / (i + 1)
    cls_loss = cls_loss / (i + 1)
    dfl_loss = dfl_loss / (i + 1)
    print(f"final loss: box_loss: {box_loss:.4f}, cls_loss: {cls_loss:.4f}, dfl_loss: {dfl_loss:.4f}")

    return box_loss, cls_loss, dfl_loss



timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp_initial = timestamp
writer = SummaryWriter('runs/parking_trainer_{}'.format(timestamp))
epoch_number = 0
epoch_best = 0

EPOCHS = 50
best_valid_loss = float('inf')

#training loop 
for epoch in range(EPOCHS): 
    v_box_loss = 0.0
    v_cls_loss = 0.0
    v_dfl_loss = 0.0
    
    print(f'EPOCH {epoch_number + 1}:')

    model.train(True)
    box_loss, cls_loss, dfl_loss = train_one_epoch(epoch_number, train_dataloader, model, optimizer, loss_fn, None)
    print("train_one_epoch complete")

    print(f"box_loss: {box_loss:.4f}, cls_loss: {cls_loss:.4f}, dfl_loss: {dfl_loss:.4f}")

    training_loss = box_loss + cls_loss + dfl_loss
    # note: validation does not work right now 
    model.eval()

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
            v_loss, v_loss_items = loss_fn(outputs, batch)
        
            v_box_loss += v_loss_items[0]
            v_cls_loss += v_loss_items[1]
            v_dfl_loss += v_loss_items[2]  

            print(f"box_loss: {v_loss_items[0]:.4f}, cls_loss: {v_loss_items[1]:.4f}, dfl_loss: {v_loss_items[2]:.4f}")

        v_box_loss = v_box_loss / (i + 1) 
        v_cls_loss = v_cls_loss / (i + 1)
        v_dfl_loss = v_dfl_loss / (i + 1)

        v_loss = v_box_loss + v_cls_loss + v_dfl_loss

        print("validation done")


        # save the best reults 
        if v_loss < best_valid_loss:
            best_valid_loss = v_loss
            epoch_best = epoch_number
            model_path = "runs/parking_trainer_{}/model_best.pt".format(timestamp_initial)
            yolo.save(model_path)
            yolo.save("model_best.pt")
            print("v_loss was less than best_valid_loss")
            print(f"new best model at epoch {epoch_number + 1}")
        else:
            print("v_loss was not less than best_valid_loss")
        
        print(f"v_loss: {v_loss:.4f} | best_valid_loss: {best_valid_loss:.4f}")
        print(f"best epoch: {epoch_best + 1}\n")
        epoch_number += 1
 

print("training and validation done, model saved to runs/parking_trainer_{}/model_best.pt".format(timestamp_initial))