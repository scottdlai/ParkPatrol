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

for name, param in model.named_parameters():
    param.requires_grad = True

# weights for the three losses that make up v8detectionloss 
model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])

#initialize training and validations sets 
training_set = ParkingDataset('data/images/train', 'data/labels/train', transform)
validation_set = ParkingDataset('data/images/val', 'data/labels/val', transform)
#helps determien how many data samples to combine into batch 
def collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)

    updated_labels = []
    for batch_i, label in enumerate(labels):
        label[:, 0] = batch_i
        updated_labels.append(label)
    labels = torch.cat(updated_labels, dim=0)
    return images, labels

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 3

train_dataloader = DataLoader(training_set, TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
valid_dataloader = DataLoader(validation_set, VAL_BATCH_SIZE, shuffle = True, collate_fn = collate)

# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)


loss_fn = v8DetectionLoss(model)

model.to(device)

def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    box = 0.0
    cls = 0.0
    dfl = 0.0
    i = 1
    for img, targets in loader:
        
        img = img.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        outputs = model(img)

        # dict format for v8detecionloss input 
        batch = {
            "batch_idx": targets[:, 0].long(),
            "cls": targets[:, 1].long(),
            "bboxes": targets[:, 2:6]
        }

        loss, loss_items = loss_fn(outputs, batch)
        total_loss = loss[0] + loss[1] + loss[2]

        total_loss.backward()

        optimizer.step()

        

        #print(f"batch {i} loss:")
        #tb_x = epoch_index * len(training_loader) + i + 1
        #tb_writer.add_scalar('Loss/train', running_loss, tb_x)
        box += loss_items[0]
        cls += loss_items[1]
        dfl += loss_items[2]

       # print(f"box: {loss_items[0]:.4f}, cls: {loss_items[1]:.4f}, dfl: {loss_items[2]:.4f}")
        i += 1

    box = box / len(loader)
    cls = cls / len(loader)
    dfl = dfl / len(loader)
   # print(f"final training loss: box: {box:.4f}, cls: {cls:.4f}, dfl: {dfl:.4f}")
    return box, cls, dfl

def validate(loader, model, loss_fn):
    box = 0.0
    cls = 0.0
    dfl = 0.0
    model.eval()
    with torch.no_grad():
        for img, targets in loader:
            img = img.to(device)
            targets = targets.to(device)
            outputs = model(img)
            batch = {
                "batch_idx": targets[:, 0].long(),
                "cls": targets[:, 1].long(),
                "bboxes": targets[:, 2:6]
            }

            loss, loss_items = loss_fn(outputs, batch)
            box += loss_items[0]
            cls += loss_items[1]
            dfl += loss_items[2]
        
        box = box / len(loader)
        cls = cls / len(loader)
        dfl = dfl / len(loader)
      #  print(f"final training loss: box: {box:.4f}, cls: {cls:.4f}, dfl: {dfl:.4f}")
        return box, cls, dfl

timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
timestamp_initial = timestamp
writer = SummaryWriter('runs/parking_trainer_{}'.format(timestamp))
EPOCHS = 200
best_valid_loss = float('inf')
patience = 0

#training + validation loop 
for epoch in range(EPOCHS): 
    if patience > 10:
        print("Model not improving, ending training loop")
        break
    # call the functions 
    t_box, t_cls, t_dfl = train_one_epoch(train_dataloader, model, optimizer, loss_fn)
    v_box, v_cls, v_dfl = validate(valid_dataloader, model, loss_fn)

    # calculate total training and validation loss (V8Detectionloss applies the weights for us)
    t_loss = t_box + t_cls + t_dfl
    v_loss = v_box + v_cls + v_dfl

    # logging values for when we make graphs (w/ tensorboard perhaps?)
  
    writer.add_scalar('Loss/train/box', t_box, epoch)
    writer.add_scalar('Loss/train/cls', t_cls, epoch)
    writer.add_scalar('Loss/train/dfl', t_dfl, epoch)

    writer.add_scalar('Loss/val/box', v_box, epoch)
    writer.add_scalar('Loss/val/cls', v_cls, epoch)
    writer.add_scalar('Loss/val/dfl', v_dfl, epoch)
    
    writer.add_scalar('Loss/train', t_loss, epoch)
    writer.add_scalar('Loss/val', v_loss, epoch)

    print(f"Loss values for epoch {epoch+1}: Training: [box = {t_box:.4f}, cls = {t_cls:.4f}, dfl = {t_dfl:.4f}] | Validation: [box = {v_box:.4f}, cls = {v_cls:.4f}, dfl = {v_dfl:.4f}]\n")

    if v_loss < best_valid_loss: 
        best_valid_loss = v_loss
        torch.save(model.state_dict(), f'runs/parking_trainer_{timestamp}/model_best.pt')
        patience = 0

    else:
        patience += 1
 
# reconstruct yolo modelb and save it
yolo.model = model
yolo.save("model_v2.pt")

print("Training and validation done, model saved to runs/parking_trainer_{}/model_best.pt".format(timestamp_initial))