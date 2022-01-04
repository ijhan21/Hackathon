import pickle
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import os

EARLY_BIRD_LIMIT= 100
EARLY_BIRD= 0
BEST_MODEL_PATH = 'best_classification_model.pth'
NUM_CLASSES = 1 # 0 정상 1 불량
BATCH_SIZE = 128
NUM_EPOCHS = 100000
LEARNING_RATE = 1e-3
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

with open('detected_img_classification_datasets.pickle', 'rb') as f:
    datasets = pickle.load(f)
train_dataset, test_dataset = train_test_split(datasets, test_size=0.2, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = models.efficientnet_b0(pretrained=True)
IN_FEATURES = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=IN_FEATURES, out_features=NUM_CLASSES)
)
model = model.to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9), weight_decay=0.0005)
criterion = nn.BCELoss()

d_time = datetime.datetime.now()
folder_name = "runs_cls/"+d_time.strftime("%Y%m%d%H%M%S")
os.mkdir(folder_name)
writer = SummaryWriter(log_dir=folder_name)
pre_loss = None
for epoch in range(NUM_EPOCHS):
    train_loss=[]
    test_loss=[]
    acc = list()
    for img, label in train_loader:
        img = img.float().to(DEVICE)
        label = label.unsqueeze(-1).long().to(DEVICE)
        output = model(img)

        loss = torch.mean(torch.square(output-label))
        # loss = criterion(output,label)
        train_loss.append(loss)
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        for img, label in test_loader:
            img = img.float().to(DEVICE)
            label = label.unsqueeze(-1).long().to(DEVICE)
            output = model(img)
            loss = torch.mean(torch.square(output-label))
            test_loss.append(loss)
            acc.append(torch.sum(label.argmax(-1)==output.argmax(-1)))
    acc = torch.sum(torch.Tensor(acc)).item()/len(test_dataset)
    loss_sum = torch.sum(torch.Tensor(test_loss)).item()
    writer.add_scalar("SUMMARY/TRAIN_LOSS",torch.mean(torch.Tensor(train_loss)).item(),epoch)
    writer.add_scalar("SUMMARY/TEST_LOSS",torch.mean(torch.Tensor(test_loss)).item(),epoch)
    writer.add_scalar("SUMMARY/ACCURACY",acc,epoch)    

    if pre_loss is None:
        pre_loss = loss_sum
    if pre_loss<loss_sum and epoch>30:
        pre_acc = loss_sum        
        torch.save(model.state_dict(), str(epoch)+BEST_MODEL_PATH)
    else:
        EARLY_BIRD+=1
        if EARLY_BIRD>EARLY_BIRD_LIMIT:
            print("Train Complete at Epoch %d!"%epoch)
            break