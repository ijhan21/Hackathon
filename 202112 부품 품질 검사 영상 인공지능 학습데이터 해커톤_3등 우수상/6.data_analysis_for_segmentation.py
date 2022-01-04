from torch import optim
from unet import *
import cv2
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision
import torch.nn.functional as F
import pickle
from sklearn.model_selection import train_test_split
import cv2

EARLY_BIRD_LIMIT= 100
EARLY_BIRD= 0
BEST_MODEL_PATH = 'best_segmentation_model.pth'
NUM_CLASSES = 1 # 0 정상 1 불량
BATCH_SIZE = 4
NUM_EPOCHS = 100000
LEARNING_RATE = 5e-7
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

with open('detected_img_segmentation_datasets.pickle', 'rb') as f:
    datasets_pre = pickle.load(f)
datasets=list()
for data in datasets_pre:
    if data[1] is None: continue
    label=data[1]/255.
    img = data[0]
    c, h, w = img.shape
    label = cv2.resize(label, (h, w))
    img = img.reshape(h, w, c).astype(np.float32)    
    img = cv2.resize(img, (h, w))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, h, w)
    datasets.append((img, label))

train_dataset, test_dataset = train_test_split(datasets, test_size=0.1, shuffle=True)
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

# unet = DoubleUNet(3,1).to(DEVICE)
# unet = DoubleUNet(1,1, bilinear=True).to(DEVICE)
unet = UNet(1,1, bilinear=True).to(DEVICE)
criterion= nn.BCEWithLogitsLoss()
d_time = datetime.datetime.now()
folder_name = "runs_seg/"+d_time.strftime("%Y%m%d%H%M%S")
os.mkdir(folder_name)
writer = SummaryWriter(log_dir=folder_name)
pre_test_loss = None
optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9), weight_decay=0.0005)
for epoch in range(NUM_EPOCHS):
    train_loss=0.
    test_loss=0.
    for img, label in train_loader:                                
            img = img.to(DEVICE)
            label = label.float().to(DEVICE)            
            label = label.unsqueeze(1)
            outputs_ = unet(img.float())
            outputs = outputs_
            # loss = torch.mean(torch.sqrt(torch.square(outputs-label)))
            loss = criterion(outputs,label)
            optimizer.zero_grad()
            train_loss+=loss        
            loss.backward()
            optimizer.step()

    outputs_test=None    
    label_test=None   
    img = None
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(DEVICE).float()
            label_test = label.unsqueeze(1).float().to(DEVICE)
                        
            outputs_test = unet(img)
            loss = criterion(outputs_test,label_test)           
            test_loss+=loss
            outputs_test_for_grid = torch.round(outputs_test)
    print(outputs_test.max())

    writer.add_scalar("SUMMARY/LOSS",test_loss/len(test_dataset),epoch)
    grid_test = torchvision.utils.make_grid(outputs_test_for_grid)
    grid_label = torchvision.utils.make_grid(label_test)
    grid_img = torchvision.utils.make_grid(img)#.permute(1,2,0)
    writer.add_image('IMAGE', grid_img, epoch)
    writer.add_image('PREDICTION', grid_test, epoch)
    writer.add_image('LABEL', grid_label, epoch)
    if pre_test_loss is None:
        pre_test_loss = test_loss
    if pre_test_loss<test_loss and epoch>50:
        pre_test_loss= test_loss
        torch.save(unet.state_dict(), str(epoch)+BEST_MODEL_PATH)
    else:
        EARLY_BIRD+=1
        if EARLY_BIRD>EARLY_BIRD_LIMIT:
            print("Train Complete at Epoch %d!"%epoch)
            break
