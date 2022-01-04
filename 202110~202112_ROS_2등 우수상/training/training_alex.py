import os
import numpy as np
import cv2
from torchvision import models
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import torchvision

DATAS_UP_PATH='./output_up_camera/'
DATAS_DOWN_PATH='./output_down_camera/'


NUM_EPOCHS=10000
BATCH_SIZE = 4
TEST_RATE = 0.2
LEARNING_RATE = 1e-3
up_list = os.listdir(DATAS_UP_PATH)
donw_list = os.listdir(DATAS_DOWN_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# follow / to_position / line_num / loaded /// position_bool / people / x / y + filename

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

dataset=list()
for i in range(len(up_list)):
    if up_list[i][-4:] != '.jpg':continue
    follow, to_position, line_num, loaded, position_bool, people = up_list[i].split('_')[0]
    x = up_list[i].split('_')[1]
    y = up_list[i].split('_')[2]
    follow, to_position, line_num, loaded, position_bool, people = float(follow), float(to_position), float(line_num), float(loaded), float(position_bool), float(people)
    x, y = float(x), float(y)
    img_up = cv2.imread(DATAS_UP_PATH+up_list[0]).reshape(3,224,224)/255.
    img_down = cv2.imread(DATAS_DOWN_PATH+up_list[0]).reshape(3,224,224)/255.
    input_data = (follow, to_position, line_num, loaded, img_up, img_down)
    input_label = (position_bool, people, x, y)
    dataset.append((input_data, input_label))


TEST_SET_NUM = int(len(dataset)*TEST_RATE)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - TEST_SET_NUM, TEST_SET_NUM])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
#     num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
#     num_workers=4
)

class AutoDrive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_classes=4 # position_bool, people, x, y
        self.unit_output = 100
        self.model_up = self.making_transfer_model()
        self.model_down = self.making_transfer_model()
        self.determine_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=204, out_features=102, bias=True),
            torch.nn.Mish(),
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=102, out_features=51, bias=True),
            torch.nn.Mish(),
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=51, out_features=self.output_classes, bias=True),
            torch.nn.ReLU(),
            # torch.nn.Sigmoid(),
        )
    
    def making_transfer_model(self):
        model = models.alexnet(pretrained=True)
        model.classifier=torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            # torch.nn.Linear(in_features=1280, out_features=self.unit_output, bias=True),
            torch.nn.Linear(in_features=9216, out_features=self.unit_output, bias=True),
            torch.nn.ReLU(),
        )
        return model

    def forward(self, follow, to_position, line_num, loaded, up_img, down_img): # follow / to_position / line_num / loaded
        # up_img = torch.from_numpy(up_img).float()
        # down_img = torch.from_numpy(down_img).float()
        x_up = self.model_up(up_img)
        x_down = self.model_down(down_img)
        follow = follow.unsqueeze(-1)
        to_position = to_position.unsqueeze(-1)
        line_num = line_num.unsqueeze(-1)
        loaded = loaded.unsqueeze(-1)
        x = torch.cat([follow,to_position, line_num, loaded, x_up, x_down], dim=-1)
        x = self.determine_model(x.float())
        return x


auto = AutoDrive()
auto = auto.to(device=device)

count = 0
optimizer = torch.optim.Adam(auto.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9), weight_decay=0.0005)
scaler = torch.cuda.amp.GradScaler()
BEST_MODEL_PATH = 'best_model.pth'

d_time = datetime.datetime.now()
folder_name = "runs/"+d_time.strftime("%Y%m%d%H%M%S")
os.mkdir(folder_name)
writer = SummaryWriter(log_dir=folder_name)

for epoch in range(NUM_EPOCHS):
    train_loss=0.
    test_loss=0.
    
    for (follow, to_position, line_num, loaded, up_img, down_img),(position_bool, people, x, y) in train_loader:
        follow = follow.to(device=device); to_position = to_position.to(device=device); line_num = line_num.to(device=device); loaded = loaded.to(device=device); up_img = normalize(up_img.float()).to(device=device); down_img = normalize(down_img.float()).to(device=device);
        position_bool=position_bool.to(device); people=people.to(device); x=x.to(device); y=y.to(device);
        position_bool=position_bool.unsqueeze(-1); people=people.unsqueeze(-1); x=x.unsqueeze(-1); y=y.unsqueeze(-1);

        labels = torch.cat([position_bool, people, x, y], dim=-1)
        # print(position_bool.shape,labels)
        optimizer.zero_grad()
        outputs = auto(follow, to_position, line_num, loaded, up_img.float(), down_img.float())
        loss = F.cross_entropy(outputs, labels)
        train_loss+=loss        
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
    
    
    with torch.no_grad():
        for (follow, to_position, line_num, loaded, up_img, down_img),(position_bool, people, x, y) in test_loader:
            follow = follow.to(device=device); to_position = to_position.to(device=device); line_num = line_num.to(device=device); loaded = loaded.to(device=device); up_img = up_img.to(device=device); down_img = down_img.to(device=device);
            position_bool=position_bool.to(device); people=people.to(device); x=x.to(device); y=y.to(device);
            position_bool=position_bool.unsqueeze(-1); people=people.unsqueeze(-1); x=x.unsqueeze(-1); y=y.unsqueeze(-1);

            labels = torch.cat([position_bool, people, x, y], dim=-1)
            outputs = auto(follow, to_position, line_num, loaded, up_img.float(), down_img.float())
    #         loss_position = F.cross_entropy(outputs[:,0], position_bool)
    #         loss_people = F.cross_entropy(outputs[:,1], people)
    #         loss_x = F.cross_entropy(outputs[:,2], x)
    #         loss_y = F.cross_entropy(outputs[:,3], y)
            loss = F.cross_entropy(outputs, labels)
            test_loss+=loss
#     print(f"{epoch} TRAIN_LOSS: {train_loss.item()} TEST LOSS: {test_loss.item()}")
    writer.add_scalar("TRAIN_LOSS",train_loss.item()/len(train_loader),epoch)
    writer.add_scalar("TEST LOSS",test_loss.item()/len(test_loader),epoch)
    torch.save(auto.state_dict(), BEST_MODEL_PATH)
