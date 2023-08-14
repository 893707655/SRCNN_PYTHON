import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


#函数作用：读取文件夹中图像数量  参数：dir为目标文件夹地址
#返回值：filenum为图像数量
def countFile(dir):
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp


#定义数据
class MyDataset(Dataset):
    def __init__(self,low_root,high_root):
        self.low_root = low_root
        self.high_root = high_root
        self.low_list = os.listdir(low_root)
        self.high_list = os.listdir(high_root)
        self.low_list.sort()
        self.high_list.sort()
        self.len = len(self.low_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        low_path = os.path.join(self.low_root,self.low_list[index])
        high_path = os.path.join(self.high_root,self.high_list[index])
        low = cv2.imread(low_path)
        low = cv2.cvtColor(low,cv2.COLOR_BGR2GRAY)
        high = cv2.imread(high_path)
        high = cv2.cvtColor(high,cv2.COLOR_BGR2GRAY)
        low_np = np.array(low)
        high_np = np.array(high)
        low_tensor = torch.from_numpy(low_np).to(torch.float32).unsqueeze(0)
        high_tensor = torch.from_numpy(high_np).to(torch.float32).unsqueeze(0)
        return low_tensor, high_tensor

#定义模型
class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN,self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, 9, 1, 4)  # 输入通道 输出通道 卷积核大小 步长 填充
        self.Conv2 = nn.Conv2d(64, 32, 1, 1, 0)
        self.Conv3 = nn.Conv2d(32, 1, 5, 1, 2)
        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Relu(self.Conv1(x))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out)
        out = x + out
        return out


def calacc(y_pred, y):
    y_pred = y_pred.detach().numpy()
    y = y.detach().numpy()
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y = np.where(y > 0.5, 1, 0)
    acc = np.sum(y_pred == y) / (y.shape[0]*y.shape[2]*y.shape[3])
    return acc


#读取数据，选用训练设备
dl = DataLoader(MyDataset('C:/Users/ROG/Desktop/textimage/lowResolution', 'C:/Users/ROG/Desktop/textimage/highResolution'), batch_size=8, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#创建模型 优化器 损失函数
model = SuperResolutionCNN()
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()


#函数：定义训练函数。参数：model为模型，dl数据，optimizer为优化器，criterion为损失函数，epochs为训练批次
def train(model, dl, optimizer, criterion, epochs):
    metric={
        'train_loss':[],
        'train_acc':[],
    }
    for epoch in range(epochs):
        for i,(x,y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            acc = calacc(y_pred.detach().cpu(),y.detach().cpu())
            if i % 10 == 0 and i != 0:
                metric['train_loss'].append(loss.item())
                metric['train_acc'].append(acc)
                print('epoch:%d,step:%d,loss:%f,acc:%f'%(epoch+1, i+1, loss.item(),acc))

        if epoch%5 == 0 and epoch!=0:
            torch.save(model.state_dict(), r'C:\Users\ROG\Desktop\textimage\srcnn_test\model.pth')


if __name__ == '__main__':
    filenum = countFile(r"C:\Users\ROG\Desktop\textimage\00")
    print(filenum)
    train(model, dl, optimizer, criterion, 5)
