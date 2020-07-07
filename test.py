import torch
import torchvision
from torchvision import transforms, datasets
import os
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 100

LEARNING_RATE = 0.1

EPOCH = 10

def conv_relu(in_channel,out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(inplace=True)
    )
    return layer

class inception1_7_8(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1):
        super(inception1_7_8, self).__init__()
        
        # 定义inception模块第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, kernel=1)
        
        # 定义inception模块第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, kernel=1),
            conv_relu(out2_1, out2_3, kernel=3, stride=1, padding=1)
        )
        
        #定义inception模块的第三条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out3_1, kernel=1)
        )
        
    def forward(self,x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3), dim=1)
        return output


class inception2(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_3, out4_1, out4_3, out5_1):
        super(inception2, self).__init__()
        
        # 定义inception模块第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, kernel=1)
        
        # 定义inception模块第二条线路
        self.branch3x3_1 = nn.Sequential(
            conv_relu(in_channel, out2_1, kernel=1),
            conv_relu(out2_1, out2_3, kernel=3, stride=1, padding=1)
        )
        
        #定义inception模块的第三条线路
        self.branch3x3_2 = nn.Sequential(
            conv_relu(in_channel, out3_1, kernel=1),
            conv_relu(out3_1, out3_3, kernel=3, stride=1, padding=1)
        )

        # 定义inception模块第四条线路
        self.branch3x3_3 = nn.Sequential(
            conv_relu(in_channel, out4_1, kernel=1),
            conv_relu(out4_1, out4_3, kernel=3, stride=1, padding=1)
        )
        # 定义inception模块第五条线路
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out5_1, kernel=1)
        )
        
    def forward(self,x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3_1(x)
        f3 = self.branch3x3_2(x)
        f4 = self.branch3x3_3(x)
        f5 = self.branch_pool(x)
        
        output = torch.cat((f1, f2, f3, f4, f5), dim=1)
        return output

class inception3_6_9(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3):
        super(inception3_6_9, self).__init__()
        
        # 定义inception模块第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, kernel=1, stride=2)
        
        # 定义inception模块第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, kernel=1),
            conv_relu(out2_1, out2_3, kernel=3, stride=2, padding=1)
        )
        
        #定义inception模块的第三条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
    def forward(self,x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3), dim=1)
        return output

class inception4_5(nn.Module):
    def __init__(self, in_channel, out1_1,out2_1, out2_3, out3_1, out3_3_1, out3_3_2, out4_1):
        super(inception4_5, self).__init__()
        
        # 定义inception模块第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, kernel=1)
        
        # 定义inception模块第二条线路
        self.branch3x3_1 = nn.Sequential(
            conv_relu(in_channel, out2_1, kernel=1),
            conv_relu(out2_1, out2_3, kernel=3, stride=1, padding=1)
        )
        
        #定义inception模块的第三条线路
        self.branch3x3_2 = nn.Sequential(
            conv_relu(in_channel, out3_1, kernel=1),
            conv_relu(out3_1, out3_3_1, kernel=3, stride=1, padding=1),
            conv_relu(out3_3_1, out3_3_2, kernel=3, stride=1, padding=1)
        )

        # 定义inception模块第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, kernel=1)
        )

    def forward(self,x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3_1(x)
        f3 = self.branch3x3_2(x)
        f5 = self.branch_pool(x)
        
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.block1 = nn.Sequential(
            conv_relu(1, 64, kernel=7, stride=1, padding=0), 
            nn.MaxPool2d(kernel_size=5, stride=3, padding=5),    
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-3)
        )
       
        self.block2 = nn.Sequential(
            inception1_7_8(64, 32, 48, 96, 16),
            nn.BatchNorm2d(144, eps=1e-3),
            nn.ReLU(inplace=True),
            inception2(144, 48, 60, 120, 32, 4, 32, 4, 16),
            nn.BatchNorm2d(192, eps=1e-3),
            nn.ReLU(inplace=True),
            inception3_6_9(192, 160, 32, 32),
            nn.BatchNorm2d(384, eps=1e-3),
            nn.ReLU(inplace=True),
            inception4_5(384, 128, 96, 192, 32, 64, 64, 32),
            nn.BatchNorm2d(416, eps=1e-3),
            nn.ReLU(inplace=True),
            inception4_5(416, 200, 100, 200, 32, 64, 64, 32),
            nn.BatchNorm2d(496, eps=1e-3),
            nn.ReLU(inplace=True),
            inception3_6_9(496, 288, 120, 68),
            nn.BatchNorm2d(992, eps=1e-3),
            nn.ReLU(inplace=True),
            inception1_7_8(992, 240, 120, 240, 120),
            nn.BatchNorm2d(600, eps=1e-3),
            nn.ReLU(inplace=True),
            inception1_7_8(600, 300, 150, 300, 120),
            nn.BatchNorm2d(720, eps=1e-3),
            nn.ReLU(inplace=True),
            inception3_6_9(720, 360, 180, 360),
            nn.BatchNorm2d(1440, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        #flatten

        self.block3 = nn.Sequential(
            nn.Linear(3*1440*1440, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 3754),
            nn.Softmax()
        )


    def forward(self, x):

        batch_size = x.size(0)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(batch_size, -1)     #3×1440×1440
        x = self.block3(x)

        return x

if __name__=='__main__':

    device = torch.device('cuda:0')
    model = GoogLeNet().to(device)
    print(model)

    input = torch.randn(1, 64, 64, 1).to(device)
    out = model(input)
    print(out.shape)