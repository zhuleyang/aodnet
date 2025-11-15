import  torch
#导入神经网络模块
import torch.nn as nn
#导入激活函数模块
import torch.functional as F


class AODnet(nn.Module):
    def __init__(self):
        #父类初始化
        super(AODnet,self).__init__()
        #第一层用一乘一的卷积核提取简单的线性特征
        self.conv1=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1)
        #第二层用三乘三的卷积核捕捉细节
        self.conv2=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,padding=1)
        #每一个卷积层都保证图像尺寸不变,第3层为前两个相加
        self.conv3=nn.Conv2d(in_channels=6,out_channels=3,kernel_size=5,padding=2)
        #继续为前两个之和
        self.conv4=nn.Conv2d(in_channels=6,out_channels=3,kernel_size=7,padding=3)
        #前面所有z卷积输出之和
        self.conv5=nn.Conv2d(in_channels=12,out_channels=3,kernel_size=3,padding=1)
        #定义偏执
        self.b=1
    #定义前向传播函数
    def forward(self,x):
        x1=F.relu(self.conv1(x))
        x2=F.relu(self.conv2(x1))
        #拼接前两层结果,shape=(b,3,h,w),在通道维数拼接，由3个特征层变为6个
        cat1=torch.cat((x1,x2),dim=1)
        x3=F.relu(self.conv3(cat1))
        #第二次拼接
        cat2=torch.catz((x2,x3),1)
        x4=F.relu(self.conv4(cat2))
        #最后一层拼接前面全部
        cat3=torch.cat((x1,x2,x3,x4),1)
        k=F.relu(self.conv5(cat3))
        #判断输出尺寸是否正确,有问题立刻报错
        if k.size() != x.size():
            raise Exception('k和雾图的尺寸不一样')
        #计算输出图像
        output=k*x-k+self.b
        return  F.relu(output)
