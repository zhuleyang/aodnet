import os
import numpy as np
import h5py
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import AODnet

#配置路径
train_dir='nyu_data/train'#训练集路径
val_dir='nyu_data/val'#测试集路径
device='cuda' if torch.cuda.is_available() else 'cpu'#训练设备
save_dir='model_pretrained'
os.makedirs(save_dir,exist_ok=True)

#超参数
batch_size=32#训练批次
num_workers=4#4线程加载
epochs=30#训练轮次
learn_rate=0.0001#学习率

#设置随机数种子（保证结果可复现）
seed=42
random.seed(seed)                           # Python 随机数种子
torch.manual_seed(seed)                     # PyTorch CPU 随机数
torch.cuda.manual_seed_all(seed)            # PyTorch GPU 随机数


#读取训练数据
class H5Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted(os.listdir(root))  # 当前目录下所有 h5 文件名

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        每次只读取一个 h5，节省内存
        """
        fname = self.files[idx]
        path = os.path.join(self.root, fname)

        # 读取 h5
        with h5py.File(path, 'r') as f:
            haze = np.array(f["haze"], dtype=np.float32)   # (H,W,3)
            gt   = np.array(f["gt"], dtype=np.float32)     # (H,W,3)

        # 转成 PyTorch tensor
        haze = torch.tensor(haze, dtype=torch.float32).permute(2, 0, 1)
        gt = torch.tensor(gt, dtype=torch.float32).permute(2, 0, 1)

        return haze, gt

train_data = H5Dataset(train_dir)
val_data   = H5Dataset(val_dir)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

if __name__ == "__main__":
    print("训练集数量：", len(train_data))
    print("验证集数量：", len(val_data))

    #初始化模型
    model = AODnet().to(device)               # 模型放到 device（GPU/CPU）
    #加载旧权重（不会重新创建模型）
    '''
    model_path = os.path.join(save_dir, "final_model.pth")
    if os.path.exists(model_path):
        print("加载已训练好的模型继续训练……")
        model.load_state_dict(torch.load(model_path, map_location=device))
    '''
    criterion = nn.MSELoss()                  # MSE Loss（AOD-Net 经典选择）
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)#学习率与优化器配置

    #训练函数
    loss_history = []      # 保存每轮 loss 画图用
    for epoch in range(epochs):
        #训练
        model.train()
        train_loss = 0
        for haze, gt in train_loader:
            haze = haze.to(device)
            gt   = gt.to(device)
            optimizer.zero_grad()               # 梯度清零
            output = model(haze)                # 前向传播（不需要写 forward）
            loss = criterion(output, gt)        # 计算损失
            loss.backward()                     # 反向传播
            optimizer.step()                    # 更新参数

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        loss_history.append(avg_train)
        print(f"[Epoch {epoch+1}/{epochs}] 训练损失: {avg_train:.6f}")


        #验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for haze, gt in val_loader:
                haze = haze.to(device)
                gt   = gt.to(device)

                output = model(haze)
                loss = criterion(output, gt)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"                 验证损失: {avg_val:.6f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"模型已保存：{save_dir}")

    # 画 Loss 曲线图
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()