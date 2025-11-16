import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

#导入测试图片、模型权重。模型以及保存的路径
input='test'
model_wb='model_pretrained/final_model.pth'
from model import AODnet
output='result_jpg'
os.makedirs(output,exist_ok=True)

#加载模型
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('当前使用',device)
net=AODnet().to(device)
state_dict = torch.load(model_wb, map_location=device)
net.load_state_dict(state_dict)
net.eval()


# 获取当前时间，开始计时
start_time = time.time()
#加载图片
transform = transforms.ToTensor()
for haze_nmae in os.listdir(input):
    haze_path=os.path.join(input,haze_nmae)
    # 加载图片并转换为 RGB
    haze = Image.open(haze_path).convert("RGB")
    # 转换为 Tensor 并加上 batch 维度 (1,3,H,W)
    haze_tensor = transform(haze).unsqueeze(0).to(device)
    #用模型推理
    with torch.no_grad():
        gt = net(haze_tensor)  # 推理
        gt = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转回 NumPy 数组
    # 限制像素值范围为 [0, 1]
    gt = np.clip(gt, 0, 1)
    # 保存结果
    gt_img = (gt * 255).astype(np.uint8)  # 转换为 [0, 255] 范围
    gt_name=haze_nmae+'_result.jpg'
    result_path = os.path.join(output, gt_name)
    Image.fromarray(gt_img).save(result_path)
    print(f"已保存去雾图：{result_path}")

# 获取结束时间并计算耗时
end_time = time.time()
total_time = end_time - start_time

print(f"\n所有图片处理完成,耗时 {total_time:.2f} 秒")