'''
创建训练集（i，j）
'''
#python3不需要这个from __future__ import division
import os
import numpy as np
import cv2
import h5py
from random import uniform #用于随机生成A和beta

#加载nyu_depth数据
def load_nyu(mat_path):
    nyu=h5py.File(mat_path,'r')
    images=nyu['images']
    depths=nyu['depths']
    return images,depths

#处理rgb
def process_rgb(image):
    #如果原格式是（3，H，W），转换一下
    if image.shape[0]==3:
        image=np.transpose(image,(1,2,0))
    #处理格式保证图片是480×640
    image=image.astype(np.uint8)
    image=cv2.resize(image,(480, 640))#注意resize的格式不一样
    #归一化
    image=image.astype(np.float32)/255.0
    return image

#处理depth
def process_depth(dpt):
    dpt = cv2.resize(dpt, (480, 640))
    #归一化
    dpt=dpt.astype(np.float32)/dpt.max()
    #因为opencv读取出来的是w×H，所以这个保证和他一样
    return dpt

#创建雾图，I(x)=J(x)t(x)+A(1−t(x))
def synthesize_haze(gt,depth,j,k):
    #gt：无雾 RGB 图像
    #depth：深度图，值越大表示越远
    #j：控制雾的浓度组别（影响 β）
    #：控制大气光强度组别（影响 A）
    bias=0.05#随机扰动
    abias=0.01
    base_beta=0.4+0.2*j#基础beta
    beta=uniform(base_beta-bias,base_beta+bias)
    base_A=0.5+0.2*k
    a=uniform(base_A-abias,base_A+abias)
    t=np.exp(-beta*depth)
    #统一尺寸
    H,W,_=gt.shape
    t=np.tile(t.reshape(H,W,1),(1,1,3))
    A = np.tile(np.array([a], dtype=np.float32).reshape(1, 1, 1), (H, W, 3))
    haze=gt*t+(1-t)*A
    return haze

#保存可视化样本
def save_demo(haze,gt,demo_dir):
    cv2.imwrite(os.path.join(demo_dir,'haze.jpg'),(haze*255).astype(np.uint8))
    cv2.imwrite(os.path.join(demo_dir,'gt.jpg'),(gt*255).astype(np.uint8))

#保存h5文件
def save_h5(haze,gt,save_dir,idx):
    h5_path=os.path.join(save_dir,f'{idx}.h5')
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("haze", data=haze)
        f.create_dataset("gt", data=gt)
    #12.h5
    #├── haze : [H, W, 3]的数组
    #└── gt   : [H, W, 3]的数组
    return

def main(nyu_path,output_dir):
    mat_path=os.path.join(nyu_path)
    #创建目录
    train=os.path.join(output_dir,'train')
    val=os.path.join(output_dir,'val')
    demo=os.path.join(output_dir,'demo')
    os.makedirs(train,exist_ok=True)
    os.makedirs(val,exist_ok=True)
    os.makedirs(demo,exist_ok=True)
    #读取数据
    images,depths=load_nyu(mat_path)

    #设置处理图片数量
    total = 0
    for idx in range(100):
        gt = process_rgb(images[idx])  # (480,640,3) float32
        dpt = process_depth(depths[idx])  # (H,W) float32, 已归一化
        for j in range(7):
            for k in range(3):
                haze = synthesize_haze(gt, dpt, j, k)
                total+=1
                save_demo(haze,gt,demo)
                save_h5(haze,gt,train,total)
                if total%21==0:
                    print(f'完成{total/21}组')
    print('train和demo生成完成')
    total=0
    for idx in range(100,120):
        gt = process_rgb(images[idx])  # (480,640,3) float32
        dpt = process_depth(depths[idx])  # (H,W) float32, 已归一化
        for j in range(7):
            for k in range(3):
                haze = synthesize_haze(gt, dpt, j, k)
                total+=1
                save_demo(haze,gt,demo)
                save_h5(haze,gt,val,total)
                if total%21==0:
                    print(f'完成{total/21}组')
    print('val生成完成')
if __name__=='__main__':
    nyu_path=r'nyu_depth_v2_labeled.mat'
    output_dir=r'D:\study\aodnet\nyu_data'
    main(nyu_path, output_dir)
                
