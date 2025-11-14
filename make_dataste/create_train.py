'''
创建训练集（i，j）
'''
#python3不需要这个from __future__ import division
import os
import numpy as np
import cv2
import h5py
from random import uniform#用于随机生成A和beta

#加载nyu_depth数据
def load_nyu(mat_path):
    nyu=h5py.File(mat_path,'r')
    images=nyu['images']
    depths=nyu['depth']
    return images,depths

#处理rgb
def process_rgb(image):
    #如果原格式是（3，H，W），转换一下
    if image.shape[0]==3:
        image=np.transpose(image,(1,2,0))
    #处理格式保证图片是480×640
    image=image.astype(np.uint8)
    image=cv2.resize(image,(640,480))
    #归一化
    image=image.astype(np.float32)/255.0
    return image

#处理depth
def process_depth(depth):
    dpt=dpt/dpt.max()
    #因为opencv读取出来的是w×H，所以这个保证和他一样
    dpt=np.transpose(dpt,(1,0))
    return dpt

#创建雾图，I(x)=J(x)t(x)+A(1−t(x))