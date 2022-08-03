from __future__ import division
import os.path
from .listdataset import  ListDataset

import numpy as np
import flow_transforms
from IPython import embed

try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

'''
Data load for bsds500 dataset:
author:Fengting Yang 
Mar.1st 2019

usage:
1. manually change the name of train.txt and val.txt in the make_dataset(dir) func. 
# 手动修改make_dataset(dir)函数中train.txt和val.txt的名称。  
2. ensure the val_dataset using the same size as the args. in the main code when performing centerCrop default value is 320*320, it is fixed to be 16*n in our project
# 确保val_dataset使用与args相同的大小。在主代码中执行centerCrop的默认值是320*320，在我们的项目中它被固定为16*n
'''

def make_dataset(dir):
    # we train and val seperately to tune the hyper-param and use all the data for the final training
    train_list_path = os.path.join(dir, 'train.txt') # use train_Val.txt for final report
    val_list_path = os.path.join(dir, 'val.txt')

    try:
        # 获取txt文件中的 训练数据集路径 和 测试数据集路径
        with open(train_list_path, 'r') as tf:
            train_list = tf.readlines()

        with open(val_list_path, 'r') as vf:
            val_list = vf.readlines()

    except IOError:
        print ('Error No avaliable list ')
        return

    return train_list, val_list



def BSD_loader(path_imgs, path_label):
    # 根据图片路径加载数据集
    # cv2.imread is faster than io.imread usually
    img = cv2.imread(path_imgs)[:, :, ::-1].astype(np.float32)
    gtseg = cv2.imread(path_label)[:, :, :1]

    # embed(header="BSD_loader----------------------------------------")
    return img, gtseg


def BSD500(root, transform=None, target_transform=None, val_transform=None,
              co_transform=None, split=None):
    train_list, val_list = make_dataset(root)  # get train_data path_list and test_data path_list

    if val_transform ==None:
        val_transform = transform

    train_dataset = ListDataset(root, 'bsd500', train_list, transform,
                                target_transform, co_transform,
                                loader=BSD_loader, datatype = 'train')

    val_dataset = ListDataset(root, 'bsd500', val_list, val_transform,
                               target_transform, flow_transforms.CenterCrop((320,320)),
                               loader=BSD_loader, datatype='val')
    # embed(header="BSD500--------------------------------------------------")
    return train_dataset, val_dataset


