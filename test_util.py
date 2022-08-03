# -*- Condeing = utf-8 -*-
# @Time : 2022/7/10 13:29
# Author : Banner(Zhijie Yan)
# @File : test_util.py
# @software : PyCharm

import argparse
import json
import logging
import math
import os
# import pdb
from os.path import exists, join, split

from datetime import datetime

import time

import numpy as np
import shutil
import threading
import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.modules import transformer
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter


import drn
import data_transforms as transforms
from model.models_new import CerberusSegmentationModelMultiHead
from train_util import *
from loss import compute_semantic_pos_loss

try:
    from modules import batchnormsync
except ImportError:
    pass

FILE_DESCRIPTION = ''

class SegListMSMultiHead(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir  # data_dir = dataset
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()  # fcn has been run
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = 640, 480
        data = np.array(data[0])
        # ================== 不太理解 ====================
        if len(data.shape) == 2:
            data = np.stack([data, data, data], axis=2)

        data = [Image.fromarray(data)]
        label_data = list()
        if self.label_list is not None:
            for it in self.label_list[index].split(','):
                label_data.append(Image.open(join(self.data_dir, it)))
            data.append(label_data)
        out_data = list(self.transforms(*data))
        # round return lower than 0.4 out/ higher than 0.5 in
        ms_images = [self.transforms(data[0].resize((round(int(w * s)/32) * 32, round(int(h * s)/32) * 32),
                                                    Image.BICUBIC))[0] for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + FILE_DESCRIPTION + '_images.txt')
        label_path = join(self.list_dir, self.phase + FILE_DESCRIPTION + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
