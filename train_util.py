# -*- Condeing = utf-8 -*-
# @Time : 2022/7/9 10:47
# Author : Banner(Zhijie Yan)
# @File : train_util.py
# @software : PyCharm

import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import mark_boundaries

import cv2
from IPython import embed

from numpy.lib import arraypad

import sys
sys.path.append('./third_party/cython')
from connectivity import enforce_connectivity

# 计算并存储平均值和当前值
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


class ConcatSegList(torch.utils.data.Dataset):
    def __init__(self, at, af, seg):
        self.at = at
        self.af = af
        self.seg = seg

    def __getitem__(self, index):
        # 三种类型的数据集的长度均为1449，这里按照索引值，每次从数据集中取出相同下标的数据
        return (self.at[index], self.af[index], self.seg[index])

    def __len__(self):
        return len(self.at)


"""<========================== mIoUAll===========================>"""
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def mIoUAll(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious), 2)
"""<========================== mIoUAll===========================>"""


"""<========================== superpixel fcn ===========================>"""
def init_spixel_grid(args, b_train=True):
    if b_train:
        # train_img_height、train_img_width default default = 208
        img_height, img_width = args.train_img_height, args.train_img_width
    else:
        img_height, img_width = args.input_img_height, args.input_img_width

    # get spixel id for the final assignment
    # np.floor()向下取整
    # downsize default = 16
    # n_spixl_h, n_spixl_w表示初始化超像素的大小
    n_spixl_h = int(np.floor(img_height/args.downsize))
    n_spixl_w = int(np.floor(img_width/args.downsize))

    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))

    # 为superpixel中像素进行编号
    spixel_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    # 获取超像素的九个方位
    # spixel_idx_tensor_的大小和spixel_values在平面上是一样大的，但是在空间上就具有了九个位置的不同性
    spixel_idx_tensor_ = shift9pos(spixel_values)  # (9, 16, 16)

    # embed(header="------------------------------ train_util 1 -----------------------------------")

    # np.repeat，分别对axis=1, axis=2的维度进行扩张，比如对行扩张到原来的208倍，则对每一个行数据都连在一起输出208次
    spix_idx_tensor = np.repeat(
        np.repeat(spixel_idx_tensor_, spixel_height, axis=1), spixel_width, axis=2)

    # np.tile表示在维度上进行复制，默认为最后一个维度，
    # torch_spix_idx_tensor的宽和高就是图片的宽和高，只是多了一个维度，且该维度表示九个位置
    torch_spix_idx_tensor = torch.from_numpy(
                np.tile(spix_idx_tensor, (args.batch_size, 1, 1, 1))).type(torch.float).cuda()


    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    # pixel coord
    # np.meshgrid：X, Y = np.meshgrid(x, y) 代表的是将x中每一个数据和y中每一个数据组合生成很多点,
    # 然后将这些点的x坐标放入到X中,y坐标放入Y中,并且相应位置是对应的
    # coords: 坐标
    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pixel_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))

    # 反过来了？？？？为什么？？？？，第一维度为2，下面的操作将第一维度对于的两个数据交换了顺序位置
    coord_tensor = np.concatenate([curr_pixel_coord[1:2, :, :], curr_pixel_coord[:1, :, :]])

    # (20, 2, 256, 256)
    all_XY_feat = (torch.from_numpy(
        np.tile(coord_tensor, (args.batch_size, 1, 1, 1)).astype(np.float32)).cuda())

    # embed(header="------------------------------ train_util 2 -----------------------------------")

    return torch_spix_idx_tensor, all_XY_feat  # 对于每个

#===================== pooling and upsampling feature ==========================================
# (n_spixl_h, n_spixl_w)
def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
    # input should be padding as (c, 1+height+1, 1+width+1)
    # ndarray = numpy.pad(array, pad_width, mode, **kwargs)
    # mode='edge'表示用边缘值填充
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)  # 在0维度上再增加一个维度

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor

def label2one_hot_torch(labels, C=14):
    # w.r.t http://jacobkimmel.github.io/pytorch_onehot/
    '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        # 将整数标签torch.autograd.Variable转换为 独热编码值
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
            # labels中每个值都是一个整数，表示正确的分类
        C : integer.
            number of classes in labels.
            C表示标签中类的数量

        Returns
        -------
        target : torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
    b, _, h, w = labels.shape
    one_hot = torch.zeros(b, C, h, w, dtype=torch.long).cuda()
    # 对labels.data中，值>0的位置填充1，否则还是为0
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1)  # require long type
    # embed(header="------------------------------label--------------------------------")

    return target.type(torch.float32)

def build_LABXY_feat(label_in, xy_coords):

    img_lab = label_in.clone().type(torch.float)
    # img_lab的维度为(20, 50, 256, 256)
    b, _, curr_img_height, curr_img_width = xy_coords.shape
    # F.interpolate 根据给定 size 或 scale_factor，上采样或下采样输入数据input
    scale_img = F.interpolate(img_lab, size=(curr_img_height, curr_img_width), mode='nearest')
    LABXY_feat = torch.cat([scale_img, xy_coords], dim=1)

    # embed(header='------------------build labxy feat ------------------')
    return LABXY_feat


def update_spixl_map (init_spixl_map_idx, output):

    model_output = output.clone()
    # torch.Size([20, 9, 256, 256])
    b, _, h, w = model_output.shape
    _, _, id_h, id_w = init_spixl_map_idx.shape

    if (id_h == h) and (id_w == w):
        spixel_map_idx = init_spixl_map_idx
    else:
        spixel_map_idx = F.interpolate(init_spixl_map_idx, size=(h, w), mode='nearest')

    model_output_max, _ = torch.max(model_output, dim=1, keepdim=True)
    assignment_ = torch.where(model_output == model_output_max, torch.ones(model_output.shape).cuda(),torch.zeros(model_output.shape).cuda())
    new_spixl_map_ = spixel_map_idx * assignment_ # winner take all
    new_spixl_map = torch.sum(new_spixl_map_, dim=1, keepdim=True).type(torch.int)

    # embed(header='------------------------update_spixel_map-------------------------------------')
    return new_spixl_map

def get_spixel_image(given_img, spix_index, n_spixels=600, b_enforce_connect=False):

    if not isinstance(given_img, np.ndarray):
        given_img_np_ = given_img.detach().cpu().numpy().transpose(1,2,0)
    else: # for cvt lab to rgb case
        given_img_np_ = given_img

    if not isinstance(spix_index, np.ndarray):
        spix_index_np = spix_index.detach().cpu().numpy().transpose(0,1)
    else:
        spix_index_np = spix_index


    h, w = spix_index_np.shape
    given_img_np = cv2.resize(given_img_np_, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    # embed(header='-------------------------- test spixel set --------------------------')
    if b_enforce_connect:
        spix_index_np = spix_index_np.astype(np.int64)
        segment_size = (given_img_np_.shape[0] * given_img_np_.shape[1]) / (int(n_spixels) * 1.0)
        min_size = int(0.06 * segment_size)
        max_size =  int(3 * segment_size)
        spix_index_np = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]
    cur_max = np.max(given_img_np)
    spixel_bd_image = mark_boundaries(given_img_np/cur_max, spix_index_np.astype(int), color = (0,1,1))
    return (cur_max*spixel_bd_image).astype(np.float32).transpose(2, 0, 1), spix_index_np #
"""<========================== superpixel fcn ============================>"""