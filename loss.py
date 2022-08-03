import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from IPython import embed

'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''

def compute_semantic_pos_loss(output, lab_xyCoords_tensor_50_2, pos_weight=0.003, kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    model_output = output.clone()

    b, c, h, w = lab_xyCoords_tensor_50_2.shape  # torch.Size([4, 52, 256, 256])
    pooled_labxy = poolfeat(lab_xyCoords_tensor_50_2, model_output, kernel_size, kernel_size)
    # reconstruct_feat重建特征
    reconstruct_feat = upfeat(pooled_labxy, model_output, kernel_size, kernel_size)

    # embed(header="-------------------------------first step-----------------------------")

    loss_position_map = reconstruct_feat[:, -2:, :, :] - lab_xyCoords_tensor_50_2[:, -2:, :, :]
    # loss_map.shape == torch.Size([16, 2, 256, 256])

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstruct_feat[:, :-2, :, :] + 1e-8)
    # loss_map == torch.Size([16, 50, 256, 256])

    # labxy_feat.shape: torch.Size([16, 50, 256, 256])
    loss_sem = -torch.sum(logit * lab_xyCoords_tensor_50_2[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_position_map, p=2, dim=1).sum() / b * m / S
    # embed(header="-------------------------------second step-----------------------------")

    # empirically we find timing 0.005 tend to better performance
    loss_sum = 0.005 * (loss_sem + loss_pos)
    loss_sem_sum = 0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    # embed(header="-------------------------------third step-----------------------------")
    return loss_sum, loss_sem_sum, loss_pos_sum

# lab_xyCoords_tensor_50_2, model_output
def poolfeat(label_1hot_xyCoords_tensor_50_2, model_output, sp_h=16, sp_w=16):

    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = label_1hot_xyCoords_tensor_50_2.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([label_1hot_xyCoords_tensor_50_2, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) # b * (n+1) * h* w
    send_to_top_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()
    # embed(header='-------------------------poolfeat 1---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)
    # embed(header='-------------------------poolfeat 2---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)
    # embed(header='-------------------------poolfeat 3---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)
    # embed(header='-------------------------poolfeat 4---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)
    # embed(header='-------------------------poolfeat 5---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)
    # embed(header='-------------------------poolfeat 6---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)
    # embed(header='-------------------------poolfeat 7---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)
    # embed(header='-------------------------poolfeat 8---------------------------------')

    prob_feat = F.avg_pool2d(feat_ * model_output.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)
    # embed(header='-------------------------poolfeat 9---------------------------------')

    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat

# pooled_labxy, model_output
def upfeat(pooled_labxy, model_output, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = pooled_labxy.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(pooled_labxy, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
    feat_sum = gt_frm_top_left * model_output.narrow(1,0,1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * model_output.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * model_output.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * model_output.narrow(1, 3, 1)

    center = F.interpolate(pooled_labxy, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * model_output.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * model_output.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * model_output.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * model_output.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * model_output.narrow(1, 8, 1)

    return feat_sum