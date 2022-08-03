import argparse
import os
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import flow_transforms
# from scipy.ndimage import imread
# from scipy.misc import imsave
import torchvision

from loss import *
import time
import random
from glob import glob

import matplotlib.pyplot as plt
# from matplotlib.image import imread
# from matplotlib.image import imsave

from imageio import imread
from imageio import imsave

from torchvision import models
from model.models_new import CerberusSegmentationModelMultiHead
from model.DPT import DPTSegmentationModel
from train_util import *

# import sys
# sys.path.append('../cython')
# from connectivity import enforce_connectivity


'''
Infer from custom dataset:
author:Fengting Yang 
last modification: Mar.5th 2020

usage:
1. set the ckpt path (--pretrained) and output
2. comment the output if do not need

results will be saved at the args.output

'''

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

# print(model_names)


parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='./demo/inputs', help='path to images folder')
parser.add_argument('--data_suffix',  default='jpg', help='suffix of the testing image')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model',
                                    default='/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/new_train_ckpt/BSD500/cerberus_3000epochs_epochSize3000_b16_lr5e-06_posW0.003_wdecay0.0001_22_08_03_00_11_train_2_adam/model_best.tar')
parser.add_argument('--output', metavar='DIR', default= './demo' , help='path to output folder')

parser.add_argument('--downsize', default=16, type=float,help='superpixel grid cell, must be same as training setting')

parser.add_argument('-nw', '--num_threads', default=1, type=int,  help='num_threads')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

args = parser.parse_args()

random.seed(100)
@torch.no_grad()
def test(args, model, img_paths, save_path, idx):
      # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    img_ = imread(load_path)[:, :, :3]
    H, W, _ = img_.shape
    H_, W_  = int(np.ceil(H/32.)*32), int(np.ceil(W/32.)*32)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / args.downsize))
    n_spixl_w = int(np.floor(W_ / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    n_spixel =  int(n_spixl_h * n_spixl_w)


    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute output
    tic = time.time()
    output = model(img1.cuda().unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    # curr_spixl_map = torch.sum(spixeIds, dim=1, keepdim=True).type(torch.int)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( H_,W_), mode='nearest').type(torch.int)

    # embed(header='----------------------------run_demo.py------------------------')

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    # 内部使用到了 enforce_connectivity
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel, b_enforce_connect=True)

    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    # img_save = (ori_img + mean_values).clamp(0, 1)
    # imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))


    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    # embed(header="------------------------------run_demo.py--------------------------------------")
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))  # 使用transpose将图片由(2,0,1)转换回原来的样子
    # imsave(spixl_save_name + "_mask", spixel_label_map)
    # save the unique maps as csv, uncomment it if needed

    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))
    output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
      # plus 1 to make it consistent with the toolkit format
    np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i',delimiter=",")


    if idx % 10 == 0:
        print("processing %d"%idx)

    return toc

def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    save_path = args.output
    print('=> will save everything to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    tst_lst = glob(args.data_dir + '/*.' + args.data_suffix)
    tst_lst.sort()

    if len(tst_lst) == 0:
        print('Wrong data dir or suffix!')
        exit(1)

    print('{} samples found'.format(len(tst_lst)))

    # create model
    network_data = torch.load(args.pretrained)  # 加载模型SpixelNet_bsd_ckpt.tar
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    print("network ==", network_data['arch'])
    # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    # model = single_model.cuda()

    single_model = DPTSegmentationModel(9, backbone="vitb_rn50_384")
    model = single_model.cuda()
    model.load_state_dict(network_data['state_dict'])
    '''
        model.eval() 作用等同于 self.train(False)
        简而言之，就是评估模式。而非训练模式。
        在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
    '''
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    mean_time = 0
    for n in range(len(tst_lst)):
      time = test(args, model, tst_lst, save_path, n)
      mean_time += time
    print("avg_time per img: %.3f"%(mean_time/len(tst_lst)))

if __name__ == '__main__':
    main()
