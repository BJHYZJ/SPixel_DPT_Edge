import argparse
import os
import torch.backends.cudnn as cudnn
import cv2
import torchvision.transforms as transforms
import flow_transforms
# from scipy.ndimage import imread
# from scipy.misc import imsave, imresize
from imageio import imread
from imageio import imsave
from model.DPT import DPTSegmentationModel
from train_util import *
from loss import *
import time
import random
from IPython import embed
import skimage

import sys
sys.path.append('./third_party/cython')
from connectivity import enforce_connectivity

'''
Infer from bsds500 dataset:
author:Fengting Yang 
last modification:  Mar.14th 2019

usage:
1. set the ckpt path (--pretrained) and output
2. comment the output if do not need

results will be saved at the args.output
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='data_preprocessing',help='path to images folder')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model', default='/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/new_train_ckpt/BSD500/cerberus_3000epochs_epochSize3000_b16_lr1e-05_posW0.0003_wdecay0.0004_22_08_03_00_20_train_4_adam/model_best.tar')
parser.add_argument('--output', metavar='DIR', default='output' ,help='path to output folder')

parser.add_argument('--downsize', default=16, type=float, help='superpixel grid cell, must be same as training setting')
parser.add_argument('-b', '--batch-size', default=1, type=int,  metavar='N', help='mini-batch size')

# the BSDS500 has two types of image, horizontal and veritical one, here I use train_img and input_img to presents them respectively
parser.add_argument('--train_img_height', '-t_imgH', default=320,  type=int, help='img height must be 16*n')
parser.add_argument('--train_img_width', '-t_imgW', default=480,  type=int, help='img width must be 16*n')
parser.add_argument('--input_img_height', '-v_imgH', default=480,   type=int, help='img height_must be 16*n')
parser.add_argument('--input_img_width', '-v_imgW', default=320,    type=int, help='img width must be 16*n')

args = parser.parse_args()
args.test_list = args.data_dir + '/test.txt'

random.seed(100)
@torch.no_grad()
def test(model, img_paths, save_path, spixeIds, idx, scale):
    """
    test
    :param model: 模型
    :param img_paths: 测试图片的所有路径
    :param save_path: 图片保存的路径
    :param spixeIds: [spixlId_1，spixlId_2]表示横着或者竖着的对于图片的处理 torch.Size([1, 9, 96, 144]) or torch.Size([1, 9, 144, 96])
    :param idx: 第几张图片
    :param scale: scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 ,1.6, 1.8]
    :return:
    """
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = img_paths[idx]  # img_file表示图片路径
    load_path = img_file
    # 图片名称
    imgId = os.path.basename(img_file)[:-4]

    # origin size 481*321 or 321*481
    img_ = imread(load_path)  # (481, 321, 3)
    H_, W_, _ = img_.shape

    # choose the right spixelIndx
    if H_ == 321 and W_==481:
        spixl_map_idx_tensor = spixeIds[0]
        img = cv2.resize(img_, (int(480 * scale), int(320 * scale)), interpolation=cv2.INTER_CUBIC)
    elif H_ == 481 and W_ == 321:
        spixl_map_idx_tensor = spixeIds[1]
        img = cv2.resize(img_, (int(320 * scale), int(480 * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        print('The image size is wrong!')
        return


    # img1.shape    torch.Size([3, 144, 96])
    # ori_img.shape torch.Size([3, 481, 321])
    img1 = input_transform(img)  #                                                   img1.shape = torch.Size([3, 144, 96])
    ori_img = input_transform(img_)   # img_是原图, ori_img是原图img_通过归一化之后的图片  ori_img.shape = torch.Size([3, 481, 321])

    # mean_values.shape torch.Size([3, 1, 1])
    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
    """
        tensor([[[0.4110]],
            [[0.4320]],
            [[0.4500]]])
    """
    # compute output
    tic = time.time()
    # embed(header="===============================test===================================")
    # img1.cuda().unsqueeze(0) == torch.Size([1, 3, 144, 96])
    # output.shape             == torch.Size([1, 9, 144, 96])
    output = model(img1.cuda().unsqueeze(0))  # output 和 img1.cuda().unsqueeze(0)的纬度值相同

    # assign the spixel map and  resize to the original size
    # spixl_map_idx_tensor.shape == torch.Size([1, 9, 96, 144]), 是从shift9pos函数中得到的九个方位的坐标信息，相当于原图信息
    # output.shape               == torch.Size([1, 9, 96, 144]), 相当于原图经过model后得到的数据
    # curr_spixl_map.shape       == torch.Size([1, 1, 144, 96]), 先将模型输出的结果output中按照dim=1获取最大值，
    # 并用这个最大值矩阵和output进行where运算，获取除了最大值所在位置为原值外，其它位置全为0的矩阵assignment_
    # 再用assignment_和9位置组合图像相乘，获取new_spixl_map(1, 9, 144, 96)
    # 最后通过torch.sum，按照dim=1的形式将维度1的内容相加得到new_spixl_map(1, 1, 144, 96)
    curr_spixl_map = update_spixl_map(spixl_map_idx_tensor, output)

    # 将curr_spixl_map resize成原图片的大小(1, 1, 481, 321)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)

    # 将ori_sz_spixel_map维度值为1的维度删除，spix_index_np.shape(481, 321)
    spix_index_np = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0, 1)
    # spix_index_np.shape(481, 321)
    spix_index_np = spix_index_np.astype(np.int64)
    # 每个像素的大小         （481 * 321）/(int(600 * scale * scale)*1.0) == 2859.27777777777778
    segment_size = (spix_index_np.shape[0] * spix_index_np.shape[1]) / (int(600 * scale * scale) * 1.0)
    # 最小值和最大值
    min_size = int(0.06 * segment_size)  # 171
    max_size = int(3 * segment_size)  # 8577

    # spix_index_np[None, :, :].shape (1, 481, 321)
    # spixel_label_map.shape(481, 321)
    spixel_label_map = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]

    # embed(header="============================ test 1 ====================================")
    torch.cuda.synchronize()
    toc = time.time() - tic

    n_spixel = len(np.unique(spixel_label_map))  # n_spixel为57

    # ori_img是原图img_通过归一化之后的图片
    # mean_values: tensor([[[0.4110]],
    #                     [[0.4320]],
    #                    [[0.4500]]])
    # clamp()函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
    # 这里将所有的数值压缩到0~1之间，如果数组大于1，则表示为1，如果小于0，则表示为0
    # given_img_np.shape(481, 321, 3)
    given_img_np = (ori_img + mean_values).clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)

    # (given_img_np / np.max(given_img_np)).shape == (481, 321, 3)
    # spixel_label_map.shape == (481, 321)
    spixel_bd_image = mark_boundaries(given_img_np / np.max(given_img_np), spixel_label_map.astype(int), color=(0, 1, 1))
    spixel_viz = spixel_bd_image.astype(np.float32).transpose(2, 0, 1)  # 转换为原来的图片

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
    imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

    # save the unique maps as csv for eval
    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))
    output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
    # plus 1 to make it consistent with the toolkit format
    np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i', delimiter=",")
    # embed(header="============================ test 2 ====================================")

    if idx % 10 == 0:
        print("processing %d"%idx)

    return toc, n_spixel

def main():
    
    global args, save_path

    test_name = '/test_4'  # remember modify it
    
    args.output = 'output/' + test_name

    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    train_img_height = args.train_img_height
    train_img_width = args.train_img_width
    input_img_height = args.input_img_height
    input_img_width = args.input_img_width

    mean_time_list = []
    # The spixel number we test
    # for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 ,1.6, 1.8]:
    for scale in [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]:
        assert (320 * scale % 16 == 0 and 480 * scale % 16 == 0)
        save_path = args.output + '/bsd/test_multiscale_enforce_connect/SPixelNet_nSpixel_{0}'.format(int(20 * scale * 30 * scale  ))

        args.train_img_height, args.train_img_width = train_img_height * scale, train_img_width * scale
        args.input_img_height, args.input_img_width = input_img_height * scale, input_img_width * scale

        print('=> will save everything to {}'.format(save_path))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        tst_lst = []
        with open(args.test_list, 'r') as tf:
            img_path = tf.readlines()
            for path in img_path:
                tst_lst.append(path[:-1])

        print('{} samples found'.format(len(tst_lst)))

        # create model
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained model '{}'".format(network_data['arch']))

        single_model = DPTSegmentationModel(9, backbone="vitb_rn50_384")
        model = single_model.cuda()

        print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        args.arch = network_data['arch']
        cudnn.benchmark = True
        # embed(header="-----------------------------------------------------------")
        # for vertical and horizontal input seperately
        spixlId_1, _ = init_spixel_grid(args, b_train=True)
        spixlId_2, _ = init_spixel_grid(args, b_train=False)
        # embed(header="-----------------------------------------------------------")

        mean_time = 0
        # the following code is for debug
        for n in range(len(tst_lst)):
          time, n_spixel = test(model, tst_lst, save_path, [spixlId_1, spixlId_2], n, scale)
          mean_time += time
        mean_time /= len(tst_lst)
        mean_time_list.append((n_spixel, mean_time))
        # embed(header="-----------------------------------------------------------")
        print("for spixel number {}: with mean_time {} , generate {} spixels".format(int(20 * scale * 30 * scale), mean_time, n_spixel))

    with open(args.output + '/bsd/test_multiscale_enforce_connect/mean_time.txt', 'w+') as f:
        for item in mean_time_list:
            tmp = "{}: {}\n".format(item[0], item[1])
            f.write(tmp)


if __name__ == '__main__':
    main()


# test
# CUDA_VISIBLE_DEVICES=2 python test_bsd.py