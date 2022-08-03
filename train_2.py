# -*- Condeing = utf-8 -*-
# @Time : 2022/7/7 13:11
# Author : Banner(Zhijie Yan)
# @File : train.py
# @software : PyCharm
import argparse
import logging
import os
import datetime
import time
import shutil
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import drn
from model.models_new import CerberusSegmentationModelMultiHead
from model.DPT import DPTSegmentationModel
from train_util import *
from loss import compute_semantic_pos_loss
import flow_transforms
import datasets
from torchvision.utils import make_grid
from IPython import embed
import matplotlib.pyplot as plt

try:
    from modules import batchnormsync
except ImportError:
    pass


"""---------------------------------------------------superpixel_fcn-------------------------------------------------------"""
dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch SpixelFCN Training on BSDS500',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ================ training setting ====================
parser.add_argument('--dataset', metavar='DATASET', default='BSD500',  choices=dataset_names,
                    help='dataset type : ' + ' | '.join(dataset_names))
parser.add_argument('--arch', '-a', metavar='ARCH', default='cerberus',  help='model architecture')
parser.add_argument('--data', metavar='DIR', default='data_preprocessing', help='path to input dataset')
parser.add_argument('--savepath', default='new_train_ckpt', help='path to save ckpt')


parser.add_argument('--train_img_height', '-t_imgH', default=256,  type=int, help='img height')
parser.add_argument('--train_img_width', '-t_imgW', default=256, type=int, help='img width')
parser.add_argument('--input_img_height', '-v_imgH', default=320, type=int, help='img height_must be 16*n')  #
parser.add_argument('--input_img_width', '-v_imgW', default=320,  type=int, help='img width must be 16*n')

# ======== learning schedule ================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=3000, type=int, metavar='N', help='number of total epoches, make it big enough to follow the iteration maxmium')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch_size', default=3000,  help='choose any value > 408 to use all the train and val data')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')

# parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'], help='solver algorithms, we use adam')
parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--step', type=int, default=200)
parser.add_argument('--lr-mode', type=str, default='poly')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameter for adam')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay')
# parser.add_argument('--bias_decay', default=0, type=float, metavar='B', help='bias decay, we never use it')
parser.add_argument('--milestones', default=[200000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--additional_step', default=100000, help='the additional iteration, after lr decay')

# ============== hyper-param ====================
# pos_weight: pos项的权重
parser.add_argument('--pos_weight', '-p_w', default=0.003, type=float, help='weight of the pos term')
parser.add_argument('--downsize', default=16, type=float, help='grid cell size for superpixel training ')

# ================= other setting ===================
# parser.add_argument('--gpu', default= '0', type=str, help='gpu id')
parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (step)')
parser.add_argument('--record_freq', '-rf', default=5, type=int, help='record frequency (epoch)')
parser.add_argument('--label_factor', default=5, type=int, help='constant multiplied to label index for viz.')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true', help='don\'t append date timestamp to folder' )

"""---------------------------------------------------superpixel_fcn-------------------------------------------------------"""

namesuffix = '_train_3_adam_adjust_lr'
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename='./logger/train_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + namesuffix + '.txt')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
TASK = 'SEGMENTATION'
TRANSFER_FROM_TASK = 'SEGMENTATION'

NYU40_PALETTE = np.asarray([
    [0, 0, 0],
    [0, 0, 80],
    [0, 0, 160],
    [0, 0, 240],
    [0, 80, 0],
    [0, 80, 80],
    [0, 80, 160],
    [0, 80, 240],
    [0, 160, 0],
    [0, 160, 80],
    [0, 160, 160],
    [0, 160, 240],
    [0, 240, 0],
    [0, 240, 80],
    [0, 240, 160],
    [0, 240, 240],
    [80, 0, 0],
    [80, 0, 80],
    [80, 0, 160],
    [80, 0, 240],
    [80, 80, 0],
    [80, 80, 80],
    [80, 80, 160],
    [80, 80, 240],
    [80, 160, 0],
    [80, 160, 80],
    [80, 160, 160],
    [80, 160, 240], [80, 240, 0], [80, 240, 80], [80, 240, 160], [80, 240, 240],
    [160, 0, 0], [160, 0, 80], [160, 0, 160], [160, 0, 240], [160, 80, 0],
    [160, 80, 80], [160, 80, 160], [160, 80, 240]], dtype=np.uint8)

task_list = ['Segmentation']
FILE_DESCRIPTION = ''
PALETTE = NYU40_PALETTE
EVAL_METHOD = 'mIoUAll'

middle_task_list = ['Segmentation']

TENSORBOARD_WRITER = SummaryWriter(comment='From_'+TRANSFER_FROM_TASK+'_TO_'+TASK)

TENSORBOARD_WRITER = SummaryWriter(comment=TASK)

best_EPE = -1
n_iter = 0
args = parser.parse_args()

def train(train_loader, model, optimizer, epoch, train_writer, init_spixl_map_idx, xy_coords):
    '''

    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param train_writer:
    :param init_spixl_map_idx:
    :param xy_coords: [4, 2, 256, 256]
    :return:
    '''
    global n_iter, args, intrinsic
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_loss = AverageMeter()
    losses_sem = AverageMeter()
    losses_pos = AverageMeter()

    # 设置epoch
    # epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)
    epoch_size = len(train_loader)
    # switch to train mode
    model.train()
    end = time.time()
    iteration = 0

    for i, (input, label) in enumerate(train_loader):
        iteration = i + epoch * epoch_size
        # ================= adjust lr if necessary ================
        if(iteration + 1) in args.milestones:  # milestones 表示学习率 / 2 的轮次数量
            state_dict = optimizer.state_dict()
            for param_group in state_dict['param_groups']:
                param_group['lr'] = args.lr * ((0.5) ** (args.milestones.index(iteration + 1) + 1))
            optimizer.load_state_dict(state_dict)
        # ========== complete data loading ================

        # label_1hot中存在物体的地方像素为1， 没有物体的地方像素为0
        # label_1hot [4, 50, 256, 256]
        label_1hot = label2one_hot_torch(label.cuda(), C=50)  # set C=50 as SSN does
        input_gpu = input.cuda()
        # xy_feat is x, y coordinates
        lab_xyCoords_tensor_50_2 = build_LABXY_feat(label_1hot, xy_coords)  # B* (50+2)* H * W
        torch.cuda.synchronize()
        data_time.update(time.time() - end)

        # embed(header="---------------------------------- before loss ------------------------------------------")

        # ============== predict association map ===============
        # model == CerberusSegmentationModelMultiHead
        # model 中的参数会直接传递给forward函数
        # 只有model中存在 出现负数的问题
        output = model(input_gpu)

        # 获取loss
        slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(output, lab_xyCoords_tensor_50_2,
                                                                  pos_weight=args.pos_weight, kernel_size=args.downsize)

        # embed(header="--------------------------------------------- after loss ---------------------------------------------------")
        # =========== back propagate ============
        # 反向传播
        # 计算梯度并进行优化步骤
        # embed(header="embed optimizer third ==============================================>")
        optimizer.zero_grad()  # 每轮开始前清空优化器
        # 损失回传过程跟新梯度，得到梯度
        slic_loss.backward()
        # 梯度更新
        optimizer.step()

        # ========  measure batch time ===========
        # torch.cuda.synchronize()同步统计pytorch调用cuda的运行时间
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # =========== record and display the loss ===========
        # record loss and EPE
        # input_gpu.size(0)表示输入数据维度中的第一个维度，即batch_size，这里为48
        total_loss.update(slic_loss.item(), input_gpu.size(0))
        losses_sem.update(loss_sem.item(), input_gpu.size(0))
        losses_pos.update(loss_pos.item(), input_gpu.size(0))

        if i % args.print_freq == 0:
            print('train Epoch: [{0}/{1}][{2}/{3}]\tTime {4}\tData {5}\tTotal_loss {6}\tLoss_sem {7}\tLoss_pos {8}\t'
                  .format(epoch, args.epoch_size, i, epoch_size, batch_time, data_time, total_loss, losses_sem, losses_pos))

            logger.info('train Epoch: [{0}/{1}][{2}/{3}]\tTime {4}\tData {5}\tTotal_loss {6}\tLoss_sem {7}\tLoss_pos {8}\t'
                  .format(epoch, args.epoch_size, i, epoch_size, batch_time, data_time, total_loss, losses_sem, losses_pos))

            train_writer.add_scalar('Train_loss', slic_loss.item(), i + epoch * epoch_size)
            train_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], i + epoch * epoch_size)

        n_iter += 1
        if i >= epoch_size:
            break

        if (iteration) >= (args.milestones[-1] + args.additional_step):
            break

    # ============== write information to tensorboard ================
    if epoch % args.record_freq == 0:
        train_writer.add_scalar('Train_loss_epoch', slic_loss.item(), epoch)
        train_writer.add_scalar('loss_sem', loss_sem.item(), epoch)
        train_writer.add_scalar('loss_pos', loss_pos.item(), epoch)

        # save image
        mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=input_gpu.dtype).view(3, 1, 1)

        input_l_save = (make_grid((input + mean_values).clamp(0, 1), nrow=args.batch_size))

        # curr_spixl_map = update_spixl_map(init_spixl_map_idx, output)
        spixel_lab_save = make_grid(update_spixl_map(init_spixl_map_idx, output), nrow=args.batch_size)[0, :, :]
        # embed(header='------------------------------get_spixel_image----------------------------------')
        spixel_viz, _ = get_spixel_image(input_l_save, spixel_lab_save)

        label_save = make_grid(args.label_factor * label)
        train_writer.add_image('Input', input_l_save, epoch)
        train_writer.add_image('label', label_save, epoch)
        train_writer.add_image('Spixel viz', spixel_viz, epoch)

        print('==> write train step %dth to tensorboard' % i)
        logger.info('==> write train step %dth to tensorboard' % i)

    return total_loss.avg, losses_sem.avg, iteration

def validate(val_loader, model, epoch, val_writer, init_spixl_map_idx, xy_feat):
    global n_iter, args, intrinsic
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_loss = AverageMeter()
    losses_sem = AverageMeter()
    losses_pos = AverageMeter()

    # set the validation epoch-size, we only randomly val. 400 batches during training to save time
    epoch_size = min(len(val_loader), 400)

    # switch to train mode
    model.eval()
    end = time.time()

    for i, (input, label) in enumerate(val_loader):
        # measure data loading time
        label_1hot = label2one_hot_torch(label.cuda(), C=50)
        input_gpu = input.cuda()
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, xy_feat)  # B* 50+2 * H * W
        torch.cuda.synchronize()
        data_time.update(time.time() - end)

        # compute output
        with torch.no_grad():
            output = model(input_gpu)
            slic_loss, loss_sem, loss_pos = compute_semantic_pos_loss(output, LABXY_feat_tensor,
                                                                      pos_weight=args.pos_weight,
                                                                      kernel_size=args.downsize)

        # measure loss and EPE
        total_loss.update(slic_loss.item(), input_gpu.size(0))
        losses_sem.update(loss_sem.item(), input_gpu.size(0))
        losses_pos.update(loss_pos.item(), input_gpu.size(0))

        if i % args.print_freq == 0:
            print('val Epoch: [{0}][{1}/{2}]\tTime {3}\tData {4}\tTotal_loss {5}\tLoss_sem {6}\tLoss_pos {7}\t'
                  .format(epoch, i, epoch_size, batch_time, data_time, total_loss, losses_sem, losses_pos))
            logger.info('val Epoch: [{0}][{1}/{2}]\tTime {3}\tData {4}\tTotal_loss {5}\tLoss_sem {6}\tLoss_pos {7}\t'
                  .format(epoch, i, epoch_size, batch_time, data_time, total_loss, losses_sem, losses_pos))

    # ================== write result to tensorboard ====================
    if epoch % args.record_freq == 0:
        val_writer.add_scalar('Train_loss_epoch', slic_loss.item(), epoch)
        val_writer.add_scalar('loss_sem', loss_sem.item(), epoch)
        val_writer.add_scalar('loss_pos', loss_pos.item(), epoch)

        mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=input_gpu.dtype).view(3, 1, 1)
        input_l_save = (make_grid((input + mean_values).clamp(0, 1), nrow=args.batch_size))

        curr_spixl_map = update_spixl_map(init_spixl_map_idx, output)

        # make_grid(, nrow=n) 表示每行展示n张图片
        spixel_lab_save = make_grid(curr_spixl_map, nrow=args.batch_size)[0, :, :]
        spixel_viz, _ = get_spixel_image(input_l_save, spixel_lab_save)

        label_save = make_grid(args.label_factor * label)

        val_writer.add_image('Input', input_l_save, epoch)
        val_writer.add_image('label', label_save, epoch)
        val_writer.add_image('Spixel viz', spixel_viz, epoch)

        print('==> write val step %dth to tensorboard' % i)
        logger.info('==> write val step %dth to tensorboard' % i)

    return total_loss.avg, losses_sem.avg

def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.tar'))

# training
# 修改任务1：将三个任务改成一个任务，只保留segmentation任务
def main():

    global best_EPE, save_path, intrinsic

    print(' '.join(sys.argv))
    logger.info(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    """ ============================== add superpixel ================================ """
    # ============= savor setting ===================
    # save_path: SpixelNet1l_bn_adam_3000epochs_epochSize6000_b48_lr5e-05_posW0.003
    save_path = '{}_{}epochs{}_b{}_lr{}_posW{}_wdecay{}'.format(
        args.arch,  # 模型结构
        # args.solver,  # adam  model optimizer
        args.epochs,  # epochs
        '_epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',  # epochSize == 6000
        args.batch_size,  # batch_size = 48
        args.lr,  # 初始化学习率 learning rate = 5e-5
        args.pos_weight,  # posW == 0.003
        args.weight_decay,  # 权重衰退值
    )

    logger.info("Experimental parameters ==> batch_size: {}, lr: {}, pos_weight: {}, weight_decay: {}".format(args.batch_size, args.lr, args.pos_weight, args.weight_decay))

    print("save_path == " + save_path)

    if not args.no_date:  # don\'t append date timestamp to folder
        timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
    else:
        timestamp = ''
    # timestamp 时间戳
    save_path = os.path.abspath(args.savepath) + '/' + os.path.join(args.dataset, save_path + '_' + timestamp + namesuffix)
    '''
        transforms.Normalize()

        功能：逐channel的对图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
        output = (input - mean) / std
        mean:各通道的均值
        std：各通道的标准差
        inplace：是否原地操作
    '''
    input_transform = transforms.Compose([
        # 将数据的维度顺序变换一下，因为正常图片的维度为（H W C，但是真正要使用的维度应该是C H W）
        # ，然后从array转换为tensor
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    val_input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
    ])

    co_transform = flow_transforms.Compose([
        # 随机裁剪
        flow_transforms.RandomCrop((args.train_img_height, args.train_img_width)),
        # 以0.5的概率随机翻转给定的图片
        flow_transforms.RandomVerticalFlip(),
        # 以0.5的概率随机水平旋转给定的图片
        flow_transforms.RandomHorizontalFlip()
    ])

    # dataset path: data_preprocessing
    print("=> loading img pairs from '{}'".format(args.data))
    # args.dataset: BSD500
    # dataset setting

    train_set, val_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        val_transform=val_input_transform,
        target_transform=target_transform,
        co_transform=co_transform
    )
    # embed(header="---------------------------------------------------------------------")
    print('{} samples found, {} train samples and {} val samples '
          .format(len(val_set) + len(train_set),
                  len(train_set),
                  len(val_set)))

    # 加载 训练 / 测试 数据集
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=True
    )


    # ============== create model ====================

    # single_model = CerberusSegmentationModelMultiHead(backbone="vitb_rn50_384")
    # model = single_model.cuda()
    # with open('CerberusSeMulti.txt', mode='w') as ctxt:
    #     ctxt.write(str(model.parameters()))

    single_model = DPTSegmentationModel(9, backbone="vitb_rn50_384")
    model = single_model.cuda()
    # with open('DPTSe.txt', mode='w') as ctxt:
    #     ctxt.write(str(model1.parameters()))


    # embed(header='----------------------------------model--------------------------------------')

    # model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # ===================== superpixel optimizer ==========================

    # optimizer = torch.optim.SGD(single_model.parameters(),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(
    #     single_model.parameters(),
    #     args.lr,
    #     weight_decay=args.weight_decay,
    #     betas=(args.momentum, args.beta))

    optimizer = torch.optim.SGD(single_model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # embed(header='-----------------------------optimizer------------------------------')

    print('=> will save everything to {}'.format(args.savepath))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer = SummaryWriter(os.path.join(save_path, 'val'))

    # spixelID: superpixel ID for visualization,
    # XY_feat: the coordinate feature for position loss term
    init_spixl_map_idx, xy_coords = init_spixel_grid(args)


    # 验证
    val_init_spixl_map_idx, val_xy_coords = init_spixel_grid(args, b_train=False)

    acc_no_longer_drops_epochs = 0  # model accuracy no longer drops' epoch number, if number >= 10, training break

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        startTime = time.time()
        train_avg_slic, train_avg_sem, iteration = train(train_loader, model, optimizer, epoch,
                                                         train_writer, init_spixl_map_idx, xy_coords)

        if epoch % args.record_freq == 0:
            train_writer.add_scalar('Mean avg_slic', train_avg_slic, epoch)

        # evaluate on validation set and save the module(and choose the best)
        with torch.no_grad():
            avg_slic, avg_sem = validate(val_loader, model, epoch, val_writer, val_init_spixl_map_idx, val_xy_coords)
            if epoch % args.record_freq == 0:
                val_writer.add_scalar('Mean avg_slic', avg_slic, epoch)


        # embed(header="-----------------------------------------------------model state_dict -------------------------------------------")
        rec_dict = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_EPE': best_EPE,
            'optimizer': optimizer.state_dict(),
            'dataset': args.dataset

        }

        # with open('SomethingOutput/modeloutput_' + str(epoch) + '.txt', 'w') as modeltxt:
        #     modeltxt.write(str(model.state_dict()))

        # args.milestones default = [200000]
        # args.additional_step) default=100000 (the additional iteration, after lr decay)
        # 在学习率下降的时候，增加迭代的次数
        print("has trained iteration: {0}/{1}".format(iteration, args.milestones[-1] + args.additional_step))
        logger.info("has trained iteration: {0}/{1}".format(iteration, args.milestones[-1] + args.additional_step))
        if (iteration) >= (args.milestones[-1] + args.additional_step):
            save_checkpoint(rec_dict, is_best=False, filename='%d_step.tar' % iteration)
            print("Train finished!")
            logger.info("Train finished!")
            break

        if best_EPE < 0:
            best_EPE = avg_sem
        is_best = avg_sem < best_EPE

        # if model accuracy no longer drops, break
        if is_best:
            acc_no_longer_drops_epochs = 0
        else:
            acc_no_longer_drops_epochs += 1

        logger.info("Accuracy did not improve in {0} epochs!".format(acc_no_longer_drops_epochs))
        print("Accuracy did not improve in {0} epochs!".format(acc_no_longer_drops_epochs))

        # if acc_no_longer_drops_epochs >= 20:
        #     logger.info("Accuracy did not drop in 20 epochs!")
        #     print("Accuracy did not drop in 20 epochs!")
        #     break

        best_EPE = min(avg_sem, best_EPE)
        save_checkpoint(rec_dict, is_best)
        endTime = time.time() - startTime
        print("epoch spend time is", endTime)

    """ ============================== add superpixel ================================ """

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr


if __name__ == '__main__':
    main()



# trianing
# CUDA_VISIBLE_DEVICES=2 python train_2.py

# continue a training process
# python train.py --pretrained=/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/inference_model/model_best.tar