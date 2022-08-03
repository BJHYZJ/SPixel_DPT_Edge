import shutil
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch StereoSpixel inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--src', default='/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/output', help='path to spixel test dir')
parser.add_argument('--dst', default='/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/eval_spixel/eval_result', help='path to collect all the evaluation results',)

args = parser.parse_args()

pattern = 'nyu'  # or nyu
test_name = '/test_4/'

if pattern == 'bsd':
    src = args.src + test_name + '/bsd/test_multiscale_enforce_connect'  # remember modify it
    dst = args.dst + test_name + '/bsd'  # remember modify it   for bsd
    list = ["96", "216", "384", "600", "864", "1176", "1536", "1944"]

else:
    src = args.src + test_name + '/nyu/test_multiscale_enforce_connect'  # remember modify it
    dst = args.dst + test_name + '/nyu'  # remember modify it   for bsd
    list = ["192" ,"432", "768", "1200", "1728", "2352", "3072", "3888"]

for l in list:
    src_pth = src + '/SPixelNet_nSpixel_' + l +'/map_csv/results.csv'
    dst_pth = dst + '/' + l
    if not os.path.isdir(dst_pth):
        os.makedirs(dst_pth)
    dst_path = dst_pth + '/results.csv'
    shutil.copy(src_pth, dst_path)
