import torch
import torch.nn as nn
from IPython import embed
import torch.nn.functional as F
from .base_model import BaseModel
from model.vit import forward_flex
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class Cerberus(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(Cerberus, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights（设置为true，你想从零开始训练，使用ImageNet权重）
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet10 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet11 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet12 = _make_fusion_block(features, use_bn)


class CerberusSegmentationModelMultiHead(Cerberus):
    def __init__(self, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)

        self.add_module('sigma', nn.Module())

        self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)

        setattr(self.scratch, "output_Segmentation", nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            # nn.Conv2d(features, 40, kernel_size=1),
            nn.Conv2d(features, 9, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Softmax(1),
        ))

        setattr(self.scratch, "output_Segmentation_upsample",
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )

        if path is not None:
            self.load(path)
        else:
            pass

    def get_attention(self, x, name):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        x = forward_flex(self.pretrained.model, x, True, name)

        return x

    def forward(self, x):
        # Cerberus batch ==         ([1, 3, 512, 512]) batch_size = 1
        # sPixel_Cerberus batch == ([32, 3, 208, 208]) batch_size = 32

        # Cerberus
        # sPixel_Cerberus x中每张图片的大小都是208 * 208
        # (batch_size, channel, width, height)
        # self.channels_last == False
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        # 从vision transformers中获取四个子层
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        """
            layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape
            (torch.Size([32, 256, 64, 64]),
             torch.Size([32, 512, 32, 32]),
             torch.Size([32, 768, 16, 16]),
             torch.Size([32, 768, 8, 8]))
        """
        # embed(header="================================================'First time!!'")
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        """
            layer_1_rn.shape, layer_2_rn.shape, layer_3_rn.shape, layer_4_rn.shape, 
            (torch.Size([32, 256, 64, 64]),
             torch.Size([32, 256, 32, 32]),
             torch.Size([32, 256, 16, 16]),
             torch.Size([32, 256, 8, 8]))
        """
        # 只有Segmentation的情况，即index == 2
        # Cerberus (path_4, layer_3_rn): (torch.Size([32, 256, 14, 14]),        (torch.Size([1, 256, 32, 32])), torch.Size([1, 256, 32, 32])) batch_size = 1
        # sPixel_Cerberus (path_4, layer_3_rn): (torch.Size([32, 256, 14, 14]), (torch.Size([32, 256, 13, 13])) batch_size = 32
        # embed(header="================================================'Second time!!'")
        path_4 = self.scratch.refinenet12(layer_4_rn)
        """
            test:
                x.shape ==  torch.Size([1, 3, 144, 96])
            
                layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape
                (torch.Size([1, 256, 36, 24]),
                 torch.Size([1, 512, 18, 12]),
                 torch.Size([1, 768, 9, 6]),
                 torch.Size([1, 768, 5, 3]))
                 
                 layer_1_rn.shape, layer_2_rn.shape, layer_3_rn.shape, layer_4_rn.shape
                 (torch.Size([1, 256, 36, 24]),
                 torch.Size([1, 256, 18, 12]),
            layer_3_rn.shape     ====>  torch.Size([1, 256, 9, 6]),
            layer_4_rn.shape     ====>  torch.Size([1, 256, 5, 3]))
                 
                 path_4.shape ==  torch.Size([1, 256, 10, 6])
        """
        # embed(header="-------------------------------------center time !!!----------------------")
        path_3 = self.scratch.refinenet11(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet10(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet09(path_2, layer_1_rn)

        """
            layer_1.shape, layer_2.shape, layer_3.shape, layer_4.shape
            (torch.Size([32, 256, 64, 64]),
             torch.Size([32, 512, 32, 32]),
             torch.Size([32, 768, 16, 16]),
             torch.Size([32, 768, 8, 8]))
            
            layer_1_rn.shape, layer_2_rn.shape, layer_3_rn.shape, layer_4_rn.shape, 
            (torch.Size([32, 256, 64, 64]),
             torch.Size([32, 256, 32, 32]),
             torch.Size([32, 256, 16, 16]),
             torch.Size([32, 256, 8, 8]))
            
            path_4.shape, path_3.shape, path_2.shape, path_1.shape
            (torch.Size([32, 256, 16, 16])
            torch.Size([32, 256, 32, 32]),
            torch.Size([32, 256, 64, 64]),
            torch.Size([32, 256, 128, 128]))
        """

        # output_task_list = self.full_output_task_list[index][1]
        # embed(header="================================================'Third time!!'")
        # outs = list()

        fun = eval("self.scratch.output_Segmentation")
        out = fun(path_1)
        fun = eval("self.scratch.output_Segmentation_upsample")
        out = fun(out)
        # outs.append(out)
        # embed(header="==============================end!=================================")
        # return outs, [self.sigma.sub_seg_sigmas,
        #               self.sigma.seg_sigmas], []

        return out

# def predict_mask(in_planes, channel=9):
#     return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

