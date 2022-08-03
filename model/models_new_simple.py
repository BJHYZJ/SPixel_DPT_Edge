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

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet12(layer_4_rn)
        path_3 = self.scratch.refinenet11(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet10(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet09(path_2, layer_1_rn)


        fun = eval("self.scratch.output_Segmentation")
        out = fun(path_1)
        fun = eval("self.scratch.output_Segmentation_upsample")
        out = fun(out)
        return out

