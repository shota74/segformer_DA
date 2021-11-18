# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
from torch import tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        num_class=2

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #decoder_params = kwargs['decoder_params']
        #embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = nn.ModuleList([MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)for i in range(num_class)])
        self.linear_c3 = nn.ModuleList([MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)for i in range(num_class)])
        self.linear_c2 = nn.ModuleList([MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)for i in range(num_class)])
        self.linear_c1 = nn.ModuleList([MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)for i in range(num_class)])
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = nn.ModuleList([ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
            #norm_cfg=dict(type='BN2d', requires_grad=True)
        )for i in range(num_class)])

        self.linear_pred = nn.ModuleList([nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)for i in range(num_class)])

    def forward(self, x, cls_lind):

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4[cls_lind](c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3[cls_lind](c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2[cls_lind](c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1[cls_lind](c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse[cls_lind](torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred[cls_lind](x)

        return x