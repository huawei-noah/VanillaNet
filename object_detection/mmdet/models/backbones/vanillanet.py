#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import (BaseModule, ModuleList, Sequential, _load_checkpoint,
                         load_state_dict)

from ...utils import get_root_logger
from ..builder import BACKBONES


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, norm_layer=nn.SyncBatchNorm):
        super(activation, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        self.bn = norm_layer(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        return self.bn(torch.nn.functional.conv2d(super(activation, self).forward(x), self.weight, padding=self.act_num, groups=self.dim))

class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, norm_layer=nn.SyncBatchNorm):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            norm_layer(dim, eps=1e-6),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=1),
            norm_layer(dim_out, eps=1e-6)
        )
        self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        self.act = activation(dim_out, act_num, norm_layer)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.act(x)
        return x


@BACKBONES.register_module()
class Vanillanet(BaseModule):
    def __init__(self, in_chans=3, act_num=3, dims=[96, 192, 384, 768], out_indices=[2,4,6],
        strides=[2,2,2,1], norm_layer=nn.SyncBatchNorm, init_cfg=None, **kwargs): 
        super().__init__()

        self.out_indices = out_indices
        self.init_cfg = init_cfg

        self.stem1 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0], eps=1e-6),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
            norm_layer(dims[0], eps=1e-6),
            activation(dims[0], act_num, norm_layer)
        )

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], norm_layer=norm_layer)
            self.stages.append(stage)
        self.depth = len(strides)

    def init_weights(self):
        if self.init_cfg is None:
            logger = get_root_logger()
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, 0, math.sqrt(2.0 / fan_out))
        else:
            state_dict = torch.load(self.init_cfg.checkpoint, map_location='cpu')
            msg = self.load_state_dict(state_dict['model_ema'], strict=False)
            print(msg)
            print('Successfully load backbone ckpt.')

    def forward(self, x):
        outs = []
        x = self.stem1(x)
        x = self.stem2(x)
        for i in range(self.depth):
            x = self.stages[i](x)

            if i in self.out_indices:
                outs.append(x)

        return outs
