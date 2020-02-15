# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as fn


class UNet(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 128, 128),
                 dropout: float = 0.5,
                 nchannels: Iterable[int] = (64, 128, 256, 512, 1024),
                 nclasses: int = 1,
                 bn: str = None,
                 activation: str = 'relu',
                 upsampling: str = 'bilinear',
                 dilation_rates: Iterable[int] = (1, 1, 1, 1, 1),
                 residual: str = None):
        super().__init__()
        self.input_shape = input_shape
        self.dropout = dropout
        self.nchannels = nchannels

    def forward(self, x):
        return x


class UNetDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int(kernel_size // 2)
        self.layers = nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ])

    def forward(self, x):
        return self.layers(x)


class UNetUpSampleBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super().__init__()
        padding = int(kernel_size // 2)
        self.layers = nn.Sequential([
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels,
                               out_channels,
                               kernel_size=2,
                               stride=2)
        ])

    def forward(self, x):
        return self.layers(x)
