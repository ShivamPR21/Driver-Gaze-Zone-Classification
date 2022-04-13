from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from moduleZoo import Conv2DNormActivation


class Encoder(nn.Module):

    def __init__(self,
                 in_shape:Tuple[int, int] = (200, 200),
                 in_channels:int = 3,
                 out_channels:int = 40) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = Conv2DNormActivation(self.in_channels, 5, kernel_size=5,
                                          stride=2, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU) # input [3, 100, 100], Output [5, 98, 98]
        self.conv2 = Conv2DNormActivation(5, 10, kernel_size=5,
                                          stride=1, padding=0, norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.SELU) # input [5, 98, 98], Output [10, 96, 96]
        self.max_pool2 = nn.MaxPool2d(kernel_size=2) # input [10, 96, 96], Output [10, 95, 95]
        self.conv3 = Conv2DNormActivation(10, 20, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU) # input [10, 95, 95], Output [20, 94, 94]
        self.conv4 = Conv2DNormActivation(20, 20, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU) # input [20, 94, 94], Output [20, 93, 93]
        self.conv5 = Conv2DNormActivation(20, 30, kernel_size=3,
                                          stride=1, padding=0, norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.SELU) # input [20, 93, 93], Output [20, 92, 92]
        self.conv6 = Conv2DNormActivation(30, self.out_channels, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU) # input [20, 91, 91], Output [10, 90, 90]

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.max_pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x

class Decoder(nn.Module):

    def __init__(self,
                 in_channels:int = 40,
                 out_size:Tuple[int, int] = (200, 200)):
        super().__init__()
        self.in_channels = in_channels
        self.out_size = out_size

        self.conv1 = Conv2DNormActivation(self.in_channels, 30, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU, transposed=True) # input [10, 186, 186], Output [20, 194, 194]
        self.conv2 = Conv2DNormActivation(30, 30, kernel_size=3,
                                          stride=2, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU, transposed=True)
        self.conv3 = Conv2DNormActivation(30, 20, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU, transposed=False)
        self.conv4 = Conv2DNormActivation(20, 10, kernel_size=3,
                                          stride=2, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU, transposed=True)
        self.conv5 = Conv2DNormActivation(10, 5, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU, transposed=True)
        self.conv6 = Conv2DNormActivation(5, 3, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.Sigmoid, transposed=False)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.interpolate(x, size=self.out_size)

        return x

class AutoEncoderFaceImages(nn.Module):

    def __init__(self, in_size = (200, 200), in_channels = 3, enc_channels = 40) -> None:
        super().__init__()

        self.in_channels, self.in_size, self.enc_channels = in_channels, in_size, enc_channels
        self.encoder = Encoder(self.in_size, self.in_channels, self.enc_channels)
        self.decoder = Decoder(self.enc_channels)

    def forward(self, x):

        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)

        return x_dec
