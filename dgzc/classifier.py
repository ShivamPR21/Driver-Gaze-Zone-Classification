from typing import Tuple

import torch
import torch.nn as nn
from moduleZoo import Conv2DNormActivation

from .autoencoder import Decoder, Encoder


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out

class ClassificationBackbone(nn.Module):

    def __init__(self, in_dim:int = 40, l:float = 1/4, img_size: Tuple[int, int] = (200, 200)) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.img_size = img_size
        self.l = l
        self.encoder = Encoder(self.img_size, 3, self.in_dim)
        self.attention1 = SelfAttention(self.in_dim, nn.SELU)
        self.conv1 = Conv2DNormActivation(self.in_dim, 50, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU)
        self.attention2 = SelfAttention(50, nn.SELU)
        self.conv2 = Conv2DNormActivation(50, 40, kernel_size=3,
                                          stride=1, padding=0, norm_layer=None,
                                          activation_layer=nn.SELU)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = Conv2DNormActivation(40, 30, kernel_size=3,
                                          stride=1, padding=0, norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.SELU)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=2)
        self.conv4 = Conv2DNormActivation(30, 10, kernel_size=3,
                                          stride=1, padding=0, norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.SELU)
        self.attention3 = SelfAttention(10, nn.SELU)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, _, _ = x.size()
        # Illumination consistency layer
        x_il = torch.pow(x, self.l)/(255.**self.l) # [B, C, H, W]

        x = torch.cat([x_il, x], dim=0) # [2*B, C, H, W]
        x = self.encoder(x) # [2*B, C, H, W]
        x_il, x = x.split(B, dim=0) # [B, C, H, W]

        x_il = self.attention1(x_il)
        x_il = self.conv1(x_il)

        x_il = self.attention2(x_il)
        x_il = self.avg_pool2(self.conv2(x_il))

        x_il = self.avg_pool3(self.conv3(x_il))

        x_il = self.attention3(self.conv4(x_il))

        return x_il, x

class AutoEncoderClassifierAmalgamation(ClassificationBackbone):

    def __init__(self, in_dim: int = 40, l: float = 1 / 4, img_size: Tuple[int, int] = (200, 200), n_class: int = 5) -> None:
        super().__init__(in_dim, l, img_size)

        self.n_class = n_class
        self.decoder = Decoder(self.in_dim, self.img_size)
        self.linear1 = nn.Linear(250, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 8)
        self.linear6 = nn.Linear(8, self.n_class)
        self.activation = nn.SELU()
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_il, x = super().forward(x)
        x_il = x_il.flatten(start_dim=1)

        x_il = self.activation(self.linear1(x_il))
        x_il = self.activation(self.linear2(x_il))
        x_il = self.activation(self.linear3(x_il))
        x_il = self.activation(self.linear4(x_il))
        x_il = self.activation(self.linear5(x_il))
        x_il = self.classifier(self.linear6(x_il))

        x = self.decoder(x)

        return x_il, x
