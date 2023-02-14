from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from thumb_instance_norm import ThumbInstanceNorm

from layer_utils import *

# dimensions of image [batch_size, channels, height, width]


class StyleSubnet(nn.Module):
    def __init__(self):
        super(StyleSubnet, self).__init__()

        # Bilinear downsampling
        #self.downsample = nn.Upsample(size=256, mode='bilinear')

        # Transform to Grayscale
        self.togray = nn.Conv2d(3, 1, kernel_size=1, stride=1)
        w = torch.nn.Parameter(torch.tensor([[[[0.299]],
                                              [[0.587]],
                                              [[0.114]]]]))
        self.togray.weight = w

        # RGB Block
        self.rgb_conv1 = ConvLayer(3, 16, kernel_size=9, stride=1)
        self.rgb_in1 = ThumbInstanceNorm(16)
        self.rgb_conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.rgb_in2 = ThumbInstanceNorm(32)
        self.rgb_conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.rgb_in3 = ThumbInstanceNorm(64)
        self.rgb_res1 = ResidualBlock(64)
        self.rgb_res2 = ResidualBlock(64)
        self.rgb_res3 = ResidualBlock(64)

        # L Block
        self.l_conv1 = ConvLayer(1, 16, kernel_size=9, stride=1)
        self.l_in1 = ThumbInstanceNorm(16)
        self.l_conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.l_in2 = ThumbInstanceNorm(32)
        self.l_conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.l_in3 = ThumbInstanceNorm(64)
        self.l_res1 = ResidualBlock(64)
        self.l_res2 = ResidualBlock(64)
        self.l_res3 = ResidualBlock(64)

        # Residual layers
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.res6 = ResidualBlock(128)

        # Upsampling Layers
        self.rezconv1 = ResizeConvLayer(128, 64, kernel_size=3, stride=1)
        self.in4 = ThumbInstanceNorm(64)
        self.rezconv2 = ResizeConvLayer(64, 32, kernel_size=3, stride=1)
        self.in5 = ThumbInstanceNorm(32)
        self.rezconv3 = ConvLayer(32, 3, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()


    def forward(self, x):

        # Bilinear downsampling
        #x = self.downsample(x)

        # Resized input image is the content target
        resized_input_img = x.clone()

        # Get RGB and L image
        x_rgb = x
        with torch.no_grad(): x_l = self.togray(x.clone())

        # RGB Block
        y_rgb = self.relu(self.rgb_in1(self.rgb_conv1(x_rgb)))
        y_rgb = self.relu(self.rgb_in2(self.rgb_conv2(y_rgb)))
        y_rgb = self.relu(self.rgb_in3(self.rgb_conv3(y_rgb)))
        y_rgb = self.rgb_res1(y_rgb)
        y_rgb = self.rgb_res2(y_rgb)
        y_rgb = self.rgb_res3(y_rgb)

        # L Block
        y_l = self.relu(self.l_in1(self.l_conv1(x_l)))
        y_l = self.relu(self.l_in2(self.l_conv2(y_l)))
        y_l = self.relu(self.l_in3(self.l_conv3(y_l)))
        y_l = self.l_res1(y_l)
        y_l = self.l_res2(y_l)
        y_l = self.l_res3(y_l)

        # Concatenate blocks along the depth dimension
        y = torch.cat((y_rgb, y_l), 1)

        # Residuals
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)

        # Decoding
        y = self.relu(self.in4(self.rezconv1(y)))
        y = self.relu(self.in5(self.rezconv2(y)))
        y = self.rezconv3(y)


        # Clamp image to be in range [0,1] after denormalization
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)

        return y, resized_input_img
