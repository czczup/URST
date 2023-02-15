import torch
import torch.nn as nn
from layer_utils import *
import sys
sys.path.append("..")
from thumb_instance_norm import ThumbInstanceNorm

# dimensions of image [batch_size, channels, height, width]

class RefineSubnet(nn.Module):
    def __init__(self):
        super(RefineSubnet, self).__init__()

        # Bilinear upsampling layer
        #self.upsample = nn.Upsample(size=1024, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = ThumbInstanceNorm(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = ThumbInstanceNorm(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = ThumbInstanceNorm(128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)

        # Upsampling Layers
        self.rezconv1 = ResizeConvLayer(128, 64, kernel_size=3, stride=1)
        self.in4 = ThumbInstanceNorm(64, affine=True)
        self.rezconv2 = ResizeConvLayer(64, 32, kernel_size=3, stride=1)
        self.in5 = ThumbInstanceNorm(32, affine=True)
        self.rezconv3 = ConvLayer(32, 3, kernel_size=3, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        in_X = X
        #if self.training == False: in_X = self.upsample(in_X)   # Only apply upsampling during test

        # resized input image is the content target
        resized_input_img = in_X.clone()

        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.relu(self.in4(self.rezconv1(y)))
        y = self.relu(self.in5(self.rezconv2(y)))
        y = self.rezconv3(y)
        y = y + resized_input_img

        # Clamp image to be in range [0,1] after denormalization
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)

        return y, resized_input_img
