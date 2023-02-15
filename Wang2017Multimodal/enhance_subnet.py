import torch
import torch.nn as nn
from layer_utils import *
import sys
sys.path.append("..")
from thumb_instance_norm import ThumbInstanceNorm

# dimensions of image [batch_size, channels, height, width]

class EnhanceSubnet(nn.Module):
    def __init__(self):
        super(EnhanceSubnet, self).__init__()

        # Bilinear upsampling layer
        #self.upsample = nn.Upsample(size=512, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)   # size = 512
        self.in1 = ThumbInstanceNorm(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)   # size = 256
        self.in2 = ThumbInstanceNorm(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)   # size = 128
        self.in3 = ThumbInstanceNorm(128, affine=True)
        self.conv4 = ConvLayer(128, 256, kernel_size=3, stride=2)   # size = 64
        self.in4 = ThumbInstanceNorm(256, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)

        # Upsampling Layers
        self.rezconv1 = ResizeConvLayer(256, 128, kernel_size=3, stride=1)
        self.in5 = ThumbInstanceNorm(128, affine=True)
        self.rezconv2 = ResizeConvLayer(128, 64, kernel_size=3, stride=1)
        self.in6 = ThumbInstanceNorm(64, affine=True)
        self.rezconv3 = ResizeConvLayer(64, 32, kernel_size=3, stride=1)
        self.in7 = ThumbInstanceNorm(32, affine=True)
        self.rezconv4 = ConvLayer(32, 3, kernel_size=9, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.upsample(X)
        # resized input image is the content target
        resized_input_img = X.clone()

        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.relu(self.in4(self.conv4(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.relu(self.in5(self.rezconv1(y)))
        y = self.relu(self.in6(self.rezconv2(y)))
        y = self.relu(self.in7(self.rezconv3(y)))
        y = self.rezconv4(y)

        # Clamp image to be in range [0,1] after denormalization
        y[0][0].clamp_((0-0.485)/0.299, (1-0.485)/0.299)
        y[0][1].clamp_((0-0.456)/0.224, (1-0.456)/0.224)
        y[0][2].clamp_((0-0.406)/0.225, (1-0.406)/0.225)

        return y, resized_input_img
