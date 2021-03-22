from torchvision import transforms
from util import *
import time
import os
import argparse
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--content',default='images/content',help='content image')
parser.add_argument('--style',default='images/style',help='style image')
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
parser.add_argument('--outf', default='output/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
parser.add_argument('--test_speed', action="store_true", help='test the speed')
parser.add_argument('--style_size', type=int, default=1024, help='style size')

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass




def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def styleTransfer(contentImg, styleImg):
    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform_original(cF5, sF5, args.alpha).cuda()
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform_original(cF4, sF4, args.alpha).cuda()
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform_original(cF3, sF3, args.alpha).cuda()
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform_original(cF2, sF2, args.alpha).cuda()
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform_original(cF1, sF1, args.alpha).cuda()
    Im1 = wct.d1(csF1)
    
    return Im1

def save_image(image, save_path):
    image = image.mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)

if __name__ == '__main__':
    
    wct = WCT(args).cuda()

    content_tf = test_transform(0, False)
    style_tf = test_transform(args.style_size, True)

    repeat = 15 if args.test_speed else 1
    time_list = []

    for i in range(repeat):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = content_tf(Image.open(args.content)).unsqueeze(0).cuda()
        style = style_tf(Image.open(args.style)).unsqueeze(0).cuda()
        with torch.no_grad():
            image = styleTransfer(image, style)
            save_image(image, os.path.join(args.outf, "original.jpg"))
            
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)

    print("time: %.2fs" % np.mean(time_list[-10:]))