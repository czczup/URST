from tqdm import tqdm
from util_wct import *
import torch.nn.functional as F
import time
import os
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import math

import sys
sys.path.append("../..")
from tools import unpadding, preprocess
from thumb_instance_norm import init_thumbnail_instance_norm

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--content', default='images/content/test6.jpg',help='path of content image')
parser.add_argument('--style', default='images/style/mosaic_hd.png',help='path of style image')
parser.add_argument('--outf', default='output', help='folder to output images')
parser.add_argument('--alpha', type=float, default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
parser.add_argument('--thumb_size', type=int, default=1024, help='thumbnail size')
parser.add_argument('--style_size', type=int, default=1024, help='style size')
parser.add_argument('--padding', type=int, default=32, help='padding size')
parser.add_argument('--test_speed', action="store_true", help='test the speed')
parser.add_argument('--URST', action="store_true", help='use URST framework')
parser.add_argument("--device", type=str, default="cuda", help="device")

args = parser.parse_args()
args.mode = "16x"
args.e5 = '../trained_models/wct_se_16x_new/5SE.pth'
args.e4 = '../trained_models/wct_se_16x_new/4SE.pth'
args.e3 = '../trained_models/wct_se_16x_new/3SE.pth'
args.e2 = '../trained_models/wct_se_16x_new/2SE.pth'
args.e1 = '../trained_models/wct_se_16x_new/1SE.pth'
args.d5 = '../trained_models/wct_se_16x_new_sd/5SD.pth'
args.d4 = '../trained_models/wct_se_16x_new_sd/4SD.pth'
args.d3 = '../trained_models/wct_se_16x_new_sd/3SD.pth'
args.d2 = '../trained_models/wct_se_16x_new_sd/2SD.pth'
args.d1 = '../trained_models/wct_se_16x_new_sd/1SD.pth'

# args.mode = "original"
# args.e5 = '../../PytorchWCT/models/vgg_normalised_conv5_1.t7'
# args.e4 = '../../PytorchWCT/models/vgg_normalised_conv4_1.t7'
# args.e3 = '../../PytorchWCT/models/vgg_normalised_conv3_1.t7'
# args.e2 = '../../PytorchWCT/models/vgg_normalised_conv2_1.t7'
# args.e1 = '../../PytorchWCT/models/vgg_normalised_conv1_1.t7'
# args.d5 = '../../PytorchWCT/models/feature_invertor_conv5_1.t7'
# args.d4 = '../../PytorchWCT/models/feature_invertor_conv4_1.t7'
# args.d3 = '../../PytorchWCT/models/feature_invertor_conv3_1.t7'
# args.d2 = '../../PytorchWCT/models/feature_invertor_conv2_1.t7'
# args.d1 = '../../PytorchWCT/models/feature_invertor_conv1_1.t7'

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


def styleTransfer(content, sFs, wct_mode):
    sF5 = sFs[0]
    if wct_mode == 'gpu':
        cF5 = wct.e5(content).squeeze(0)
        csF5 = wct.transform_v2(cF5, sF5, args.alpha, index=1, wct_mode=wct_mode)
    else:
        cF5 = wct.e5(content).data.cpu().squeeze(0)
        csF5 = wct.transform_v2(cF5, sF5, args.alpha, index=1, wct_mode=wct_mode).to(device)
    Im5 = wct.d5(csF5)
    del csF5, cF5; torch.cuda.empty_cache()

    sF4 = sFs[1]
    if wct_mode == 'gpu':
        cF4 = wct.e4(Im5).squeeze(0)
        csF4 = wct.transform_v2(cF4, sF4, args.alpha, index=2, wct_mode=wct_mode)
    else:
        cF4 = wct.e4(Im5).data.cpu().squeeze(0)
        csF4 = wct.transform_v2(cF4, sF4, args.alpha, index=2, wct_mode=wct_mode).to(device)
    Im4 = wct.d4(csF4)
    del csF4, cF4; torch.cuda.empty_cache()

    sF3 = sFs[2]
    if wct_mode == 'gpu':
        cF3 = wct.e3(Im4).squeeze(0)
        csF3 = wct.transform_v2(cF3, sF3, args.alpha, index=3, wct_mode=wct_mode)
    else:
        cF3 = wct.e3(Im4).data.cpu().squeeze(0)
        csF3 = wct.transform_v2(cF3, sF3, args.alpha, index=3, wct_mode=wct_mode).to(device)
    Im3 = wct.d3(csF3)
    del csF3, cF3; torch.cuda.empty_cache()

    sF2 = sFs[3]
    if wct_mode == 'gpu':
        cF2 = wct.e2(Im3).squeeze(0)
        csF2 = wct.transform_v2(cF2, sF2, args.alpha, index=4, wct_mode=wct_mode)
    else:
        cF2 = wct.e2(Im3).data.cpu().squeeze(0)
        csF2 = wct.transform_v2(cF2, sF2, args.alpha, index=4, wct_mode=wct_mode).to(device)
    Im2 = wct.d2(csF2)
    del csF2, cF2; torch.cuda.empty_cache()

    sF1 = sFs[4]
    if wct_mode == 'gpu':
        cF1 = wct.e1(Im2).squeeze(0)
        csF1 = wct.transform_v2(cF1, sF1, args.alpha, index=5, wct_mode=wct_mode)
    else:
        cF1 = wct.e1(Im2).data.cpu().squeeze(0)
        csF1 = wct.transform_v2(cF1, sF1, args.alpha, index=5, wct_mode=wct_mode).to(device)
    Im1 = wct.d1(csF1)
    del csF1, cF1; torch.cuda.empty_cache()

    return Im1


def save_image(image, save_path):
    image = image.mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)
    
    
def style_transfer_thumbnail(thumb, sFs, save_path, save=True, wct_mode="cpu"):
    init_thumbnail_instance_norm(wct, collection=True)
    stylized = styleTransfer(thumb, sFs, wct_mode)
    if save:
        save_image(stylized, save_path)


def style_transfer_high_resolution(patches, sFs, padding, collection, save_path, save=True, wct_mode="cpu"):
    stylized_patches = []
    init_thumbnail_instance_norm(wct, collection=collection)
    for patch in tqdm(patches):
        patch = patch.unsqueeze(0).to(device)
        stylized_patch = styleTransfer(patch, sFs, wct_mode)
        stylized_patch = F.interpolate(stylized_patch, patch.shape[2:], mode='bilinear', align_corners=True)
        stylized_patch = unpadding(stylized_patch, padding=padding)
        stylized_patches.append(stylized_patch.cpu())

    stylized_patches = torch.cat(stylized_patches, dim=0)
    
    b, c, h, w = stylized_patches.shape
    stylized_patches = stylized_patches.unsqueeze(dim=0)
    stylized_patches = stylized_patches.view(1, b, c * h * w).permute(0, 2, 1).contiguous()
    output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
    stylized_image = F.fold(stylized_patches, output_size=output_size,
                            kernel_size=(h, w), stride=(h, w))
    if save:
        save_image(stylized_image, save_path)


def generate_sFs(style, mode="cpu"):
    if mode == 'cpu':
        sF5 = wct.e5(style).data.cpu().squeeze(0)
        sF4 = wct.e4(style).data.cpu().squeeze(0)
        sF3 = wct.e3(style).data.cpu().squeeze(0)
        sF2 = wct.e2(style).data.cpu().squeeze(0)
        sF1 = wct.e1(style).data.cpu().squeeze(0)
    else:
        sF5 = wct.e5(style).squeeze(0)
        sF4 = wct.e4(style).squeeze(0)
        sF3 = wct.e3(style).squeeze(0)
        sF2 = wct.e2(style).squeeze(0)
        sF1 = wct.e1(style).squeeze(0)
    return [sF5, sF4, sF3, sF2, sF1]


if __name__ == '__main__':
    device = torch.device(args.device)
    wct = WCT(args).to(device)
    
    PATCH_SIZE = args.patch_size
    PADDING = args.padding
    content_tf = test_transform(0, False)
    style_tf = test_transform(args.style_size, True)

    repeat = 15 if args.test_speed else 1
    time_list = []

    for i in range(repeat):
        image = Image.open(args.content)
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size
        style = Image.open(args.style)

        torch.cuda.synchronize()
        start_time = time.time()
        
        style = style_tf(style).unsqueeze(0).to(device)

        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = image.resize((int(aspect_ratio * args.thumb_size), args.thumb_size))
            patches = preprocess(image, padding=PADDING, patch_size=PATCH_SIZE, transform=content_tf, cuda=False)
            thumbnail = content_tf(thumbnail).unsqueeze(0).to(device)
            print("content:", patches.shape)
            print("thumb:", thumbnail.shape)
            print("style:", style.shape)
            
            # Stylize image
            with torch.no_grad():
                sFs = generate_sFs(style, mode="gpu")
                style_transfer_thumbnail(
                    thumbnail, sFs, save=False if args.test_speed else True, wct_mode="gpu",
                    save_path=os.path.join(args.outf, "thumb-%d.jpg" % args.thumb_size)
                )
                style_transfer_high_resolution(
                    patches, sFs, padding=PADDING, collection=False, wct_mode="gpu",
                    save_path=os.path.join(args.outf, "ours-patch%d-padding%d.jpg" % (PATCH_SIZE, PADDING)),
                    save=False if args.test_speed else True
                )
                # style_transfer_high_resolution(
                #     patches, sFs, padding=PADDING, collection=True, wct_mode=args.wct_mode,
                #     save_path=os.path.join(args.outf, "baseline-width%d-padding%d.jpg" % (PATCH_SIZE, PADDING))
                # )
        else:
            image = content_tf(image).unsqueeze(0).to(device)
            print("image:", image.shape)
            print("style:", style.shape)
    
            # Stylize image
            with torch.no_grad():
                sFs = generate_sFs(style, mode="cpu")
                style_transfer_thumbnail(
                    image, sFs, save=False if args.test_speed else True, wct_mode="cpu",
                    save_path=os.path.join(args.outf, "original_result.jpg")
                )
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)
    
    print("time: %.2fs" % np.mean(time_list[-10:]))
    # print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
    # print("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))

    

