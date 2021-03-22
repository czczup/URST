import os
import torch
import argparse
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, encoder5
from libs.models import decoder3,decoder4, decoder5
from thumb_instance_norm import init_thumbnail_instance_norm
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import math

import sys
sys.path.append("..")
from tools import unpadding, preprocess


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def save_image(image, save_path):
    image = image.mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)

def style_transfer_high_resolution(patches, sF, padding, save_path, collection=False, save=True):
    stylized_patches = []
    init_thumbnail_instance_norm(matrix, collection=collection)

    for patch in tqdm(patches):
        patch = patch.unsqueeze(0).to(device)
        cF = vgg(patch)
        if (args.layer == 'r41'):
            feature = matrix(cF[args.layer], sF[args.layer])
        else:
            feature = matrix(cF, sF)
        stylized = dec(feature)
        stylized = F.interpolate(stylized, patch.shape[2:], mode='bilinear', align_corners=True)
        stylized = unpadding(stylized, padding=padding)
        stylized_patches.append(stylized.cpu())

    stylized_patches = torch.cat(stylized_patches, dim=0)
    b, c, h, w = stylized_patches.shape
    stylized_patches = stylized_patches.unsqueeze(dim=0)
    stylized_patches = stylized_patches.view(1, b, c * h * w).permute(0, 2, 1).contiguous()
    output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
    stylized_image = F.fold(stylized_patches, output_size=output_size, kernel_size=(h, w), stride=(h, w))
    if save:
        save_image(stylized_image, save_path)
    
    
def style_transfer_thumbnail(thumb, sF, save_path, save=False):
    cF = vgg(thumb)
    init_thumbnail_instance_norm(matrix, collection=True)
    if (args.layer == 'r41'):
        feature = matrix(cF[args.layer], sF[args.layer])
    else:
        feature = matrix(cF, sF)
    stylized_thumb = dec(feature)
    if save:
        save_image(stylized_thumb, save_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                        help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                        help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                        help='pre-trained model path')
    parser.add_argument("--style", default="data/style/mosaic.jpg",
                        help='path to style image')
    parser.add_argument("--content", default="data/content/test.jpg",
                        help='path to frames')
    parser.add_argument("--outf", default="output/",
                        help='path to transferred images')
    parser.add_argument("--layer", default="r41",
                        help='which features to transfer, either r31 or r41')
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
    parser.add_argument('--thumb_size', type=int, default=1024, help='thumbnail size')
    parser.add_argument('--style_size', type=int, default=1024, help='style size')
    parser.add_argument('--padding', type=int, default=32, help='padding')
    parser.add_argument('--test_speed', action="store_true", help='test the speed')
    parser.add_argument('--URST', action="store_true", help='use URST framework')
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument('--resize', type=int, default=0, help='resize')

    ################# PREPARATIONS #################
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print_options(args)
    os.makedirs(args.outf, exist_ok=True)
    content_name = args.content.split("/")[-1].split(".")[0]
    style_name = args.style.split("/")[-1].split(".")[0]
    device = torch.device(args.device)

    ################# MODEL #################
    if(args.layer == 'r31'):
        vgg = encoder3().to(device)
        dec = decoder3().to(device)
    elif(args.layer == 'r41'):
        vgg = encoder4().to(device)
        dec = decoder4().to(device)
    matrix = MulLayer(args.layer).to(device)
    vgg.load_state_dict(torch.load(args.vgg_dir))
    dec.load_state_dict(torch.load(args.decoder_dir))
    matrix.load_state_dict(torch.load(args.matrixPath))
    
    PATCH_SIZE = args.patch_size
    PADDING = args.padding
    
    content_tf = test_transform(0, False)
    style_tf = test_transform(args.style_size, True)

    repeat = 15 if args.test_speed else 1
    time_list = []

    for i in range(repeat):
        image = Image.open(args.content)
        if args.resize != 0:
            image = image.resize((args.resize, args.resize))
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size
        style = Image.open(args.style)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = image.resize((int(aspect_ratio * args.thumb_size), args.thumb_size))
            patches = preprocess(image, padding=PADDING, patch_size=PATCH_SIZE, transform=content_tf, cuda=False)
            thumbnail = content_tf(thumbnail).unsqueeze(0).to(device)
            style = style_tf(style).unsqueeze(0).to(device)
            
            print("content:", patches.shape)
            print("thumb:", thumbnail.shape)
            print("style:", style.shape)
            
            with torch.no_grad():
                sF = vgg(style)
                style_transfer_thumbnail(thumbnail, sF, save=False if args.test_speed else True,
                                         save_path=os.path.join(args.outf, "thumb-%d.jpg" % args.thumb_size))
                style_transfer_high_resolution(
                    patches, sF, padding=PADDING, collection=False,
                    save_path=os.path.join(args.outf, "ours-patch%d-padding%d.jpg" % (PATCH_SIZE, PADDING)),
                    save=False if args.test_speed else True
                )
                # style_transfer_high_resolution(
                #     patches, sF, padding=PADDING, collection=True,
                #     save_path=os.path.join(args.outf, "baseline-width%d-padding%d.jpg" % (PATCH_SIZE, PADDING))
                # )
        else:
            image = content_tf(image).unsqueeze(0).to(device)
            style = style_tf(style).unsqueeze(0).to(device)

            print("image:", image.shape)
            print("style:", style.shape)
            
            with torch.no_grad():
                sF = vgg(style)
                style_transfer_thumbnail(image, sF, save=False if args.test_speed else True,
                                         save_path=os.path.join(args.outf, "original_result.jpg"))
            
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)

    print("time: %.2fs" % np.mean(time_list[-10:]))
    # print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
    # print("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))
