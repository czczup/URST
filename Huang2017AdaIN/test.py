import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import model
from function import coral
import torch.nn.functional as F
import numpy as np
import os
import time
import math

import sys
sys.path.append("..")
from thumb_instance_norm import ThumbAdaptiveInstanceNorm, init_thumbnail_instance_norm
from tools import preprocess, unpadding



def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, help='path to the content image')
    parser.add_argument('--style', type=str, help='path to the style image')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
    parser.add_argument('--thumb_size', type=int, default=1024, help='thumbnail size')
    parser.add_argument('--style_size', type=int, default=1024, help='style size')
    parser.add_argument('--padding', type=int, default=32, help='padding size')
    parser.add_argument('--outf', type=str, default='output', help='directory to save the output image')
    parser.add_argument('--preserve_color', action='store_true', help='if specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0, help='the weight that controls the degree of stylization. Should be between 0 and 1')
    parser.add_argument('--test_speed', action="store_true", help='test the speed')
    parser.add_argument('--resize', type=int, default=0, help='resize')
    parser.add_argument('--URST', action="store_true", help='use URST framework')
    parser.add_argument("--device", type=str, default="cuda", help="device")
    args = parser.parse_args()

    return args


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style_f, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    feat = thumb_adaptive_instance_norm.forward(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    
    return decoder(feat)


def save_image(image, save_path):
    image = image.mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)
    
def style_transfer_thumbnail(content, style_f, save_path, save=False):
    content = content.unsqueeze(0)
    init_thumbnail_instance_norm(thumb_adaptive_instance_norm, collection=True)
    stylized_thumb = style_transfer(vgg, decoder, content, style_f, args.alpha)
    if save:
        save_image(stylized_thumb, save_path)

def style_transfer_high_resolution(patches, style_f, padding, save_path, collection=False, save=True):
    stylized_patches = []
    init_thumbnail_instance_norm(thumb_adaptive_instance_norm, collection=collection)
    for patch in tqdm(patches):
        patch = patch.unsqueeze(0).to(device)
        stylized_patch = style_transfer(vgg, decoder, patch, style_f, args.alpha)
        stylized_patch = F.interpolate(stylized_patch, patch.shape[2:], mode='bilinear', align_corners=True)
        stylized_patch = unpadding(stylized_patch, padding=padding)
        stylized_patches.append(stylized_patch.cpu())

    stylized_patches = torch.cat(stylized_patches, dim=0)
    b, c, h, w = stylized_patches.shape
    stylized_patches = stylized_patches.unsqueeze(dim=0)
    stylized_patches = stylized_patches.view(1, b, c * h * w).permute(0, 2, 1).contiguous()
    output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
    stylized_image = F.fold(stylized_patches, output_size=output_size, kernel_size=(h, w), stride=(h, w))
    if save:
        save_image(stylized_image, save_path)


if __name__ == '__main__':
    
    thumb_adaptive_instance_norm = ThumbAdaptiveInstanceNorm()
    thumb_adaptive_instance_norm.eval()
    args = init_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.outf)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    PATCH_SIZE = args.patch_size
    PADDING = args.padding
    
    decoder = model.decoder
    vgg = model.vgg
    decoder.eval()
    vgg.eval()
    
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    
    vgg.to(device)
    decoder.to(device)
    
    content_tf = test_transform(0, False)
    style_tf = test_transform(args.style_size, True)

    repeat = 15 if args.test_speed else 1
    time_list = []
    
    for i in range(repeat):
        image = Image.open(args.content)
        style = Image.open(args.style)
        if args.resize != 0:
            image = image.resize((args.resize, args.resize))
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size

        torch.cuda.synchronize()
        start_time = time.time()
        
        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = image.resize((int(aspect_ratio * args.thumb_size), args.thumb_size))
            thumbnail = content_tf(thumbnail).to(device)
            patches = preprocess(image, patch_size=PATCH_SIZE, padding=PADDING, transform=content_tf, cuda=False)
            style = style_tf(style).unsqueeze(0).to(device)
            
            print("patch:", patches.shape)
            print("thumbnail:", thumbnail.shape)
            print("style:", style.shape)
            
            if args.preserve_color:
                style = coral(style, thumbnail)
                
            with torch.no_grad():
                style_f = vgg(style)
                style_transfer_thumbnail(thumbnail, style_f, save=False if args.test_speed else True,
                                         save_path=os.path.join(args.outf, "thumb-%d.jpg" % args.thumb_size))
                style_transfer_high_resolution(patches, style_f, padding=PADDING, collection=False,
                    save_path=os.path.join(args.outf, "ours-patch%d-padding%d.jpg" % (PATCH_SIZE, PADDING)),
                    save=False if args.test_speed else True)
                # style_transfer_high_resolution(patches, style_f, padding=PADDING, collection=True,
                #                                save_path=os.path.join(args.outf, "baseline-width%d-padding%d.jpg" % (PATCH_SIZE, PADDING)))
        else:
            image = content_tf(image).to(device)
            style = style_tf(style).unsqueeze(0).to(device)
            print("image:", image.shape)
            print("style:", style.shape)
            
            with torch.no_grad():
                style_f = vgg(style)
                style_transfer_thumbnail(image, style_f, save_path=os.path.join(args.outf, "original_result.jpg"),
                                         save=False if args.test_speed else True)
        
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)
    
    print("time: %.2fs" % np.mean(time_list[-10:]))
    # print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
    # print("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))



