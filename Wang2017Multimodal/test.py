import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from utils import save_image, tensor_normalizer
import torch.nn.functional as F
from mt import MT
import numpy as np
import os
import time
import math

import sys
sys.path.append("..")
from tools import preprocess, unpadding
from thumb_instance_norm import init_thumbnail_instance_norm


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True,
                        help="path to content image")
    parser.add_argument("--model", type=str,
                        default='still_life', help="path to checkpoint model")
    parser.add_argument('--patch_size', type=int,
                        default=1000, help='patch size')
    parser.add_argument('--thumb_size', type=int,
                        default=1024, help='thumbnail size')
    parser.add_argument('--padding', type=int, default=32, help='padding size')
    parser.add_argument('--test_speed', action="store_true",
                        help='test the speed')
    parser.add_argument('--outf', type=str,
                        default="output", help='path to save')
    parser.add_argument('--URST', action="store_true",
                        help='use URST framework')
    parser.add_argument("--device", type=str, default="cuda", help="device")
    args = parser.parse_args()

    return args


def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(tensor_normalizer())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer_thumbnail(image_thumb, save_path, save=True):
    init_thumbnail_instance_norm(net, collection=True)
    # stylized_thumb = net.forward(F.interpolate(image_thumb,256,mode='bilinear',align_corners=True))
    stylized_thumb = net.forward(image_thumb)
    if save:
        stylized_thumb = F.interpolate(
            stylized_thumb, image_thumb.shape[2:], mode='bilinear', align_corners=True)
        save_image(stylized_thumb, save_path)


def style_transfer_high_resolution(patches, padding, save_path, collection=False, save=True):
    stylized_patches = []
    init_thumbnail_instance_norm(net, collection=collection)
    for patch in tqdm(patches):
        patch = patch.unsqueeze(0).to(device)
        # stylized_patch = net.forward(F.interpolate(patch,256,mode='bilinear',align_corners=True))
        stylized_patch = net.forward(patch)
        stylized_patch = F.interpolate(
            stylized_patch, patch.shape[2:], mode='bilinear', align_corners=True)
        stylized_patch = unpadding(stylized_patch, padding=padding)
        stylized_patches.append(stylized_patch.cpu())

    stylized_patches = torch.cat(stylized_patches, dim=0)
    b, c, h, w = stylized_patches.shape
    stylized_patches = stylized_patches.unsqueeze(dim=0)
    stylized_patches = stylized_patches.view(
        1, b, c * h * w).permute(0, 2, 1).contiguous()
    output_size = (int(math.sqrt(b) * h), int(math.sqrt(b) * w))
    stylized_image = F.fold(stylized_patches, output_size=output_size,
                            kernel_size=(h, w), stride=(h, w))
    if save:
        save_image(stylized_image, save_path)


if __name__ == '__main__':
    args = init_args()

    device = torch.device(args.device)
    output_dir = Path(args.outf)
    output_dir.mkdir(exist_ok=True, parents=True)

    PATCH_SIZE = args.patch_size
    PADDING = args.padding

    net = MT()
    net.load(args.model)
    net = net.eval().to(device)

    tf = test_transform()

    repeat = 15 if args.test_speed else 1
    time_list = []

    for i in range(repeat):
        image = Image.open(args.content).convert('RGB')
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size

        torch.cuda.synchronize()
        start_time = time.time()

        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = image.resize(
                (int(aspect_ratio * args.thumb_size), args.thumb_size))
            thumbnail = tf(thumbnail).unsqueeze(0).to(device)
            patches = preprocess(image, padding=PADDING,
                                 transform=tf, patch_size=PATCH_SIZE, cuda=False)

            print("patch:", patches.shape)
            print("thumbnail:", thumbnail.shape)

            with torch.no_grad():
                style_transfer_thumbnail(thumbnail, save_path=os.path.join(args.outf, f"thumb-{args.thumb_size}-model-{args.model}.jpg"),
                                         save=False if args.test_speed else True)
                style_transfer_high_resolution(patches, padding=PADDING, collection=False,
                                               save_path=os.path.join(
                                                   args.outf, f"ours-patch{PATCH_SIZE}-padding{PADDING}-model-{args.model}.jpg"),
                                               save=False if args.test_speed else True)
        else:
            image = image.resize((256, 256))
            image = tf(image).unsqueeze(0).to(device)
            print("image:", image.shape)
            with torch.no_grad():
                style_transfer_thumbnail(image, save_path=os.path.join(args.outf, "original_result.jpg"),
                                         save=False if args.test_speed else True)

        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)

    print("time: %.2fs" % np.mean(time_list[-10:]))
