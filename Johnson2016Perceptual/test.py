from model_test import TransformerNet
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from utils import *
import torch
import argparse
import os
import time
import math

import sys
sys.path.append("..")
from tools import unpadding, preprocess
from thumb_instance_norm import init_thumbnail_instance_norm


def save_image(image, save_path):
    image = denormalize(image).mul_(255.0).add_(0.5).clamp_(0, 255)
    image = image.squeeze(0).permute(1, 2, 0).to(torch.uint8)
    image = image.cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)
    
    
def style_transfer_thumbnail(image_thumb, save_path, save=True):
    init_thumbnail_instance_norm(net, collection=True)
    stylized_thumb = net.forward(image_thumb)
    if save:
        save_image(stylized_thumb, save_path)
    

def style_transfer_high_resolution(patches, padding, save_path, collection=False, save=True):
    stylized_patches = []
    init_thumbnail_instance_norm(net, collection=collection)
    for patch in tqdm(patches):
        patch = patch.unsqueeze(0).to(device)
        stylized_patch = net.forward(patch)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True, help="path to content image")
    parser.add_argument("--model", type=str, required=True, help="path to checkpoint model")
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
    parser.add_argument('--thumb_size', type=int, default=1024, help='thumbnail size')
    parser.add_argument('--padding', type=int, default=32, help='padding size')
    parser.add_argument('--test_speed', action="store_true", help='test the speed')
    parser.add_argument('--outf', type=str, default="output", help='path to save')
    parser.add_argument('--URST', action="store_true", help='use URST framework')
    parser.add_argument("--device", type=str, default="cuda", help="device")

    args = parser.parse_args()
    print(args)
    
    device = torch.device(args.device)
    os.makedirs(args.outf, exist_ok=True)
    transform = style_transform()
    
    # Define model and load model checkpoint
    net = TransformerNet().to(device)
    checkpoint = torch.load(args.model)
    new_checkpoint = dict()
    for k, v in checkpoint.items():
        if not "norm" in k:
            new_checkpoint[k.replace("module.", "")] = v
        else:
            new_checkpoint[k.replace("module.", "")] = v.reshape(1, -1, 1, 1)
    checkpoint = new_checkpoint
    net.load_state_dict(checkpoint)
    net.eval()
    
    repeat = 15 if args.test_speed else 1
    time_list = []
    
    for i in range(repeat):
        PATCH_SIZE = args.patch_size
        PADDING = args.padding
        image = Image.open(args.content)
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size

        torch.cuda.synchronize()
        start_time = time.time()
        
        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = image.resize((int(aspect_ratio * args.thumb_size), args.thumb_size))
            thumbnail = transform(thumbnail).unsqueeze(0).to(device)
            patches = preprocess(image, padding=PADDING, transform=transform, patch_size=PATCH_SIZE, cuda=False)

            print("patch:", patches.shape)
            print("thumbnail:", thumbnail.shape)
    
            with torch.no_grad():
                style_transfer_thumbnail(thumbnail, save_path=os.path.join(args.outf, "thumb-%d.jpg" % args.thumb_size),
                                         save=False if args.test_speed else True)
                style_transfer_high_resolution(patches, padding=PADDING, collection=False,
                                               save_path=os.path.join(args.outf, "ours-patch%d-padding%d.jpg"%(PATCH_SIZE, PADDING)),
                                               save=False if args.test_speed else True)
                # style_transfer_high_resolution(patches, padding=PADDING, collection=True,
                #     save_path=os.path.join(args.outf, "baseline-width%d-padding%d.jpg"%(PATCH_SIZE, PADDING))
                # )
        else:
            image = transform(image).unsqueeze(0).to(device)
            print("image:", image.shape)
            with torch.no_grad():
                style_transfer_thumbnail(image, save_path=os.path.join(args.outf, "original_result.jpg"),
                                         save=False if args.test_speed else True)
        torch.cuda.synchronize()
        time_list.append(time.time()-start_time)
    
    print("time: %.2fs" % np.mean(time_list[-10:]))
    # print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
    # print("Total memory of the current GPU: %.4f GB" % (
    #             torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))
