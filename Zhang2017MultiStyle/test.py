import torch
import utils
import argparse
import numpy as np
from model import Net
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
import os

import sys
sys.path.append("..")
from tools import unpadding, preprocess
from thumb_instance_norm import init_thumbnail_instance_norm

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def set_style_target(style, model):
    style = utils.tensor_rgb2bgr(style)
    style = style.unsqueeze(0)
    init_thumbnail_instance_norm(model, collection=True)
    model.setTarget(style)

def save_image(image, save_path):
    image = image.add_(0.5).clamp_(0, 255).squeeze(0)
    image = utils.tensor_bgr2rgb(image)
    image = image.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(save_path)
    
def style_transfer_thumbnail(thumb, model, save_path, save=True):
    thumb = thumb.unsqueeze(0)
    thumb = utils.preprocess_batch(thumb)
    init_thumbnail_instance_norm(model, collection=True)
    stylized_thumb = model.forward(thumb)
    if save:
        save_image(stylized_thumb, save_path)
    
    
def style_transfer_high_resolution(patches, model, padding, save_path,
                                   collection=False, save=True):
    stylized_patches = []
    init_thumbnail_instance_norm(model, collection=collection)
    for patch in tqdm(patches):
        patch = utils.tensor_rgb2bgr(patch).unsqueeze(0).to(device)
        stylized_patch = model.forward(patch)
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


def load_model(model_path):
    style_model = Net(ngf=128)
    model_dict = torch.load(model_path)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model = style_model.to(device)
    return style_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True, help="path to content image")
    parser.add_argument("--style", type=str, required=True, help="path to style image")
    parser.add_argument("--model", type=str, default="models/21styles.model",
                        help="path to checkpoint model")
    parser.add_argument('--patch_size', type=int, default=1000, help='patch size')
    parser.add_argument('--thumb_size', type=int, default=1024, help='thumbnail size')
    parser.add_argument('--style_size', type=int, default=1024, help='style size')
    parser.add_argument('--padding', type=int, default=32, help='padding size')
    parser.add_argument('--test_speed', action="store_true", help='test the speed')
    parser.add_argument('--outf', type=str, default="images/outputs", help='path to save')
    parser.add_argument('--URST', action="store_true", help='use URST framework')
    parser.add_argument("--device", type=str, default="cuda", help="device")

    args = parser.parse_args()
    print(args)
    
    device = torch.device(args.device)
    os.makedirs(args.outf, exist_ok=True)

    PATCH_SIZE = args.patch_size
    PADDING = args.padding
    content_tf = test_transform(0, False)
    style_tf = test_transform(args.style_size, True)

    model = load_model(model_path=args.model)
    model.eval()

    repeat = 15 if args.test_speed else 1
    time_list = []
    
    for i in range(repeat):
        # Prepare input
        image = Image.open(args.content)
        IMAGE_WIDTH, IMAGE_HEIGHT = image.size
        style = Image.open(args.style)

        torch.cuda.synchronize()
        start_time = time.time()
        
        style = np.array(style).transpose(2, 0, 1)
        style = torch.from_numpy(style).float().to(device)
        
        if args.URST:
            aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
            thumbnail = image.resize((int(aspect_ratio * args.thumb_size), args.thumb_size))
            patches = preprocess(image, padding=PADDING, patch_size=PATCH_SIZE,
                                 transform=content_tf, cuda=False).mul_(255.0)
            thumbnail = np.array(thumbnail).transpose(2, 0, 1)
            thumbnail = torch.from_numpy(thumbnail).float().to(device)
            
            print("patch:", patches.shape)
            print("thumb:", thumbnail.shape)
            print("style:", style.shape)
    
            with torch.no_grad():
                set_style_target(style=style, model=model)
                style_transfer_thumbnail(thumbnail, model=model,
                                         save_path=os.path.join(args.outf, "thumb-%d.jpg" % args.thumb_size),
                                         save=False if args.test_speed else True)
                style_transfer_high_resolution(
                    patches, model, padding=PADDING, collection=False,
                    save_path=os.path.join(args.outf, "ours-patch%d-padding%d.jpg" % (PATCH_SIZE, PADDING)),
                    save=False if args.test_speed else True
                )
                # style_transfer_high_resolution(patches, model, padding=PADDING, collection=True,
                #     save_path=os.path.join(args.outf, "baseline-width%d-padding%d.jpg" % (PATCH_SIZE, PADDING))
                # )
        else:
            image = np.array(image).transpose(2, 0, 1)
            image = torch.from_numpy(image).float().to(device)

            print("image:", image.shape)
            print("style:", style.shape)
    
            with torch.no_grad():
                set_style_target(style=style, model=model)
                style_transfer_thumbnail(image, model=model,
                                         save_path=os.path.join(args.outf, "original_result.jpg"),
                                         save=False if args.test_speed else True)
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)

    print("time: %.2fs" % np.mean(time_list[-10:]))
    # print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
    # print("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))