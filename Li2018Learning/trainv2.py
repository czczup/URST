import os
import torch
import argparse
import torch.optim as optim
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder3,encoder4
from libs.models import decoder3,decoder4
from libs.models import encoder5 as loss_network
import numpy as np
import torch.nn.functional as F
import time
import datetime
from tensorboardX import SummaryWriter
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_process(image):
    image = image.permute(1, 2, 0) * 255.0
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.to(torch.uint8).cpu().data.numpy()
    return image

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                        help='pre-trained encoder path')
    parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth',
                        help='used for loss network')
    parser.add_argument("--stylePath", default="datasets/wikiart/train/",
                        help='path to wikiArt dataset')
    parser.add_argument("--contentPath", default="datasets/coco2014/train2014/train2014/",
                        help='path to MSCOCO dataset')
    parser.add_argument("--matrixPath", default='models/r41.pth',
                        help='pre-trained model path')
    parser.add_argument("--outf", default="checkpoints/",
                        help='folder to output images and model checkpoints')
    parser.add_argument("--content_layers", default="r41",
                        help='layers for content')
    parser.add_argument("--style_layers", default="r11,r21,r31,r41",
                        help='layers for style')
    parser.add_argument("--batchSize", type=int,default=8,
                        help='batch size')
    parser.add_argument("--niter", type=int,default=100000,
                        help='iterations to train the model')
    parser.add_argument('--loadSize', type=int, default=512,
                        help='scale image size')
    parser.add_argument('--fineSize', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--thumbSize', type=int, default=256,
                        help='thumbnail image size')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--content_weight", type=float, default=1.0,
                        help='content loss weight')
    parser.add_argument("--sp_weight", type=float, default=1.0,
                        help='stroke perceptual loss weight')
    parser.add_argument("--style_weight", type=float, default=0.02,
                        help='style loss weight')
    parser.add_argument("--log_interval", type=int, default=100,
                        help='log interval')
    parser.add_argument("--gpu_id", type=int, default=0,
                        help='which gpu to use')
    parser.add_argument("--save_interval", type=int, default=5000,
                        help='checkpoint save interval')
    parser.add_argument("--layer", default="r41",
                        help='which features to transfer, either r31 or r41')
    parser.add_argument('--log_dir', default='./logs/trainv2',
                        help='Directory to save the log')
    
    ################# PREPARATIONS #################
    opt = parser.parse_args()
    opt.content_layers = opt.content_layers.split(',')
    opt.style_layers = opt.style_layers.split(',')
    opt.cuda = torch.cuda.is_available()
    if(opt.cuda):
        torch.cuda.set_device(opt.gpu_id)
    
    os.makedirs(opt.outf,exist_ok=True)
    cudnn.benchmark = True
    print_options(opt)
    log_dir = Path(opt.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    ################# DATA #################
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader_ = torch.utils.data.DataLoader(dataset     = content_dataset,
                                                  batch_size  = opt.batchSize,
                                                  shuffle     = True,
                                                  num_workers = 1,
                                                  drop_last   = True)
    content_loader = iter(content_loader_)
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style_loader_ = torch.utils.data.DataLoader(dataset     = style_dataset,
                                                batch_size  = opt.batchSize,
                                                shuffle     = True,
                                                num_workers = 1,
                                                drop_last   = True)
    style_loader = iter(style_loader_)
    
    ################# MODEL #################
    vgg5 = loss_network()
    if(opt.layer == 'r31'):
        matrix = MulLayer('r31')
        vgg = encoder3()
        dec = decoder3()
    elif(opt.layer == 'r41'):
        matrix = MulLayer('r41')
        vgg = encoder4()
        dec = decoder4()
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    # dec.load_state_dict(torch.load(opt.decoder_dir))
    vgg5.load_state_dict(torch.load(opt.loss_network_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))
    for param in vgg.parameters():
        param.requires_grad = False
    for param in vgg5.parameters():
        param.requires_grad = False
    for param in matrix.parameters():
        param.requires_grad = False
    # for param in dec.parameters():
    #     param.requires_grad = False
    
    ################# LOSS & OPTIMIZER #################
    criterion = LossCriterion(opt.style_layers,
                              opt.content_layers,
                              opt.style_weight,
                              opt.content_weight,
                              opt.sp_weight)
    optimizer = optim.Adam(dec.parameters(), opt.lr)
    
    
    ################# GPU  #################
    if(opt.cuda):
        vgg.cuda()
        dec.cuda()
        vgg5.cuda()
        matrix.cuda()
    
    ################# TRAINING #################
    def adjust_learning_rate(optimizer, iteration):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr / (1 + iteration * 1e-5)
    
    start_time = time.time()
    
    for iteration in range(1, opt.niter+1):
        optimizer.zero_grad()
        try:
            content, _ = content_loader.next()
        except IOError:
            content, _ = content_loader.next()
        except StopIteration:
            content_loader = iter(content_loader_)
            content, _ = content_loader.next()
        except:
            continue
    
        try:
            style, _ = style_loader.next()
        except IOError:
            style, _ = style_loader.next()
        except StopIteration:
            style_loader = iter(style_loader_)
            style, _ = style_loader.next()
        except:
            continue
    
    
        content = content.cuda()  # [8, 3, 512, 512]
        randx = np.random.randint(0, opt.fineSize - opt.thumbSize)
        randy = np.random.randint(0, opt.fineSize - opt.thumbSize)
        position = [randx, randx + opt.thumbSize, randy, randy + opt.thumbSize]
        patch = content[..., randx:randx + opt.thumbSize,
                          randy:randy + opt.thumbSize]  # [8, 3, 256, 256]
        thumb = F.interpolate(content, (opt.thumbSize, opt.thumbSize),
                              mode='bilinear', align_corners=False)  # [8, 3, 256, 256]
    
        """ style images """
        style = style.cuda()  # [8, 3, 512, 512]
        style = F.interpolate(style, (opt.thumbSize, opt.thumbSize),
                                     mode='bilinear', align_corners=False)  # [8, 3, 256, 256]
    
        # forward
        sF = vgg(style)
        pF = vgg(patch)
        tF = vgg(thumb)
    
        if(opt.layer == 'r41'):
            feature_t, feature_p = matrix.forwardv2(tF[opt.layer], pF[opt.layer], sF[opt.layer])
        else:
            feature_t, feature_p = matrix.forwardv2(tF, pF, sF)
        transfer_t = dec(feature_t)
        transfer_p = dec(feature_p)
        
        transfer_tp = F.interpolate(transfer_t, scale_factor=2, mode='bilinear', align_corners=False)
        transfer_tp = transfer_tp[..., position[0]:position[1], position[2]:position[3]]
        
        
        sF = vgg5(style)
        tF = vgg5(thumb)
        ttF = vgg5(transfer_t)
        tpF = vgg5(transfer_p)
        ttpF = vgg5(transfer_tp)
        loss, styleLoss, contentLoss, spLoss = criterion.forwardv2(tF, sF, ttF, tpF, ttpF)
    
        # backward & optimization
        loss.backward()
        optimizer.step()
    
        writer.add_scalar('loss_content', contentLoss.item(), iteration)
        writer.add_scalar('loss_style', styleLoss.item(), iteration)
        writer.add_scalar('loss_sp', spLoss.item(), iteration)
        
        eta_seconds = ((time.time() - start_time) / (iteration + 1)) * (opt.niter - (iteration + 1))
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        print("[trainv2] Iters: %d/%d || Content Loss: %.2f || Style Loss: %.2f || SP Loss: %.2f || Estimated Time: %s]"
              % (iteration, opt.niter, contentLoss.item(), styleLoss.item(), spLoss.item(), eta_string))
        
        adjust_learning_rate(optimizer, iteration)
    
        if((iteration) % opt.log_interval == 0):
            # transfer = transfer.clamp(0, 1)
            # concat = torch.cat((content, style, transfer.cpu()),dim=0)
            # vutils.save_image(concat, '%s/%d.png' % (opt.outf, iteration), normalize=True, scale_each=True, nrow=opt.batchSize)
    
            writer.add_image('trainv2/content_patch',
                             image_process(patch[0]),
                             global_step=iteration, dataformats='HWC')
            writer.add_image('trainv2/content_thumb',
                             image_process(thumb[0]),
                             global_step=iteration, dataformats='HWC')
            writer.add_image('trainv2/style_thumb',
                             image_process(style[0]),
                             global_step=iteration, dataformats='HWC')
            writer.add_image('trainv2/stylized_thumb',
                             image_process(transfer_t[0]),
                             global_step=iteration, dataformats='HWC')
            writer.add_image('trainv2/stylized_patch',
                             image_process(transfer_p[0]),
                             global_step=iteration, dataformats='HWC')
    
        if(iteration > 0 and (iteration) % opt.save_interval == 0):
            torch.save(dec.state_dict(), '%s/dec_%s.pth' % (opt.outf, opt.layer))
