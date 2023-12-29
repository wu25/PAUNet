

"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""
import time

import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from basicsr.archs.PAUNet_arch import PAUNet
from dataloaders.data_rgb import get_validation_data
import utils
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/home/amax/Desktop/Dataset/Private/SIDD_patches/val',
    type=str, help='Directory of validation images')

parser.add_argument('--result_dir', default='./results/denoising/',
    type=str, help='Directory for results')


parser.add_argument('--weights', default='/usr/disk/real_models/.pth',
    type=str, help='Path to weights')

# parser.add_argument('--weights', default='/usr/disk/wjf/mlp_2/real_models/MIRNet-v2.pth',
#     type=str, help='Path to weights')


# parser.add_argument('--weights', default='/usr/disk/wjf/mlp_2/real_models/ffdnet_color_clip.pth',
#     type=str, help='Path to weights')

# parser.add_argument('--weights', default='/usr/disk/wjf/mlp_2/real_models/rid.pth',
#     type=str, help='Path to weights')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images',default='false', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)




model_restoration=PAUNet()


utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
# model_restoration.cpu()
model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()





from utils.ssim_utils import calculate_ssim
# test_dataset = get_validation_data(args.input_dir)
# test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)


with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    start=time.time()
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].cuda()
        rgb_noisy = data_test[1].cuda()
        # rgb_gt=data_test[0].cpu()
        # rgb_noisy=data_test[1].cpu()
        filenames = data_test[2]
        # print("***",rgb_noisy)
        rgb_restored = model_restoration(rgb_noisy)
        # print("&&&&&&&&",rgb_restored)
        rgb_restored = torch.clamp(rgb_restored,0,1)

        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))
        ssim_val_rgb.append(calculate_ssim(rgb_restored,rgb_gt,0))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                # denoised_img = rgb_restored[batch]
                denoised_img = img_as_ubyte(rgb_restored[batch])
                # print(denoised_img)
                utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)
    end=time.time()
    print("推理时间: {:.4f} 秒".format((end - start)/7822))
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
print("PSNR: %.3f " %(psnr_val_rgb))
print("PSNR: %.4f " %(ssim_val_rgb))

