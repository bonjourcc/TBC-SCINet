from re import A
from my_tools import *
from dataLoadess import Imgdataset
from torch.utils.data import DataLoader

from utils import generate_masks, time2file_name
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import argparse
import random
from torch.autograd import Variable
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# from sk import *
from network_swinir_copy0104 import SwinIR
from network_swinir_copy0104 import SwinIR_2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
n_gpu = torch.cuda.device_count()
print('The number of GPU is {}'.format(n_gpu))

data_path = "/home1/wjx/wenjxcc/train"
test_path1 = "/home1/wjx/wenjxcc/test"


# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count()>1:
#     model=torch.nn.DataParallel(model,device_ids=[0,1])
# model.to(device)
# data_path="D:/桌面/wenjxcc/train"
# test_path1="D:/桌面/wenjxcc/test"
mask, mask_s = generate_masks(data_path)

parser = argparse.ArgumentParser(description='Setting, compressive rate, size, and mode')

parser.add_argument('--last_train', default=0, type=int, help='pretrain model')
parser.add_argument('--model_save_filename', default='2021_12_05_21_22_46', type=str, help='pretrain model save folder name')
parser.add_argument('--max_iter', default=100, type=int, help='max epoch')
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--B', default=8, type=int, help='compressive rate')
# parser.add_argument('--num_block', default=18, type=int, help='the number of reversible blocks')
# parser.add_argument('--num_group', default=2, type=int, help='the number of groups')
parser.add_argument('--size', default=[256, 256], type=int, help='input image resolution')
parser.add_argument('--mode', default='normal', type=str, help='training mode: reverse or normal')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net')


args = parser.parse_args()

class model_1129(nn.Module):

    def __init__(self,LayerNo):
        super(model_1129, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(8,8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.SwinIR = SwinIR(upscale=1,
                             window_size=8, img_range=1.,
                             embed_dim=60, mlp_ratio=2, upsampler='pixelshuffledirect')
        self.SwinIR_2 = SwinIR_2(upscale=1,
                             window_size=8, img_range=1.,
                             embed_dim=60, mlp_ratio=2, upsampler='pixelshuffledirect')

        self.conv2 = nn.Sequential(
            # nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
            #                    output_padding=(0, 1, 1)),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(inplace=True),
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_2 = nn.Sequential(
            # nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
            #                    output_padding=(0, 1, 1)),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
        )
        # self.re = ResidualBlock(nf=8)
        # self.re3 = ResidualBlock3(nf=8)
        onelayer=[]
        self.LayerNo = 10
        # if self.LayerNo <=0:
        #     return


        self.fcs = nn.ModuleList(onelayer)


    def forward(self,meas1,meas_re, args):
        # PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)

        # PhiTb = torch.mm(Phix, Phi)


        # meas1 = torch.unsqueeze(meas1,1)
        # print(meas1.shape)
        batch_size = meas_re.shape[0]
        # print(meas_re.shape)
        mask = self.mask.to(meas_re.cuda())
        
        
        maskt = mask.expand([batch_size, args.B, args.size[0], args.size[1]])
        maskt1 = maskt.mul(meas_re)
        # print(maskt.shape)1 8 256 256
        data_ = meas_re + maskt1
        # print("data",data_.shape)1 8 256 256


        [x_final_1,x6]= self.SwinIR(data_)


        meas = torch.unsqueeze(meas1, 1)
        out2 = meas - maskt.mul(x_final_1)
        # print(out2.shape)
        out2 = torch.sum(out2, dim=1)
        
        out2_re = torch.div(out2.cuda(), mask_s.cuda())
        
        
        out2_re = torch.unsqueeze(out2_re, 1)
        out2_re_ = out2_re + maskt.mul(out2_re)




        x_final_2 = self.SwinIR_2(out2_re_,x6)

        x_final = x_final_2 + x_final_1

        x_final = torch.unsqueeze(x_final,1)
        out_final = self.conv2(x_final)


        return [out_final, x_final_1]
























