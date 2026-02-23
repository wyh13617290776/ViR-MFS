import os
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from models.common import RGB2YCrCb
resize_size = (640,480)
task = 'train'
task1 = 'test'

#---FMB数据集的路径---#
# /root/autodl-tmp/wyh_datasets/FMB_origin/vi/train
# /root/autodl-tmp/wyh_datasets/FMB_origin/ir/train
# /root/autodl-tmp/wyh_datasets/FMB_origin/Label/train
# self.vi_dir = os.path.join(rf'/root/autodl-tmp/wyh_datasets/FMB_origin/vi/{task}')
# self.ir_dir = os.path.join(rf'/root/autodl-tmp/wyh_datasets/FMB_origin/ir/{task}')
# self.vi_gt_dir = os.path.join(rf'/root/autodl-tmp/wyh_datasets/FMB_origin/vi_gt/{task}')
# self.label_dir = os.path.join(rf'/root/autodl-tmp/wyh_datasets/FMB_origin/Label/{task}')
#---MSRS数据集的路径
#/root/wyh_code_workspace/our_code/MSRS/Visible/{task}/MSRS
#/root/wyh_code_workspace/our_code/MSRS/Infrared/{task}/MSRS
#/root/wyh_code_workspace/our_code/MSRS/Label/{task}/MSRS

class vifs_dataloder(Dataset):
    def __init__(self):
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        # --- 1. 路径定义 ---
        self.vi_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Visible/{task}/MSRS')
        self.ir_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Infrared/{task}/MSRS')
        self.vi_gt_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Visible/{task}/MSRS')
        self.label_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Label/{task}/MSRS')
        
        # self.vi_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Visible/{task}/MSRS')
        # self.ir_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Infrared/{task}/MSRS')
        # self.vi_gt_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Visible_gt/{task}/MSRS')
        # self.label_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Label/{task}/MSRS')

        #获取全部的文件名：
        self.file_list =  os.listdir(self.vi_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 获取文件名
        file_name = self.file_list[index]
        #加载图像和标签,直接对灰度图进行处理，融合阶段的彩色用Ycrcb分解解决
        if resize_size == (None,None):
            vi_image = Image.open(os.path.join(self.vi_dir, file_name)).convert('L')
            ir_image = Image.open(os.path.join(self.ir_dir, file_name)).convert('L')
            label_image = Image.open(os.path.join(self.label_dir, file_name))
            try:
                vis_gt = Image.open(os.path.join(self.vi_gt_dir, file_name)).convert('L')
                #print('enhance')
            except:
                vis_gt = Image.open(os.path.join(self.vi_dir, file_name)).convert('L')
        else:
            vi_image = Image.open(os.path.join(self.vi_dir, file_name)).convert('L').resize(resize_size)
            ir_image = Image.open(os.path.join(self.ir_dir, file_name)).convert('L').resize(resize_size)
            label_image = Image.open(os.path.join(self.label_dir, file_name)).resize(resize_size)
            try:
                vis_gt = Image.open(os.path.join(self.vi_gt_dir, file_name)).convert('L').resize(resize_size)
            except:
                vis_gt = Image.open(os.path.join(self.vi_dir, file_name)).convert('L').resize(resize_size)
        # gt_image = Image.open(os.path.join(self.vi_gt_dir, file_name)).convert('L')


        #处理标签，确保为 Long Tensor 且形状正确 ---
        label_np = np.array(label_image, dtype=np.uint8)
        label_tensor = torch.from_numpy(label_np).long()

        vi_image = self.to_tensor(vi_image)
        ir_image = self.to_tensor(ir_image)
        vis_gt = self.to_tensor(vis_gt)
        # gt_image = self.to_tensor(gt_image)
        return vi_image, ir_image, label_tensor, vis_gt


class vifs_dataloder_test(Dataset):
    def __init__(self):
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        # --- 1. 路径定义 ---
        self.vi_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Visible/{task1}/MSRS')
        self.ir_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Infrared/{task1}/MSRS')
        self.label_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Label/{task1}/MSRS')
        
        # self.vi_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Visible/{task1}/MSRS')
        # self.ir_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Infrared/{task1}/MSRS')
        # self.label_dir = os.path.join(rf'/root/wyh_code_workspace/our_code/MSRS/Label/{task1}/MSRS')


        #获取全部的文件名：
        self.file_list =  os.listdir(self.vi_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 获取文件名
        file_name = self.file_list[index]
        #加载图像和标签,直接对灰度图进行处理，融合阶段的彩色用Ycrcb分解解决

        vi_image = Image.open(os.path.join(self.vi_dir, file_name)).resize(resize_size)
        ir_image = Image.open(os.path.join(self.ir_dir, file_name)).convert('L').resize(resize_size)
        label_image = Image.open(os.path.join(self.label_dir, file_name)).resize(resize_size)


        #处理标签，确保为 Long Tensor 且形状正确 ---
        label_np = np.array(label_image, dtype=np.uint8)
        label_tensor = torch.from_numpy(label_np).long()

        vi_image = self.to_tensor(vi_image)
        ir_image = self.to_tensor(ir_image)
        vi_y, cr, cb = RGB2YCrCb(vi_image)
        # gt_image = self.to_tensor(gt_image)
        return vi_y, ir_image, label_tensor, file_name, cr, cb