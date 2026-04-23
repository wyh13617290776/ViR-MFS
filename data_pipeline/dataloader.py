import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.common import RGB2YCrCb

class VIFSDataset(Dataset):
    """
    统一的可见光与红外图像融合分割数据集加载器
    """
    def __init__(self, mode, vi_dir, ir_dir, label_dir, resize_size=(640, 480)):
        """
        初始化数据集。
        
        Args:
            mode (str): 'train' 或 'test'，决定返回值的组合。
            vi_dir (str): 可见光图像文件夹路径。
            ir_dir (str): 红外图像文件夹路径。
            label_dir (str): 语义分割标签文件夹路径。
            resize_size (tuple): 图像缩放尺寸 (W, H)，若为(None, None)则不缩放。
        """
        super().__init__()
        self.mode = mode
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.label_dir = label_dir
        self.resize_size = resize_size
        self.to_tensor = transforms.ToTensor()

        # 获取全部的文件名
        if not os.path.exists(self.vi_dir):
            raise FileNotFoundError(f"目录不存在: {self.vi_dir}")
        self.file_list = sorted(os.listdir(self.vi_dir)) # 加上 sorted 保证顺序稳定

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        
        # 1. 路径拼接
        vi_path = os.path.join(self.vi_dir, file_name)
        ir_path = os.path.join(self.ir_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        # 2. 读取图像与标签 (针对不同的 mode 做微调)
        if self.mode == 'train':
            vi_image = Image.open(vi_path).convert('L')
            ir_image = Image.open(ir_path).convert('L')
            label_image = Image.open(label_path)
            
            if self.resize_size[0] is not None:
                vi_image = vi_image.resize(self.resize_size)
                ir_image = ir_image.resize(self.resize_size)
                label_image = label_image.resize(self.resize_size)
                
            vi_image = self.to_tensor(vi_image)
            ir_image = self.to_tensor(ir_image)
            
            # 处理标签，确保为 Long Tensor 且形状正确
            label_np = np.array(label_image, dtype=np.uint8)
            label_tensor = torch.from_numpy(label_np).long()
            
            return vi_image, ir_image, label_tensor

        elif self.mode == 'test':
            # 测试阶段：保留可见光的彩色信息用于色彩恢复
            vi_image_color = Image.open(vi_path)
            ir_image = Image.open(ir_path).convert('L')
            label_image = Image.open(label_path)
            
            if self.resize_size[0] is not None:
                vi_image_color = vi_image_color.resize(self.resize_size)
                ir_image = ir_image.resize(self.resize_size)
                label_image = label_image.resize(self.resize_size)

            label_np = np.array(label_image, dtype=np.uint8)
            label_tensor = torch.from_numpy(label_np).long()

            vi_image_tensor = self.to_tensor(vi_image_color)
            ir_image_tensor = self.to_tensor(ir_image)
            
            # 将 RGB 转换为 YCrCb 空间，分离出亮度(Y)与色度(Cr, Cb)
            vi_y, cr, cb = RGB2YCrCb(vi_image_tensor)
            
            # 返回包含了色度通道和文件名的拓展信息，用于后续重构彩色图像
            return vi_y, ir_image_tensor, label_tensor, file_name, cr, cb
        else:
            raise ValueError(f"不支持: {self.mode}")