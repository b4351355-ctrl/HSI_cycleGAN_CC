import os
import torch
import random
import numpy as np
import scipy.io as sio
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms.functional as TF


def make_mat_dataset(dir, max_dataset_size=float("inf")):
    """查找目录下的所有 .mat 文件"""
    mats = []
    if not os.path.isdir(dir):
        return []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.mat'):
                path = os.path.join(root, fname)
                mats.append(path)
    return mats[:min(max_dataset_size, len(mats))]


class PairedHyperspectralDataset(BaseDataset):
    """
    专门用于 '有监督 Identity Loss' 的数据集加载器。

    A域: 仍然是随机读取 (Unpaired)
    B域: 读取 RGB 图片的同时，强制读取对应的原始高光谱 (.mat) 作为 B_raw
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # 必须存在 trainB_raw 文件夹，否则不仅无法计算ID Loss，逻辑也跑不通
        self.dir_B_raw = os.path.join(opt.dataroot, opt.phase + 'B_raw')

        if not os.path.isdir(self.dir_B_raw):
            raise ValueError(
                f"【错误】无法找到 B_raw 文件夹: {self.dir_B_raw}。使用 paired_hyperspectral 模式必须准备此数据。")

        self.A_paths = sorted(make_mat_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # 300 -> 3 模式
        self.input_nc = opt.input_nc  # 300
        self.output_nc = opt.output_nc  # 3

    def __load_mat(self, path):
        """读取 .mat 并归一化"""
        try:
            mat_data = sio.loadmat(path)
            # 假设 key 是 'data'，如果您的key不同请修改这里
            img_np = mat_data['data'].astype(np.float32)

            d_min, d_max = img_np.min(), img_np.max()
            if d_max > d_min:
                img_np = (img_np - d_min) / (d_max - d_min)
            img_np = (img_np - 0.5) / 0.5
            return torch.from_numpy(img_np)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((self.input_nc, 256, 256))

    def __getitem__(self, index):
        # ---------------- A 域 (白片, Unpaired) ----------------
        # 逻辑：随机读取，独立裁剪
        A_path = self.A_paths[index % self.A_size]
        A_tensor = self.__load_mat(A_path)

        # A 的预处理 (Crop, Flip)
        if 'crop' in self.opt.preprocess:
            h, w = A_tensor.shape[1], A_tensor.shape[2]
            crop_size = self.opt.crop_size
            y_A = random.randint(0, np.maximum(0, h - crop_size))
            x_A = random.randint(0, np.maximum(0, w - crop_size))
            A_tensor = A_tensor[:, y_A:y_A + crop_size, x_A:x_A + crop_size]

        if 'flip' in self.opt.preprocess and random.random() > 0.5:
            A_tensor = torch.flip(A_tensor, [2])  # flip width

        # ---------------- B 域 (RGB + Raw, Paired) ----------------
        # 逻辑：读取 B (RGB)，根据 B 的文件名找 B_raw (HS)，两者做完全相同的裁剪

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # 1. 读取 RGB B
        B_img = Image.open(B_path).convert('RGB')

        # 2. 读取对应的 HS B (B_raw)
        B_name = os.path.basename(B_path)
        B_raw_name = os.path.splitext(B_name)[0] + '.mat'  # 替换后缀
        B_raw_path = os.path.join(self.dir_B_raw, B_raw_name)

        # 加载 B_raw (如果文件不存在会报错，保证数据严谨性)
        if not os.path.exists(B_raw_path):
            raise FileNotFoundError(f"找不到对应的 B_raw 文件: {B_raw_path}")
        B_raw_tensor = self.__load_mat(B_raw_path)

        # 3. 同步变换 (Sync Transform)
        # 这一步非常关键：必须保证 RGB 和 HS 切的是同一个位置！

        # 生成随机参数 (比如: crop位置是 (10, 20), flip=True)
        transform_params = get_params(self.opt, B_img.size)

        # A. 对 RGB 图片应用变换
        B_transform = get_transform(self.opt, params=transform_params, grayscale=False)
        B_tensor = B_transform(B_img)

        # B. 对 HS Tensor 手动应用相同的变换
        # B_raw_tensor shape: (300, H, W)

        # Sync Crop
        if 'crop' in self.opt.preprocess:
            crop_x, crop_y = transform_params['crop_pos']
            crop_size = self.opt.crop_size
            B_raw_tensor = B_raw_tensor[:, crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

        # Sync Flip
        if 'flip' in self.opt.preprocess and transform_params['flip']:
            # Tensor翻转: dim 2 是宽度
            B_raw_tensor = torch.flip(B_raw_tensor, [2])

        return {
            'A': A_tensor,
            'B': B_tensor,
            'B_raw': B_raw_tensor,  # 这里的 B_raw 和 B 是严格空间对齐的
            'A_paths': A_path,
            'B_paths': B_path
        }

    def __len__(self):
        return max(self.A_size, self.B_size)