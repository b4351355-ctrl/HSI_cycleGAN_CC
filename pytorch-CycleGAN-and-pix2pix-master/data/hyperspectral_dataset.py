import os
import torch
import random
import numpy as np
import scipy.io as sio
import cv2  # 【新增】用于读取 16-bit PNG
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
# from PIL import Image # PIL 不再用于读取主训练数据
import torchvision.transforms.functional as TF


def make_mat_dataset(dir, max_dataset_size=float("inf")):
    """保持不变：查找 .mat 文件"""
    mats = []
    if not os.path.isdir(dir):
        return []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.mat'):
                path = os.path.join(root, fname)
                mats.append(path)
    return mats[:min(max_dataset_size, len(mats))]


class HyperspectralDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_B_raw = os.path.join(opt.dataroot, opt.phase + 'B_raw')

        # A 是高光谱 .mat
        self.A_paths = sorted(make_mat_dataset(self.dir_A, opt.max_dataset_size))

        self.is_B_hsi = (opt.output_nc == opt.input_nc == 300)

        if self.is_B_hsi:
            self.B_paths = sorted(make_mat_dataset(self.dir_B, opt.max_dataset_size))
        else:
            # 【修改点1】这里会查找 .png (你新生成的) 或 .jpg
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
            print(f"B域加载: {len(self.B_paths)} 张图片 (支持 16-bit PNG)")

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # 预加载 crop_size，避免在 getitem 里反复调用 self.opt
        self.crop_size = opt.crop_size if 'crop' in opt.preprocess else 256

    def __load_mat(self, path):
        """保持不变：加载 .mat 并归一化到 [-1, 1]"""
        try:
            mat_data = sio.loadmat(path)
            # 假设 key 是 'data'，如果不确定可以用 list(mat_data.keys())[-1]
            if 'data' in mat_data:
                img_np = mat_data['data'].astype(np.float32)
            else:
                # 自动寻找非元数据的 key
                key = [k for k in mat_data.keys() if not k.startswith('__')][0]
                img_np = mat_data[key].astype(np.float32)

            # 动态 Min-Max 归一化 (根据你的原始逻辑)
            d_min, d_max = img_np.min(), img_np.max()
            if d_max > d_min:
                img_np = (img_np - d_min) / (d_max - d_min)

            # [0, 1] -> [-1, 1]
            img_np = (img_np - 0.5) / 0.5

            # 转换为 Tensor (C, H, W)
            # 注意：scipy读取的mat通常是 (H, W, C)，如果是这样需要 permute
            # 如果你的 mat 本身就是 (C, H, W) 则不需要 permute
            tensor = torch.from_numpy(img_np)
            if tensor.shape[0] != 300 and tensor.shape[2] == 300:
                tensor = tensor.permute(2, 0, 1)

            return tensor
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((300, 256, 256))

    def __load_16bit_png(self, path):
        """【新增函数】专门读取 16-bit PNG 并归一化到 [-1, 1]"""
        # 1. 使用 OpenCV 读取原始数据 (uint16)
        # flags=-1 等同于 IMREAD_UNCHANGED，保留位深
        img_np = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img_np is None:
            raise ValueError(f"无法读取图片: {path}")

        # 2. BGR -> RGB
        # OpenCV 读彩色图默认是 BGR
        if len(img_np.shape) == 3:  # 彩色
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # 3. 归一化逻辑
        if img_np.dtype == np.uint16:
            # 16-bit: 除以 65535.0 归一化到 [0, 1]
            img_np = img_np.astype(np.float32) / 65535.0
        else:
            # 兼容 8-bit: 除以 255.0
            img_np = img_np.astype(np.float32) / 255.0

        # 4. [0, 1] -> [-1, 1] (CycleGAN 标准输入范围)
        img_np = (img_np - 0.5) / 0.5

        # 5. 转 Tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        return tensor

    def __getitem__(self, index):
        # 1. Load A
        A_path = self.A_paths[index % self.A_size]
        A_tensor = self.__load_mat(A_path)

        # 2. Load B
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        if self.is_B_hsi:
            B_tensor = self.__load_mat(B_path)
            B_raw_tensor = B_tensor.clone()
        else:
            # 【修改点2】使用新的 16-bit 读取函数
            B_tensor = self.__load_16bit_png(B_path)

            # 加载对应的 Raw mat 数据 (如果有)
            B_name = os.path.basename(B_path)
            # 假设 png 名字是 Sample_Ref_A.png，对应的 raw 是 Sample_Ref_A.mat
            B_raw_name = os.path.splitext(B_name)[0] + '.mat'
            B_raw_path = os.path.join(self.dir_B_raw, B_raw_name)

            if os.path.exists(B_raw_path):
                B_raw_tensor = self.__load_mat(B_raw_path)
            else:
                # 如果没有 raw mat，造一个占位符，避免 DataLoader 报错
                # 注意通道数要和 input_nc 一致，这里假设 A 是 300
                B_raw_tensor = torch.zeros((300, B_tensor.shape[1], B_tensor.shape[2]))

        # ---------------- Sync Crop (随机裁剪) ----------------
        if 'crop' in self.opt.preprocess:
            h, w = A_tensor.shape[1], A_tensor.shape[2]
            h_B, w_B = B_tensor.shape[1], B_tensor.shape[2]

            # 保护机制：如果图片比 crop_size 小，就不裁了或者 resize
            if h < self.crop_size or w < self.crop_size:
                # 简单处理：跳过裁剪
                pass
            else:
                x = random.randint(0, np.maximum(0, w - self.crop_size))
                y = random.randint(0, np.maximum(0, h - self.crop_size))

                A_tensor = A_tensor[:, y:y + self.crop_size, x:x + self.crop_size]

                # 如果 A 和 B 尺寸不一样（非成对数据），B 应该独立裁剪
                # 如果是成对数据 (pix2pix模式)，则应该用同一个 x, y
                # CycleGAN 通常是非成对的，所以这里 B 独立生成坐标
                x_B = random.randint(0, np.maximum(0, w_B - self.crop_size))
                y_B = random.randint(0, np.maximum(0, h_B - self.crop_size))

                B_tensor = B_tensor[:, y_B:y_B + self.crop_size, x_B:x_B + self.crop_size]

                # B_raw 跟随 B 的裁剪 (假设它们空间对齐)
                if B_raw_tensor.shape[1] == h_B and B_raw_tensor.shape[2] == w_B:
                    B_raw_tensor = B_raw_tensor[:, y_B:y_B + self.crop_size, x_B:x_B + self.crop_size]

        return {
            'A': A_tensor,
            'B': B_tensor,
            'B_raw': B_raw_tensor,
            'A_paths': A_path,
            'B_paths': B_path
        }

    def __len__(self):
        return max(self.A_size, self.B_size)