import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm import tqdm  # 进度条


def tensor2im_3ch(input_image):
    """强制将Tensor转为3通道numpy图片用于计算PSNR/SSIM"""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()

    # 如果是300通道，取您设定的波段 [217, 156, 63]
    if image_numpy.shape[0] == 300:
        select_bands = [217, 156, 63]
        image_numpy = image_numpy[select_bands, :, :]

    # 归一化转uint8
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


if __name__ == '__main__':
    opt = TestOptions().parse()

    # --------【修复点：添加设备设置】--------
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # -------------------------------------

    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1  # no visdom display

    # # 强制设置 output_nc 为 300 以匹配您的模型
    # opt.input_nc = 300
    # opt.output_nc = 3

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # 初始化指标记录器
    metrics = {
        'RMSE_300ch': [],  # 300通道的光谱误差
        'PSNR_RGB': [],  # 3通道RGB的信噪比
        'SSIM_RGB': []  # 3通道RGB的结构相似度
    }

    # 创建保存目录 (用于FID计算)
    fake_b_dir = os.path.join(opt.results_dir, opt.name, 'fid_fake_B')
    real_b_dir = os.path.join(opt.results_dir, opt.name, 'fid_real_B')
    util.mkdirs([fake_b_dir, real_b_dir])

    print(f"开始测试 {len(dataset)} 张图片...")

    for i, data in tqdm(enumerate(dataset)):
        model.set_input(data)
        model.test()

        # --- 1. 计算 300通道 RMSE (结构/光谱一致性) ---
        # 获取原始数据 (real_A) 和 循环重建数据 (rec_A)
        # 形状: (1, 300, H, W), 范围 [-1, 1]
        real_A = model.real_A
        rec_A = model.rec_A

        # 计算 MSE -> RMSE
        mse = torch.mean((real_A - rec_A) ** 2).item()
        rmse = np.sqrt(mse)
        metrics['RMSE_300ch'].append(rmse)

        # --- 2. 计算 3通道 PSNR & SSIM (视觉一致性) ---
        # 转换为可视化的 RGB numpy 数组 (0-255)
        img_real_A = tensor2im_3ch(real_A)
        img_rec_A = tensor2im_3ch(rec_A)

        # PSNR
        p = psnr_metric(img_real_A, img_rec_A, data_range=255)
        metrics['PSNR_RGB'].append(p)

        # SSIM (多通道需设置 channel_axis)
        s = ssim_metric(img_real_A, img_rec_A, channel_axis=2, data_range=255)
        metrics['SSIM_RGB'].append(s)

        # --- 3. 保存图片用于 FID 计算 ---
        # 保存 Fake B (生成的染色图)
        fake_B = model.fake_B
        img_fake_B = tensor2im_3ch(fake_B)
        save_path = os.path.join(fake_b_dir, f"fake_B_{i:04d}.png")
        util.save_image(img_fake_B, save_path)

        # 保存 Real B (真实的染色图，来自数据集)
        real_B = model.real_B
        img_real_B = tensor2im_3ch(real_B)
        save_path_b = os.path.join(real_b_dir, f"real_B_{i:04d}.png")
        util.save_image(img_real_B, save_path_b)

    # --- 输出平均指标 ---
    print("\n" + "=" * 20 + " 测试报告 " + "=" * 20)
    print(f"测试集数量: {len(dataset)}")
    print(f"1. 结构与光谱一致性 (越低越好):")
    print(f"   RMSE (300ch): {np.mean(metrics['RMSE_300ch']):.4f}")

    print(f"2. 循环重建视觉质量 (越高越好):")
    print(f"   PSNR (RGB)  : {np.mean(metrics['PSNR_RGB']):.2f} dB")
    print(f"   SSIM (RGB)  : {np.mean(metrics['SSIM_RGB']):.4f}")
    print("=" * 50)

    print(f"\n提示: 图片已保存至 {opt.results_dir}/{opt.name}/")
    print(f"请使用 pytorch-fid 计算 FID 分数以评估染色真实度。")