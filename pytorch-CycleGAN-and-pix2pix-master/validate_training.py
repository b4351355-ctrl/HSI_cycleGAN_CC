import os
import glob
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# 引入项目模块
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


# --- 辅助函数：将 Tensor 转为 RGB numpy ---
def tensor2im_3ch(input_image):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()

    # 适配 300 通道：提取指定波段 [217, 156, 63]
    if image_numpy.shape[0] == 300:
        select_bands = [217, 156, 63]
        # 防止越界
        select_bands = [min(b, image_numpy.shape[0] - 1) for b in select_bands]
        image_numpy = image_numpy[select_bands, :, :]

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def main():
    # 1. 解析参数
    opt = TestOptions().parse()

    # -------- 【修复点：最安全的设备设置方式】 --------
    # 不再依赖 opt.gpu_ids，直接问 Pytorch 有没有显卡
    if torch.cuda.is_available():
        opt.device = torch.device("cuda:0")
    else:
        opt.device = torch.device("cpu")
    # -----------------------------------------------

    # 强制设置测试参数
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.eval = True

    # 定义 FID 参考集路径
    fid_real_dir = os.path.join(opt.results_dir, opt.name, 'fid_ref_real')
    util.mkdirs(fid_real_dir)
    has_generated_ref = len(os.listdir(fid_real_dir)) > 0

    # 2. 扫描 checkpoints
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_files = glob.glob(os.path.join(ckpt_dir, "*_net_G_A.pth"))

    # 提取 epoch
    epochs = []
    for f in model_files:
        match = re.search(r'(\d+)_net_G_A.pth', f)
        if match:
            epochs.append(int(match.group(1)))
    epochs = sorted(list(set(epochs)))

    # 筛选目标 Epoch (从第5轮开始测)
    target_epochs = [e for e in epochs if e >= 5]
    print(f"检测到 {len(target_epochs)} 个模型检查点: {target_epochs}")

    # 准备 CSV
    csv_path = os.path.join(opt.results_dir, opt.name, 'training_metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("加载已有测试记录...")
    else:
        df = pd.DataFrame(columns=['epoch', 'FID', 'PSNR', 'SSIM', 'RMSE'])

    # 3. 循环测试
    dataset = create_dataset(opt)

    for epoch in target_epochs:
        if epoch in df['epoch'].values:
            continue

        print(f"\n>>> 正在评估 Epoch {epoch} ...")

        opt.epoch = str(epoch)
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        fake_b_dir = os.path.join(opt.results_dir, opt.name, f'fid_fake_epoch_{epoch}')
        util.mkdirs(fake_b_dir)

        psnr_list, ssim_list, rmse_list = [], [], []

        for i, data in enumerate(tqdm(dataset)):
            model.set_input(data)
            model.test()

            # --- 获取数据 ---
            real_A = model.real_A  # 原图 (300ch)
            rec_A = model.rec_A  # 循环重建图 (300ch)
            fake_B = model.fake_B  # 生成图 (3ch)
            real_B = model.real_B  # 真实 RGB (3ch)

            # 1. 计算 RMSE
            mse = torch.mean((real_A - rec_A) ** 2).item()
            rmse_list.append(np.sqrt(mse))

            # 2. 转换图片
            img_real_A = tensor2im_3ch(real_A)
            img_rec_A = tensor2im_3ch(rec_A)
            img_fake_B = tensor2im_3ch(fake_B)
            img_real_B = tensor2im_3ch(real_B)

            # 保存图片 (用于 FID)
            save_name = f"img_{i:04d}.png"
            util.save_image(img_fake_B, os.path.join(fake_b_dir, save_name))

            if not has_generated_ref:
                util.save_image(img_real_B, os.path.join(fid_real_dir, save_name))

            # 3. 计算 PSNR & SSIM (计算 Rec_A 循环一致性)
            psnr_list.append(psnr_metric(img_real_A, img_rec_A, data_range=255))
            ssim_list.append(ssim_metric(img_real_A, img_rec_A, channel_axis=2, data_range=255))

        has_generated_ref = True

        # --- 计算 FID ---
        print(f"正在计算 Epoch {epoch} 的 FID...")
        try:
            # 直接使用 device 字符串，例如 "cuda:0"
            device_str = str(opt.device)
            cmd = f"python -m pytorch_fid \"{fid_real_dir}\" \"{fake_b_dir}\" --device {device_str}"

            result = subprocess.check_output(cmd, shell=True).decode()
            fid_score = float(re.search(r'FID:\s*(\d+\.\d+)', result).group(1))
        except Exception as e:
            print(f"FID 计算失败: {e}")
            fid_score = float('nan')

        # --- 记录结果 ---
        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        mean_rmse = np.mean(rmse_list)

        print(f"[Epoch {epoch}] FID: {fid_score:.2f} | PSNR(Rec_A): {mean_psnr:.2f} | SSIM(Rec_A): {mean_ssim:.4f}")

        new_row = pd.DataFrame([{
            'epoch': epoch,
            'FID': fid_score,
            'PSNR': mean_psnr,
            'SSIM': mean_ssim,
            'RMSE': mean_rmse
        }])
        df = pd.concat([df, new_row], ignore_index=True)

        df.sort_values(by='epoch', inplace=True)
        df.to_csv(csv_path, index=False)

    # 4. 画图
    plot_metrics(df, os.path.join(opt.results_dir, opt.name, 'metrics_plot.png'))


def plot_metrics(df, save_path):
    epochs = df['epoch'].tolist()

    plt.figure(figsize=(15, 10))

    # 1. FID
    plt.subplot(2, 2, 1)
    plt.plot(epochs, df['FID'], 'r-o', label='FID (Lower is Better)')
    plt.title('FID Score (Realism)')
    plt.grid(True)
    plt.legend()

    # 2. PSNR (Rec_A)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, df['PSNR'], 'g-o', label='PSNR (Higher is Better)')
    plt.title('PSNR (Reconstruction A)')
    plt.grid(True)
    plt.legend()

    # 3. SSIM (Rec_A)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, df['SSIM'], 'b-o', label='SSIM (Higher is Better)')
    plt.title('SSIM (Reconstruction A)')
    plt.grid(True)
    plt.legend()

    # 4. RMSE
    plt.subplot(2, 2, 4)
    plt.plot(epochs, df['RMSE'], 'k-o', label='RMSE (Lower is Better)')
    plt.title('RMSE (Spectral)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n图表已保存: {save_path}")


if __name__ == '__main__':
    main()