import os
import glob
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import concurrent.futures
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# 引入您项目中的模块
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


# --- 辅助函数：将 Tensor 转为 RGB numpy (CPU版) ---
def tensor2im_3ch_cpu(image_tensor):
    image_numpy = image_tensor[0].float().numpy()

    # 适配 300 通道：提取指定波段 [217, 156, 63] 用于可视化
    if image_numpy.shape[0] == 300:
        select_bands = [217, 156, 63]
        # 防止越界，做个限制
        select_bands = [min(b, image_numpy.shape[0] - 1) for b in select_bands]
        image_numpy = image_numpy[select_bands, :, :]

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


# --- 线程任务：处理单张图片 ---
def process_single_image(i, real_A_cpu, rec_A_cpu, fake_B_cpu, real_B_cpu, fake_b_dir, fid_real_dir, save_ref_flag):
    """
    1. 保存 Fake_B 用于计算 FID (反映真实度)
    2. 计算 Real_A vs Rec_A 的 PSNR/SSIM (反映结构一致性/稳定性)
    """
    # 转换图片
    img_real_A = tensor2im_3ch_cpu(real_A_cpu)
    img_rec_A = tensor2im_3ch_cpu(rec_A_cpu)
    img_fake_B = tensor2im_3ch_cpu(fake_B_cpu)
    img_real_B = tensor2im_3ch_cpu(real_B_cpu)

    # --- A. 保存图片用于 FID 计算 ---
    save_name = f"img_{i:04d}.png"
    util.save_image(img_fake_B, os.path.join(fake_b_dir, save_name))

    # 仅在第一次运行时保存真实参考集
    if save_ref_flag:
        util.save_image(img_real_B, os.path.join(fid_real_dir, save_name))

    # --- B. 计算循环一致性指标 (Cycle Consistency) ---
    # 比较 "原图 A" 和 "循环重建 A"。对于非配对数据，这是监控模型是否崩坏的最佳指标。
    # 如果 PSNR 下降，说明模型丢失了病理结构信息。
    psnr = psnr_metric(img_real_A, img_rec_A, data_range=255)
    ssim = ssim_metric(img_real_A, img_rec_A, channel_axis=2, data_range=255)

    return psnr, ssim


def main():
    # 1. 解析参数
    opt = TestOptions().parse()

    # -------- 修复 Device 设置 --------
    if len(opt.gpu_ids) > 0:
        opt.device = torch.device(f"cuda:{opt.gpu_ids[0]}")
    else:
        opt.device = torch.device("cpu")
    # --------------------------------

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

    # 2. 自动扫描 Checkpoints
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_files = glob.glob(os.path.join(ckpt_dir, "*_net_G_A.pth"))

    epochs = []
    for f in model_files:
        match = re.search(r'(\d+)_net_G_A.pth', f)
        if match:
            epochs.append(int(match.group(1)))
    epochs = sorted(list(set(epochs)))

    # 筛选：从第 5 轮开始测 (您可以根据需要修改这里)
    target_epochs = [e for e in epochs if e >= 5]
    print(f"检测到 {len(target_epochs)} 个模型检查点: {target_epochs}")

    # 准备 CSV 记录
    csv_path = os.path.join(opt.results_dir, opt.name, 'training_metrics.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['epoch', 'FID', 'PSNR_Cycle', 'SSIM_Cycle'])

    dataset = create_dataset(opt)

    # 3. 逐个 Epoch 测试
    for epoch in target_epochs:
        if epoch in df['epoch'].values:
            continue  # 跳过已测过的

        print(f"\n>>> [Epoch {epoch}] 正在推理与计算...")

        # 加载特定 Epoch 的模型
        opt.epoch = str(epoch)
        model = create_model(opt)
        model.setup(opt)
        model.eval()

        fake_b_dir = os.path.join(opt.results_dir, opt.name, f'fid_fake_epoch_{epoch}')
        util.mkdirs(fake_b_dir)

        # 检查是否需要生成参考集
        has_generated_ref = len(os.listdir(fid_real_dir)) > (opt.num_test * 0.9)

        psnr_list, ssim_list = [], []

        # --- 多线程流水线 ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []

            # GPU 推理循环
            for i, data in enumerate(tqdm(dataset, desc="GPU Inference")):
                model.set_input(data)
                model.test()

                # 搬运数据到 CPU
                real_A_cpu = model.real_A.data.cpu().clone()
                rec_A_cpu = model.rec_A.data.cpu().clone()
                fake_B_cpu = model.fake_B.data.cpu().clone()
                real_B_cpu = model.real_B.data.cpu().clone()

                # 提交给 CPU 线程
                future = executor.submit(
                    process_single_image,
                    i, real_A_cpu, rec_A_cpu, fake_B_cpu, real_B_cpu,
                    fake_b_dir, fid_real_dir, not has_generated_ref
                )
                futures.append(future)

            # 等待 CPU 计算完成
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="CPU Calculating"):
                p, s = f.result()
                psnr_list.append(p)
                ssim_list.append(s)

        # --- 计算 FID ---
        print(f"正在计算 Epoch {epoch} 的 FID...")
        try:
            # 修复：正确传递 device 字符串
            device_str = str(opt.device)
            cmd = f"python -m pytorch_fid \"{fid_real_dir}\" \"{fake_b_dir}\" --device {device_str}"

            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"FID Error: {result.stderr}")
                fid_score = float('nan')
            else:
                # 解析 FID 输出
                fid_match = re.search(r'FID:\s*(\d+\.\d+)', result.stdout)
                fid_score = float(fid_match.group(1)) if fid_match else float('nan')
        except Exception as e:
            print(f"FID Exception: {e}")
            fid_score = float('nan')

        # --- 记录与保存 ---
        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)

        print(f"★ 结果 [Epoch {epoch}]: FID: {fid_score:.2f} (越低越好) | Cycle-PSNR: {mean_psnr:.2f} (越高越好)")

        new_row = pd.DataFrame([{
            'epoch': epoch,
            'FID': fid_score,
            'PSNR_Cycle': mean_psnr,
            'SSIM_Cycle': mean_ssim
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.sort_values(by='epoch', inplace=True)
        df.to_csv(csv_path, index=False)

        # 画图
        plot_metrics(df, os.path.join(opt.results_dir, opt.name, 'metrics_plot.png'))


def plot_metrics(df, save_path):
    epochs = df['epoch'].tolist()
    plt.figure(figsize=(12, 6))

    # FID
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['FID'], 'r-o', label='FID (Realism)')
    plt.title('FID Score (Lower is Better)')
    plt.xlabel('Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # PSNR
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df['PSNR_Cycle'], 'g-o', label='Cycle PSNR (Stability)')
    plt.title('Cycle Consistency PSNR (Higher is Better)')
    plt.xlabel('Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图表已更新: {save_path}")


if __name__ == '__main__':
    main()