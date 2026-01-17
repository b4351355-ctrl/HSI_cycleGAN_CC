import os
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage.color import rgb2gray

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


def tensor2im_3ch(input_image):
    """
    将Tensor转换为3通道numpy数组，用于可视化或RGB计算
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    image_numpy = image_tensor[0].cpu().float().numpy()

    if image_numpy.shape[0] == 300:
        select_bands = [217, 156, 63]
        image_numpy = image_numpy[select_bands, :, :]

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(np.uint8)


def get_hsi_aip_gray(input_tensor):
    """输入高光谱 Tensor [1, C, H, W] -> 输出平均强度投影灰度图 (uint8)"""
    aip_tensor = torch.mean(input_tensor, dim=1, keepdim=True)
    aip_tensor = (aip_tensor + 1.0) / 2.0 * 255.0
    aip_tensor = torch.clamp(aip_tensor, 0, 255)
    return aip_tensor.cpu().numpy()[0, 0].astype(np.uint8)


def get_rgb_gray(rgb_numpy_uint8):
    """输入 RGB numpy uint8 [H, W, 3] -> 输出灰度图 (uint8)"""
    gray = rgb2gray(rgb_numpy_uint8)
    gray = (gray * 255.0).astype(np.uint8)
    return gray


if __name__ == '__main__':
    opt = TestOptions().parse()

    # -------- 设备配置 --------
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # -------------------------

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    metrics = {
        'RMSE_300ch': [],
        'PSNR_RGB': [],
        'SSIM_RGB': [],
        'Sup_PSNR_RGB': [],
        'Sup_SSIM_RGB': [],
        'Content_PSNR': [],
        'Content_SSIM': []
    }

    # 结果保存路径
    # 这里的 opt.results_dir 默认是 ./results/，opt.name 是实验名称
    result_dir_root = os.path.join(opt.results_dir, opt.name)
    fake_b_dir = os.path.join(result_dir_root, 'fid_fake_B')
    real_b_dir = os.path.join(result_dir_root, 'fid_real_B')
    util.mkdirs([fake_b_dir, real_b_dir])

    print(f"开始测试 {len(dataset)} 张图片...")

    skipped_cycle_count = 0
    skipped_sup_count = 0

    # 使用 tqdm 显示进度条
    for i, data in tqdm(enumerate(dataset)):
        model.set_input(data)
        model.test()

        real_A = model.real_A
        fake_B = model.fake_B
        rec_A = model.rec_A
        real_B = model.real_B

        # === 1. Cycle Consistency ===
        if real_A.shape == rec_A.shape:
            mse = torch.mean((real_A - rec_A) ** 2).item()
            metrics['RMSE_300ch'].append(np.sqrt(mse))

        img_real_A_vis = tensor2im_3ch(real_A)
        img_rec_A_vis = tensor2im_3ch(rec_A)

        if img_real_A_vis.shape == img_rec_A_vis.shape:
            metrics['PSNR_RGB'].append(psnr_metric(img_real_A_vis, img_rec_A_vis, data_range=255))
            metrics['SSIM_RGB'].append(ssim_metric(img_real_A_vis, img_rec_A_vis, channel_axis=2, data_range=255))
        else:
            skipped_cycle_count += 1

        # === 2. Supervised ===
        if 'B_raw' in data and 'B' in data:
            real_B_raw = data['B_raw'].to(model.device)
            real_B_gt_tensor = data['B'].to(model.device)
            with torch.no_grad():
                fake_B_sup = model.netG_A(real_B_raw)
            img_fake_B_sup = tensor2im_3ch(fake_B_sup)
            img_real_B_gt = tensor2im_3ch(real_B_gt_tensor)

            if img_fake_B_sup.shape == img_real_B_gt.shape:
                metrics['Sup_PSNR_RGB'].append(psnr_metric(img_real_B_gt, img_fake_B_sup, data_range=255))
                metrics['Sup_SSIM_RGB'].append(
                    ssim_metric(img_real_B_gt, img_fake_B_sup, channel_axis=2, data_range=255))
            else:
                skipped_sup_count += 1
                if skipped_sup_count <= 5:
                    tqdm.write(f"[Warning] Skip Sup-Metrics img {i}: Size mismatch")

        # === 3. Content Consistency ===
        gray_real_A = get_hsi_aip_gray(real_A)
        img_fake_B = tensor2im_3ch(fake_B)
        gray_fake_B = get_rgb_gray(img_fake_B)

        if gray_real_A.shape == gray_fake_B.shape:
            metrics['Content_PSNR'].append(psnr_metric(gray_real_A, gray_fake_B, data_range=255))
            metrics['Content_SSIM'].append(ssim_metric(gray_real_A, gray_fake_B, data_range=255))

        # === 4. 保存图片 ===
        # 这里静默保存，不打印 log，以免干扰进度条
        save_path = os.path.join(fake_b_dir, f"fake_B_{i:04d}.png")
        util.save_image(img_fake_B, save_path)

        img_real_B = tensor2im_3ch(real_B)
        save_path_b = os.path.join(real_b_dir, f"real_B_{i:04d}.png")
        util.save_image(img_real_B, save_path_b)

    # ================= 循环结束，统一输出 =================

    # 1. 打印保存完成日志
    print(f"\n[Info] 所有测试图片已保存至: {fake_b_dir}")

    # 2. 构建指标报告内容
    log_lines = []
    log_lines.append("=" * 20 + " 测试报告 (Test Report) " + "=" * 20)
    log_lines.append(f"图片数量 (Count): {len(dataset)}")

    if len(metrics['RMSE_300ch']) > 0:
        log_lines.append("\n1. 循环一致性 (Cycle Consistency, A->B->A):")
        log_lines.append(f"   RMSE (300ch): {np.mean(metrics['RMSE_300ch']):.4f}")
        log_lines.append(f"   PSNR (RGB)  : {np.mean(metrics['PSNR_RGB']):.2f} dB")
        log_lines.append(f"   SSIM (RGB)  : {np.mean(metrics['SSIM_RGB']):.4f}")

    if len(metrics['Sup_PSNR_RGB']) > 0:
        log_lines.append("\n2. 监督指标 (Supervised, B_raw->FakeB vs RealB):")
        log_lines.append(f"   PSNR (RGB)  : {np.mean(metrics['Sup_PSNR_RGB']):.2f} dB")
        log_lines.append(f"   SSIM (RGB)  : {np.mean(metrics['Sup_SSIM_RGB']):.4f}")

    if len(metrics['Content_PSNR']) > 0:
        log_lines.append("\n3. 内容一致性 (Content Consistency, HSI-AIP vs Fake-Gray):")
        log_lines.append(f"   PSNR (Gray) : {np.mean(metrics['Content_PSNR']):.2f} dB")
        log_lines.append(f"   SSIM (Gray) : {np.mean(metrics['Content_SSIM']):.4f}")

    if skipped_sup_count > 0:
        log_lines.append(f"\n[Warning] {skipped_sup_count} images skipped in Supervised Metrics due to size mismatch.")

    log_lines.append("=" * 60)

    # 将列表拼接成字符串
    log_content = "\n".join(log_lines)

    # 3. 打印到控制台
    print(log_content)

    # 4. 保存到文件 (保存到 results/experiment_name/test_metrics.txt)
    metrics_save_path = os.path.join(result_dir_root, 'test_metrics.txt')
    try:
        with open(metrics_save_path, 'w') as f:
            f.write(log_content)
        print(f"[Info] 指标数值已保存至: {metrics_save_path}")
    except Exception as e:
        print(f"[Error] 保存指标文件失败: {e}")