import os
import glob
import scipy.io as sio
from PIL import Image
from collections import defaultdict
import numpy as np
from tqdm import tqdm  # 如果没有安装tqdm，可以去掉这行和下面的tqdm()包裹


def check_dataset_dimensions(dataroot):
    # 定义四个子文件夹及其对应的文件类型
    phases = {
        'trainA': '.mat',
        'trainB': '.jpg',
        'testA': '.mat',
        'testB': '.jpg'
    }

    print(f"🚀 开始检查数据集路径: {dataroot}")
    print("=" * 60)

    # 遍历每个阶段 (trainA, trainB, testA, testB)
    for phase_name, ext in phases.items():
        dir_path = os.path.join(dataroot, phase_name)

        # 检查文件夹是否存在
        if not os.path.exists(dir_path):
            print(f"⚠️  警告: 目录不存在 {dir_path}，跳过。")
            print("-" * 60)
            continue

        print(f"📂 正在检查 {phase_name} (文件类型: {ext}) ...")

        # 查找所有文件 (支持递归查找子文件夹)
        files = sorted(glob.glob(os.path.join(dir_path, '**', '*' + ext), recursive=True))

        if len(files) == 0:
            print(f"   ❌ 未在该目录下找到 {ext} 文件！")
            print("-" * 60)
            continue

        # 字典用于记录: 尺寸 -> [文件路径列表]
        size_counter = defaultdict(list)

        # 使用 tqdm 显示进度条 (如果报错就把 tqdm(files) 改成 files)
        for file_path in tqdm(files, desc="Scanning"):
            try:
                # --- 处理 .mat 文件 (高光谱) ---
                if ext == '.mat':
                    mat = sio.loadmat(file_path)
                    if 'data' not in mat:
                        print(f"   ❌ 错误: {os.path.basename(file_path)} 中没有 'data' 键")
                        continue
                    data = mat['data']

                    # 您的数据通常是 (Channels, Height, Width) -> 取 H, W
                    # 为了兼容性，判断一下维度
                    if data.ndim == 3:
                        # 假设格式为 (C, H, W)
                        h, w = data.shape[1], data.shape[2]
                    elif data.ndim == 2:
                        h, w = data.shape[0], data.shape[1]
                    else:
                        print(f"   ❓ 未知维度 {data.shape}: {os.path.basename(file_path)}")
                        continue

                    size = (h, w)

                # --- 处理 .jpg 文件 (RGB) ---
                elif ext == '.jpg':
                    with Image.open(file_path) as img:
                        w_pil, h_pil = img.size  # PIL 返回的是 (Width, Height)
                        size = (h_pil, w_pil)  # 统一转换为 (Height, Width) 以便对比

                # 记录该尺寸
                size_counter[size].append(file_path)

            except Exception as e:
                print(f"   ❌ 读取失败 {os.path.basename(file_path)}: {e}")

        # --- 输出统计结果 ---
        distinct_sizes = list(size_counter.keys())
        total_files = len(files)

        if len(distinct_sizes) == 1:
            h, w = distinct_sizes[0]
            print(f"   ✅ 完美！所有 {total_files} 个文件尺寸一致: 高={h}, 宽={w}")
        else:
            print(f"   ⚠️  注意！发现 {len(distinct_sizes)} 种不同的尺寸：")
            # 按数量从多到少排序
            sorted_sizes = sorted(distinct_sizes, key=lambda s: len(size_counter[s]), reverse=True)

            for size in sorted_sizes:
                count = len(size_counter[size])
                ratio = count / total_files * 100
                print(f"      - 尺寸 {size[0]}x{size[1]}: {count} 张 ({ratio:.1f}%)")

                # 如果这种尺寸的文件很少（少于10个），很可能是异常值，打印出来方便你删除
                if count < 10:
                    print(f"        └── 文件名: {[os.path.basename(p) for p in size_counter[size]]}")

        print("-" * 60)


if __name__ == "__main__":
    # 🔴 请在这里修改为您数据集的实际根目录
    target_dataroot = "./datasets/cyclegan_dataset_HSI"

    check_dataset_dimensions(target_dataroot)