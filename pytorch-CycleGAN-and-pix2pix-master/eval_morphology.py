import os
import glob
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial import cKDTree
from stardist.models import StarDist2D
from tqdm import tqdm
import argparse


def calculate_compactness(area, perimeter):
    """
    计算紧凑度 (Compactness)。
    公式: 4 * pi * Area / (Perimeter^2)。越接近 1 越圆。
    """
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def extract_morphology_metrics(image_dir, label, model):
    """
    从指定文件夹中提取所有图像的形态学指标
    """
    image_paths = glob.glob(os.path.join(image_dir, '*.png'))[:200]  # 为加快测试，可限制图片数量
    if not image_paths:
        print(f"警告：在 {image_dir} 中未找到图片！")
        return []

    all_metrics = []

    print(f"正在处理 {label} 组图像...")
    for path in tqdm(image_paths):
        # 1. 读取 RGB 图像
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. 预测细胞核 Mask
        # prob 是概率图，labels 是实例分割的掩码图 (每个细胞核一个独立ID)
        labels, details = model.predict_instances(img)

        # 3. 使用 skimage 提取形态学特征
        props = measure.regionprops(labels)

        # 提取质心用于计算核间距
        centroids = [prop.centroid for prop in props]

        # 计算最近邻核间距 (Internuclear distance)
        distances = []
        if len(centroids) > 1:
            tree = cKDTree(centroids)
            # k=2 因为最近的总是自己(距离为0)，所以取第二近的
            dists, _ = tree.query(centroids, k=2)
            distances = dists[:, 1].tolist()
        else:
            distances = [0] * len(centroids)

        # 4. 汇总当前图的所有细胞指标
        for i, prop in enumerate(props):
            area = prop.area
            perimeter = prop.perimeter
            eccentricity = prop.eccentricity
            compactness = calculate_compactness(area, perimeter)

            # 过滤掉太小或边缘的噪点 (可选)
            if area < 20:
                continue

            all_metrics.append({
                'Domain': label,
                'Area': area,
                'Eccentricity': eccentricity,
                'Compactness': compactness,
                'Internuclear Distance': distances[i]
            })

    return all_metrics


def plot_violins(df, save_path):
    """
    绘制并保存小提琴图
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['Internuclear Distance', 'Area', 'Eccentricity', 'Compactness']
    titles = ['(d) Internuclear Distance', '(e) Cross-sectional Nuclear Area',
              '(g) Eccentricity', '(h) Compactness']
    y_labels = ['Distance (pixels)', 'Area (pixels^2)', 'Ratio (0-1)', 'Ratio (0-1)']

    # 自定义颜色：真实用蓝色系，生成用橙色系
    palette = {"Real H&E": "#4C72B0", "Fake H&E": "#DD8452"}

    for i, ax in enumerate(axes.flatten()):
        metric = metrics[i]
        sns.violinplot(
            data=df,
            x='Domain',
            y=metric,
            ax=ax,
            palette=palette,
            split=False,
            inner="quartile",  # 显示四分位数虚线，和论文一致
            linewidth=1.5
        )
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_ylabel(y_labels[i], fontsize=12)
        ax.set_xlabel('')
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[成功] 小提琴图已保存至：{save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Morphological Metrics")
    parser.add_argument('--real_dir', type=str, required=True, help="真实H&E图像文件夹路径")
    parser.add_argument('--fake_dir', type=str, required=True, help="生成的Fake H&E图像文件夹路径")
    parser.add_argument('--output_name', type=str, default='morphology_violins.png', help="输出图表名称")
    args = parser.parse_args()

    # 自动下载或加载专门针对 H&E 的预训练模型
    print("正在加载 StarDist 预训练模型 (2D_versatile_he)...")
    model = StarDist2D.from_pretrained('2D_versatile_he')

    # 提取特征
    real_metrics = extract_morphology_metrics(args.real_dir, 'Real H&E', model)
    fake_metrics = extract_morphology_metrics(args.fake_dir, 'Fake H&E', model)

    # 合并数据
    all_data = real_metrics + fake_metrics
    df = pd.DataFrame(all_data)

    if df.empty:
        print("错误：未能提取到任何细胞特征，请检查图片路径和内容。")
    else:
        # 输出简单的统计对比表格
        print("\n=== 指标中位数对比 (Median Comparison) ===")
        print(df.groupby('Domain').median())

        # 绘图
        plot_violins(df, args.output_name)