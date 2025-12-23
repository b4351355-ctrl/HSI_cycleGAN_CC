import os
import time
import torch
from data import create_dataset
from models import create_model


# 简单的配置类，模拟 argparse 的功能
class Config:
    def __init__(self):
        # ----------------- 基础配置 -----------------
        self.name = 'experiment_cyclegan_demo'  # 实验名称
        self.dataroot = './datasets/HSI_sup'  # 数据集路径
        self.checkpoints_dir = './checkpoints'  # 模型保存路径
        self.results_dir = './results'  # 结果保存路径
        self.gpu_ids = [0]  # GPU ID, 例如 [0], 空列表为CPU
        self.num_threads = 4  # 数据加载线程数

        # ----------------- 模型配置 -----------------
        self.model = 'cycle_gan'  # 模型类型
        self.input_nc = 3  # 输入通道数
        self.output_nc = 3  # 输出通道数
        self.ngf = 64  # 生成器滤波器数
        self.ndf = 64  # 判别器滤波器数
        self.netD = 'basic'  # 判别器架构
        self.netG = 'resnet_9blocks'  # 生成器架构
        self.n_layers_D = 3  # 仅当 netD=='n_layers' 时有效
        self.norm = 'instance'  # 归一化层类型
        self.init_type = 'normal'  # 初始化方法
        self.init_gain = 0.02  # 初始化增益
        self.no_dropout = True  # 是否禁用 dropout

        # ----------------- 训练配置 -----------------
        self.dataset_mode = 'unaligned'  # 数据集模式
        self.direction = 'AtoB'  # 转换方向
        self.serial_batches = False  # 是否顺序读取数据
        self.pool_size = 50  # 图像缓冲池大小
        self.gan_mode = 'lsgan'  # GAN 损失类型
        self.lambda_A = 10.0  # Cycle loss A 权重
        self.lambda_B = 10.0  # Cycle loss B 权重
        self.lambda_identity = 0.5  # Identity loss 权重

        self.max_dataset_size = float("inf")  # 最大数据集大小，默认无穷大

        # 优化器参数
        self.n_epochs = 5  # 保持初始学习率的 epoch 数
        self.n_epochs_decay = 5  # 学习率衰减的 epoch 数
        self.beta1 = 0.5  # Adam beta1
        self.lr = 0.0002  # 初始学习率
        self.lr_policy = 'linear'  # 学习率策略
        self.lr_decay_iters = 50  # 仅当 lr_policy=='step' 时有效

        # 图像处理参数
        self.batch_size = 1  # 批大小
        self.load_size = 286  # 加载尺寸
        self.crop_size = 256  # 裁剪尺寸
        self.preprocess = 'resize_and_crop'  # 预处理方式
        self.no_flip = False  # 是否禁用翻转

        # 可视化与保存频率
        self.display_freq = 400  # 显示频率
        self.print_freq = 100  # 打印频率
        self.save_latest_freq = 5000  # 保存 latest 模型频率
        self.save_epoch_freq = 1  # 保存 epoch 模型频率
        self.save_by_iter = False  # 是否按迭代保存
        self.continue_train = False  # 是否继续训练
        self.epoch_count = 1  # 起始 epoch
        self.phase = 'train'  # 阶段
        self.verbose = False  # 详细模式

        # ----------------- 系统设置 -----------------
        self.device = torch.device(
            'cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
        self.isTrain = True  # 训练标志

        # 其他兼容性参数 (Visdom/WandB/HTML)
        self.use_wandb = False
        self.wandb_project_name = 'CycleGAN-and-pix2pix'
        self.update_html_freq = 1000
        self.display_id = 1
        self.display_winsize = 256
        self.display_ncols = 4
        self.display_server = "http://localhost"
        self.display_env = 'main'
        self.display_port = 8097
        self.no_html = False
        self.suffix = ''


def run_training(opt):
    print("----------- 开始训练 -----------")
    opt.isTrain = True
    opt.phase = 'train'

    # [新增修复] 手动创建模型保存目录
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
        print(f"已创建模型保存目录: {expr_dir}")

    # 1. 创建数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'训练集图片数量: {dataset_size}')

    # 2. 创建模型
    model = create_model(opt)
    model.setup(opt)  # 加载网络，打印网络结构

    # 3. 初始化可视化工具 (可选)
    total_iters = 0

    # 4. 训练循环
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # --- 核心训练逻辑 ---
            model.set_input(data)  # 解包数据
            model.optimize_parameters()  # 优化参数

            # --- 打印 Loss ---
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = f'(epoch: {epoch}, iters: {epoch_iter}, time: {t_comp:.3f}, data: {t_data:.3f}) '
                for k, v in losses.items():
                    message += f'{k}: {v:.3f} '
                print(message)

            # --- 保存 Latest 模型 ---
            if total_iters % opt.save_latest_freq == 0:
                print(f'保存 latest 模型 (epoch {epoch}, total_iters {total_iters})')
                save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # 更新学习率
        model.update_learning_rate()

        # --- 按 Epoch 保存模型 ---
        if epoch % opt.save_epoch_freq == 0:
            print(f'保存模型 (epoch {epoch})')
            model.save_networks('latest')
            model.save_networks(epoch)

        print(
            f'Epoch {epoch} 完成 / 总共 {opt.n_epochs + opt.n_epochs_decay} \t 耗时: {time.time() - epoch_start_time:.2f} 秒')


def run_testing(opt):
    print("\n----------- 开始测试 -----------")
    # 修改配置以适应测试模式
    opt.isTrain = False
    opt.phase = 'test'
    opt.serial_batches = True  # 测试时不打乱顺序
    opt.no_flip = True  # 测试时不翻转
    opt.load_size = opt.crop_size  # 测试时不缩放再裁剪
    opt.num_test = 50  # 测试生成的图片数量
    opt.eval = True  # 使用 eval 模式

    # 1. 创建测试数据集
    dataset = create_dataset(opt)
    print(f'测试集图片数量: {len(dataset)}')

    # 2. 创建模型
    # 注意：这里重新创建模型以确保加载正确的各种标志
    model = create_model(opt)
    model.setup(opt)

    # 切换到 eval 模式
    if opt.eval:
        model.eval()

    # 3. 创建结果网页生成器
    from util import html
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'
    print(f'创建结果目录: {web_dir}')

    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')

    # 4. 测试推理循环
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)  # 解包测试数据
        model.test()  # 前向推理

        visuals = model.get_current_visuals()  # 获取结果
        img_path = model.get_image_paths()  # 获取图片路径

        if i % 5 == 0:
            print(f'处理第 {i} 张图片... {img_path}')

        # 保存图片到磁盘
        from util.visualizer import save_images
        save_images(webpage, visuals, img_path, aspect_ratio=1.0, width=opt.display_winsize)

    webpage.save()
    print("测试完成。结果已保存。")


if __name__ == '__main__':
    # 实例化配置
    opt = Config()

    # 确保 GPU 可用
    if not torch.cuda.is_available() and len(opt.gpu_ids) > 0:
        print("警告: 未检测到 GPU，将使用 CPU 运行。")
        opt.gpu_ids = []
        opt.device = torch.device('cpu')

    # 第一步：运行训练
    run_training(opt)

    # 第二步：运行测试
    run_testing(opt)