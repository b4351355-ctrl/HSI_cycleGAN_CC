import torch
from models.cycle_gan_model import CycleGANModel
from models.losses import VGGLoss, SSIMLoss, FrequencyLoss,ColorConsistencyLoss

class HsCycleGANModel(CycleGANModel):
    """
    修改版的高光谱 CycleGAN 模型 (修复 WandB 300通道报错版)
    """

    def __init__(self, opt):
        super().__init__(opt)

        # 1. 添加 B_raw 的可视化 (如果训练中)
        # 注意：这里我们直接用 'real_B_raw'，不再加 _vis 后缀，保持逻辑统一
        if self.isTrain:
            self.visual_names.append('real_B_raw')

            # === 【新增：将新 Loss 添加到监控列表】 ===
            if opt.lambda_perceptual > 0:
                self.loss_names.append('idt_A_VGG')
            if opt.lambda_ssim > 0:
                self.loss_names.append('idt_A_SSIM')

        # 2. 移除 idt_B (无法计算，避免报错)
        if 'idt_B' in self.visual_names:
            self.visual_names.remove('idt_B')

        # === 【新增：根据参数初始化 Loss 模块】 ===
        if self.isTrain:
            # 只有当权重 > 0 时才加载网络，节省显存
            if opt.lambda_perceptual > 0:
                self.criterionVGG = VGGLoss().to(self.device)
                print(f"✅ Enabled Perceptual Loss (weight={opt.lambda_perceptual})")

            if opt.lambda_ssim > 0:
                self.criterionSSIM = SSIMLoss().to(self.device)
                print(f"✅ Enabled SSIM Loss (weight={opt.lambda_ssim})")

            if opt.lambda_freq > 0:
                self.criterionFreq = FrequencyLoss().to(self.device)
                self.loss_names.append('idt_A_Freq')  # 添加到 WandB 监控列表
                print(f"✅ Enabled Frequency Loss (weight={opt.lambda_freq})")

            if opt.lambda_color > 0:
                self.criterionColor = ColorConsistencyLoss().to(self.device)
                self.loss_names.append('idt_A_Color')
                print(f"✅ Enabled Color Consistency Loss (weight={opt.lambda_color})")

    def set_input(self, input):
        """继承父类方法，并额外获取 B_raw"""
        super().set_input(input)

        # 尝试获取 B_raw
        if 'B_raw' in input:
            self.real_B_raw = input['B_raw'].to(self.device)
        else:
            self.real_B_raw = None

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # --- Identity Loss 计算 ---
        # === 【修改开始：复合监督 Identity Loss】 ===
        if lambda_idt > 0:
            # G_A 的监督路径: B_raw (300) -> G_A -> B_rgb (3)
            if self.real_B_raw is not None:
                self.idt_A = self.netG_A(self.real_B_raw)

                # 1. 基础 L1 Loss (始终计算)
                loss_l1 = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_L1_sup

                # 2. Perceptual Loss (VGG)
                loss_vgg = 0
                if self.opt.lambda_perceptual > 0:
                    loss_vgg = self.criterionVGG(self.idt_A, self.real_B) * self.opt.lambda_perceptual
                    self.loss_idt_A_VGG = loss_vgg  # 记录用于显示

                # 3. SSIM Loss
                loss_ssim = 0
                if self.opt.lambda_ssim > 0:
                    loss_ssim = self.criterionSSIM(self.idt_A, self.real_B) * self.opt.lambda_ssim
                    self.loss_idt_A_SSIM = loss_ssim  # 记录用于显示

                # 4. freq Loss
                loss_freq = 0
                if self.opt.lambda_freq > 0:
                    # 计算频域损失
                    loss_freq = self.criterionFreq(self.idt_A, self.real_B) * self.opt.lambda_freq
                    self.loss_idt_A_Freq = loss_freq  # 记录用于显示

                # 5. Color Consistency Loss (新增)
                loss_color = 0
                if self.opt.lambda_color > 0:
                    loss_color = self.criterionColor(self.idt_A, self.real_B) * self.opt.lambda_color
                    self.loss_idt_A_Color = loss_color

                    # === 总 Loss 更新 ===
                    # 将 loss_color 加入
                    self.loss_idt_A = (loss_l1 + loss_vgg + loss_ssim + loss_freq + loss_color) * lambda_idt

                # 总 Loss: 三者之和，再乘以 lambda_idt
                # 这样设计是为了保持和 CycleGAN 原有参数量级的兼容性
                self.loss_idt_A = (loss_l1 + loss_vgg + loss_ssim + loss_freq) * lambda_idt
            else:
                self.loss_idt_A = 0
                self.idt_A = self.fake_B

            # G_B 的 Identity Loss 无法计算，保持为 0
            self.loss_idt_B = 0
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # --- GAN & Cycle Loss ---
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # --- 总 Loss ---
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def compute_visuals(self):
        """
        核心修复：遍历所有要显示的变量，如果是 300 通道，强制转成 3 通道。
        """
        super().compute_visuals()

        # 定义转换函数：300ch -> 3ch RGB
        def process(t):
            # 只有当它是 Tensor 且通道数是 300 时才处理
            if t is not None and isinstance(t, torch.Tensor) and t.shape[1] == 300:
                select_bands = [217, 156, 63]  # R, G, B
                select_bands = [min(b, t.shape[1] - 1) for b in select_bands]
                return t[:, select_bands, :, :]
            return t

        # 【重点】直接覆盖原变量
        # visual_names 里有什么（比如 'real_A', 'fake_A'），我们就处理什么
        for name in self.visual_names:
            if hasattr(self, name):
                raw_data = getattr(self, name)
                # 处理并覆盖回去
                processed_data = process(raw_data)
                setattr(self, name, processed_data)