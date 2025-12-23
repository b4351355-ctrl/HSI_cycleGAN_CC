import torch
from models.cycle_gan_model import CycleGANModel


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

        # 2. 移除 idt_B (无法计算，避免报错)
        if 'idt_B' in self.visual_names:
            self.visual_names.remove('idt_B')

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
        if lambda_idt > 0:
            # 【G_A 的 Identity Loss】: B_raw (300) -> G_A -> B_rgb (3)
            if self.real_B_raw is not None:
                self.idt_A = self.netG_A(self.real_B_raw)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            else:
                self.loss_idt_A = 0
                # 给一个占位符，防止可视化时找不到变量
                self.idt_A = self.fake_B

                # 【G_B 的 Identity Loss】: 无法计算，置0
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