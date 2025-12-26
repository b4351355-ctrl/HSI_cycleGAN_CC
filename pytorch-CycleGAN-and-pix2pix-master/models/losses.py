import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class VGGLoss(nn.Module):
    """
    感知损失 (Perceptual Loss): 使用预训练 VGG19 提取特征并计算 L1 距离
    """

    def __init__(self, layer_ids=[4, 9, 18, 27, 36]):
        super(VGGLoss, self).__init__()
        # 加载预训练 VGG19，只取特征提取部分
        vgg = models.vgg19(pretrained=True).features

        # 冻结参数，不参与训练
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg_layers = vgg
        self.layer_ids = layer_ids
        # ImageNet 归一化参数
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # 输入 x, y 范围通常是 [-1, 1]，需转换到 [0, 1] 并进行归一化
        x = (x + 1) / 2.0
        y = (y + 1) / 2.0
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)

            if i >= max(self.layer_ids):
                break
        return loss


class SSIMLoss(nn.Module):
    """
    结构相似性损失 (SSIM Loss): 返回 1 - SSIM
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # 转换到 [0, 1] 计算更有物理意义
        img1 = (img1 + 1) / 2.0
        img2 = (img2 + 1) / 2.0

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)