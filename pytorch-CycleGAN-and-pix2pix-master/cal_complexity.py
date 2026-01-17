import torch
import time
from thop import profile
from models import networks

# === 配置 ===
input_nc = 300  # 输入高光谱通道数
output_nc = 3  # 输出RGB通道数
ngf = 128  # 基础特征数
img_size = 256  # 测试图片大小 (256x256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name):
    """
    修改版：直接调用 networks.define_G，自动支持所有在 networks.py 中注册的模型
    """
    # 这里使用默认参数，需确保与您训练时的 norm 设置一致 (通常是 instance)
    net = networks.define_G(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        netG=model_name,
        norm='instance',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02
    )
    return net.to(device)


def measure_fps(model, input_tensor, warm_up=10, test_time=50):
    """测量实际推理速度 (FPS)"""
    model.eval()
    with torch.no_grad():
        # 预热 GPU
        for _ in range(warm_up):
            _ = model(input_tensor)

        # 同步时间
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(test_time):
            _ = model(input_tensor)

        torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / test_time
    fps = 1.0 / avg_time
    return avg_time * 1000, fps  # 返回 ms 和 FPS


if __name__ == '__main__':
    print(f"=== 模型复杂度对比 (Input: {input_nc}x{img_size}x{img_size}) ===\n")

    # 请在此处添加您想测试的所有模型名称 (必须要在 models/networks.py 的 define_G 中存在)
    model_list = [
        'resnet_9blocks',
        'resnet_12blocks',  # 之前报错的，现在应该可以了
        'resnet_12blocks_se',  # 您之前提到的 SE 版本
        'dual_stream',
        'hsi_starnet'# 注意：如果 models/networks.py 里没有 'dual_stream' 这个名字，这里还是会报错
    ]

    dummy_input = torch.randn(1, input_nc, img_size, img_size).to(device)

    print(f"{'Model':<20} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'Time (ms)':<10} | {'FPS':<10}")
    print("-" * 80)

    for name in model_list:
        try:
            net = get_model(name)

            # 1. 计算 FLOPs 和 Params
            flops, params = profile(net, inputs=(dummy_input,), verbose=False)

            # 2. 计算 FPS
            latency, fps = measure_fps(net, dummy_input)

            # 3. 打印结果
            print(f"{name:<20} | {params / 1e6:10.2f} M | {flops / 1e9:10.2f} G | {latency:8.2f} ms | {fps:8.2f}")

        except Exception as e:
            print(f"{name:<20} | 报错: {e}")
            print(f"  (请检查 models/networks.py 的 define_G 函数中是否已添加 '{name}')")