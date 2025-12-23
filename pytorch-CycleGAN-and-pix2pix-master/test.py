"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
# 添加必要的导入
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # 初始化存储指标的列表
    psnr_values = []
    ssim_values = []

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print(f"processing ({i:04d})-th image... {img_path}")
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        # 计算PSNR和SSIM（仅当存在真实图像B时）
        if 'B' in data:
            real_B = data['B'].squeeze(0).cpu().numpy()
            fake_B = visuals['fake_B'].cpu().numpy()
            
            # 确保图像维度正确
            if len(real_B.shape) == 3 and len(fake_B.shape) == 3:
                real_B = np.transpose(real_B, (1, 2, 0))
                fake_B = np.transpose(fake_B, (1, 2, 0))

                # 转换为灰度图进行比较
                real_B_gray = np.dot(real_B[...,:3], [0.2989, 0.5870, 0.1140])
                fake_B_gray = np.dot(fake_B[...,:3], [0.2989, 0.5870, 0.1140])

                current_psnr = psnr(real_B_gray, fake_B_gray)
                current_ssim = ssim(real_B_gray, fake_B_gray)

                psnr_values.append(current_psnr)
                ssim_values.append(current_ssim)

    webpage.save()  # save the HTML

    # 保存指标到文件（仅当有指标值时）
    if psnr_values:
        metrics_file = web_dir / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("Image\tPSNR\tSSIM\n")
            for idx, (p, s) in enumerate(zip(psnr_values, ssim_values)):
                f.write(f"{idx}\t{p:.4f}\t{s:.4f}\n")
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values) if ssim_values else 0
            f.write(f"\nAverage\t{avg_psnr:.4f}\t{avg_ssim:.4f}\n")

        print(f"Saved metrics to {metrics_file}")
    else:
        print("No metrics to save.")