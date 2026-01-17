#!/bin/bash

echo "Starting Experiment 1: HSI2png_16bit_L1_FFL_color_vgg1_starnet"
python train.py --dataroot ./datasets/HSI_sup --name HSI2png_16bit_L1_FFL_color_vgg1_starnet --model hs_cycle_gan --dataset_mode paired_hyperspectral --input_nc 300 --output_nc 3 --netG hsi_starnet --preprocess crop --crop_size 256 --ngf 128 --num_threads 16 --display_freq 100 --print_freq 100 --save_epoch_freq 20 --lambda_identity 1 --no_flip --n_epochs 100 --n_epochs_decay 100 --use_wandb --lambda_L1_sup 1.0 --lambda_perceptual 5.0 --lambda_ssim 0 --lambda_freq 1.0 --lambda_color 4.0

echo "Experiment 1 finished. Starting Experiment 2: HSI2png_16bit_L1_FFL_color1_starnet"
python train.py --dataroot ./datasets/HSI_sup --name HSI2png_16bit_L1_FFL_color1_starnet --model hs_cycle_gan --dataset_mode paired_hyperspectral --input_nc 300 --output_nc 3 --netG hsi_starnet --preprocess crop --crop_size 256 --ngf 128 --num_threads 16 --display_freq 100 --print_freq 100 --save_epoch_freq 20 --lambda_identity 1 --no_flip --n_epochs 100 --n_epochs_decay 100 --use_wandb --lambda_L1_sup 1.0 --lambda_perceptual 0 --lambda_ssim 0 --lambda_freq 1.0 --lambda_color 4.0
echo "All experiments finished!"