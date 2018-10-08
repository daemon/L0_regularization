#!/bin/sh
# python train_wide_resnet.py --lamba 0 --lat_lambda 1E3 > wrn_log_1e3
# python train_wide_resnet.py --lat_lambda 0 > wrn_regular_run
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights > wrn_regular_run_30
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights > wrn_tied_run_30
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights --resume runs/wrn-tied-run-30/checkpoint.pth.tar --finetune > wrn_tied_run_30_finetune
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights --resume runs/wrn-tied-run-30/checkpoint.pth.tar --finetune > wrn_tied_run_30_finetune_
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights --finetune_dropout 0.15 --resume runs/wrn-tied-run-30/checkpoint.pth.tar --finetune > wrn_tied_run_30_finetune_dropout0.15

# NEED TO RUN: 32-32-1-0-1-False.dat, 32-32-3-1-2-False.dat, 32-32-1-0-2-False.dat, 16-16-3-1-2-False.dat, 16-16-1-0-2-False.dat, 

# python train_wide_resnet.py --lat_lambda 10 --lamba 0 --dataset c100 > wrn_notied_i7-8700k-cifar100
# python train_wide_resnet.py --lat_lambda 10 --lamba 0 --dataset c100 --resume runs/L0WideResNet_28_10_c100/checkpoint.pth.tar --finetune > wrn_notied_i7-8700k-cifar100-finetune

# python train_wide_resnet.py --lat_lambda 10 --lamba 0 > wrn_notied_i7-8700k
# python train_wide_resnet.py --lat_lambda 10 --lamba 0 --finetune --resume runs/wrn-latency/checkpoint.pth.tar > wrn_notied_i7-8700k_finetune

python train_wide_resnet.py --lat_lambda 5 --lamba 0 --name wrn-latlamb5 > wrn_notied_i7-8700k-latlamb5
python train_wide_resnet.py --lat_lambda 5 --lamba 0 --name wrn-latlamb5 --resume runs/wrn-latlamb5_28_10/checkpoint.pth.tar --finetune > wrn_notied_i7-8700k-latlamb5-finetune