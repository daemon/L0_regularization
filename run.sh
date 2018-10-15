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

# python train_wide_resnet.py --lat_lambda 5 --lamba 0 --name wrn-latlamb5 > wrn_notied_i7-8700k-latlamb5
# python train_wide_resnet.py --lat_lambda 5 --lamba 0 --name wrn-latlamb5 --resume runs/wrn-latlamb5_28_10/checkpoint.pth.tar --finetune > wrn_notied_i7-8700k-latlamb5-finetune

# python train_wide_resnet.py --lat_lambda 5 --lamba 0 --name wrn-latlamb5-real #> wrn_notied_i7-8700k-latlamb5_2
# python train_wide_resnet.py --lat_lambda 5 --lamba 0 --name wrn-latlamb5-real --resume runs/wrn-latlamb5-real_28_10/checkpoint.pth.tar --finetune # > wrn_notied_i7-8700k-latlamb5-finetune_2

# python train_wide_resnet.py --lat_lambda 50 --lamba 0 --name wrn-latlamb50-target --target_latency 0.2 > latlamb50-target
# python train_wide_resnet.py --lat_lambda 50 --lamba 0 --finetune --resume runs/wrn-latlamb50-target_28_10/checkpoint.pth.tar --name wrn-latlamb50-target --target_latency 0.2 > latlamb50-target-finetune


# python train_wide_resnet.py --lat_lambda 20 --lamba 0 --name x # --target_latency 0
# python train_wide_resnet.py --lat_lambda 50 --lamba 0 --finetune --resume runs/wrn-latlamb50-target_28_10/checkpoint.pth.tar --name wrn-latlamb50-target --target_latency 0.2 > 190-with-pruning-bn_log

# python train_wide_resnet.py --resume_boiled wrn_boiled_acc_2.pt --lat_lambda 0 --lamba 0 --lr 0.0008 --finetune > boiling_log_wrn

# python train_wide_resnet.py --boil --resume runs/wideresnet-paper-original-run/checkpoint.pth.tar --batch-size 32

# python train_wide_resnet.py --resume_boiled wrn_boiled_cuda.pt --lat_lambda 0 --lamba 0 --lr 0.0008 --finetune > cuda_boiling_log_wrn2


# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 4000 --lamba 0 --name wrn-flopslamb1E-4 > flops_log
# python train_wide_resnet.py --flops_lambda 0 --target_flops 4000 --lamba 0 --name wrn-flopslamb1E-4 --resume runs/wrn-flopslamb1E-4_28_10/checkpoint.pth.tar --finetune > flops_log-finetune

# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 4000 --lamba 0 --name wrn-flopslamb1E-4 --dataset c100 > flops_log-c100
# python train_wide_resnet.py --flops_lambda 0 --target_flops 4000 --lamba 0 --name wrn-flopslamb1E-4 --dataset c100 --resume runs/wrn-flopslamb1E-4_28_10_c100/model_189.pth.tar --finetune > flops_log-finetune-c100
# python train_wide_resnet.py --flops_lambda 0 --target_flops 4000 --lamba 0 --name wrn-flopslamb1E-4 --dataset c100 --resume runs/wrn-flopslamb1E-4_28_10_c100/checkpoint.pth.tar --finetune > flops_log-finetune-c100

# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k > flops_log-2.5k
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/checkpoint.pth.tar --finetune > flops_log-2.5k-finetune
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/model_189.pth.tar --finetune > flops_log-2.5k-finetune
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --compute_flops --resume runs/wrn-flopslamb1E-425k_28_10/model_189.pth.tar --finetune > flops_log-2.5k-finetune



# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-4-2.5k --dataset c100 > flops_log-c100-2.5k
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-4-2.5k --dataset c100 --resume runs/wrn-flopslamb1E-4_28_10_c100/model_189.pth.tar --finetune > flops_log-finetune-c100-2.5k


# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k > flops_log-2.5k-2
# # python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/checkpoint.pth.tar --finetune > flops_log-2.5k-finetune
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/model_189.pth.tar --finetune > flops_log-2.5k-finetune-2


# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k > flops_log-2.5k-3
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/checkpoint.pth.tar --finetune > flops_log-2.5k-finetune
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/model_189.pth.tar --finetune > flops_log-2.5k-finetune-3

# python train_wide_resnet.py --flops_lambda 3E-3 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k > flops_log-2.5k-4
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/checkpoint.pth.tar --finetune > flops_log-2.5k-finetune
# python train_wide_resnet.py --flops_lambda 0 --target_flops 2500 --lamba 0 --name wrn-flopslamb1E-425k --resume runs/wrn-flopslamb1E-425k_28_10/model_189.pth.tar --finetune > flops_log-2.5k-finetune-4
