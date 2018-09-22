#!/bin/sh
# python train_wide_resnet.py --lamba 0 --lat_lambda 1E3 > wrn_log_1e3
# python train_wide_resnet.py --lat_lambda 0 > wrn_regular_run
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights > wrn_regular_run_30
# python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights > wrn_tied_run_30
python train_wide_resnet.py --lat_lambda 30 --lamba 0 --tie_all_weights --resume runs/wrn-tied-run-30/checkpoint.pth.tar --finetune > wrn_tied_run_30_finetune


# NEED TO RUN: 32-32-1-0-1-False.dat, 32-32-3-1-2-False.dat, 32-32-1-0-2-False.dat, 16-16-3-1-2-False.dat, 16-16-1-0-2-False.dat, 
