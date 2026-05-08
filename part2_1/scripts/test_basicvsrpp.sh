#!/usr/bin/env bash
set -e

cd /home/fc/Coding/CV/part2_1/mmagic

CONFIG=configs/basicvsr_pp/basicvsr-pp_c64n7_fc_finetune.py
CHECKPOINT=/home/fc/Coding/CV/part2_1/mmagic/work_dirs/basicvsr-pp_c64n7_fc_finetune/basicvsr-pp_c64n7_fc_finetune/best_PSNR_iter_20000.pth

python tools/test.py $CONFIG $CHECKPOINT
