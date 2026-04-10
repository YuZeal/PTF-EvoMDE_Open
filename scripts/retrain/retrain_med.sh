# !/usr/bin/env sh

torchrun --nproc_per_node=1 --master_port=21506 ./tools/retrain.py \
            ./configs/evomde_retrain_med.py \
            --launcher pytorch \
            --job_name retrain_med \
            --seed 1 \
            --devices 6 \
            --work_dir ./log/medical/ \
            --dataset colon \
            --input_height 256 \
            --input_width 256 \
            --max_depth 20 \
            --max_depth_eval 20 \
            --batch_size 16 \
            --do_online_eval \
            --num_threads 1 \
            --validate \
            --total_epochs 40 \
            --net_arch """[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k3_e6', 'skip', 'k3_e6', 'skip'], 2]|
[[24, 32], ['k5_e6', 'k5_e6', 'k3_e3', 'skip'], 2]|
[[32, 64], ['k7_e3', 'k5_e6', 'skip', 'k3_e3', 'k3_e3', 'k7_e3'], 2]|
[[64, 96], ['k3_e6', 'skip', 'skip', 'skip', 'k7_e6', 'k5_e3'], 1]|
[[96, 160], ['k5_e6', 'k3_e3', 'k7_e3', 'k3_e6'], 2]|
[[160, 320], ['k7_e6'], 1]"""
