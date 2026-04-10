# !/usr/bin/env sh

torchrun --nproc_per_node=2 --master_port=29511 ./tools/retrain.py \
            ./configs/evomde_retrain_nyu.py \
            --launcher pytorch \
            --job_name retrain_nyu \
            --seed 1 \
            --devices 0,1 \
            --work_dir ./log/ \
            --dataset nyu \
            --data_path /data/dataset/NYU_Depth_V2/sync/ \
            --gt_path /data/dataset/NYU_Depth_V2/sync/ \
            --filenames_file data_splits/nyudepthv2_train_files_with_gt_dense.txt \
            --input_height 480 \
            --input_width 640 \
            --max_depth 10.0 \
            --max_depth_eval 10.0 \
            --batch_size 8 \
            --do_online_eval \
            --data_path_eval /data/dataset/NYU_Depth_V2/test/ \
            --gt_path_eval /data/dataset/NYU_Depth_V2/test/ \
            --filenames_file_eval data_splits/nyudepthv2_test_files_with_gt.txt \
            --do_random_rotate \
            --degree 2.5 \
            --num_threads 1 \
            --eigen_crop \
            --validate \
            --net_arch """[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k3_e3', 'k7_e3', 'k7_e3', 'k7_e3'], 2]|
[[24, 32], ['k5_e6', 'skip', 'k3_e3', 'skip'], 2]|
[[32, 64], ['k5_e3', 'skip', 'skip', 'k3_e6', 'k7_e6', 'k5_e6'], 2]|
[[64, 96], ['k7_e6', 'skip', 'skip', 'skip', 'skip', 'skip'], 1]|
[[96, 160], ['k3_e6', 'k7_e3', 'k3_e3', 'k7_e3'], 2]|
[[160, 320], ['k3_e3'], 1]"""
