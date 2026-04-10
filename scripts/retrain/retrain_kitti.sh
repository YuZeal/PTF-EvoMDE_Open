# !/usr/bin/env sh

torchrun --nproc_per_node=2 --master_port=29503 ./tools/retrain.py \
            ./configs/evomde_retrain_kitti.py \
            --launcher pytorch \
            --job_name retrain_kitti \
            --seed 1 \
            --devices 4,5 \
            --work_dir ./log/ \
            --dataset kitti \
            --data_path /data/dataset/KITTI/ \
            --gt_path /data/dataset/KITTI/data_depth_annotated/ \
            --filenames_file data_splits/eigen_train_files_with_gt.txt \
            --input_height 352 \
            --input_width 1120 \
            --max_depth 80.0 \
            --batch_size 8 \
            --do_online_eval \
            --data_path_eval /data/dataset/KITTI/ \
            --gt_path_eval /data/dataset/KITTI/data_depth_annotated/ \
            --filenames_file_eval data_splits/eigen_test_files_with_gt.txt \
            --do_random_rotate \
            --degree 1.0 \
            --do_kb_crop \
            --num_threads 1 \
            --garg_crop \
            --validate \
            --net_arch """[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k7_e6', 'k5_e6', 'skip', 'skip'], 2]|
[[24, 32], ['k7_e6', 'k7_e6', 'skip', 'skip'], 2]|
[[32, 64], ['k7_e6', 'k7_e6', 'k7_e6', 'skip', 'skip', 'skip'], 2]|
[[64, 96], ['k7_e6', 'k7_e6', 'skip', 'skip', 'skip', 'skip'], 1]|
[[96, 160], ['k3_e6', 'k7_e6', 'k7_e6', 'k7_e6'], 2]|
[[160, 320], ['k7_e6'], 1]"""
