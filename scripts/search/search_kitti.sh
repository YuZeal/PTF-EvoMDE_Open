# !/usr/bin/env sh

torchrun --nproc_per_node=4 --master_port=23333 ./tools/search.py \
            ./configs/evomde_search_kitti.py \
            --launcher pytorch \
            --job_name ex_search_kitti \
            --seed 1 \
            --devices 0,1,2,3 \
            --work_dir ./log/ \
            --dataset kitti \
            --data_path /data/dataset/KITTI/ \
            --gt_path /data/dataset/KITTI/data_depth_annotated/ \
            --filenames_file data_splits/eigen_train_files_with_gt.txt \
            --input_height 352 \
            --input_width 1120 \
            --max_depth 80.0 \
            --batch_size 4 \
            --do_online_eval \
            --data_path_eval /data/dataset/KITTI/ \
            --gt_path_eval /data/dataset/KITTI/data_depth_annotated/ \
            --filenames_file_eval data_splits/eigen_test_files_with_gt.txt \
            --do_random_rotate \
            --degree 1.0 \
            --do_kb_crop \
            --num_threads 1 \
            --garg_crop \
            --validate

