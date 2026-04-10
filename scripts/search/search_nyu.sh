# !/usr/bin/env sh

torchrun --nproc_per_node=4 --master_port=24323 ./tools/search.py \
            ./configs/evomde_search_nyu.py \
            --launcher pytorch \
            --job_name search_nyu \
            --seed 1 \
            --devices 0,1,2,3 \
            --work_dir ./log/ \
            --dataset nyu \
            --data_path /data/dataset/NYU_Depth_V2/sync/ \
            --gt_path /data/dataset/NYU_Depth_V2/sync/ \
            --filenames_file data_splits/nyudepthv2_train_files_with_gt_dense.txt \
            --input_height 480 \
            --input_width 640 \
            --max_depth 10.0 \
            --max_depth_eval 10.0 \
            --batch_size 16 \
            --do_online_eval \
            --data_path_eval /data/dataset/NYU_Depth_V2/test/ \
            --gt_path_eval /data/dataset/NYU_Depth_V2/test/ \
            --filenames_file_eval data_splits/nyudepthv2_test_files_with_gt.txt \
            --do_random_rotate \
            --degree 2.5 \
            --num_threads 1 \
            --eigen_crop \
            --validate

