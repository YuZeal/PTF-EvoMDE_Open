# !/usr/bin/env sh

torchrun --nproc_per_node=4 --master_port=23334 ./tools/search.py \
            ./configs/evomde_search_med.py \
            --launcher pytorch \
            --job_name search_colon \
            --seed 1 \
            --devices 4,5,6,7 \
            --work_dir ./log/medical/ \
            --dataset colon \
            --input_height 256 \
            --input_width 256 \
            --max_depth 20.0 \
            --max_depth_eval 20 \
            --batch_size 4 \
            --do_online_eval \
            --num_threads 1 \
            --validate