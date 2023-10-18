#!/bin/bash

python test.py \
--model_name google/mt5-base \
--data_version complete_v3 \
--context_num 3 \
--per_context_len 768 \
--output_dir ./tmp/google/test_multi \
--overwrite_output_dir "false" \
--resume_from_checkpoint "./tmp/google/mt5-base-3-768_raw_data/checkpoint-4004" \
--do_train "false" \
--do_predict "true" \
--seed 42 \
--predict_with_generate true \
--generation_max_length 64 \
--generation_num_beams 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 24 \
--data_mode "raw_data" \
--num_proc 8