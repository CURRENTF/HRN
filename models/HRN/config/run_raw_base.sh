#!/bin/bash

python main.py \
--model_name google/mt5-base \
--data_version complete_v3 \
--context_num 3 \
--per_context_len 768 \
--output_dir ./tmp/google/mt5-base-3-768_raw_data \
--overwrite_output_dir "true" \
--resume_from_checkpoint "false" \
--do_train \
--do_predict \
--evaluation_strategy epoch \
--save_strategy epoch \
--gradient_accumulation_steps 32 \
--learning_rate 3e-4 \
--weight_decay 1e-4 \
--num_train_epochs 10 \
--lr_scheduler_type linear \
--warmup_ratio 0.01 \
--save_total_limit 2 \
--seed 42 \
--load_best_model_at_end true \
--metric_for_best_model acc_f1_sum \
--predict_with_generate true \
--generation_max_length 64 \
--generation_num_beams 1 \
--logging_steps 10 \
--use_weight "false" \
--per_device_train_batch_size 2 \
--data_mode "raw_data" \
--num_proc 8 \
--stage_weight "0.6,0.8,1.0,1.4"