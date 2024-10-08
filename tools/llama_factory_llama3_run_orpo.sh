#!/bin/bash

deepspeed --num_gpus 4 ../../src/train_bash.py \
  --deepspeed ../deepspeed/ds_z3_config.json \
  --stage orpo \
  --do_train \
  --model_name_or_path /home/image_team/image_team_docker_home/lgd/e_commerce_llm/weights/Llama3_Chinese_Sft/ \
  --dataset dpo_mix_en,dpo_mix_zh \
  --dataset_dir ../../data \
  --template llama3 \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir ../../saves/Llama3_Chinese_Sft/lora/dpo \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 1024 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --warmup_steps 20 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --evaluation_strategy steps \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_samples 3000 \
  --val_size 0.1 \
  --ddp_timeout 180000000 \
  --plot_loss \
  --do_eval False \
  --bf16 true
  --orpo_beta 0.05
