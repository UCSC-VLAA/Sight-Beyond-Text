#!/usr/bin/env bash
#SBATCH --job-name=lora-lama
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=compute
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --output=textllamalora.txt

source /data/haoqin_tu/.bashrc
conda activate llava
cd /data/haoqin_tu/codes/Sight-Beyond-Text

deepspeed --master_port 61329 llava/train/train.py --deepspeed scripts/zero2.json \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_enable True \
    --cache_dir /data/haoqin_tu/.cache/torch/transformers \
    --version llama_2 \
    --data_path /data/haoqin_tu/datasets/llava-instruct-data/llava_text_instruct_80k.json \
    --image_folder /data/haoqin_tu/datasets/mscoco/train2017/ \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /data/haoqin_tu/weights/llava/llama-2-7b-lora/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
