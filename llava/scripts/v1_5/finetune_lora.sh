#!/bin/bash
LLAMA2_WEIGHTS=/path/to/llama2
PT_CKPT=/path/to/amharic/llama2
DATA_JSON=/path/to/llava_665k_amharic.json
IMG_FOLDER=/path/to/ft/data
CLIP_PATH=/path/to/clip
ADAPTER_PATH=/path/to/pretrained/adapter
OUTPUTS=/path/to/ft/outputs

deepspeed ../../llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ../zero3.json \
    --model_name_or_path $LLAMA2_WEIGHTS \
    --pretrained_checkpoint $PT_CKPT \
    --version v1 \
    --data_path $DATA_JSON \
    --image_folder $IMG_FOLDER \
    --vision_tower $CLIP_PATH \
    --pretrain_mm_mlp_adapter $ADAPTER_PATH \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUTS \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4729 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
