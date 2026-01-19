################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

deepspeed --include localhost:1 --master_port 29601 llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 48 --lora_alpha 96 --mm_projector_lr 2e-5 \
    --expert_num 6 \
    --model_name_or_path /root/Desktop/code/HiDe/HiDe-LLaVA-main/llava_1.5_7b \
    --previous_task_model_path /root/Desktop/code/HiDe/HiDe-LLaVA-main/output/HiDe/Task3_llava_lora_ours \
    --version $PROMPT_VERSION \
    --data_path /root/Desktop/code/HiDe/HiDe-LLaVA-main/instructions/IconQA/train.json \
    --image_folder /root/Desktop/code/HiDe/HiDe-LLaVA-main/datasets \
    --vision_tower /root/Desktop/code/HiDe/HiDe-LLaVA-main/clip-vit-large-patch14-336 \
    --text_tower /root/Desktop/code/HiDe/HiDe-LLaVA-main/clip-vit-large-patch14-336 \
    --cur_task 3 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /root/Desktop/code/HiDe/HiDe-LLaVA-main/output/HiDe/Task4_llava_lora_ours \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
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
    --report_to none