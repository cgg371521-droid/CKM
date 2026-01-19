#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=1
IDX=0

if [ ! -n "$1" ] ;then
    STAGE='hide'
else
    STAGE=$1
fi

MODELPATH=$2

if [ ! -n "$3" ] ;then
    GPU=0
else
    GPU=$3
fi

RESULT_DIR="/root/Desktop/code/HiDe/HiDe-LLaVA-main/output/UCIT/each_dataset/Flickr30k"

# for IDX in $(seq 0 $((CHUNKS-1))); do
CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_answer \
    --model_path $MODELPATH \
    --model_base /root/Desktop/code/CKM/CKM-LLaVA-main/llava_1.5_7b \
    --question_file /root/Desktop/code/CKM/CKM-LLaVA-main/instructions/Flickr30k/test_3000.json \
    --image_folder /root/Desktop/code/CKM/CKM-LLaVA-main/datasets \
    --text_tower /root/Desktop/code/CKM/CKM-LLaVA-main/clip-vit-large-patch14-336 \
    --answers_file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --temperature 0 \
    --conv_mode vicuna_v1 &

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_caption \
    --annotation-file /root/Desktop/code/CKM/CKM-LLaVA-main/instructions/Flickr30k/val_coco_type_3000.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \
