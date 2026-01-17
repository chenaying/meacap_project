#!/bin/bash
# Script to run inference with local models on Linux

# Set local model paths
VL_MODEL="/home/cyp/project/mea_cos/MeaCap/checkpoints/clip-vit-base-patch32"
PARSER_CHECKPOINT="/home/cyp/project/mea_cos/MeaCap/checkpoints/flan-t5-base-VG-factual-sg"
WTE_MODEL_PATH="/home/cyp/project/mea_cos/MeaCap/checkpoints/all-Mini-L6-v2"
LM_MODEL_PATH="./checkpoints/CBART_COCO"

# Run inference
python inference.py \
    --memory_id coco \
    --img_path ./image_example \
    --lm_model_path "${LM_MODEL_PATH}" \
    --vl_model "${VL_MODEL}" \
    --parser_checkpoint "${PARSER_CHECKPOINT}" \
    --wte_model_path "${WTE_MODEL_PATH}" \
    "$@"

