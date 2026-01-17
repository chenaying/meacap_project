#!/bin/bash
# Script to find local model paths

echo "Searching for local models in common locations..."
echo ""

# Common locations to check
LOCATIONS=(
    "/home/cyp/project/mea_cos/MeaCap/checkpoints"
    "/home/cyp/.cache/huggingface/hub"
    "$HOME/.cache/huggingface/hub"
    "./checkpoints"
    "../checkpoints"
)

# Model names to search for
CLIP_MODELS=("clip-vit-base-patch32" "clip-vit-large-patch14")
PARSER_MODELS=("flan-t5-base-VG-factual-sg" "flan-t5-base")
WTE_MODELS=("all-Mini-L6-v2" "all-MiniLM-L6-v2")

echo "=== CLIP Models ==="
for location in "${LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        for model in "${CLIP_MODELS[@]}"; do
            if [ -d "$location/$model" ] || [ -d "$location/models--*$model" ]; then
                echo "Found: $location/$model"
            fi
        done
    fi
done

echo ""
echo "=== Parser Models (Flan-T5) ==="
for location in "${LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        for model in "${PARSER_MODELS[@]}"; do
            if [ -d "$location/$model" ] || [ -d "$location/models--*$model" ]; then
                echo "Found: $location/$model"
            fi
        done
    fi
done

echo ""
echo "=== WTE Models (SentenceTransformer) ==="
for location in "${LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        for model in "${WTE_MODELS[@]}"; do
            if [ -d "$location/$model" ] || [ -d "$location/models--*$model" ]; then
                echo "Found: $location/$model"
            fi
        done
    fi
done

echo ""
echo "=== HuggingFace Cache ==="
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    echo "Hub cache: $HOME/.cache/huggingface/hub"
    echo "Listing CLIP models:"
    find "$HOME/.cache/huggingface/hub" -type d -name "*clip*" -maxdepth 2 2>/dev/null | head -5
    echo ""
    echo "Listing Flan-T5 models:"
    find "$HOME/.cache/huggingface/hub" -type d -name "*flan*" -maxdepth 2 2>/dev/null | head -5
    echo ""
    echo "Listing SentenceTransformer models:"
    find "$HOME/.cache/huggingface/hub" -type d -name "*MiniLM*" -maxdepth 2 2>/dev/null | head -5
fi

