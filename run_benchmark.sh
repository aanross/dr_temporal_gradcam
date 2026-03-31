#!/bin/bash
# run_benchmark.sh
# Orchestrates the training and evaluation of all 6 Spatiotemporal Architectures for the DR Study

export WANDB_MODE=disabled
export PYTHONPATH=.

MODELS=("resnet_baseline" "resnet50_lstm" "efficientnet_bilstm" "vit_temporal" "timesformer" "convlstm")

echo "========================================================="
echo "Starting Full Study Benchmark Across All 6 Architectures"
echo "========================================================="

for model in "${MODELS[@]}"
do
    echo "[$(date)] Launching Training Pipeline for $model..."
    # Running 2 epochs, 5 folds, default 'gradcam' (which evaluates the best checkpoint at the end)
    python3 src/train.py \
        --model_type "$model" \
        --cam_method gradcam++ \
        --epochs 2 \
        --num_folds 5 \
        --batch_size 4 \
        --generate_plots
        
    echo "[$(date)] Completed Evaluation for $model."
    echo "---------------------------------------------------------"
done

echo "========================================================="
echo "All 6 Models Benchmarked Successfully. Check results/ folder."
echo "========================================================="
