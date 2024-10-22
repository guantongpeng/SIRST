#!/bin/bash

# GPU devices to use for each dataset
declare -A DATASET_GPU_MAP
DATASET_GPU_MAP[SIRST]=0
DATASET_GPU_MAP[NUDT-SIRST]=1
DATASET_GPU_MAP[IRSTD-1k]=2

# Datasets to use
# DATASETS=('SIRST' 'NUDT-SIRST' 'IRSTD-1k')
DATASETS=('IRSTD-1k')
# Model names to test
# MODELS=('ACM' 'ALCNet' 'DNANet' 'UIUNet' 'SCTransNet' 'ResUNet' 'UNet' 'NestedUNet' 'EffiSegNet' 'MSHNet' 'RDIAN')
MODELS=('DNANet' 'ResUNet' 'NestedUNet' 'EffiSegNet')
# Training parameters
BATCHSIZE=16
EPOCHS=1000
LR=0.01
LOSS_FUNC='SoftIoULoss'
OPTIMIZER_NAME='Adam'
TEST_EPOCH=50

# Function to start training with given parameters
train_model() {
  local gpu=$1
  local dataset=$2
  local model_name=$3
  local base_size=256
  local crop_size=256

  if [ "$dataset" == "IRSTD-1k" ]; then
    base_size=512
    crop_size=512
  fi

  CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --dataset "$dataset" \
    --model_name "$model_name" \
    --loss_func "$LOSS_FUNC" \
    --train 1 \
    --test 0 \
    --deep_supervision True \
    --batchsize "$BATCHSIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --base_size "$base_size" "$base_size" \
    --crop_size "$crop_size" \
    --optimizer_name "$OPTIMIZER_NAME" \
    --test_epoch "$TEST_EPOCH"
}

# Function to handle the training tasks for each dataset
train_dataset() {
  local dataset=$1
  local gpus="${DATASET_GPU_MAP[$dataset]}"

  for gpu in $gpus; do
    for model_name in "${MODELS[@]}"; do
      echo "Starting training for $model_name on $dataset using GPU $gpu"
      train_model $gpu $dataset $model_name
    done
  done
}

# Start training tasks for each dataset in parallel
for dataset in "${DATASETS[@]}"; do
  train_dataset $dataset &
done

# Wait for all background processes to finish
wait
