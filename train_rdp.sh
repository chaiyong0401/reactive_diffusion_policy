#!/bin/bash

GPU_ID=0
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "PYTHONPATH is $PYTHONPATH"

TASK_NAME="umi"
DATASET_PATH="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/rdp_dataset/dataset_zarr_normalized_replaybuffer.zarr"
LOGGING_MODE="online"

TIMESTAMP=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

# # # Stage 1: Train Asymmetric Tokenizer
# echo "Stage 1: training Asymmetric Tokenizer..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
#     --config-name=train_at_workspace \
#     task=real_${TASK_NAME}_image_xela_wrench_at_24fps \
#     task.dataset_path=${DATASET_PATH} \
#     task.name=real_${TASK_NAME}_image_xela_wrench_at_24fps_${TIMESTAMP} \
#     at=at_peel \
#     logging.mode=${LOGGING_MODE}

# find the latest checkpoint
echo ""
echo "Searching for the latest AT checkpoint..."
# AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${TIMESTAMP}*" -type d)/checkpoints/latest.ckpt
AT_LOAD_DIR="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/2025.06.26/11.04.49_train_vae_real_umi_image_xela_wrench_at_24fps_0626110449/checkpoints/latest.ckpt"
# AT_LOAD_DIR="/home/embodied-ai/mcy/reactive_diffusion_policy/data/outputs/2025.05.30/20.58.18_train_vae_real_umi_image_gelsight_emb_at_24fps_0530205817/checkpoints/latest.ckpt"
if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
    exit 1
fi

# Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=real_${TASK_NAME}_image_xela_wrench_ldp_24fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK_NAME}_image_xela_wrench_ldp_24fps_${TIMESTAMP} \
    at=at_peel \
    at_load_dir=${AT_LOAD_DIR} \
    logging.mode=${LOGGING_MODE}