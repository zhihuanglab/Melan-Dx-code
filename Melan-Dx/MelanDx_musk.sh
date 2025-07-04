#!/bin/bash

nvidia-smi


eval "$(conda shell.bash hook)"
conda activate melanDx


CONFIG_PATH="config/melandx_musk_config.json"
BACKBONE_TYPE="musk"
BACKBONE_PATH=""
EMBEDDING_DIR="embeddings/musk_embeddings"
LOSS_TYPE="basic"
SAVE_DIR="VLM_musk_model_weighted"


# # Stage 1: Data preprocessing
# echo "Starting data preprocessing stage..."
# python -u train_model.py \
#     --config ${CONFIG_PATH} \
#     --stage preprocess \
#     --backbone_type ${BACKBONE_TYPE} \
#     --backbone_path ${BACKBONE_PATH} \
#     --embedding_dir ${EMBEDDING_DIR}



echo "Starting embedding generation..."
python -u train_model.py \
    --config ${CONFIG_PATH} \
    --stage embedding \
    --backbone_type ${BACKBONE_TYPE} \
    --backbone_path ${BACKBONE_PATH} \
    --embedding_dir ${EMBEDDING_DIR} \
    --batch_size 32 \
    --save_dir ${SAVE_DIR}

# Check exit status of the previous command
if [ $? -ne 0 ]; then
    echo "Embedding generation stage failed, exiting execution"
    exit 1
fi

# Stage 3: Model training
echo "Starting model training..."
python -u train_model.py \
    --config ${CONFIG_PATH} \
    --stage train \
    --backbone_type ${BACKBONE_TYPE} \
    --backbone_path ${BACKBONE_PATH} \
    --embedding_dir ${EMBEDDING_DIR} \
    --loss_type ${LOSS_TYPE} \
    --learning_rates 1e-5 1e-4 1e-3 \
    --save_dir ${SAVE_DIR} \


# Check final execution status
if [ $? -ne 0 ]; then
    echo "Model training stage failed"
    exit 1
else
    echo "All stages completed successfully"
fi



