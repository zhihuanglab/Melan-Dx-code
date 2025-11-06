#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate melandx

# Configuration parameters
CONFIG_PATH="config/melandx_musk_config.json"
TRAIN_EMBEDDING="/xxx/xxx/train_embeddings.pt"
VAL_EMBEDDING="/xxx/xxx/val_embeddings.pt"
TEST_EMBEDDING="/xxx/xxx/test_embeddings.pt"
KNOWLEDGE_EMBEDDING="/xxx/xxx/knowledge_embeddings.pt"
TREE_JSON_PATH="config/who_44_classes_tree.json"
LOSS_TYPE="basic"
SAVE_DIR="melandx_musk_model_results"

# Model training (directly from saved embeddings)
echo "Starting model training from saved embeddings..."
echo "Train embedding: ${TRAIN_EMBEDDING}"
echo "Val embedding: ${VAL_EMBEDDING}"
echo "Test embedding: ${TEST_EMBEDDING}"
echo "Knowledge embedding: ${KNOWLEDGE_EMBEDDING}"
echo "Save directory: ${SAVE_DIR}"

python -u train_model.py \
    --config ${CONFIG_PATH} \
    --train_embedding ${TRAIN_EMBEDDING} \
    --val_embedding ${VAL_EMBEDDING} \
    --test_embedding ${TEST_EMBEDDING} \
    --knowledge_embedding ${KNOWLEDGE_EMBEDDING} \
    --tree_json_path ${TREE_JSON_PATH} \
    --loss_type ${LOSS_TYPE} \
    --learning_rates 1e-5 1e-4 1e-3 \
    --save_dir ${SAVE_DIR}


# Check execution status
if [ $? -ne 0 ]; then
    echo "Model training failed"
    exit 1
else
    echo "Training completed successfully"
fi
