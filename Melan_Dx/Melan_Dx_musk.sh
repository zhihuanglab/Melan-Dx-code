#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --output=logs/run_musk_finetune_output_%j.txt
#SBATCH --error=logs/run_musk_finetune_error_%j.txt
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=47:59:00
#SBATCH --partition=ltdai


nvidia-smi

eval "$(conda shell.bash hook)"
conda activate llava

# Configuration parameters
CONFIG_PATH="config/melandx_musk_config.json"
TRAIN_EMBEDDING="/cbica/home/yaoji/Projects/VLM_4_26/基础算法/2_musk_embeddings/train_embeddings.pt"
VAL_EMBEDDING="/cbica/home/yaoji/Projects/VLM_4_26/基础算法/2_musk_embeddings/val_embeddings.pt"
TEST_EMBEDDING="/cbica/home/yaoji/Projects/VLM_4_26/基础算法/2_musk_embeddings/test_embeddings.pt"
KNOWLEDGE_EMBEDDING="/cbica/home/yaoji/Projects/VLM_4_26/基础算法/2_musk_embeddings/knowledge_embeddings.pt"
TREE_JSON_PATH="config/who_44_classes_tree.json"
LOSS_TYPE="basic"
SAVE_DIR="VLM_musk_model_weighted"

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
