# Melan-Dx: a knowledge-enhanced vision-language framework improves differential diagnosis of melanocytic neoplasm pathology

A training framework for melanocytic neoplasm classification using pre-computed embeddings and hierarchical disease taxonomy.
## 🔬 Related Publication

This framework accompanies the following peer-reviewed publication:

**Melan-Dx: a knowledge-enhanced vision-language framework improves differential diagnosis of melanocytic neoplasm pathology**
Jialu Yao, Songhao Li, Peixian Liang, Xiaowei Xu, David Elder, Zhi Huang
*npj Digital Medicine*, 2026

📄 Paper: https://www.nature.com/articles/s41746-026-02357-3

🔗 DOI: https://doi.org/10.1038/s41746-026-02357-3

If you use this code or the Melan-Dx framework in your research, please cite the paper above.

---
## Overview

This version allows you to **train directly from pre-computed embedding files** without data preprocessing or embedding generation steps.


## Input File Requirements

### 1. Embedding Files (Required)

Four `.pt` files, each containing:
- `embeddings`: torch.Tensor with shape `(N, embed_dim)`
- `disease_names`: List[str] with length N

Required files:
- `train_embeddings.pt` - Training set embeddings
- `val_embeddings.pt` - Validation set embeddings  
- `test_embeddings.pt` - Test set embeddings
- `knowledge_embeddings.pt` - Knowledge base embeddings

**Important: Preparing Your Embeddings**

If you have a single merged embedding file, you MUST split it into train/val/test sets using stratified sampling:

```bash
# Split embeddings with 70/15/15 ratio (stratified by disease label)
python split_embeddings.py merged_all_embeddings.pt ./output_dir --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```


### 2. Disease Hierarchy JSON (Required)

`config/who_44_classes_tree.json` - A 3-level hierarchical structure:

```json
{
  "Level 2 (Grandparent)": {
    "Level 3 (Parent)": [
      "Level 4 (Disease 1)",
      "Level 4 (Disease 2)",
      ...
    ]
  }
}
```

**Example:**
```json
{
  "Melanocytic neoplasms in intermittently sun-exposed skin": {
    "Naevi": [
      "Junctional, compound, and dermal naevi",
      "Simple lentigo and lentiginous melanocytic naevus",
      "Dysplastic naevus"
    ]
  }
}
```


## Quick Start

### Method 1: Using Shell Script (Recommended)

1. Edit `Melan_Dx_musk.sh` to set your embedding file paths:

```bash
TRAIN_EMBEDDING="/path/to/train_embeddings.pt"
VAL_EMBEDDING="/path/to/val_embeddings.pt"
TEST_EMBEDDING="/path/to/test_embeddings.pt"
KNOWLEDGE_EMBEDDING="/path/to/knowledge_embeddings.pt"
SAVE_DIR="output_model"
```

2. Run the script:

```bash
bash Melan_Dx_musk.sh
```

### Method 2: Direct Python Execution

```bash
python train_model.py \
    --config config/melandx_musk_config.json \
    --train_embedding /path/to/train_embeddings.pt \
    --val_embedding /path/to/val_embeddings.pt \
    --test_embedding /path/to/test_embeddings.pt \
    --knowledge_embedding /path/to/knowledge_embeddings.pt \
    --tree_json_path config/who_44_classes_tree.json \
    --loss_type basic \
    --learning_rates 1e-5 1e-4 1e-3 \
    --save_dir output_model
```



## Output Files

After training, the following files will be generated in `{SAVE_DIR}/`:

```
{SAVE_DIR}/
├── best_model_lr_1e_5.pth          # Best model for each learning rate
├── val_metrics_lr_1e_5.csv         # Validation metrics per epoch
├── test_metrics_lr_1e_5.csv        # Test metrics per epoch
└── predictions/                     # Prediction results
    ├── val_predictions_epoch_X_lr_1e_5.npz
    └── test_predictions_epoch_X_lr_1e_5.npz
```



## Data Flow

```
Input Files
├── Embedding Files (.pt)
│   ├── embeddings (Tensor)
│   └── disease_names (List)
│
└── Hierarchy JSON
    └── 3-level tree structure

        ↓

Automatic Data Structure Construction
├── train_data
│   ├── paths: Placeholder list
│   ├── disease_names: From embedding file
│   ├── disease_to_parent: Built from JSON
│   └── parent_to_grandparent: Built from JSON
│
├── val_data, test_data
│   ├── paths: Placeholder list
│   └── disease_names: From embedding file
│
└── knowledge_data
    ├── texts: Placeholder list
    └── disease_names: From embedding file

        ↓

Training Loop
├── Initialize MainModel
├── Initialize ModelTrainer
└── Start training iterations
```




## Embedding Processing Workflow

### Step 1: Split Merged Embeddings (If Needed)

If you have a single merged embedding file, split it into train/val/test:

```bash
# Basic usage (70/15/15 split)
python split_embeddings.py merged_all_embeddings.pt ./split_output

# Custom split ratios
python split_embeddings.py merged_all_embeddings.pt ./split_output \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 42
```
**Output:**
```
./split_output/
├── train_embeddings.pt
├── val_embeddings.pt
└── test_embeddings.pt
```

### Step 2: Update Training Script

Edit `Melan_Dx_musk.sh` to point to your split embeddings:

```bash
TRAIN_EMBEDDING="./split_output/train_embeddings.pt"
VAL_EMBEDDING="./split_output/val_embeddings.pt"
TEST_EMBEDDING="./split_output/test_embeddings.pt"
KNOWLEDGE_EMBEDDING="/path/to/knowledge_embeddings.pt"
```

## Example: Complete Training Workflow

```bash
# 1. Split embeddings (if needed)
python split_embeddings.py merged_all_embeddings.pt ./split_output

# 2. Start training
bash Melan_Dx_musk.sh

# 3. Monitor training progress (if using WandB)
# Open WandB link in browser
```

