# Melan-Dx Code
The MUSK model embeddings for knowledge and images are available [here](https://drive.google.com/file/d/1zEYjr8QB7oy3TK9-7HZSQq9N3pYfmLA5/view?usp=sharing).


## Project Structure

### Root Directory

#### `train_model.py`
Main entry point for the entire training pipeline. Handles:
- Command-line argument parsing
- Configuration loading
- Three-stage execution flow: preprocessing, embedding generation, and model training
- Backbone model initialization (PLIP, MUSK, etc.)
- Coordination between data processing, embedding generation, and model training
- Multi-learning rate experiment management

#### `MelanDx_musk.sh`
Bash script for running the complete training pipeline with MUSK backbone. Automates:
- Environment activation
- Sequential execution of preprocessing, embedding generation, and training stages
- Error handling and status checking between stages

---

## Directory Structure

### `/models/`

Core model architecture and components.

#### `main_model.py`
Main model class (`MainModel`) that orchestrates the complete classification pipeline:
- Manages disease-to-index mappings
- Pre-computes label masks for images and knowledge
- Integrates embedding selection, fusion, and loss computation
- Implements forward pass with image and knowledge retrieval
- Provides prediction interface with top-k results and probability distributions

#### `model_components.py`
Modular components used by the main model:
- `EmbeddingSelector`: Retrieves similar samples using class-specific attention mechanisms
- `MultiClassAttention`: Independent attention layers for each disease class
- `EmbeddingFusion_weighted`: Weighted fusion of embeddings using transformer blocks
- `LossModule`: Handles loss computation with contrastive learning (basic loss and hierarchical loss options)

#### `model_TransformerAttentionBlock.py`
Transformer-based attention blocks for embedding fusion:

---

### `/data_processing/`

Data loading, validation, and preprocessing.


---

### `/training/`

Training loop and evaluation logic.

#### `trainer.py`
Complete training and evaluation pipeline:
- `TrainerConfig`: Training hyperparameters and settings
- `PreprocessedImageData`: Custom dataset class for image paths and labels
- `PreprocessedKnowledgeData`: Knowledge data container
- `ModelTrainer`: Main trainer class
  - Manages training loop with early stopping
  - Handles batch processing and loss computation
  - Implements evaluation with multiple metrics (top-k accuracy, F1, precision, recall, specificity)
  - Calculates hierarchical accuracy based on disease taxonomy
  - Saves checkpoints and metrics to CSV
  - Saves prediction results with probabilities
  - Integrates with Weights & Biases for experiment tracking

---

### `/utils/`

Utility functions and helpers.

#### `scheduler.py`
Learning rate scheduling:
- `cosine_lr()`: Implements cosine annealing with warmup for learning rate adjustment

#### `early_stopping.py`
Training optimization:
- `EarlyStopping`: Monitors validation metrics and stops training when no improvement is observed
  - Supports both 'min' and 'max' modes for different metrics
  - Configurable patience and minimum delta

#### `time_utils.py`
Execution time tracking:
- `TimeTracker`: Records and saves execution time for different pipeline stages
  - Tracks individual component timing
  - Saves statistics to JSON format

#### `seed_utils.py`
Reproducibility utilities:
- `set_seed()`: Sets random seeds for Python, NumPy, and PyTorch for reproducible results

---

### `/backbone_model/`

Wrapper classes for pre-trained vision-language models.

#### `plip_for_train.py`
PLIP (Pathology Language-Image Pretraining) model wrapper:
- `PLIP`: Wrapper class providing unified interface
  - `encode_images()`: Batch encoding of images to embeddings
  - `encode_text()`: Batch encoding of text to embeddings
  - Handles model initialization and inference

#### `musk_for_train.py`
MUSK model wrapper:
- `MUSK`: Wrapper class with similar interface to PLIP
  - Supports configurable encoding parameters (layer selection, pooling strategy)
  - `encode_images()`: Image encoding with flexible layer and pooling options
  - `encode_text()`: Text encoding with similar configurability

---

### `/config/`

JSON configuration files for different experiments.

#### `melandx_plip_config.json`
Configuration for PLIP backbone experiments:
- Data paths for train/val/test sets and knowledge base
- Model hyperparameters (embedding dimension, attention heads, retrieval numbers)
- Trainer settings (batch size, learning rate, epochs)

#### `melandx_musk_config.json`
Configuration for MUSK backbone experiments:
- Similar structure to PLIP config
- Includes encoder-specific settings for MUSK model

