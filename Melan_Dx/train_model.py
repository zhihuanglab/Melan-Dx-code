import argparse
import json
import logging
import sys
import os
from pathlib import Path

import torch

from utils.time_utils import TimeTracker
from utils.seed_utils import set_seed
from models.main_model import MainModel
from models.model_components import ModelConfig
from training.trainer import ModelTrainer, TrainerConfig


def load_config(config_path: str) -> dict:
    """Load JSON configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def build_hierarchy_from_tree(tree_json_path: str) -> tuple:
    """Build hierarchy mapping from who_44_classes_tree.json
    
    Args:
        tree_json_path: Path to who_44_classes_tree.json file
        
    Returns:
        disease_to_parent: dict mapping disease name to parent name (Level 4 -> Level 3)
        parent_to_grandparent: dict mapping parent name to grandparent name (Level 3 -> Level 2)
    """
    with open(tree_json_path, 'r') as f:
        tree = json.load(f)
    
    disease_to_parent = {}
    parent_to_grandparent = {}
    
    # Traverse the 3-level structure
    for grandparent, parent_dict in tree.items():
        for parent, diseases in parent_dict.items():
            # Build parent -> grandparent mapping
            parent_to_grandparent[parent] = grandparent
            
            # Build disease -> parent mapping
            for disease in diseases:
                disease_to_parent[disease] = parent
    
    return disease_to_parent, parent_to_grandparent

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(description='Self-Supervised Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--learning_rates', 
        type=float, 
        nargs='+',  # Allow multiple values
        default=[1e-5],  # Default learning rate
        help='List of learning rates to try, e.g., 1e-5 1e-4 1e-3'
    )
    parser.add_argument('--train_embedding', type=str, required=True,
                       help='Path to train embeddings file')
    parser.add_argument('--val_embedding', type=str, required=True,
                       help='Path to validation embeddings file')
    parser.add_argument('--test_embedding', type=str, required=True,
                       help='Path to test embeddings file')
    parser.add_argument('--knowledge_embedding', type=str, required=True,
                       help='Path to knowledge embeddings file')
    parser.add_argument('--loss_type', type=str, 
                       choices=['basic', 'hierarchy', 'hierarchy_3level', 'weighted', 'combined'],
                       default='basic',
                       help='Type of loss function to use')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--tree_json_path', type=str, 
                       default='config/who_44_classes_tree.json',
                       help='Path to disease hierarchy tree JSON file')
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    model_config = ModelConfig(**config['model_config'])
    trainer_config = TrainerConfig(**config['trainer_config'])
    
    ############################################################
    # If save_dir is specified in command line arguments, override the value in config file
    if args.save_dir is not None:
        trainer_config.save_dir = args.save_dir
        trainer_config.wandb_name = args.save_dir
    
    # # Set number of training epochs
    # trainer_config.num_epochs = 100   
    # model_config.num_hidden_layers = 8
    
    # model_config.image_retrieval_number = 2
    # model_config.knowledge_retrieval_number = 2
    
    # # Set whether to use projection layer
    # model_config.use_projection = True
    # trainer_config.batch_size = 64
    
    # Set loss type
    model_config.loss_type = args.loss_type
    

    ############################################################
    
    logger = setup_logging()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config.device = device
    
    # Create save directories
    os.makedirs(trainer_config.save_dir, exist_ok=True)
    
    # Initialize time tracker
    time_tracker = TimeTracker(trainer_config.save_dir)

    # 训练阶段 - 直接从embedding文件加载
    print("Number of hidden layers:", model_config.num_hidden_layers)
    train_model_with_embeddings_only(
        args, model_config, trainer_config,
        device, logger, time_tracker, args.tree_json_path
    )
    time_tracker.save_stats()

def train_model_with_embeddings_only(args, model_config, trainer_config,
                               device, logger, time_tracker, tree_json_path):
    """Train model using pre-generated embeddings"""
    # Load pre-generated embeddings
    embedding_files = {
        'train': args.train_embedding,
        'val': args.val_embedding,
        'test': args.test_embedding,
        'knowledge': args.knowledge_embedding
    }
    
    # Verify all required embedding files exist and load them
    embeddings_data = {}
    
    for name, path in embedding_files.items():
        if not os.path.exists(path):
            logger.error(f"Missing {name} embeddings file: {path}")
            return
            
        # Load saved data
        saved_data = torch.load(path)
        
        # Move embeddings to correct device
        embeddings = saved_data['embeddings'].to(device)
        
        print(f"{name} embeddings shape: {embeddings.shape}, device: {embeddings.device}")
        embeddings_data[name] = embeddings
    
    # Reconstruct data dictionaries from embedding files
    logger.info("Reconstructing data dictionaries from embeddings...")
    
    # Load embedding data
    train_saved = torch.load(embedding_files['train'])
    val_saved = torch.load(embedding_files['val'])
    test_saved = torch.load(embedding_files['test'])
    knowledge_saved = torch.load(embedding_files['knowledge'])
    
    # Load hierarchy information from JSON file
    logger.info(f"Loading hierarchy information from {tree_json_path}")
    disease_to_parent, parent_to_grandparent = build_hierarchy_from_tree(tree_json_path)
    logger.info(f"Loaded {len(disease_to_parent)} disease mappings and {len(parent_to_grandparent)} parent mappings")
    
    # Build training data dictionary
    train_data = {
        'paths': [f"train_sample_{i}" for i in range(len(train_saved['disease_names']))],  # Placeholder paths
        'disease_names': train_saved['disease_names'],
        'disease_to_parent': disease_to_parent,
        'parent_to_grandparent': parent_to_grandparent
    }
    
    val_data = {
        'paths': [f"val_sample_{i}" for i in range(len(val_saved['disease_names']))],
        'disease_names': val_saved['disease_names']
    }
    
    test_data = {
        'paths': [f"test_sample_{i}" for i in range(len(test_saved['disease_names']))],
        'disease_names': test_saved['disease_names']
    }
    
    knowledge_data = {
        'texts': [f"knowledge_{i}" for i in range(len(knowledge_saved['disease_names']))],  # Placeholder texts
        'disease_names': knowledge_saved['disease_names']
    }
    
    logger.info(f"Train samples: {len(train_data['paths'])}")
    logger.info(f"Val samples: {len(val_data['paths'])}")
    logger.info(f"Test samples: {len(test_data['paths'])}")
    logger.info(f"Knowledge entries: {len(knowledge_data['texts'])}")
    
    # Verify hierarchy information
    logger.info(f"Hierarchy information: {len(train_data['disease_to_parent'])} diseases, {len(train_data['parent_to_grandparent'])} parents")
    
    # Check if all diseases have hierarchy mapping
    missing_diseases = [d for d in train_data['disease_names'] if d not in train_data['disease_to_parent']]
    if missing_diseases:
        logger.warning(f"Found {len(missing_diseases)} diseases without hierarchy mapping")
        logger.warning(f"Example missing diseases: {missing_diseases[:5]}")
    
    # Train with each learning rate
    all_results = {}
    for lr in args.learning_rates:
        logger.info(f"\nStarting training with learning rate {lr}")
        trainer_config.learning_rate = lr
        
        time_tracker.start_track(f"model_training_lr_{lr}")
        
        # Initialize model
        model = MainModel(
            config=model_config,
            backbone_model=None,  # No backbone model needed
            train_data=train_data,
            knowledge_data=knowledge_data,
            device=device
        )
        
        # Load model checkpoint (if available)
        if args.model_checkpoint:
            logger.info(f"Loading model checkpoint from {args.model_checkpoint}")
            checkpoint = torch.load(args.model_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize trainer and start training
        trainer = ModelTrainer(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            knowledge_data=knowledge_data,
            trainer_config=trainer_config,
            device=device,
            data_processor=None,  
            precomputed_embeddings=embeddings_data
        )
        
        logger.info(f"Starting training with learning rate {lr}...")
        results = trainer.train()
        time_tracker.stop_track(f"model_training_lr_{lr}")
        all_results[lr] = results
        
        # Log best results for current learning rate
        log_training_results(logger, lr, results)
    
    return all_results

def log_training_results(logger, learning_rate: float, results: dict):
    """Log training results
    
    Args:
        logger: Logger instance
        learning_rate: Current learning rate
        results: Training results dictionary
    """
    logger.info(f"\nTraining results for learning rate {learning_rate}:")
    logger.info("-" * 50)
    
    # Log evaluation metrics
    logger.info("Evaluation metrics:")
    for metric_name, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"{metric_name}: {value:.4f}")
        else:
            logger.info(f"{metric_name}: {value}")
    
    logger.info("-" * 50)

if __name__ == "__main__":
    main()