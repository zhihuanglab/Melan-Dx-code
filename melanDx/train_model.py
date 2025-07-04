import argparse
import json
import logging
import sys
import os
from pathlib import Path

import torch
from backbone_model.plip_for_train import PLIP
from backbone_model.musk_for_train import MUSK

from utils.time_utils import TimeTracker
from utils.seed_utils import set_seed
from models.main_model import MainModel
from models.model_components import ModelConfig
from training.trainer import ModelTrainer, TrainerConfig

# project_root = str(Path(__file__).parents[1])
# sys.path.append(project_root)
from data_processing.data_processor import DataProcessor, DataConfig


def load_config(config_path: str) -> dict:
    """Load JSON configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)

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
    parser.add_argument('--backbone_type', type=str, required=True, choices=['plip', 'conch','musk','pathgen'], help='Type of backbone model')
    parser.add_argument('--backbone_path', type=str, required=True, help='Path to backbone model')
    parser.add_argument('--backbone_checkpoint', type=str, help='Path to finetuned backbone checkpoint')
    parser.add_argument('--model_checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plip_orignial_path', type=str, default="/cbica/home/yaoji/Projects/VLM_2_3/checkpoints/plip", help='Path to plip original model to initialize parameters')
    parser.add_argument(
        '--learning_rates', 
        type=float, 
        nargs='+',  # Allow multiple values
        default=[1e-5],  # Default learning rate
        help='List of learning rates to try, e.g., 1e-5 1e-4 1e-3'
    )
    parser.add_argument('--stage', type=str, choices=['preprocess', 'embedding', 'train'], 
                       help='Stage to run: preprocess/embedding/train')
    parser.add_argument('--embedding_dir', type=str, default='embeddings',
                       help='Directory to save/load embeddings')
    parser.add_argument('--loss_type', type=str, 
                       choices=['basic', 'hierarchy', 'hierarchy_3level', 'weighted', 'combined'],
                       default='hierarchy',
                       help='Type of loss function to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding generation')
    parser.add_argument('--save_dir', type=str,
                       help='Directory to save model checkpoints')
    parser.add_argument('--use_projection', type=bool, default=False,
                       help='Whether to use projection layer for embedding dimension conversion')
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    data_config = DataConfig(**config['data_config'])
    model_config = ModelConfig(**config['model_config'])
    trainer_config = TrainerConfig(**config['trainer_config'])
    
    ############################################################
    # If save_dir is specified in command line arguments, override the value in config file
    if args.save_dir is not None:
        trainer_config.save_dir = args.save_dir
        trainer_config.wandb_name = args.save_dir
    
    # Set number of training epochs
    trainer_config.num_epochs = 100   
    model_config.num_hidden_layers = 12
    
    model_config.image_retrieval_number = 2
    model_config.knowledge_retrieval_number = 2
    
    # Set whether to use projection layer
    model_config.use_projection = False
    trainer_config.batch_size = 64
    # Set loss type
    model_config.loss_type = args.loss_type
    
    # Set bootstrap configuration (set directly in code, not using command line arguments)
    trainer_config.use_bootstrap = True
    trainer_config.bootstrap_n_samples = 10
    trainer_config.bootstrap_sample_ratio = 0.8
    ############################################################
    
    
    
    logger = setup_logging()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config.device = device
    
    # Create save directories
    os.makedirs(trainer_config.save_dir, exist_ok=True)
    os.makedirs(args.embedding_dir, exist_ok=True)
    
    # Initialize time tracker
    time_tracker = TimeTracker(trainer_config.save_dir)

    if args.stage == 'preprocess':
        # Data preprocessing stage
        time_tracker.start_track("data_processing")
        data_processor = DataProcessor(data_config)
        time_tracker.stop_track("data_processing")
        time_tracker.save_stats()
        return

    # Load data processor
    time_tracker.start_track("data_processing")
    data_processor = DataProcessor(data_config)
    time_tracker.stop_track("data_processing")
    if not data_processor.metadata:
        logger.error("Failed to initialize data processor")
        return

    # Prepare data dictionaries
    train_data, val_data, test_data, knowledge_data = prepare_data_dicts(data_processor)
    
    if args.stage == 'embedding':
        # Embedding generation stage
        generate_and_save_embeddings(
            args, train_data, val_data, test_data, knowledge_data,
            data_processor, device, logger, time_tracker, config
        )
        time_tracker.save_stats()
        return

    if args.stage == 'train':
        print("Number of hidden layers:", model_config.num_hidden_layers)
        # Training stage
        train_model_with_embeddings(
            args, train_data, val_data, test_data, knowledge_data,
            data_processor, model_config, trainer_config,
            device, logger, time_tracker
        )
        time_tracker.save_stats()
        return

def prepare_data_dicts(data_processor):
    """Prepare data dictionaries"""
    train_data = {
        'paths': [img.path for img in data_processor.metadata.train_images],
        'disease_names': [img.disease_name for img in data_processor.metadata.train_images],
        'disease_to_parent': data_processor.metadata.disease_to_parent,
        'parent_to_grandparent': data_processor.metadata.parent_to_grandparent
    }
    
    val_data = {
        'paths': [img.path for img in data_processor.metadata.val_images],
        'disease_names': [img.disease_name for img in data_processor.metadata.val_images]
    }
    
    test_data = {
        'paths': [img.path for img in data_processor.metadata.test_images],
        'disease_names': [img.disease_name for img in data_processor.metadata.test_images]
    }
    
    knowledge_data = {
        'texts': data_processor.metadata.knowledge_texts,
        'disease_names': data_processor.metadata.knowledge_disease_names
    }
    
    print(f"train_data count: {len(train_data['paths'])}")
    print(f"val_data count: {len(val_data['paths'])}")
    print(f"test_data count: {len(test_data['paths'])}")
    print(f"knowledge_data count: {len(knowledge_data['texts'])}")
    return train_data, val_data, test_data, knowledge_data

def generate_and_save_embeddings(args, train_data, val_data, test_data, knowledge_data, 
                               data_processor, device, logger, time_tracker, config):
    """Generate and save embeddings"""
    # Initialize backbone model
    logger.info(f"Loading {args.backbone_type} model from {args.backbone_path}")
    backbone_model = initialize_backbone_model(args, device)
    
    # Check if saved embeddings exist
    embedding_files = {
        'train': f"{args.embedding_dir}/train_embeddings.pt",
        'val': f"{args.embedding_dir}/val_embeddings.pt",
        'test': f"{args.embedding_dir}/test_embeddings.pt",
        'knowledge': f"{args.embedding_dir}/knowledge_embeddings.pt"
    }
    
    # Get encoder configuration
    encoder_config = {}
    if args.backbone_type == 'musk' and 'encoder_config' in config:
        encoder_config = config['encoder_config'].get('musk', {})
        logger.info(f"Using MUSK encoder config: {encoder_config}")
    
    time_tracker.start_track("embedding_generation")
    
    # Use torch.no_grad() to generate embeddings
    with torch.no_grad():
        # Generate and save embeddings
        if not os.path.exists(embedding_files['train']):
            train_embeddings = backbone_model.encode_images(
                train_data['paths'], 
                batch_size=args.batch_size,
                **encoder_config if args.backbone_type == 'musk' else {}
            )
            # Save only filenames
            train_filenames = [os.path.basename(path) for path in train_data['paths']]
            torch.save({
                'embeddings': train_embeddings,
                'paths': train_filenames,
                'disease_names': train_data['disease_names']
            }, embedding_files['train'])
            logger.info(f"Saved train embeddings to {embedding_files['train']}")
        
        if not os.path.exists(embedding_files['val']):
            val_embeddings = backbone_model.encode_images(
                val_data['paths'], 
                batch_size=args.batch_size,
                **encoder_config if args.backbone_type == 'musk' else {}
            )
            val_filenames = [os.path.basename(path) for path in val_data['paths']]
            torch.save({
                'embeddings': val_embeddings,
                'paths': val_filenames,
                'disease_names': val_data['disease_names']
            }, embedding_files['val'])
            logger.info(f"Saved validation embeddings to {embedding_files['val']}")
        
        if not os.path.exists(embedding_files['test']):
            test_embeddings = backbone_model.encode_images(
                test_data['paths'], 
                batch_size=args.batch_size,
                **encoder_config if args.backbone_type == 'musk' else {}
            )
            test_filenames = [os.path.basename(path) for path in test_data['paths']]
            torch.save({
                'embeddings': test_embeddings,
                'paths': test_filenames,
                'disease_names': test_data['disease_names']
            }, embedding_files['test'])
            logger.info(f"Saved test embeddings to {embedding_files['test']}")
        
        if not os.path.exists(embedding_files['knowledge']):
            knowledge_embeddings = backbone_model.encode_text(
                knowledge_data['texts'], 
                batch_size=args.batch_size,
                **encoder_config if args.backbone_type == 'musk' else {}
            )
            torch.save({
                'embeddings': knowledge_embeddings,
                'texts': knowledge_data['texts'],
                'disease_names': knowledge_data['disease_names']
            }, embedding_files['knowledge'])
            logger.info(f"Saved knowledge embeddings to {embedding_files['knowledge']}")
    
    time_tracker.stop_track("embedding_generation")

def verify_embeddings_data(saved_data: dict, current_data: dict, data_type: str) -> bool:
    """Verify if saved embeddings data matches current data"""
    if data_type == 'knowledge':
        paths_match = saved_data['texts'] == current_data['texts']
    else:
        # Convert current data paths to filenames for comparison
        current_filenames = [os.path.basename(path) for path in current_data['paths']]
        paths_match = saved_data['paths'] == current_filenames
    
    diseases_match = saved_data['disease_names'] == current_data['disease_names']
    
    return paths_match and diseases_match

def train_model_with_embeddings(args, train_data, val_data, test_data, knowledge_data,
                              data_processor, model_config, trainer_config,
                              device, logger, time_tracker):
    """Train model using pre-generated embeddings"""
    # Load pre-generated embeddings
    embedding_files = {
        'train': f"{args.embedding_dir}/train_embeddings.pt",
        'val': f"{args.embedding_dir}/val_embeddings.pt",
        'test': f"{args.embedding_dir}/test_embeddings.pt",
        'knowledge': f"{args.embedding_dir}/knowledge_embeddings.pt"
    }
    
    # Verify all required embedding files exist and load them
    embeddings_data = {}
    current_data = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'knowledge': knowledge_data
    }
    
    for name, path in embedding_files.items():
        if not os.path.exists(path):
            logger.error(f"Missing {name} embeddings file: {path}")
            return
            
        # Load saved data
        saved_data = torch.load(path)
        
        # Verify data matches
        if not verify_embeddings_data(saved_data, current_data[name], name):
            logger.error(f"Mismatch in {name} data between saved embeddings and current data")
            return
        
        print(f"{name} embeddings shape: {saved_data['embeddings'].shape}")
        embeddings_data[name] = saved_data['embeddings']
    
    # Ensure train_data contains necessary hierarchy information
    if args.loss_type in ['hierarchy', 'hierarchy_3level', 'combined']:
        if 'disease_to_parent' not in train_data:
            logger.error("Hierarchy loss requires disease_to_parent in train_data")
            return
        
        if args.loss_type == 'hierarchy_3level' and 'parent_to_grandparent' not in train_data:
            logger.error("3-level hierarchy loss requires parent_to_grandparent in train_data")
            return
    
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
            device=device,
            plip_orignial_path=args.plip_orignial_path
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
            data_processor=data_processor,
            precomputed_embeddings=embeddings_data
        )
        
        logger.info(f"Starting training with learning rate {lr}...")
        results = trainer.train()
        time_tracker.stop_track(f"model_training_lr_{lr}")
        all_results[lr] = results
        
        # Log best results for current learning rate
        log_training_results(logger, lr, results)
    
    return all_results

def initialize_backbone_model(args, device):
    """Initialize backbone model"""
    if args.backbone_type == "plip":
        backbone_model = PLIP(args.backbone_path)
    elif args.backbone_type == "musk":
        backbone_model = MUSK(args.backbone_path)
    else:
        raise ValueError(f"Unknown backbone type: {args.backbone_type}")
    
    backbone_model.device = device
    
    if args.backbone_checkpoint:
        checkpoint = torch.load(args.backbone_checkpoint, map_location=device)
        backbone_model.model.load_state_dict(checkpoint['model_state_dict'])
    
    return backbone_model

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