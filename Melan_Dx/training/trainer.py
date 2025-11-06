# training/trainer.py
import os

import torch
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from utils.scheduler import cosine_lr
from utils.early_stopping import EarlyStopping

@dataclass
class TrainerConfig:
    num_epochs: int = 1000
    batch_size: int = 5
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.1
    scheduler_patience: int = 3
    eval_steps: int = 300 
    embedding_update_steps: int = 50  
    save_dir: str = "checkpoints"
    wandb_login: str = "wandb_login"
    wandb_project: str = "self_supervised_model"
    wandb_name: str = "image_knowledge_model"
    wandb_notes: str = "experiment"
    use_wandb: bool = True
    backbone_batch_size: int = 40
    test_train_disease_match: bool = True 





def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function to handle PIL images
    Args:
        batch: List of dictionaries, each containing 'image', 'path', 'disease_idx'
    Returns:
        Merged dictionary
    """
    # Collect each type of data separately
    paths = [item['path'] for item in batch]
    disease_indices = torch.stack([item['disease_idx'] for item in batch])

    return {
        'path': paths,
        'disease_idx': disease_indices
    }


class PreprocessedImageData(Dataset):
    """Custom dataset class for preprocessed data"""

    def __init__(self, image_data, disease_to_idx):
        """
        Args:
            image_data: dict containing paths and disease names
            disease_to_idx: mapping from disease names to indices
        """
        self.paths = image_data['paths']  # List of image paths
        self.disease_names = image_data['disease_names']  # List of disease names
        self.disease_to_idx = disease_to_idx

        # Pre-compute disease indices
        self.disease_indices = torch.tensor([
            self.disease_to_idx[name] for name in self.disease_names
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {
            'path': self.paths[idx],
            'disease_idx': self.disease_indices[idx]
        }



@dataclass
class PreprocessedKnowledgeData:
    """Structure for preprocessed knowledge data"""
    texts: List[str]  # Knowledge texts
    disease_names: List[str]  # Corresponding disease names


class ModelTrainer:
    def __init__(
            self,
            model,
            train_data: Dict,
            val_data: Dict,
            test_data: Dict,
            knowledge_data: Dict,
            trainer_config: TrainerConfig,
            device: str = "cuda",
            data_processor = None,
            precomputed_embeddings: Dict[str, torch.Tensor] = None
    ):
        self.model = model
        self.config = trainer_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.data_processor = data_processor
        self.precomputed_embeddings = precomputed_embeddings


        # Create datasets
        self.train_dataset = PreprocessedImageData(train_data, self.model.disease_to_idx)
        self.val_dataset = PreprocessedImageData(val_data, self.model.disease_to_idx) 
        self.test_dataset = PreprocessedImageData(test_data, self.model.disease_to_idx)

        # Record metrics for each epoch
        self.val_metrics = []
        self.test_metrics = []
        self.best_metrics = {
            'top1_accuracy': 0.0,
            'top3_accuracy': 0.0,
            'top5_accuracy': 0.0,
            'top10_accuracy': 0.0,
            'f1_weighted': 0.0,
            'f1_macro': 0.0
        }

        
        self.knowledge_data = PreprocessedKnowledgeData(
            texts=knowledge_data['texts'],
            disease_names=knowledge_data['disease_names']
        )

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trainer_config.learning_rate,
            weight_decay=trainer_config.weight_decay
        )

        # Calculate total training steps
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        total_steps = len(train_loader) * trainer_config.num_epochs
        warmup_steps = len(train_loader) * 5  # Set 5 epochs for warmup

        # Use cosine learning rate scheduler
        self.scheduler = cosine_lr(
            optimizer=self.optimizer,
            base_lr=trainer_config.learning_rate,
            warmup_length=warmup_steps,
            steps=total_steps
        )

        # Initialize EarlyStopping, change mode to 'max' because higher accuracy is better
        self.early_stopping = EarlyStopping(
            patience=20,
            mode='max',  # Change to max because higher accuracy is better
            min_delta=0.0,
            verbose=True
        )



    def update_embeddings(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update and return all required embeddings"""
        with torch.no_grad():
            all_train_embeddings = self.model.embedding_updater.update_embeddings_from_images(
                images=self.train_dataset.paths,
                batch_size=self.config.backbone_batch_size
            )
            all_val_embeddings = self.model.embedding_updater.update_embeddings_from_images(
                images=self.val_dataset.paths,
                batch_size=self.config.backbone_batch_size
            )
            all_test_embeddings = self.model.embedding_updater.update_embeddings_from_images(
                images=self.test_dataset.paths,
                batch_size=self.config.backbone_batch_size
            )
            all_knowledge_embeddings = self.model.embedding_updater.update_embeddings_from_texts(
                texts=self.knowledge_data.texts,
                batch_size=self.config.backbone_batch_size
            )

            return (all_train_embeddings.to(self.device),
                    all_val_embeddings.to(self.device),
                    all_test_embeddings.to(self.device),
                    all_knowledge_embeddings.to(self.device))

    def _format_lr_for_filename(self, lr: float) -> str:
        """Convert learning rate to exponential format string"""
        # Convert learning rates like 1e-5 to '1e_5' format
        return f"{lr:.0e}".replace('-', '_')

    def _save_checkpoint(
            self,
            epoch: int,
            global_step: int,
            metrics: Dict[str, float],
            is_best: bool = False
    ):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'disease_to_idx': self.model.disease_to_idx  # Add disease_to_idx mapping
        }

        # Get formatted learning rate string
        lr_str = self._format_lr_for_filename(self.config.learning_rate)

        if is_best:
            torch.save(
                checkpoint,
                os.path.join(self.config.save_dir, f'best_model_lr_{lr_str}.pth')
            )
        # else:
        #     torch.save(
        #         checkpoint,
        #         os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}_lr_{lr_str}.pth')
        #     )

    def _save_metrics_to_csv(self, epoch):
        """Save metrics to CSV file"""
        import pandas as pd
        
        # Get formatted learning rate string
        lr_str = self._format_lr_for_filename(self.config.learning_rate)
        
        # Save validation set results
        val_df = pd.DataFrame(self.val_metrics)
        val_df.to_csv(
            f"{self.config.save_dir}/val_metrics_lr_{lr_str}.csv",
            index=False
        )
        
        # Save test set results
        test_df = pd.DataFrame(self.test_metrics)
        test_df.to_csv(
            f"{self.config.save_dir}/test_metrics_lr_{lr_str}.csv",
            index=False
            )

    def train(self):
        """Train the model"""
        if self.precomputed_embeddings is None:
            raise ValueError("No precomputed embeddings provided")
        
        # Use pre-computed embeddings - ensure they are on the correct device
        all_train_embeddings = self.precomputed_embeddings['train'].to(self.device)
        all_val_embeddings = self.precomputed_embeddings['val'].to(self.device)
        all_test_embeddings = self.precomputed_embeddings['test'].to(self.device)
        all_knowledge_embeddings = self.precomputed_embeddings['knowledge'].to(self.device)
        
        # Print device information for confirmation
        self.logger.info(f"Embeddings devices - train: {all_train_embeddings.device}, val: {all_val_embeddings.device}, test: {all_test_embeddings.device}, knowledge: {all_knowledge_embeddings.device}")
        
        # Initialize wandb
        if self.config.use_wandb:
            lr_str = self._format_lr_for_filename(self.config.learning_rate)
            exp_name = f"{self.config.wandb_name}_lr_{lr_str}"
            wandb.login(key=self.config.wandb_login)
            wandb.init(
                project=self.config.wandb_project,
                name=exp_name
            )
            
            
        global_step = 0
        best_val_accuracy = 0  


        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )

        for epoch in range(self.config.num_epochs):
            # Ensure training mode at the beginning of each epoch
            self.model.train()
            total_loss = 0
            total_accuracy = 0
            num_batches = 0

            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"):

                
                global_step += 1


                # Train batch
                batch_loss, batch_accuracy = self._train_batch(
                    batch_data,
                    all_train_embeddings,
                    all_knowledge_embeddings
                )

                total_loss += batch_loss
                total_accuracy += batch_accuracy
                
                num_batches += 1

                # Update learning rate
                self.scheduler(global_step)

                # Record training metrics
                if self.config.use_wandb:
                    wandb.log({
                        "train_loss": batch_loss.item(),
                        "train_accuracy": batch_accuracy,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1
                    }, step=global_step)

                # Periodic evaluation
                if global_step % self.config.eval_steps == 0:
                    val_metrics = self.evaluate(
                        all_train_embeddings,
                        all_val_embeddings,
                        all_knowledge_embeddings,
                        "val"
                    )
                    self.model.train()

            # Evaluate validation and test sets at the end of each epoch
            val_metrics = self.evaluate(
                all_train_embeddings,
                all_val_embeddings,
                all_knowledge_embeddings,
                "val"
            )
            test_metrics = self.evaluate(
                all_train_embeddings,
                all_test_embeddings,
                all_knowledge_embeddings,
                "test"
            )
            
            # Early stopping check
            if self.early_stopping(epoch, val_metrics["top1_accuracy"]):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best validation top1 accuracy: {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}")
                break
            

            self.val_metrics.append({
                'epoch': epoch + 1,
                **val_metrics
            })
            
            self.test_metrics.append({
                'epoch': epoch + 1,
                **test_metrics
            })
            

            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'epoch_train_loss': total_loss / num_batches,
                    **{f'epoch_val_{k}': v for k, v in val_metrics.items()},
                    **{f'epoch_test_{k}': v for k, v in test_metrics.items()},
                    'early_stopping_counter': self.early_stopping.counter
                }, step=global_step)


            if val_metrics["top1_accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["top1_accuracy"]
                self.best_metrics = val_metrics
                self._save_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    metrics=val_metrics,
                    is_best=True
                )


            self._save_metrics_to_csv(epoch + 1)

        final_results = {
            'best_metrics': self.best_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
            'learning_rate': self.config.learning_rate 
        }
        

        
        if self.config.use_wandb:
            wandb.finish()
        
        return final_results

    def _train_batch(
            self,
            batch_data: Dict,
            all_train_embeddings: torch.Tensor,
            all_knowledge_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, float]:

        self.optimizer.zero_grad()

        batch_paths = batch_data['path']
        target_indices = batch_data['disease_idx'].to(self.device)

        batch_indices = [self.train_dataset.paths.index(path) for path in batch_paths]
        batch_embeddings = all_train_embeddings[batch_indices]

        enhanced_image_embeddings, enhanced_knowledge_embeddings = self.model(
            batch_embeddings,
            all_train_embeddings,
            all_knowledge_embeddings,
            training=True,
            exclude_indices=batch_indices
        )

        loss, accuracy = self.model.compute_loss(
            enhanced_image_embeddings,
            enhanced_knowledge_embeddings,
            target_indices
        )

        loss.backward()
        self.optimizer.step()

        return loss, accuracy.item()

    def calculate_hierarchy_accuracy(
            self,
            pred_disease: str,
            true_disease: str,
            disease_to_parent: Dict[str, str],
            parent_to_grandparent: Optional[Dict[str, str]] = None
    ) -> float:
        """Calculate hierarchy accuracy score for a single prediction
        
        Args:
            pred_disease: Predicted disease name
            true_disease: True disease name
            disease_to_parent: Dictionary mapping diseases to parent nodes
            parent_to_grandparent: Dictionary mapping parent nodes to grandparent nodes (optional)
            
        Returns:
            float: Hierarchy accuracy score (1.0 for exact match, 0.5 for same parent, 0.25 for same grandparent)
        """

        if pred_disease == true_disease:
            return 1.0
        
        pred_parent = disease_to_parent[pred_disease]
        true_parent = disease_to_parent[true_disease]
        
        if pred_parent == true_parent:
            return 0.5
        
        if parent_to_grandparent is not None:
            if pred_parent in parent_to_grandparent and true_parent in parent_to_grandparent:
                pred_grandparent = parent_to_grandparent[pred_parent]
                true_grandparent = parent_to_grandparent[true_parent]
                if pred_grandparent == true_grandparent:
                    return 0.25
                
        return 0.0



    def evaluate(
            self,
            all_train_embeddings: torch.Tensor,
            all_eval_embeddings: torch.Tensor,
            all_knowledge_embeddings: torch.Tensor,
            dataset: str = "test"
    ) -> Dict[str, float]:

        was_training = self.model.training
        self.model.eval()
        
        # Ensure embeddings are on the correct device
        all_train_embeddings = all_train_embeddings.to(self.device)
        all_eval_embeddings = all_eval_embeddings.to(self.device)
        all_knowledge_embeddings = all_knowledge_embeddings.to(self.device)

        batch_size = self.config.batch_size


        if dataset == "val":
            eval_dataset = self.val_dataset
        elif dataset == "test":
            eval_dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid dataset: {dataset}")


        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        correct = {
            'top1': 0,
            'top3': 0,
            'top5': 0,
            'top10': 0
        }
        total_samples = len(eval_dataset)
        all_preds = []
        all_labels = []
        all_probs = [] 
        all_paths = [] 


        hierarchy_accuracy_sum = 0.0

        with torch.no_grad():
            for batch_data in tqdm(eval_loader, desc=f"Evaluating {dataset}"):

                batch_paths = batch_data['path']
                batch_indices = [eval_dataset.paths.index(path) for path in batch_paths]
                

                batch_embeddings = all_eval_embeddings[batch_indices]
                true_class_indices = batch_data['disease_idx'].to(self.device)


                pred_indices, _, probabilities = self.model.predict_with_scores(
                    batch_embeddings,
                    all_train_embeddings,
                    all_knowledge_embeddings,
                    top_k=10
                )


                all_probs.extend(probabilities.cpu().numpy())
                all_paths.extend(batch_paths)


                top1_preds = pred_indices[:, 0]
                all_preds.extend(top1_preds.cpu().numpy())
                all_labels.extend(true_class_indices.cpu().numpy())

                for i, true_idx in enumerate(true_class_indices):
                    if true_idx == pred_indices[i, 0]:
                        correct['top1'] += 1
                    if true_idx in pred_indices[i, :3]:
                        correct['top3'] += 1
                    if true_idx in pred_indices[i, :5]:
                        correct['top5'] += 1
                    if true_idx in pred_indices[i, :10]:
                        correct['top10'] += 1

                top1_preds = pred_indices[:, 0]
                
                for pred, true in zip(top1_preds, true_class_indices):
                    pred_disease = self.model.idx_to_disease[pred.item()]
                    true_disease = self.model.idx_to_disease[true.item()]
                    
                    hierarchy_score = self.calculate_hierarchy_accuracy(
                        pred_disease,
                        true_disease,
                        self.model.disease_to_parent,
                        parent_to_grandparent=getattr(self.model, 'parent_to_grandparent', None)
                    )
                    hierarchy_accuracy_sum += hierarchy_score

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        n_classes = len(self.model.disease_to_idx)
        conf_matrix = confusion_matrix(
            all_labels, 
            all_preds,
            labels=list(range(n_classes))
        )
        
        specificities = []
        class_weights = []
        
        for i in range(n_classes):
            class_weight = np.sum(all_labels == i)
            class_weights.append(class_weight)
            
            if class_weight > 0:
                true_neg = conf_matrix.sum() - conf_matrix[i,:].sum() - conf_matrix[:,i].sum() + conf_matrix[i,i]
                false_pos = conf_matrix[:,i].sum() - conf_matrix[i,i]
                specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) != 0 else 0
            else: 
                specificity = 0 
            
            specificities.append(specificity)
        

        specificity_macro = np.mean(specificities)
        specificity_weighted = (np.array(specificities) * np.array(class_weights)).sum() / np.sum(class_weights) if np.sum(class_weights) > 0 else 0.0
        

        hierarchy_accuracy = hierarchy_accuracy_sum / total_samples
        
        metrics = {
            "top1_accuracy": correct['top1'] / total_samples,
            "top3_accuracy": correct['top3'] / total_samples,
            "top5_accuracy": correct['top5'] / total_samples,
            "top10_accuracy": correct['top10'] / total_samples,
            "hierarchy_accuracy": hierarchy_accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "precision_macro": precision_macro,
            "recall_weighted": recall_weighted,
            "recall_macro": recall_macro,
            "specificity_macro": specificity_macro,
            "specificity_weighted": specificity_weighted
            # "num_classes_total": n_classes,
            # "num_classes_present": len(np.unique(all_labels))
        }


        predictions_dir = os.path.join(self.config.save_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        

        epoch = len(self.val_metrics) + 1 
        lr_str = self._format_lr_for_filename(self.config.learning_rate)
        


        

        prediction_results = {
            'path': all_paths,
            'true_label': [self.model.idx_to_disease[idx] for idx in all_labels],
            'probabilities': all_probs,
            'epoch': epoch,
            'metrics': metrics  
        }
        



        save_path = os.path.join(
            predictions_dir,
            f'{dataset}_predictions_epoch_{epoch}_lr_{lr_str}.npz'
        )
        np.savez_compressed(save_path, **prediction_results)

        self.model.train(was_training)

        return metrics

