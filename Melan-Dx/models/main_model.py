import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
from .model_TransformerAttentionBlock import CLIPModel_my
from .model_components import (
    ModelConfig,
    EmbeddingSelector,
    EmbeddingFusion_weighted,
    EmbeddingFusion_average,
    LossModule
)
from backbone_model.plip_for_train import PLIP


class MainModel(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
            backbone_model,
            # fusion_model,
            train_data: Dict,
            knowledge_data: Dict,
            device: str = "cuda",
            plip_orignial_path: str = "VLM/checkpoints/plip"
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.backbone_model = backbone_model

        # Create disease_to_idx mapping from train data
        unique_diseases = sorted(set(train_data['disease_names']))
        self.disease_to_idx = {name: idx for idx, name in enumerate(unique_diseases)}

        # Pre-compute image label masks
        num_samples = len(train_data['paths'])
        num_classes = len(self.disease_to_idx)

        # Create image label masks
        self.image_label_mask = torch.zeros((num_samples, num_classes), device=device)
        image_label_indices = torch.tensor(
            [self.disease_to_idx[label] for label in train_data['disease_names']],
            device=device
        )


        self.image_label_mask.scatter_(
            1,
            image_label_indices.unsqueeze(1),  # (num_samples, 1)
            torch.ones(num_samples, 1, device=device)  
        )

        # Pre-compute sample count per class
        self.samples_per_class = self.image_label_mask.sum(0)  # (num_classes,)

        # Pre-compute knowledge label masks
        num_knowledge = len(knowledge_data['texts'])
        self.knowledge_class_masks = torch.zeros((num_classes, num_knowledge), device=device)

        knowledge_label_indices = torch.tensor(
            [self.disease_to_idx[label] for label in knowledge_data['disease_names']],
            device=device
        )


        self.knowledge_class_masks.scatter_(
            0, 
            knowledge_label_indices.unsqueeze(0),  # (1, num_knowledge)
            torch.ones(1, num_knowledge, device=device)
        )
        # Convert to boolean type
        self.knowledge_class_masks = self.knowledge_class_masks.bool()
        

        
        
        # Initialize other components
        self.embedding_selector = EmbeddingSelector(
            config,
            self.image_label_mask,
            self.samples_per_class
        )
        if config.fusion_model == "average":    
            self.image_fusion = EmbeddingFusion_average(config, CLIPModel_my)
            self.knowledge_fusion = EmbeddingFusion_average(config, CLIPModel_my)
        elif config.fusion_model == "weighted":
            self.image_fusion = EmbeddingFusion_weighted(config, CLIPModel_my)
            self.knowledge_fusion = EmbeddingFusion_weighted(config, CLIPModel_my)
        else:
            raise ValueError(f"Invalid fusion model: {config.fusion_model}")
        self.loss_module = LossModule(config)


        self.to(device)

        # Add reverse mapping
        self.idx_to_disease = {idx: name for name, idx in self.disease_to_idx.items()}
        self.disease_to_parent = train_data['disease_to_parent']  
        self.parent_to_grandparent = train_data['parent_to_grandparent']



    def forward(
            self,
            query_embeddings: torch.Tensor,  # (batch_size, embed_dim)
            all_image_embeddings: torch.Tensor,  # (num_images, embed_dim)
            all_knowledge_embeddings: torch.Tensor,  # (num_knowledge, embed_dim)
            training: bool = True,
            exclude_indices: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch processing forward pass"""
        batch_size = query_embeddings.size(0)
        num_classes = len(self.disease_to_idx)

        # 1. Get similar image indices and attention weights (batch_size, num_classes, k_img)
        similar_images_indices, image_attention_weights = self.embedding_selector.get_similar_samples(
            query_embeddings,
            all_image_embeddings,
            n_samples=self.config.image_retrieval_number,
            exclude_indices=exclude_indices if training else None
        )

        # 2. Batch retrieve similar image embeddings
        # valid_mask: (batch_size, num_classes, k_img)
        valid_mask = (similar_images_indices != -1)
        # padded_indices: (batch_size, num_classes, k_img)
        padded_indices = torch.where(valid_mask, similar_images_indices, 0)
        # similar_images: (batch_size, num_classes, k_img, embed_dim)
        similar_images = all_image_embeddings[padded_indices]
        similar_images = torch.where(
            valid_mask.unsqueeze(-1),
            similar_images,
            torch.zeros_like(similar_images)
        )

        # Apply softmax to image attention weights
        image_attention_weights = F.softmax(image_attention_weights, dim=-1)  # Softmax over k_img dimension

        # 3. Fuse image embeddings: (batch_size, num_classes, embed_dim)
        enhanced_images = self.image_fusion.fuse_embeddings_batch(
            query_embeddings = query_embeddings,
            context_embeddings = similar_images,
            attention_weights = image_attention_weights,
            is_knowledge = False
        )

        # 4. Prepare input for knowledge selection
        # (batch_size, num_classes, k_img, embed_dim) -> (batch_size * num_classes * k_img, embed_dim)
        flattened_images = similar_images.reshape(-1, similar_images.size(-1))

        # Create class masks for each image
        # First create class index matrix
        class_indices = torch.arange(num_classes, device=self.device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_classes)
        # Expand to each retrieved image
        class_indices = class_indices.unsqueeze(2).expand(-1, -1, self.config.image_retrieval_number)  # (batch_size, num_classes, k_img)
        # Flatten to match flattened_images
        flat_class_indices = class_indices.reshape(-1)  # (batch_size*num_classes*k_img)

        # Create new knowledge mask, each image can only access knowledge of its corresponding class
        knowledge_mask = torch.zeros(
            (len(flattened_images), len(all_knowledge_embeddings)),
            # (batch_size*num_classes*k_img, num_knowledge)
            dtype=torch.bool,
            device=self.device
        )

        # For each class, find its corresponding knowledge
        for class_idx in range(num_classes):
            # Find all image positions for current class
            class_positions = (flat_class_indices == class_idx)
            # Get knowledge mask for this class
            class_knowledge = self.knowledge_class_masks[class_idx]  # (num_knowledge,)
            # Assign class knowledge mask to corresponding image positions
            knowledge_mask[class_positions] = class_knowledge

        # 5. Get knowledge indices and attention weights for each image
        knowledge_indices, knowledge_attention_weights = self.embedding_selector.get_similar_samples_for_knowledge(
            flattened_images,  # (batch_size*num_classes*k_img, embed_dim)
            all_knowledge_embeddings,  # (num_knowledge, embed_dim)
            knowledge_mask,  # (batch_size*num_classes*k_img, num_knowledge)
            self.config.knowledge_retrieval_number
        )

        # 6. Get selected knowledge embeddings
        # valid_k_mask: (batch_size*k_img, num_classes, k_knowledge)
        valid_k_mask = (knowledge_indices != -1)
        padded_k_indices = torch.where(valid_k_mask, knowledge_indices, 0)

        # Reshape indices and mask to match batch structure
        # padded_k_indices: (batch_size, num_classes, k_img, k_knowledge)
        padded_k_indices = padded_k_indices.view( # padded is used to set missing knowledge to 0 for subsequent indexing
            batch_size,
            num_classes,
            self.config.image_retrieval_number,
            self.config.knowledge_retrieval_number
        )
        knowledge_attention_weights = knowledge_attention_weights.view(
            batch_size,
            num_classes,
            self.config.image_retrieval_number,
            self.config.knowledge_retrieval_number
        )
        # valid_k_mask: (batch_size, num_classes, k_img, k_knowledge)
        valid_k_mask = valid_k_mask.view( # valid is used to indicate which ones are valid
            batch_size,
            num_classes,
            self.config.image_retrieval_number,
            self.config.knowledge_retrieval_number
        )

        # Get knowledge embeddings: (batch_size, num_classes, k_img, k_knowledge, embed_dim)
        selected_knowledge = all_knowledge_embeddings[padded_k_indices]
        selected_knowledge = torch.where(
            valid_k_mask.unsqueeze(-1),
            selected_knowledge,
            torch.zeros_like(selected_knowledge)
        )

        # 8. Fuse knowledge embeddings: (batch_size, num_classes, embed_dim)
        enhanced_knowledge = self.knowledge_fusion.fuse_embeddings_batch(
            query_embeddings = query_embeddings ,
            context_embeddings = selected_knowledge,
            attention_weights = knowledge_attention_weights,
            image_weights = image_attention_weights,  # Directly use softmaxed image_attention_weights
            is_knowledge = True
        )

        return enhanced_images, enhanced_knowledge

    def compute_loss(
            self,
            enhanced_images: torch.Tensor,  # (batch_size, num_classes, embed_dim)
            enhanced_knowledge: torch.Tensor,  # (batch_size, num_classes, embed_dim)
            target_indices: torch.Tensor  # (batch_size,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate loss and accuracy"""
        return self.loss_module.compute_loss(
            enhanced_images,
            enhanced_knowledge,
            target_indices,
            self.hierarchy_targets,
            self.hierarchy_3level_targets
        )

    def predict_with_scores(
            self,
            query_embeddings: torch.Tensor,  # (batch_size, embed_dim)
            all_image_embeddings: torch.Tensor,  # (num_images, embed_dim)
            all_knowledge_embeddings: torch.Tensor,  # (num_knowledge, embed_dim)
            top_k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # (batch_size, top_k), (batch_size, top_k), (batch_size, num_classes)
        """
        Predict categories and return similarity scores and probability distributions for all classes

        Returns:
            - Predicted category indices: (batch_size, top_k)
            - Similarity scores: (batch_size, top_k)
            - Probability distribution for all classes: (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            # 1. Get enhanced embeddings
            enhanced_images, enhanced_knowledge = self.forward(
                query_embeddings,
                all_image_embeddings,
                all_knowledge_embeddings,
                training=False
            )

            # 2. Normalize embeddings
            image_norm = F.normalize(enhanced_images, p=2, dim=2)
            knowledge_norm = F.normalize(enhanced_knowledge, p=2, dim=2)

            # 3. Calculate similarity for each class
            similarities = torch.sum(image_norm * knowledge_norm, dim=2)
            
            # 4. Calculate probability distribution
            probabilities = F.softmax(similarities, dim=1)
            
            # 5. Get top-k predictions and their scores
            scores, indices = torch.topk(similarities, k=top_k, dim=1)

            return indices, scores, probabilities





