import logging

# models/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

from PIL import Image

from backbone_model.plip_for_train import PLIP


@dataclass
class ModelConfig:
    """Model configuration"""
    embed_dim: int = 512
    num_heads: int = 8
    dropout_rate: float = 0.1
    logit_scale_init_value: float = 2.6592
    num_hidden_layers: int = 3
    image_retrieval_number: int = 5
    knowledge_retrieval_number: int = 10
    device: str = "cuda"
    fusion_model: str = "average"
    loss_type: str = "hierarchy"  # Add loss type
    use_projection: bool = False  # Whether to use projection layer
    projection_dim: int = 512  # Dimension after projection


    
class MultiClassAttention(nn.Module):
    """Create independent attention for each class"""

    def __init__(self, embed_dim: int, num_heads: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        # Create independent attention layers for each class
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_classes)
        ])

    def forward(self, query, key, value, class_idx):
        """
        Args:
            query: (L, N, E) Query embeddings
            key: (S, N, E) Key embeddings for specific class
            value: (S, N, E) Value embeddings for specific class
            class_idx: Which class attention to use
        """
        return self.attention_layers[class_idx](query, key, value)


class EmbeddingSelector(nn.Module):
    def __init__(
            self,
            config: ModelConfig,
            label_mask: torch.Tensor, 
            samples_per_class: torch.Tensor  
    ):
        super().__init__()
        self.config = config
        self.label_mask = label_mask
        self.samples_per_class = samples_per_class
        self.num_classes = label_mask.size(1)

        # 创建类别特定的attention
        self.class_attention = MultiClassAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_classes=self.num_classes,
            dropout=config.dropout_rate
        ).to(config.device)



    def get_similar_samples(
            self,
            query_embeddings: torch.Tensor,  # shape: (batch_size, embed_dim)
            all_embeddings: torch.Tensor,  # shape: (num_samples, embed_dim)
            n_samples: int,  # Number of samples to select per class
            exclude_indices: Optional[List[int]] = None  # Sample indices to exclude
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Return (indices, attention_weights)
        """
        Modified return values to include both selected sample indices and corresponding attention weights
        """
        bsz, embed_dim = query_embeddings.size()

        # Initialize result tensors
        result_indices = torch.zeros(
            (bsz, self.num_classes, n_samples),
            dtype=torch.long,
            device=self.config.device
        )
        result_weights = torch.zeros(
            (bsz, self.num_classes, n_samples),
            device=self.config.device
        )

        # Handle exclude indices
        exclude_mask = None
        if exclude_indices:
            exclude_mask = torch.zeros(len(all_embeddings), dtype=torch.bool, device=self.config.device)
            exclude_mask[exclude_indices] = True

        # Process each class separately
        for class_idx in range(self.num_classes):
            # Get sample mask for current class
            class_mask = self.label_mask[:, class_idx]  # (num_samples,)

            # Skip if this class has no samples
            if not class_mask.any():
                result_indices[:, class_idx] = -1
                result_weights[:, class_idx] = 0
                continue

            # Get samples for current class
            class_sample_indices = torch.where(class_mask)[0]
            class_embeddings = all_embeddings[class_sample_indices]  # (class_samples, embed_dim)

            # Exclude specified samples
            if exclude_mask is not None:
                class_exclude_mask = exclude_mask[class_sample_indices]
                if class_exclude_mask.any():
                    class_embeddings = class_embeddings[~class_exclude_mask]
                    class_sample_indices = class_sample_indices[~class_exclude_mask]

            if len(class_embeddings) == 0:
                result_indices[:, class_idx] = -1
                result_weights[:, class_idx] = 0
                continue

            # Prepare attention input
            query = query_embeddings.unsqueeze(0)  # (1, batch_size, embed_dim)
            key = class_embeddings.unsqueeze(1)  # (class_samples, 1, embed_dim)
            key = key.expand(-1, bsz, -1)  # (class_samples, batch_size, embed_dim)

            # Use attention for current class
            _, attn_weights = self.class_attention(
                query,
                key,
                key,
                class_idx
            )  # attn_weights: (batch_size, 1, class_samples)
            attn_weights = attn_weights.squeeze(1)  # (batch_size, class_samples)

            # Select top-k samples and their weights
            k = min(n_samples, len(class_embeddings))
            if k > 0:
                weights, top_k_indices = torch.topk(attn_weights, k=k, dim=1)
                for batch_idx in range(bsz):
                    result_indices[batch_idx, class_idx, :k] = class_sample_indices[top_k_indices[batch_idx]]
                    result_weights[batch_idx, class_idx, :k] = weights[batch_idx]
                if k < n_samples:
                    result_indices[:, class_idx, k:] = -1
                    result_weights[:, class_idx, k:] = 0
            else:
                result_indices[:, class_idx] = -1
                result_weights[:, class_idx] = 0

        return result_indices, result_weights

    def get_similar_samples_for_knowledge(
            self,
            query_embeddings: torch.Tensor,  # (batch_size*num_classes*k_img, embed_dim)
            all_knowledge_embeddings: torch.Tensor,  # (num_knowledge, embed_dim)
            knowledge_mask: torch.Tensor,  # (batch_size*num_classes*k_img, num_knowledge)
            n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (indices, attention_weights)
        """Select most similar knowledge samples for each query image in its corresponding class, and return attention weights"""
        total_queries = query_embeddings.size(0)

        # Step 1: Create num_classes index sequence, repeat class indices by k_img
        class_indices = torch.arange(self.num_classes, device=self.config.device).repeat_interleave(self.config.k_img)
        # Step 2: Repeat by batch_size to get batch_size*num_classes*k_img long sequence
        class_indices = class_indices.repeat(total_queries //(self.num_classes * self.config.k_img))

        # Initialize result tensors
        result_indices = torch.zeros(
            (total_queries, n_samples),
            dtype=torch.long,
            device=self.config.device
        )
        result_weights = torch.zeros(
            (total_queries, n_samples),
            device=self.config.device
        )

        # Process queries by class in batches
        for class_idx in range(self.num_classes):
            # Get queries for current class
            class_mask = (class_indices == class_idx)
            if not class_mask.any():
                continue

            class_queries = query_embeddings[class_mask]
            class_knowledge_mask = knowledge_mask[class_mask]

            # Get available knowledge for current class
            valid_knowledge_mask = class_knowledge_mask.any(dim=0)  # (num_knowledge,)
            if not valid_knowledge_mask.any():
                result_indices[class_mask] = -1
                result_weights[class_mask] = 0
                continue

            valid_knowledge = all_knowledge_embeddings[valid_knowledge_mask]

            # Prepare attention input
            query = class_queries.unsqueeze(0)  # (1, class_queries, embed_dim)
            key = valid_knowledge.unsqueeze(1)  # (valid_knowledge, 1, embed_dim)
            key = key.expand(-1, len(class_queries), -1)

            # Use class-specific attention
            _, attn_weights = self.class_attention(
                query,
                key,
                key,
                class_idx
            )
            attn_weights = attn_weights.squeeze(1)  # (class_queries, valid_knowledge)

            # Select top-k knowledge and their weights
            k = min(n_samples, valid_knowledge.size(0))
            if k > 0:
                weights, top_indices = torch.topk(attn_weights, k=k, dim=1)  # (class_queries, k)
                # Convert local indices to global indices
                valid_indices = torch.where(valid_knowledge_mask)[0]
                global_indices = valid_indices[top_indices]
                
                # Save indices and weights
                result_indices[class_mask, :k] = global_indices
                result_weights[class_mask, :k] = weights
                
                if k < n_samples:
                    result_indices[class_mask, k:] = -1
                    result_weights[class_mask, k:] = 0
            else:
                result_indices[class_mask] = -1
                result_weights[class_mask] = 0

        return result_indices, result_weights
    
class EmbeddingFusion_weighted(nn.Module):
    """Component for handling embedding fusion"""

    def __init__(self, config: ModelConfig, fusion_model: nn.Module):
        super().__init__()
        self.config = config
        self.fusion_block = fusion_model(
            number_hidden_layers=config.num_hidden_layers,
            embed_dim=config.embed_dim,
            use_projection=config.use_projection
        ).to(config.device)


    def fuse_embeddings_batch(
            self,
            query_embeddings: torch.Tensor,  # (batch_size, embed_dim)
            context_embeddings: torch.Tensor,  # (batch_size, num_classes, k, embed_dim) for images
            # or (batch_size, num_classes, k, m, embed_dim) for knowledge
            attention_weights: torch.Tensor,  # (batch_size, num_classes, k) for images
            # or (batch_size, num_classes, k, m) for knowledge
            image_weights: Optional[torch.Tensor] = None,  # (batch_size, num_classes, k) weights from image attention
            is_knowledge: bool = False
    ) -> torch.Tensor:  # (batch_size, num_classes, embed_dim)
        """Batch fusion of embeddings"""
        batch_size = query_embeddings.size(0)

        projected_dim = 512 if self.config.use_projection else self.config.embed_dim
        
        if is_knowledge:
            # Knowledge embeddings: (batch_size, num_classes, k, m, embed_dim)
            num_classes, k, m, embed_dim = context_embeddings.shape[1:]
            
            # Step 1: Normalize knowledge weights for each class
            # (batch_size, num_classes, k, m) -> (batch_size * num_classes * k, m)
            reshaped_weights = attention_weights.view(-1, m)
            # Apply softmax to each group of knowledge weights: (batch_size * num_classes * k, m)
            normalized_k_weights = F.softmax(reshaped_weights, dim=1)
            # Reshape back to original dimensions: (batch_size, num_classes, k, m)
            normalized_k_weights = normalized_k_weights.view(batch_size, num_classes, k, m)
            
            # Step 2: Use passed image weights
            # Expand image weights dimensions: (batch_size, num_classes, k, 1)
            image_weights_expanded = image_weights.unsqueeze(-1)
            
            # Step 3: Calculate two-level weights
            # Combine two-level weights: (batch_size, num_classes, k, m)
            combined_weights = image_weights_expanded * normalized_k_weights
            
            # Step 4: Re-normalize combined weights
            # (batch_size, num_classes, k, m) -> (batch_size, num_classes, k * m)
            combined_weights = combined_weights.view(batch_size, num_classes, -1)
            # Normalize weights for each class: (batch_size, num_classes, k * m)
            combined_weights = F.normalize(combined_weights, p=1, dim=2)
            
            # (batch_size, num_classes, k * m) -> (batch_size * num_classes, k * m)
            reshaped_weights = combined_weights.view(-1, k * m)
            # Reshape data for fusion
            # (batch_size, num_classes, k, m, embed_dim) -> (batch_size * num_classes, k * m, embed_dim)
            reshaped = context_embeddings.view(-1, k * m, embed_dim)

            # Apply fusion model: (batch_size * num_classes, k * m, embed_dim)
            fused = self.fusion_block(reshaped)
            
            # Use combined weights for weighted average
            # (batch_size * num_classes, k * m, 1)
            weights_expanded = reshaped_weights.unsqueeze(-1)
            # (batch_size * num_classes, embed_dim)
            pooled = (fused * weights_expanded).sum(dim=1)
        else:
            # Image embeddings: (batch_size, num_classes, k, embed_dim)
            num_classes, k, embed_dim = context_embeddings.shape[1:]
            
            # Step 1: Normalize retrieved image weights
            # (batch_size, num_classes, k) -> (batch_size, num_classes, k)
            # normalized_weights = F.normalize(attention_weights, p=1, dim=2)
            normalized_weights = attention_weights
            
            # Step 2: Add query_embedding
            # (batch_size, embed_dim) -> (batch_size, num_classes, 1, embed_dim)
            query_expanded = query_embeddings.unsqueeze(1).unsqueeze(1)
            query_expanded = query_expanded.expand(-1, num_classes, 1, -1)
            # Concatenate: (batch_size, num_classes, k+1, embed_dim)
            combined = torch.cat([query_expanded, context_embeddings], dim=2)
            
            # Step 3: Combine weights (query weight is 1, others are normalized weights)
            # (batch_size, num_classes, 1)
            query_weight = torch.ones(batch_size, num_classes, 1, device=attention_weights.device)
            # (batch_size, num_classes, k+1)
            combined_weights = torch.cat([query_weight, normalized_weights], dim=2)
            
            # Step 4: Re-normalize combined weights
            # (batch_size, num_classes, k+1)
            combined_weights = F.normalize(combined_weights, p=1, dim=2)
            
            # Reshape data for fusion
            # (batch_size, num_classes, k+1, embed_dim) -> (batch_size * num_classes, k+1, embed_dim)
            reshaped = combined.view(-1, k + 1, embed_dim)
            # (batch_size, num_classes, k+1) -> (batch_size * num_classes, k+1)
            reshaped_weights = combined_weights.view(-1, k + 1)
            
            # Apply fusion model: (batch_size * num_classes, k+1, embed_dim)
            fused = self.fusion_block(reshaped)
            
            # Calculate final weighted average
            # (batch_size * num_classes, k+1, 1)
            weights_expanded = reshaped_weights.unsqueeze(-1)
            # (batch_size * num_classes, embed_dim)
            pooled = (fused * weights_expanded).sum(dim=1)

        # Reshape to final output: (batch_size, num_classes, embed_dim/projected_dim)
        enhanced = pooled.view(batch_size, num_classes, projected_dim)

        return enhanced
    
class EmbeddingFusion_average(nn.Module):
    """Component for handling embedding fusion"""

    def __init__(self, config: ModelConfig, fusion_model: nn.Module):
        super().__init__()
        self.config = config
        self.fusion_block = fusion_model(
            number_hidden_layers=config.num_hidden_layers,
            embed_dim=config.embed_dim,
            use_projection=config.use_projection
        ).to(config.device)


    def fuse_embeddings_batch(
            self,
            query_embeddings: torch.Tensor,  # (batch_size, embed_dim)
            context_embeddings: torch.Tensor,  # (batch_size, num_classes, k, embed_dim) for images
            # or (batch_size, num_classes, k, m, embed_dim) for knowledge
            attention_weights: torch.Tensor,  # (batch_size, num_classes, k) for images
            # or (batch_size, num_classes, k, m) for knowledge
            image_weights: Optional[torch.Tensor] = None,  # (batch_size, num_classes, k) 
            is_knowledge: bool = False
    ) -> torch.Tensor:  # (batch_size, num_classes, embed_dim)
        """Batch fusion of embeddings"""
        batch_size = query_embeddings.size(0)

        projected_dim = 512 if self.config.use_projection else self.config.embed_dim
        
        if is_knowledge:
            # (batch_size, num_classes, k, m, embed_dim)
            num_classes, k, m, embed_dim = context_embeddings.shape[1:]
            # (batch_size * num_classes, k*m, embed_dim)
            reshaped = context_embeddings.view(-1, k * m, embed_dim)
        else:
            # (batch_size, num_classes, k, embed_dim)
            num_classes, k, embed_dim = context_embeddings.shape[1:]
            # (batch_size, num_classes, 1, embed_dim)
            query_expanded = query_embeddings.unsqueeze(1).unsqueeze(1)
            query_expanded = query_expanded.expand(-1, num_classes, 1, -1)
            # (batch_size, num_classes, k+1, embed_dim)
            combined = torch.cat([query_expanded, context_embeddings], dim=2)
            # (batch_size * num_classes, k+1, embed_dim)
            reshaped = combined.view(-1, k + 1, embed_dim)

        # (batch_size * num_classes, k/k+1, embed_dim)
        fused = self.fusion_block(reshaped)

        # (batch_size * num_classes, embed_dim)
        pooled = torch.mean(fused, dim=1)

        # (batch_size, num_classes, embed_dim/projected_dim)
        enhanced = pooled.view(batch_size, num_classes, projected_dim)

        return enhanced


class LossModule(nn.Module):
    """Component for handling loss computation"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.logit_scale_1 = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.logit_scale_2 = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.logit_scale_3 = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        # Add global contrastive learning balance parameters
        self.global_contrast_weight = nn.Parameter(torch.tensor(0.5))
        self.local_contrast_weight = nn.Parameter(torch.tensor(0.5))
        self.global_contrast_weight_1 = nn.Parameter(torch.tensor(0.5))
        self.global_contrast_weight_2 = nn.Parameter(torch.tensor(0.5))


    
    def compute_basic_loss(
            self,
            image_embeddings: torch.Tensor,  # (batch_size, num_classes, embed_dim)
            knowledge_embeddings: torch.Tensor,  # (batch_size, num_classes, embed_dim)
            target_indices: torch.Tensor  # (batch_size,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss function - flattened similarity matrix loss + global contrastive learning using only ground truth classes"""


        image_norm = F.normalize(image_embeddings, p=2, dim=2)  # (batch_size, num_classes, embed_dim)
        knowledge_norm = F.normalize(knowledge_embeddings, p=2, dim=2)  # (batch_size, num_classes, embed_dim)


        logit_scale_1 = self.logit_scale_1.exp()
        similarity_matrices = torch.bmm(
            image_norm,  # (batch_size, num_classes, embed_dim)
            knowledge_norm.transpose(1, 2)  # (batch_size, embed_dim, num_classes)
        ) * logit_scale_1  # (batch_size, num_classes, num_classes)

        num_classes = image_embeddings.size(1)


        flattened_similarities = similarity_matrices.view(
            -1, num_classes * num_classes
        )  # (batch_size, num_classes * num_classes)
        target_indices_flat = target_indices * num_classes + target_indices  # (batch_size,)

        original_loss = self.cross_entropy(flattened_similarities, target_indices_flat)

        batch_size = image_embeddings.size(0)
        
        # Extract ground truth class representations
        # (batch_size, embed_dim)
        gt_images = image_norm[torch.arange(batch_size), target_indices]
        gt_knowledge = knowledge_norm[torch.arange(batch_size), target_indices]
        

        # Reshape all embeddings for contrastive learning
        # (batch_size, num_classes, embed_dim) -> (batch_size * num_classes, embed_dim)
        all_knowledge = knowledge_norm.reshape(-1, knowledge_norm.size(-1))
        all_images = image_norm.reshape(-1, image_norm.size(-1))
        
        # Create label matrix (positive sample positions)
        labels_matrix = torch.zeros(
            batch_size, 
            batch_size * num_classes, 
            device=image_embeddings.device
        )
        
        # Create mask matrix to mark samples to exclude
        # i.e., embeddings from other samples (not self) that have the same class as current sample
        exclude_mask = torch.zeros(
            batch_size,
            batch_size * num_classes,
            device=image_embeddings.device,
            dtype=torch.bool
        )
        
        # Set positive sample positions and positions to exclude
        for i in range(batch_size):
            current_class = target_indices[i]
            
            # Set positive sample position - sample i's own corresponding class embedding
            positive_idx = i * num_classes + current_class
            labels_matrix[i, positive_idx] = 1
            
            # Set positions to exclude - embeddings from other samples with same class as current sample
            for j in range(batch_size):
                if j != i:  # Not self
                    exclude_idx = j * num_classes + current_class
                    exclude_mask[i, exclude_idx] = True
        
        # Compute global contrastive learning loss using logit_scale_2
        logit_scale_2 = self.logit_scale_2.exp()
        # logit_scale_3 = self.logit_scale_3.exp()
        # 1. Image-to-knowledge contrastive learning
        # (batch_size, batch_size*num_classes)
        img_to_know_sim = torch.mm(gt_images, all_knowledge.t()) * logit_scale_2
        
        # Apply mask - set positions to exclude to very small values
        img_to_know_sim = img_to_know_sim.masked_fill(exclude_mask, float('-inf'))
        
        # Compute loss - only consider non-excluded positions
        img_to_know_loss = self.cross_entropy(img_to_know_sim, torch.argmax(labels_matrix, dim=1))
        
        # 2. Knowledge-to-image contrastive learning
        # (batch_size, batch_size*num_classes)
        know_to_img_sim = torch.mm(gt_knowledge, all_images.t()) * logit_scale_2
        
        # Apply mask - set positions to exclude to very small values
        know_to_img_sim = know_to_img_sim.masked_fill(exclude_mask, float('-inf'))

        # Compute loss - only consider non-excluded positions
        know_to_img_loss = self.cross_entropy(know_to_img_sim, torch.argmax(labels_matrix, dim=1))
        
        # Combine contrastive losses
        # global_contrastive_loss = (img_to_know_loss + know_to_img_loss) / 2.0
        
        # Total loss = local_weight*original_loss + global_weight*global_contrastive_loss
        total_loss = original_loss + self.global_contrast_weight_1 * img_to_know_loss + self.global_contrast_weight_2 * know_to_img_loss
        
        # Compute accuracy (still based on original task)
        predictions = torch.argmax(flattened_similarities, dim=1)  # (batch_size,)
        accuracy = (predictions == target_indices_flat).float().mean()

        return total_loss, accuracy
    
    
    def compute_loss(self, image_embeddings, knowledge_embeddings, 
                target_indices, hierarchy_targets=None, hierarchy_3level_targets=None):
        """Select and compute loss based on configuration"""
        if self.config.loss_type == 'basic':
            return self.compute_basic_loss(image_embeddings, knowledge_embeddings, 
                                        target_indices)   
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

