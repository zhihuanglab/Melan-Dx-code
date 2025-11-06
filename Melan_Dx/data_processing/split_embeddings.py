#!/usr/bin/env python3
"""Split merged embeddings into train, val, test files with stratified sampling"""

import torch
import os
import sys
import numpy as np
from collections import defaultdict

def split_embeddings(merged_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split merged embeddings into train, val, test files with stratified sampling by disease label
    
    Args:
        merged_file: Path to merged embeddings file
        output_dir: Directory to save split embedding files
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.15)
        test_ratio: Ratio for test set (default: 0.15)
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("Splitting Embeddings (Stratified by Label)")
    print("=" * 80)
    
    # Check ratios sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        print(f"ERROR: Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        sys.exit(1)
    
    # Check if merged file exists
    if not os.path.exists(merged_file):
        print(f"ERROR: Merged file not found: {merged_file}")
        sys.exit(1)
    
    print(f"\nLoading merged embeddings from: {merged_file}")
    print(f"Split ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")
    print(f"Random seed: {seed}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Load merged data
    merged_data = torch.load(merged_file)
    all_embeddings = merged_data['embeddings']
    all_disease_names = merged_data['disease_names']
    
    print(f"\nTotal samples: {len(all_disease_names)}")
    print(f"Embedding shape: {all_embeddings.shape}")
    
    # Group samples by disease label
    disease_groups = defaultdict(list)
    for idx, disease in enumerate(all_disease_names):
        disease_groups[disease].append(idx)
    
    print(f"\nFound {len(disease_groups)} unique diseases")
    print(f"Samples per disease:")
    for disease, indices in sorted(disease_groups.items()):
        print(f"  {disease}: {len(indices)} samples")
    
    # Stratified split
    train_indices = []
    val_indices = []
    test_indices = []
    
    print(f"\nPerforming stratified split...")
    for disease, indices in disease_groups.items():
        # Shuffle indices for this disease
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        n_samples = len(indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        # Remaining goes to test to ensure we use all samples
        n_test = n_samples - n_train - n_val
        
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train+n_val].tolist())
        test_indices.extend(indices[n_train+n_val:].tolist())
        
        print(f"  {disease}: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Convert to sorted arrays
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)
    
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/len(all_disease_names)*100:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples ({len(val_indices)/len(all_disease_names)*100:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_indices)/len(all_disease_names)*100:.1f}%)")
    
    # Split embeddings and disease names
    train_embeddings = all_embeddings[train_indices]
    val_embeddings = all_embeddings[val_indices]
    test_embeddings = all_embeddings[test_indices]
    
    train_disease_names = [all_disease_names[i] for i in train_indices]
    val_disease_names = [all_disease_names[i] for i in val_indices]
    test_disease_names = [all_disease_names[i] for i in test_indices]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare train data (with hierarchy info if available)
    train_data = {
        'embeddings': train_embeddings,
        'disease_names': train_disease_names
    }
    if 'disease_to_parent' in merged_data:
        train_data['disease_to_parent'] = merged_data['disease_to_parent']
    if 'parent_to_grandparent' in merged_data:
        train_data['parent_to_grandparent'] = merged_data['parent_to_grandparent']
    
    # Prepare val and test data
    val_data = {
        'embeddings': val_embeddings,
        'disease_names': val_disease_names
    }
    
    test_data = {
        'embeddings': test_embeddings,
        'disease_names': test_disease_names
    }
    
    # Save split files
    train_file = os.path.join(output_dir, "train_embeddings.pt")
    val_file = os.path.join(output_dir, "val_embeddings.pt")
    test_file = os.path.join(output_dir, "test_embeddings.pt")
    
    torch.save(train_data, train_file)
    torch.save(val_data, val_file)
    torch.save(test_data, test_file)
    
    print(f"\nSplit embeddings saved to: {output_dir}")
    print(f"  Train: {train_file} (shape: {train_embeddings.shape})")
    print(f"  Val:   {val_file} (shape: {val_embeddings.shape})")
    print(f"  Test:  {test_file} (shape: {test_embeddings.shape})")
    
    print("\n" + "=" * 80)
    print("Split completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split merged embeddings with stratified sampling')
    parser.add_argument('merged_file', type=str, help='Path to merged embeddings file')
    parser.add_argument('output_dir', type=str, help='Directory to save split embedding files')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_embeddings(
        args.merged_file, 
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
