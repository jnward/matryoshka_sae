#!/usr/bin/env python3
"""
Script to generate cosine similarity data between same-sized SAEs with different suffixes.

This script:
1. Processes SAE models of sizes from 768 to 49152
2. For each size, compares models with base_suffix vs test_suffix
3. Computes cosine similarities between the paired models
4. Saves the similarity data for visualization
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from scipy.optimize import linear_sum_assignment

BATCH_SIZE = 4096
def get_normalized_weights(sae, use_decoder=True):
    """
    Get normalized decoder weights from SAE
    
    Args:
        sae: SparseAutoencoder model
        
    Returns:
        Normalized decoder weights
    """
    if use_decoder:
        weights = sae.W_dec.data
    else:
        weights = sae.W_enc.data.T
    return weights / weights.norm(dim=1, keepdim=True)

def run_hungarian_alignment(base_weights, other_weights, batch_size=BATCH_SIZE):
    """
    Run Hungarian alignment between two sets of weights
    
    Args:
        base_weights: First set of weights
        other_weights: Second set of weights
        batch_size: Batch size for processing
        
    Returns:
        tuple: (cost_matrix, row_indices, col_indices, average_score)
    """
    n_batches = (base_weights.shape[0] + batch_size - 1) // batch_size
    cost_matrix = torch.zeros(base_weights.shape[0], other_weights.shape[0], device="cpu")
    
    # Calculate cost matrix in batches
    for i in tqdm(range(n_batches), desc="Computing alignment"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, base_weights.shape[0])
        value = base_weights[start_idx:end_idx] @ other_weights.T
        cost_matrix[start_idx:end_idx] = value.cpu()
    
    # Handle NaN values and run linear_sum_assignment
    cost_matrix = torch.nan_to_num(cost_matrix, nan=0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy(), maximize=True)
    avg_score = cost_matrix[row_ind, col_ind].mean().item()

    similarities = cost_matrix[row_ind, col_ind].numpy()

    
    return cost_matrix, row_ind, col_ind, avg_score, similarities
