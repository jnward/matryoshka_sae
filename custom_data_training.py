import copy
import multiprocessing as mp
import random
from typing import List, Union

import numpy as np
import torch

from config import post_init_cfg
from custom_data import CustomDataLoader
from custom_data_config import get_custom_data_cfg
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from training import train_sae_group_seperate_wandb


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_custom_data(
    data: torch.Tensor,
    act_size: int,
    dict_sizes: List[int],
    topks: List[int],
    group_sizes: List[int],
    seed: int = 42,
    device: str = "cuda",
    wandb_project: str = "custom-data-matryoshka",
    num_tokens: int = int(1e6),
    batch_size: int = 4096,
    lr: float = 3e-4,
    checkpoint_freq: int = 10000,
    n_signals: int = 10,
    signal_strength: float = 1.0,
    noise_level: float = 0.1,
) -> None:
    """
    Train SAEs on custom data.

    Args:
        data: Input data tensor of shape (num_samples, act_size)
        act_size: Size of the input activations
        dict_sizes: List of dictionary sizes for the BatchTopK SAEs
        topks: List of top-k values for the BatchTopK SAEs
        group_sizes: List of group sizes for the Matryoshka SAE
        seed: Random seed
        device: Device to run on
        wandb_project: Weights & Biases project name
        num_tokens: Number of tokens to train on
        batch_size: Batch size
        lr: Learning rate
        checkpoint_freq: Frequency of checkpoints
        n_signals: Number of signal components in synthetic data
        signal_strength: Strength of signal components
        noise_level: Level of noise in synthetic data

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If CUDA is requested but not available
    """
    # Validate input data
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor")
    if data.dim() != 2:
        raise ValueError("data must be 2-dimensional (num_samples, act_size)")
    if data.shape[1] != act_size:
        raise ValueError(f"data shape {data.shape} does not match act_size {act_size}")
    if data.shape[0] == 0:
        raise ValueError("data must contain at least one sample")

    # Validate lists
    if not dict_sizes or not topks or not group_sizes:
        raise ValueError("dict_sizes, topks, and group_sizes must not be empty")
    if len(dict_sizes) != len(topks):
        raise ValueError("dict_sizes and topks must have the same length")
    if any(size <= 0 for size in dict_sizes):
        raise ValueError("all dict_sizes must be positive")
    if any(k <= 0 for k in topks):
        raise ValueError("all topks must be positive")
    if any(size <= 0 for size in group_sizes):
        raise ValueError("all group_sizes must be positive")

    # Validate other parameters
    if act_size <= 0:
        raise ValueError("act_size must be positive")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq must be positive")
    if n_signals <= 0:
        raise ValueError("n_signals must be positive")
    if signal_strength < 0:
        raise ValueError("signal_strength must be non-negative")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    # Set up configuration
    cfg = get_custom_data_cfg()
    cfg["act_size"] = act_size
    cfg["dict_sizes"] = dict_sizes
    cfg["topks"] = topks
    cfg["group_sizes"] = group_sizes
    cfg["device"] = device
    cfg["wandb_project"] = wandb_project
    cfg["num_tokens"] = num_tokens
    cfg["batch_size"] = batch_size
    cfg["lr"] = lr
    cfg["checkpoint_freq"] = checkpoint_freq
    cfg["seed"] = seed
    cfg["n_signals"] = n_signals
    cfg["signal_strength"] = signal_strength
    cfg["noise_level"] = noise_level

    # Set the seed for reproducibility
    set_seed(cfg["seed"])
    print(f"Using seed: {cfg['seed']}")

    # Create data loader
    data_loader = CustomDataLoader(data, cfg)

    # Initialize SAEs
    saes: List[Union[BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE]] = []
    cfgs = []

    # Train the BatchTopK SAEs
    for i, (dict_size, topk) in enumerate(zip(dict_sizes, topks)):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["sae_type"] = "batch-topk"
        cfg_copy["dict_size"] = dict_size
        cfg_copy["top_k"] = topk

        cfg_copy = post_init_cfg(cfg_copy)
        sae = BatchTopKSAE(cfg_copy)
        saes.append(sae)
        cfgs.append(cfg_copy)

    # Train the Matryoshka SAE
    dict_size = dict_sizes[-1]
    topk = topks[-1]
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["sae_type"] = "global-matryoshka-topk"
    cfg_copy["dict_size"] = dict_size
    cfg_copy["top_k"] = topk
    cfg_copy["group_sizes"] = group_sizes

    cfg_copy = post_init_cfg(cfg_copy)
    global_sae = GlobalBatchTopKMatryoshkaSAE(cfg_copy)
    saes.append(global_sae)
    cfgs.append(cfg_copy)

    # Train the SAEs
    train_sae_group_seperate_wandb(saes, data_loader, None, cfgs)


if __name__ == "__main__":
    # Example usage
    mp.freeze_support()  # Needed for multiprocessing to work with frozen executables

    # Create some random data for demonstration
    act_size = 100
    num_samples = 10000
    data = torch.randn(num_samples, act_size)

    # Example configuration
    dict_sizes = [200, 400, 800, 1600, 3200]
    topks = [20, 25, 30, 35, 40]
    group_sizes = [200, 200, 400, 800, 1600]

    train_custom_data(
        data=data,
        act_size=act_size,
        dict_sizes=dict_sizes,
        topks=topks,
        group_sizes=group_sizes,
        seed=42,
        device="cuda",
        wandb_project="custom-data-matryoshka",
        num_tokens=int(1e6),
        batch_size=4096,
        lr=3e-4,
        checkpoint_freq=10000,
        n_signals=10,
        signal_strength=1.0,
        noise_level=0.1,
    )
