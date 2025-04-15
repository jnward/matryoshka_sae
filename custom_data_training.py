import copy
import multiprocessing as mp
import random
from typing import Any

import numpy as np
import torch

from config import post_init_cfg
from custom_data import CustomDataLoader
from custom_data_config import get_custom_data_cfg
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from training import train_sae_group_seperate_wandb


def set_seed(seed):
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
    dict_sizes: list,
    topks: list,
    group_sizes: list,
    seed: int = 42,
    device: str = "cuda",
    wandb_project: str = "custom-data-matryoshka",
    num_tokens: int = int(1e6),
    batch_size: int = 4096,
    lr: float = 3e-4,
    checkpoint_freq: int = 10000,
):
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
    """
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

    # Set the seed for reproducibility
    set_seed(cfg["seed"])
    print(f"Using seed: {cfg['seed']}")

    # Create data loader
    data_loader = CustomDataLoader(data, cfg)

    # Initialize SAEs
    saes: list[BatchTopKSAE | GlobalBatchTopKMatryoshkaSAE] = []
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
    global_sae: Any = GlobalBatchTopKMatryoshkaSAE(cfg_copy)
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
    )
