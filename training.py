import copy
import multiprocessing as mp
import random
from typing import List, Union

import numpy as np
import torch

from config import get_default_cfg, post_init_cfg
from custom_data import load_synthetic_data
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from training_functions import train_sae_group


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    cfg = get_default_cfg()
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = False
    cfg["l1_coeff"] = 0.0
    cfg["act_size"] = 768
    cfg["group_sizes"] = [1536 // 4, 1536 // 4, 1536 // 2, 1536, 1536 * 2, 1536 * 4, 1536 * 8]
    cfg["num_tokens"] = int(2e8)  # Ensure num_tokens is an integer
    cfg["model_dtype"] = torch.bfloat16
    cfg["checkpoint_freq"] = 10000

    # Update synthetic data parameters
    cfg["n_signals"] = 10
    cfg["signal_strength"] = 1.0
    cfg["noise_level"] = 0.1
    # New synthetic data parameters
    cfg["non_euclidean"] = 0.0  # Keep data in Euclidean space
    cfg["superposition"] = 0.5  # Moderate superposition of signals
    cfg["non_orthogonal"] = 0.3  # Somewhat orthogonal signals
    cfg["hierarchical"] = 0.2  # Slight hierarchical structure

    # Set the seed for reproducibility
    set_seed(cfg["seed"])
    print(f"Using seed: {cfg['seed']}")

    # Generate synthetic data using load_synthetic_data
    print("Generating synthetic data...")
    data_loader, activation_size = load_synthetic_data(cfg)
    print(f"Generated synthetic data with {activation_size} activation size")

    # Initialize SAEs
    saes: List[Union[BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE]] = []
    cfgs = []

    # Train the BatchTopK SAEs
    dict_sizes = [1536, 1536 * 2, 1536 * 4, 1536 * 8, 1536 * 16]
    topks = [22, 25, 27, 29, 32]

    # Initialize the BatchTopK SAEs
    for dict_size, topk in zip(dict_sizes, topks):
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["sae_type"] = "batch-topk"
        cfg_copy["dict_size"] = dict_size
        cfg_copy["top_k"] = topk

        cfg_copy = post_init_cfg(cfg_copy)
        sae = BatchTopKSAE(cfg_copy)
        saes.append(sae)
        cfgs.append(cfg_copy)

    # Initialize the Matryoshka SAE
    dict_size = dict_sizes[-1]
    topk = topks[-1]
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["sae_type"] = "global-matryoshka-topk"
    cfg_copy["dict_size"] = dict_size
    cfg_copy["top_k"] = topk
    cfg_copy["group_sizes"] = [dict_size // 16, dict_size // 16, dict_size // 8, dict_size // 4, dict_size // 2]

    cfg_copy = post_init_cfg(cfg_copy)
    sae_matryoshka = GlobalBatchTopKMatryoshkaSAE(cfg_copy)
    saes.append(sae_matryoshka)
    cfgs.append(cfg_copy)

    # Train all SAEs together
    print(f"Training {len(saes)} SAEs...")
    train_sae_group(saes, data_loader, cfgs)


if __name__ == "__main__":
    mp.freeze_support()  # Needed for multiprocessing to work with frozen executables
    main()
