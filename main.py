import copy
import multiprocessing as mp
import os
import random
import sys

import numpy as np
import torch
from transformer_lens import HookedTransformer  # type: ignore

from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from training import train_sae_group_seperate_wandb

os.environ["HF_TOKEN"] = "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"

if len(sys.argv) != 2:
    seed = 42
else:
    seed = int(sys.argv[1])


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cfg = get_default_cfg()
    cfg["model_name"] = "gemma-2-2b"
    cfg["layer"] = 14
    cfg["site"] = "resid_post"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["aux_penalty"] = 1 / 32
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = False
    # cfg["dict_size"] = 12288
    cfg["wandb_project"] = "batch-topk-matryoshka-seeds"
    cfg["l1_coeff"] = 0.0
    cfg["act_size"] = 768
    cfg["device"] = "cuda"
    cfg["bandwidth"] = 0.001
    cfg["top_k_matryoshka"] = [10, 10, 10, 10, 10]
    # cfg["group_sizes"] = [768//4, 768 // 4 ,768 // 2, 768, 768*2, 768*4, 768*8]
    cfg["group_sizes"] = [1536 // 4, 1536 // 4, 1536 // 2, 1536, 1536 * 2, 1536 * 4, 1536 * 8]
    cfg["num_tokens"] = 2e8
    cfg["model_batch_size"] = 32
    cfg["model_dtype"] = torch.bfloat16
    cfg["num_batches_in_buffer"] = 10
    cfg["seq_len"] = 128
    cfg["seed"] = seed
    cfg["checkpoint_freq"] = 10000

    # Apply post-initialization to set the hook_point and name
    cfg = post_init_cfg(cfg)

    # Set the seed for reproducibility
    set_seed(cfg["seed"])
    print(f"Using seed: {cfg['seed']}")

    # Train the BatchTopK SAEs
    dict_sizes = [1536, 1536 * 2, 1536 * 4, 1536 * 8, 1536 * 16]
    topks = [22, 25, 27, 29, 32]

    model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"]).to(cfg["model_dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    saes = []
    cfgs = []

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
    cfg_copy["group_sizes"] = [dict_size // 16, dict_size // 16, dict_size // 8, dict_size // 4, dict_size // 2]

    cfg_copy = post_init_cfg(cfg_copy)
    sae = GlobalBatchTopKMatryoshkaSAE(cfg_copy)
    saes.append(sae)
    cfgs.append(cfg_copy)

    # train_sae_group_seperate_wandb([saes[-2]], activations_store, model, [cfgs[-2]])
    train_sae_group_seperate_wandb(saes, activations_store, model, cfgs)


if __name__ == "__main__":
    mp.freeze_support()  # Needed for multiprocessing to work with frozen executables
    main()
