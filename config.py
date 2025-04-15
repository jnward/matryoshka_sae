import torch


def get_default_cfg():
    default_cfg = {
        "seed": 95,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e6),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "act_size": 4096,
        "dict_size": 12288,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "input_unit_norm": True,
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 20,
        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1 / 32),
        # Synthetic data parameters
        "signal_to_noise_ratio": 10.0,  # Signal-to-noise ratio: higher means cleaner signals
        "non_euclidean": 0.0,  # 0: Euclidean; 1: fully warped
        "superposition_multiplier": 1.0,  # Controls #signals: n_signals = activation_size * superposition_multiplier
        "non_orthogonal": 0.0,  # 0: fully orthogonal signals; 1: as generated (non-orthogonal)
        "hierarchical": 0.0,  # 0: independent signals; 1: signals grouped in clusters
    }

    return default_cfg


def post_init_cfg(cfg):
    cfg["name"] = f"{cfg['sae_type']}_{cfg['dict_size']}_{cfg['top_k']}_{cfg['lr']}_{cfg['seed']}"
    return cfg
