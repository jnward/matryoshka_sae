import torch
import transformer_lens.utils as utils


def get_custom_data_cfg():
    """Get default configuration for custom data training."""
    cfg = {
        # Core training parameters
        "model_name": "gemma-2-2B",
        "dtype": torch.float32,
        "model_dtype": torch.float32,
        "site": "resid_post",
        "layer": 12,
        "num_tokens": int(1e6),
        "batch_size": 4096,
        "lr": 3e-4,
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "act_size": 4096,
        "dict_size": 12288,
        "model_batch_size": 512,
        "num_batches_in_buffer": 10,
        "checkpoint_freq": 10000,
        "perf_log_freq": 1000,
        "n_batches_to_dead": 20,
        # Device and logging
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_project": "custom-data-matryoshka",
        "seed": 42,
        # Data parameters
        "input_unit_norm": False,
        "sae_type": "topk",
        # TopKSAE specific parameters
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1 / 32),
        # JumpReLU specific
        "bandwidth": 0.001,
        # Synthetic data specific parameters
        "n_samples": 10000,
        "n_signals": 10,
        "signal_strength": 1.0,
        "noise_level": 0.1,
    }

    # Post-initialization
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = (
        f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_"
        f"{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}_{cfg['seed']}"
    )

    return cfg
