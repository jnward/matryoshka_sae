import torch
import transformer_lens.utils as utils  # type: ignore


def get_default_cfg():
    default_cfg = {
        "seed": 42,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e8),
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
        "n_signals": 10,
        "signal_strength": 1.0,
        "noise_level": 0.1,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg


def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = (
        f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_"
        f"{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}_{cfg['seed']}"
    )
    return cfg
