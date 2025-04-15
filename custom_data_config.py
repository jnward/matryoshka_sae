from typing import Any, Dict

import torch
import transformer_lens.utils as utils  # type: ignore


def get_custom_data_cfg() -> Dict[str, Any]:
    """
    Get default configuration for custom data training.

    Returns:
        Dictionary containing the default configuration

    Raises:
        ValueError: If any required parameters are missing or invalid
    """
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

    # Validate required parameters
    required_params = [
        "model_name",
        "dtype",
        "model_dtype",
        "site",
        "layer",
        "num_tokens",
        "batch_size",
        "lr",
        "l1_coeff",
        "beta1",
        "beta2",
        "max_grad_norm",
        "seq_len",
        "act_size",
        "dict_size",
        "model_batch_size",
        "num_batches_in_buffer",
        "checkpoint_freq",
        "perf_log_freq",
        "n_batches_to_dead",
        "device",
        "wandb_project",
        "seed",
        "input_unit_norm",
        "sae_type",
        "top_k",
        "top_k_aux",
        "aux_penalty",
        "bandwidth",
        "n_samples",
        "n_signals",
        "signal_strength",
        "noise_level",
    ]

    for param in required_params:
        if param not in cfg:
            raise ValueError(f"Missing required parameter: {param}")

    # Validate parameter types and values
    if not isinstance(cfg["model_name"], str):
        raise ValueError("model_name must be a string")
    if not isinstance(cfg["dtype"], torch.dtype):
        raise ValueError("dtype must be a torch.dtype")
    if not isinstance(cfg["model_dtype"], torch.dtype):
        raise ValueError("model_dtype must be a torch.dtype")
    if not isinstance(cfg["site"], str):
        raise ValueError("site must be a string")
    if not isinstance(cfg["layer"], int) or cfg["layer"] < 0:
        raise ValueError("layer must be a non-negative integer")
    if not isinstance(cfg["num_tokens"], int) or cfg["num_tokens"] <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if not isinstance(cfg["batch_size"], int) or cfg["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(cfg["lr"], float) or cfg["lr"] <= 0:
        raise ValueError("lr must be a positive float")
    if not isinstance(cfg["l1_coeff"], (int, float)) or cfg["l1_coeff"] < 0:
        raise ValueError("l1_coeff must be a non-negative number")
    if not isinstance(cfg["beta1"], float) or not 0 <= cfg["beta1"] <= 1:
        raise ValueError("beta1 must be a float between 0 and 1")
    if not isinstance(cfg["beta2"], float) or not 0 <= cfg["beta2"] <= 1:
        raise ValueError("beta2 must be a float between 0 and 1")
    if not isinstance(cfg["max_grad_norm"], (int, float)) or cfg["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be a positive number")
    if not isinstance(cfg["seq_len"], int) or cfg["seq_len"] <= 0:
        raise ValueError("seq_len must be a positive integer")
    if not isinstance(cfg["act_size"], int) or cfg["act_size"] <= 0:
        raise ValueError("act_size must be a positive integer")
    if not isinstance(cfg["dict_size"], int) or cfg["dict_size"] <= 0:
        raise ValueError("dict_size must be a positive integer")
    if not isinstance(cfg["model_batch_size"], int) or cfg["model_batch_size"] <= 0:
        raise ValueError("model_batch_size must be a positive integer")
    if not isinstance(cfg["num_batches_in_buffer"], int) or cfg["num_batches_in_buffer"] <= 0:
        raise ValueError("num_batches_in_buffer must be a positive integer")
    if not isinstance(cfg["checkpoint_freq"], int) or cfg["checkpoint_freq"] <= 0:
        raise ValueError("checkpoint_freq must be a positive integer")
    if not isinstance(cfg["perf_log_freq"], int) or cfg["perf_log_freq"] <= 0:
        raise ValueError("perf_log_freq must be a positive integer")
    if not isinstance(cfg["n_batches_to_dead"], int) or cfg["n_batches_to_dead"] <= 0:
        raise ValueError("n_batches_to_dead must be a positive integer")
    if not isinstance(cfg["device"], str):
        raise ValueError("device must be a string")
    if not isinstance(cfg["wandb_project"], str):
        raise ValueError("wandb_project must be a string")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer")
    if not isinstance(cfg["input_unit_norm"], bool):
        raise ValueError("input_unit_norm must be a boolean")
    if not isinstance(cfg["sae_type"], str):
        raise ValueError("sae_type must be a string")
    if not isinstance(cfg["top_k"], int) or cfg["top_k"] <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(cfg["top_k_aux"], int) or cfg["top_k_aux"] <= 0:
        raise ValueError("top_k_aux must be a positive integer")
    if not isinstance(cfg["aux_penalty"], (int, float)) or cfg["aux_penalty"] < 0:
        raise ValueError("aux_penalty must be a non-negative number")
    if not isinstance(cfg["bandwidth"], float) or cfg["bandwidth"] <= 0:
        raise ValueError("bandwidth must be a positive float")
    if not isinstance(cfg["n_samples"], int) or cfg["n_samples"] <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(cfg["n_signals"], int) or cfg["n_signals"] <= 0:
        raise ValueError("n_signals must be a positive integer")
    if not isinstance(cfg["signal_strength"], (int, float)) or cfg["signal_strength"] < 0:
        raise ValueError("signal_strength must be a non-negative number")
    if not isinstance(cfg["noise_level"], (int, float)) or cfg["noise_level"] < 0:
        raise ValueError("noise_level must be a non-negative number")

    # Post-initialization
    try:
        cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    except Exception as e:
        raise ValueError(f"Failed to get activation name: {str(e)}")

    cfg["name"] = (
        f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_"
        f"{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}_{cfg['seed']}"
    )

    return cfg
