import json
import os
from typing import List, Union

import torch
import tqdm  # type: ignore

from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE


def save_checkpoint_mp(
    sae: Union[BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE], cfg: dict, step: int, checkpoint_dir: str = "checkpoints"
):
    """
    Save checkpoint.
    """
    save_dir = f"{checkpoint_dir}/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    print(f"Model and config saved at step {step} in {save_dir}")
    return save_dir, sae_path, config_path


def train_sae_group(
    saes: List[Union[BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE]], activation_store: any, cfgs: List[dict]
) -> None:
    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    print(f"Number of batches: {num_batches}")
    optimizers = [
        torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        for sae, cfg in zip(saes, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    for i in pbar:
        batch = activation_store.next_batch()

        for idx, (sae, cfg, optimizer) in enumerate(zip(saes, cfgs, optimizers)):
            sae_output = sae(batch)
            loss = sae_output["loss"]

            if i % cfg["checkpoint_freq"] == 0:
                # Save checkpoint
                save_checkpoint_mp(sae, cfg, i, checkpoint_dir="custom_data_checkpoints")

            pbar.set_postfix(
                {
                    f"Loss_{idx}": f"{loss.item():.4f}",
                    f"L0_{idx}": f"{sae_output['l0_norm']:.4f}",
                    f"L2_{idx}": f"{sae_output['l2_loss']:.4f}",
                    f"L1_{idx}": f"{sae_output['l1_loss']:.4f}",
                }
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

    # Final checkpoints
    for idx, (sae, cfg) in enumerate(zip(saes, cfgs)):
        save_checkpoint_mp(sae, cfg, i, checkpoint_dir="custom_data_checkpoints")
