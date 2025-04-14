import torch
import tqdm
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
import multiprocessing as mp
from queue import Empty
import wandb
import json
import os

def train_sae(sae, activation_store, model, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)
    
    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch)
        log_wandb(sae_output, i, wandb_run)
        if i % cfg["perf_log_freq"]  == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, i)
    

def train_sae_group(saes, activation_store, model, cfgs):
    num_batches = cfgs[0]["num_tokens"] // cfgs[0]["batch_size"]
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfgs[0])

    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        batch = activation_store.next_batch()
        counter = 0
        for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_wandb(sae_output, i, wandb_run, index=counter)
            if i % cfg["perf_log_freq"]  == 0:
                log_model_performance(wandb_run, i, model, activation_store, sae, index=counter, batch_tokens=batch_tokens)

            if i % cfg["checkpoint_freq"] == 0:
                save_checkpoint(wandb_run, sae, cfg, i)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
   
    for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
        save_checkpoint(wandb_run, sae, cfg, i)


def save_checkpoint_mp(sae, cfg, step):
    """
    Save checkpoint without requiring a wandb run object.
    Creates an artifact but doesn't log it to wandb directly.
    """
    save_dir = f"checkpoints/{cfg['name']}_{step}"
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

def wandb_process_target(config, log_queue, entity, project):
    run = wandb.init(
        entity=entity, 
        project=project, 
        config=config, 
        name=config["name"]
    )
    
    while True:
        try:
            # Wait up to 1 second for new data
            log = log_queue.get(timeout=1)
            
            # Check for termination signal
            if log == "DONE":
                break
            
            # Check if this is a checkpoint signal
            if isinstance(log, dict) and log.get("checkpoint"):
                # Create and log artifact
                artifact = wandb.Artifact(
                    name=f"{config['name']}_{log['step']}",
                    type="model",
                    description=f"Model checkpoint at step {log['step']}",
                )
                save_dir = log["save_dir"]
                artifact.add_file(os.path.join(save_dir, "sae.pt"))
                artifact.add_file(os.path.join(save_dir, "config.json"))
                run.log_artifact(artifact)
            else:
                # Log regular metrics
                wandb.log(log)
        except Empty:
            continue
            
    wandb.finish()

def train_sae_group_seperate_wandb(saes, activation_store, model, cfgs):
    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    print(f"Number of batches: {num_batches}")
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) 
                 for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    # Initialize wandb processes and queues
    wandb_processes = []
    log_queues = []
    
    for i, cfg in enumerate(cfgs):
        log_queue = mp.Queue()
        log_queues.append(log_queue)
        wandb_config = cfg
        wandb_process = mp.Process(
            target=wandb_process_target,
            args=(wandb_config, log_queue, cfg.get("wandb_entity", ""), cfg["wandb_project"]),
        )
        wandb_process.start()
        wandb_processes.append(wandb_process)

    for i in pbar:
        batch = activation_store.next_batch()
        
        for idx, (sae, cfg, optimizer) in enumerate(zip(saes, cfgs, optimizers)):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            
            # Log metrics to appropriate wandb process
            log_dict = {
                k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
                for k, v in sae_output.items() if isinstance(v, (int, float)) or 
                (isinstance(v, torch.Tensor) and v.dim() == 0)
            }
            log_queues[idx].put(log_dict)

            if i % cfg["checkpoint_freq"] == 0:
                # Save checkpoint and send artifact info to wandb process
                save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
                log_queues[idx].put({
                    "checkpoint": True,
                    "step": i,
                    "save_dir": save_dir
                })

            pbar.set_postfix({
                f"Loss_{idx}": f"{loss.item():.4f}", 
                f"L0_{idx}": f"{sae_output['l0_norm']:.4f}",
                f"L2_{idx}": f"{sae_output['l2_loss']:.4f}", 
                f"L1_{idx}": f"{sae_output['l1_loss']:.4f}",
            })
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

    # Final checkpoints
    for idx, (sae, cfg) in enumerate(zip(saes, cfgs)):
        save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
        log_queues[idx].put({
            "checkpoint": True,
            "step": i,
            "save_dir": save_dir
        })

    # Clean up wandb processes
    for queue in log_queues:
        queue.put("DONE")
    for process in wandb_processes:
        process.join()