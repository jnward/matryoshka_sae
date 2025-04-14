# %% Imports
import json
import os

import torch

from config import post_init_cfg
from sae import GlobalBatchTopKMatryoshkaSAE

# %% Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Define paths and load configuration
checkpoint_dir = "checkpoints/gpt2_blocks.6.hook_resid_post_12288_global-matryoshka-topk_32_0.0003_10000"  # change this to your checkpoint path
model_path = os.path.join(checkpoint_dir, "sae.pt")
config_path = os.path.join(checkpoint_dir, "config.json")

# Load config
with open(config_path, "r") as f:
    cfg = json.load(f)

print(f"Loaded config from {config_path}")

# %% Initialize and load the SAE
# Convert string representations back to lists if needed
for k, v in cfg.items():
    if isinstance(v, str) and v.startswith("["):
        try:
            cfg[k] = json.loads(v)
        except Exception as e:
            print(f"Error converting {k} to list: {e}")
            pass
    elif isinstance(v, str) and v.startswith("torch."):
        try:
            module_path = v.split(".")
            if module_path[0] == "torch":
                cfg[k] = getattr(torch, module_path[1])
        except Exception as e:
            print(f"Error converting {k} to torch: {e}")
            pass

# Update device and finalize config
cfg["device"] = device
cfg = post_init_cfg(cfg)

# Create and load SAE model
sae = GlobalBatchTopKMatryoshkaSAE(cfg)
sae.load_state_dict(torch.load(model_path, map_location=device))
sae.eval()
print(f"SAE model loaded from {model_path}")
# %%
sae.W_enc.shape

# %%
sae.training = True
acts = torch.randn(128, 768).to(device)

test_features = sae.encode(acts)
test_features.shape

test_recon = sae.decode(test_features)
test_recon.shape
# %%
# compute variance explained
error = acts - test_recon
variance_explained = 1 - torch.var(error) / torch.var(acts)
variance_explained
# %%
(test_features > 1).sum(dim=-1).float().mean()
# %%
test_features = sae.encode(acts[:1, :1, :])
test_features.shape

# (test_features[0, 0] > 1).sum()
# %%
