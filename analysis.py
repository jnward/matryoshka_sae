# %% Imports
import json
import os

import torch

from config import post_init_cfg
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE

# %% Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Define paths and load configuration
# checkpoint_dir_42 = "original_checkpoints/gpt2_blocks.6.hook_resid_post_12288_global-matryoshka-topk_32_0.0003_42_48827"  # Change this to your checkpoint path
n_features = 768
k = 22

checkpoint_dir_42 = f"original_checkpoints/gpt2_blocks.6.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_42_48827"  # Change this to your checkpoint path
model_path_42 = os.path.join(checkpoint_dir_42, "sae.pt")
config_path_42 = os.path.join(checkpoint_dir_42, "config.json")

# checkpoint_dir_43 = "original_checkpoints/gpt2_blocks.6.hook_resid_post_12288_global-matryoshka-topk_32_0.0003_43_48827"  # Change this to your checkpoint path
checkpoint_dir_43 = f"original_checkpoints/gpt2_blocks.6.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_43_48827"  # Change this to your checkpoint path
model_path_43 = os.path.join(checkpoint_dir_43, "sae.pt")
config_path_43 = os.path.join(checkpoint_dir_43, "config.json")

# Load config
with open(config_path_42, "r") as f:
    cfg_42 = json.load(f)

with open(config_path_43, "r") as f:
    cfg_43 = json.load(f)

print(f"Loaded config from {config_path_42}")
print(f"Loaded config from {config_path_43}")


# %% Initialize and load the SAE
# Convert string representations back to lists if needed
def process_config(cfg):
    for k, v in cfg.items():
        if isinstance(v, str) and v.startswith("["):
            try:
                cfg[k] = json.loads(v)
            except:
                pass
        elif isinstance(v, str) and v.startswith("torch."):
            try:
                module_path = v.split(".")
                if module_path[0] == "torch":
                    cfg[k] = getattr(torch, module_path[1])
            except:
                pass


process_config(cfg_42)
process_config(cfg_43)

# Update device and finalize config
cfg_42["device"] = device
cfg_42 = post_init_cfg(cfg_42)
# cfg_42["n_features"] = n_features

cfg_43["device"] = device
cfg_43 = post_init_cfg(cfg_43)
# cfg_43["n_features"] = n_features

# Create and load SAE model
# sae_42 = GlobalBatchTopKMatryoshkaSAE(cfg_42)
sae_42 = BatchTopKSAE(cfg_42)
sae_42.load_state_dict(torch.load(model_path_42, map_location=device))
sae_42.eval()
print(f"SAE model loaded from {model_path_42}")

# sae_43 = GlobalBatchTopKMatryoshkaSAE(cfg_43)
sae_43 = BatchTopKSAE(cfg_43)
sae_43.load_state_dict(torch.load(model_path_43, map_location=device))
sae_43.eval()
print(f"SAE model loaded from {model_path_43}")
# %%
from transformer_lens import HookedTransformer

from activation_store import ActivationsStore

model = (
    HookedTransformer.from_pretrained_no_processing(cfg_42["model_name"]).to(cfg_42["model_dtype"]).to(cfg_42["device"])
)
activations_store = ActivationsStore(model, cfg_42)

# %%
test_acts = activations_store.next_batch()

# %%
out_42 = sae_42(test_acts)
recon_42 = out_42["sae_out"]
features_42 = out_42["feature_acts"]

mse = torch.nn.functional.mse_loss(recon_42, test_acts)
mse

e = recon_42 - test_acts
total_var = torch.var(test_acts)
fvu = torch.var(e) / total_var
print(f"FVU: {fvu}")

(features_42 > 0).sum(dim=-1).float().mean()

# %%

sub_W_enc = sae_42.W_enc  # [:, sae_42.group_indices[2]:sae_42.group_indices[3]]
sub_b_enc = sae_42.b_enc  # [sae_42.group_indices[2]:sae_42.group_indices[3]]
sub_W_dec = sae_42.W_dec  # [sae_42.group_indices[2]:sae_42.group_indices[3], :]
sub_b_dec = sae_42.b_dec

print(sub_W_enc.shape, sub_b_enc.shape, sub_W_dec.shape, sub_b_dec.shape)

sub_features = test_acts.float() @ sub_W_enc + sub_b_enc
sub_features[sub_features <= sae_42.threshold] = 0
print(sub_features.shape)
sub_features

sub_recon = sub_features @ sub_W_dec + sub_b_dec
e = sub_recon - test_acts
total_var = torch.var(test_acts)
fvu = torch.var(e) / total_var
print(f"FVU: {fvu}")

# make sure the full SAE works w/ this code ^

# %%
sub_W_enc.sum(dim=0).abs().mean()

# (test_features[0, 0] > 1).sum()
# %%
from hungarian import get_normalized_weights, run_hungarian_alignment

decoder_42 = get_normalized_weights(sae_42)
decoder_43 = get_normalized_weights(sae_43)

encoder_42 = get_normalized_weights(sae_42, use_decoder=False)
encoder_43 = get_normalized_weights(sae_43, use_decoder=False)

decoder_42 = decoder_42[:768]
decoder_43 = decoder_43[:768]

encoder_42 = encoder_42[:768]
encoder_43 = encoder_43[:768]

dec_cost_matrix, dec_row_ind, dec_col_ind, dec_avg_score, dec_similarities = run_hungarian_alignment(
    decoder_42, decoder_43, 4096
)
enc_cost_matrix, enc_row_ind, enc_col_ind, enc_avg_score, enc_similarities = run_hungarian_alignment(
    encoder_42, encoder_43, 4096
)
# %%
dec_similarities.shape
import numpy as np

# %%
import plotly.express as px

fig = px.histogram(
    dec_similarities,
    nbins=100,
    range_x=[0, 1],
    # log_x=True,
    title="Decoder Similarities (log scale)",
)
fig.show()

# Keep the second histogram as is
fig = px.histogram(enc_similarities, nbins=100, range_x=[0, 1], title="Encoder Similarities")
fig.show()
# %%
identical = enc_col_ind == dec_col_ind

import numpy as np
import pandas as pd

# %%
import plotly.express as px
import plotly.graph_objects as go

# Create dataframe for the scatter plot
df = pd.DataFrame(
    {
        "Decoder alignment": dec_similarities,
        "Encoder alignment": enc_similarities,
        "Category": ["Equal" if i else "Different" for i in identical],
    }
)

# Split data by category
df_equal = df[df["Category"] == "Equal"]
df_different = df[df["Category"] == "Different"]

# Create figure with secondary y-axis
fig = px.scatter(
    df,
    x="Decoder alignment",
    y="Encoder alignment",
    color="Category",
    color_discrete_map={"Different": "orange", "Equal": "blue"},
    opacity=0.6,
    marginal_x="histogram",
    marginal_y="histogram",
    range_x=[0, 1],
    range_y=[0, 1],
    size_max=3,  # Make points smaller
)

# Update the scatter points separately to make them even smaller
fig.update_traces(marker=dict(size=2), selector=dict(mode="markers"))

# Add contours for better visibility of density
contour_params = dict(showscale=False, ncontours=8, line_width=1, contours=dict(showlabels=False, coloring="lines"))

# Add contour for "Equal" points
fig.add_trace(
    go.Histogram2dContour(
        x=df_equal["Decoder alignment"],
        y=df_equal["Encoder alignment"],
        # colorscale='Blues',
        opacity=0.7,
        **contour_params,
    )
)

# Add contour for "Different" points
fig.add_trace(
    go.Histogram2dContour(
        x=df_different["Decoder alignment"],
        y=df_different["Encoder alignment"],
        # colorscale='Oranges',
        opacity=0.7,
        **contour_params,
    )
)

# Update layout
fig.update_layout(
    xaxis_title="Decoder alignment",
    yaxis_title="Encoder alignment",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.00, bgcolor="rgba(255, 255, 255, 0.8)"),
    yaxis=dict(scaleanchor="x", scaleratio=1),
)

fig.show()

# %%


def plot_alignment_comparison(sae_1, sae_2, start_idx=0, end_idx=768, hungarian_batch_dim=4096):
    """
    Create a scatter plot comparing decoder and encoder alignments for a slice of features.

    Args:
        sae_1: First SAE model
        sae_2: Second SAE model
        start_idx: Starting index for feature slice
        end_idx: Ending index for feature slice
        hungarian_batch_dim: Dimension for Hungarian alignment

    Returns:
        Plotly figure object
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from hungarian import get_normalized_weights, run_hungarian_alignment

    # Get normalized weights and slice them
    decoder_1 = get_normalized_weights(sae_1)[start_idx:end_idx]
    decoder_2 = get_normalized_weights(sae_2)[start_idx:end_idx]

    encoder_1 = get_normalized_weights(sae_1, use_decoder=False)[start_idx:end_idx]
    encoder_2 = get_normalized_weights(sae_2, use_decoder=False)[start_idx:end_idx]

    # Run Hungarian alignment
    dec_cost_matrix, dec_row_ind, dec_col_ind, dec_avg_score, dec_similarities = run_hungarian_alignment(
        decoder_1, decoder_2, hungarian_batch_dim
    )
    enc_cost_matrix, enc_row_ind, enc_col_ind, enc_avg_score, enc_similarities = run_hungarian_alignment(
        encoder_1, encoder_2, hungarian_batch_dim
    )

    # Check identical alignments
    identical = enc_col_ind == dec_col_ind

    # Create dataframe for the scatter plot
    df = pd.DataFrame(
        {
            "Decoder alignment": dec_similarities,
            "Encoder alignment": enc_similarities,
            "Category": ["Equal" if i else "Different" for i in identical],
        }
    )

    # Split data by category
    df_equal = df[df["Category"] == "Equal"]
    df_different = df[df["Category"] == "Different"]

    # Create figure with secondary y-axis
    fig = px.scatter(
        df,
        x="Decoder alignment",
        y="Encoder alignment",
        color="Category",
        color_discrete_map={"Different": "orange", "Equal": "blue"},
        opacity=0.6,
        marginal_x="histogram",
        marginal_y="histogram",
        range_x=[0, 1],
        range_y=[0, 1],
        size_max=3,
    )

    # Update the scatter points separately to make them even smaller
    fig.update_traces(marker=dict(size=2), selector=dict(mode="markers"))

    # Contour parameters
    contour_params = dict(showscale=False, ncontours=8, line_width=1, contours=dict(showlabels=False, coloring="lines"))

    # Add contour for "Equal" points
    fig.add_trace(
        go.Histogram2dContour(
            x=df_equal["Decoder alignment"], y=df_equal["Encoder alignment"], opacity=0.7, **contour_params
        )
    )

    # Add contour for "Different" points
    fig.add_trace(
        go.Histogram2dContour(
            x=df_different["Decoder alignment"], y=df_different["Encoder alignment"], opacity=0.7, **contour_params
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Alignment Comparison (features {start_idx}-{end_idx})",
        xaxis_title="Decoder alignment",
        yaxis_title="Encoder alignment",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.00, bgcolor="rgba(255, 255, 255, 0.8)"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


# # Features 0-768
fig = plot_alignment_comparison(sae_42, sae_43, start_idx=0, end_idx=768)
fig.show()

# Features 768-1536
# fig = plot_alignment_comparison(sae_42, sae_43, start_idx=3072, end_idx=6144)
# fig.show()
# %%
