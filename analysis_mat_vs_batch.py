# %% Imports
import torch
import os
import json
from sae import GlobalBatchTopKMatryoshkaSAE, BatchTopKSAE
from config import post_init_cfg

# %% Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Define paths and load configuration
checkpoint_dir_42 = "original_checkpoints/gpt2_blocks.6.hook_resid_post_12288_global-matryoshka-topk_32_0.0003_42_48827"  # Change this to your checkpoint path
n_features = 12288
k = 32

# checkpoint_dir_42 = f"original_checkpoints/gpt2_blocks.6.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_42_48827"  # Change this to your checkpoint path
model_path_42 = os.path.join(checkpoint_dir_42, "sae.pt")
config_path_42 = os.path.join(checkpoint_dir_42, "config.json")

# checkpoint_dir_43 = "original_checkpoints/gpt2_blocks.6.hook_resid_post_12288_global-matryoshka-topk_32_0.0003_43_48827"  # Change this to your checkpoint path
checkpoint_dir_43 = f"original_checkpoints/gpt2_blocks.6.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_42_48827"  # Change this to your checkpoint path
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
                module_path = v.split('.')
                if module_path[0] == 'torch':
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
sae_42 = GlobalBatchTopKMatryoshkaSAE(cfg_42)
# sae_42 = BatchTopKSAE(cfg_42)
sae_42.load_state_dict(torch.load(model_path_42, map_location=device))
sae_42.eval()
print(f"SAE model loaded from {model_path_42}") 

# sae_43 = GlobalBatchTopKMatryoshkaSAE(cfg_43)
sae_43 = BatchTopKSAE(cfg_43)
sae_43.load_state_dict(torch.load(model_path_43, map_location=device))
sae_43.eval()
print(f"SAE model loaded from {model_path_43}") 
# %%

def plot_alignment_comparison(sae_1, sae_2, start_idx=0, end_idx=768, hungarian_batch_dim=4096, title=None):
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
    from hungarian import run_hungarian_alignment, get_normalized_weights
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    # Get normalized weights and slice them
    decoder_1 = get_normalized_weights(sae_1)[start_idx:end_idx]
    decoder_2 = get_normalized_weights(sae_2)[start_idx:end_idx]
    
    encoder_1 = get_normalized_weights(sae_1, use_decoder=False)[start_idx:end_idx]
    encoder_2 = get_normalized_weights(sae_2, use_decoder=False)[start_idx:end_idx]
    
    # Run Hungarian alignment
    dec_cost_matrix, dec_row_ind, dec_col_ind, dec_avg_score, dec_similarities = run_hungarian_alignment(
        decoder_1, decoder_2, hungarian_batch_dim)
    enc_cost_matrix, enc_row_ind, enc_col_ind, enc_avg_score, enc_similarities = run_hungarian_alignment(
        encoder_1, encoder_2, hungarian_batch_dim)
    
    # Define matryoshka slices
    slices = [(0, 768), (768, 1536), (1536, 3072), (3072, 6144), (6144, 12288)]
    slice_names = ['Slice 1 (0-768)', 'Slice 2 (768-1536)', 'Slice 3 (1536-3072)', 
                  'Slice 4 (3072-6144)', 'Slice 5 (6144-12288)']
    
    # Determine which slice each feature belongs to
    feature_indices = np.arange(start_idx, end_idx)
    slice_categories = []
    
    for idx in feature_indices:
        slice_found = False
        for i, (slice_start, slice_end) in enumerate(slices):
            if slice_start <= idx < slice_end:
                slice_categories.append(slice_names[i])
                slice_found = True
                break
        if not slice_found:
            slice_categories.append('Other')
    
    # Create dataframe for the scatter plot
    df = pd.DataFrame({
        'Decoder alignment': dec_similarities,
        'Encoder alignment': enc_similarities,
        'Slice': slice_categories,
        'Feature Index': feature_indices
    })

    # Sort dataframe to control drawing order (higher slice numbers will be drawn first, lower numbers on top)
    df['Slice_Order'] = df['Slice'].map({name: i for i, name in enumerate(slice_names)})
    df = df.sort_values('Slice_Order', ascending=False)
    
    # Create figure with secondary y-axis
    fig = px.scatter(
        df, 
        x='Decoder alignment', 
        y='Encoder alignment',
        color='Slice',
        hover_data=['Feature Index'],
        opacity=0.6,
        marginal_x='histogram',
        marginal_y='histogram',
        range_x=[0, 1],
        range_y=[0, 1],
        size_max=3,
        category_orders={'Slice': slice_names[::-1]}  # Reverse order in legend
    )

    # Update the scatter points separately to make them even smaller
    fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Decoder alignment",
        yaxis_title="Encoder alignment",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.00,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    return fig


# # Features 0-768
fig = plot_alignment_comparison(sae_42, sae_43, start_idx=0, end_idx=12288, title="Matryoshka Full Dictionary vs BatchTopK")
fig.show()

# Features 768-1536
# fig = plot_alignment_comparison(sae_42, sae_43, start_idx=3072, end_idx=6144)
# fig.show()
# %%