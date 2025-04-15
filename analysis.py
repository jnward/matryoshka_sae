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
# checkpoint_dir_42 = "original_checkpoints/gpt2_blocks.6.hook_resid_post_12288_global-matryoshka-topk_32_0.0003_42_48827"  # Change this to your checkpoint path
# checkpoint_dir_42 = "checkpoints/gemma-2-2b_blocks.14.hook_resid_post_24576_batch-topk_32_0.0003_42_48827"
# checkpoint_dir_42 = "checkpoints/gemma-2-2b_blocks.14.hook_resid_post_24576_global-matryoshka-topk_32_0.0003_42_48827"

n_features = 24576
k = 32

checkpoint_dir_42 = f"checkpoints/gemma-2-2b_blocks.14.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_42_48827"  # Change this to your checkpoint path
# checkpoint_dir_42 = f"original_checkpoints/gpt2_blocks.6.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_42_48827"  # Change this to your checkpoint path
model_path_42 = os.path.join(checkpoint_dir_42, "sae.pt")
config_path_42 = os.path.join(checkpoint_dir_42, "config.json")

# checkpoint_dir_43 = "checkpoints/gemma-2-2b_blocks.14.hook_resid_post_24576_batch-topk_32_0.0003_43_48827"
# checkpoint_dir_43 = "checkpoints/gemma-2-2b_blocks.14.hook_resid_post_24576_global-matryoshka-topk_32_0.0003_43_48827"
# checkpoint_dir_43 = f"original_checkpoints/gpt2_blocks.6.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_43_48827"  # Change this to your checkpoint path
checkpoint_dir_43 = f"checkpoints/gemma-2-2b_blocks.14.hook_resid_post_{n_features}_batch-topk_{k}_0.0003_43_48827"
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
    from hungarian import run_hungarian_alignment, get_normalized_weights
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
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
    
    # Check identical alignments
    identical = (enc_col_ind == dec_col_ind)
    
    # Create dataframe for the scatter plot
    df = pd.DataFrame({
        'Decoder alignment': dec_similarities,
        'Encoder alignment': enc_similarities,
        'Category': ['Equal' if i else 'Different' for i in identical]
    })

    # Split data by category
    df_equal = df[df['Category'] == 'Equal']
    df_different = df[df['Category'] == 'Different']

    # Create figure with secondary y-axis
    fig = px.scatter(
        df, 
        x='Decoder alignment', 
        y='Encoder alignment',
        color='Category',
        color_discrete_map={'Different': 'orange', 'Equal': 'blue'},
        opacity=0.6,
        marginal_x='histogram',
        marginal_y='histogram',
        range_x=[0, 1],
        range_y=[0, 1],
        size_max=3
    )

    # Update the scatter points separately to make them even smaller
    fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))

    # Contour parameters
    contour_params = dict(
        showscale=False,
        ncontours=8,
        line_width=1,
        contours=dict(
            showlabels=False,
            coloring='lines'
        )
    )

    # Add contour for "Equal" points
    fig.add_trace(go.Histogram2dContour(
        x=df_equal['Decoder alignment'],
        y=df_equal['Encoder alignment'],
        opacity=0.7,
        **contour_params
    ))

    # Add contour for "Different" points
    fig.add_trace(go.Histogram2dContour(
        x=df_different['Decoder alignment'],
        y=df_different['Encoder alignment'],
        opacity=0.7,
        **contour_params
    ))

    # Update layout
    fig.update_layout(
        title=f"BatchTopK Seed Alignment ({n_features} features)",
        # title=f"Matryoshka Seed Alignment\n(features {start_idx}-{end_idx})",
        xaxis_title="Decoder alignment",
        yaxis_title="Encoder alignment",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.1,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )
    mean_decoder_alignment = df['Decoder alignment'].mean()
    return fig, mean_decoder_alignment


# # Features 0-768
fig, mean_alignment = plot_alignment_comparison(sae_42, sae_43, start_idx=0, end_idx=-1)
fig.show()
print(f"Mean alignment: {mean_alignment}")

# Features 768-1536
# fig = plot_alignment_comparison(sae_42, sae_43, start_idx=3072, end_idx=6144)
# fig.show()
# %%
mean_alignments = []
for group_idx in range(6):
    start_idx = 0
    end_idx = 2**(group_idx) * 768
    fig, mean_alignment = plot_alignment_comparison(sae_42, sae_43, start_idx=start_idx, end_idx=end_idx)
    fig.show()
    mean_alignments.append(mean_alignment)
# %%
import plotly.express as px
px.line(mean_alignments, x=range(6), y=mean_alignments, title="Mean Alignment by Group")
# %%
matryoshka_mean_alignments = [0.63246906, 0.58439595, 0.55029696, 0.5238667, 0.49258527]
batch_topk_mean_alignments = [0.79798245, 0.78008759, 0.75952810, 0.7231605, 0.62281191]

# %% Plot alignment comparison with doubling group sizes
import plotly.graph_objects as go

# Define x-axis labels (doubling each time)
group_sizes = [1536, 3072, 6144, 12288, 24576]
x_labels = [f"{size}" for size in group_sizes]  # Double each size for x-axis labels

# Create the figure
fig = go.Figure()

# Add line for matryoshka
fig.add_trace(go.Scatter(
    x=x_labels,
    y=matryoshka_mean_alignments,
    mode='lines+markers',
    name='Matryoshka',
    line=dict(color='blue', width=2),
    marker=dict(size=8)
))

# Add line for batch_topk
fig.add_trace(go.Scatter(
    x=x_labels,
    y=batch_topk_mean_alignments,
    mode='lines+markers',
    name='BatchTopK',
    line=dict(color='red', width=2),
    marker=dict(size=8)
))

# Update layout
fig.update_layout(
    title="Decoder Alignment Comparison by Dictionary Size",
    xaxis_title="Dictionary/Group Size",
    yaxis_title="Average Alignment Score",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)'
    ),
    yaxis=dict(
        range=[0, 1]
    )
)

fig.show()
# %%

