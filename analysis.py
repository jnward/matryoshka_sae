import json
import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import torch
from scipy.optimize import linear_sum_assignment  # type: ignore

from config import post_init_cfg
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE


def get_normalized_weights(sae: torch.nn.Module, use_decoder: bool = True) -> torch.Tensor:
    """
    Get normalized weights from an SAE model.

    Args:
        sae: The SAE model
        use_decoder: If True, use decoder weights, otherwise use encoder weights

    Returns:
        Normalized weights tensor
    """
    if use_decoder:
        weights = sae.W_dec.data
    else:
        weights = sae.W_enc.data
    # Normalize the weights
    return weights / torch.norm(weights, dim=1, keepdim=True)


def run_hungarian_alignment(
    weights_1: torch.Tensor,
    weights_2: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Run Hungarian algorithm to align features between two SAEs.

    Args:
        weights_1: First set of weights
        weights_2: Second set of weights

    Returns:
        Tuple of (cost_matrix, row_ind, col_ind, avg_score, similarities)
    """
    # Convert to numpy
    weights_1_np = weights_1.cpu().numpy()
    weights_2_np = weights_2.cpu().numpy()

    # For decoder weights, features are along dim 0
    # For encoder weights, features are along dim 1
    # We want to compare features along the same dimension
    if weights_1_np.shape[0] != weights_2_np.shape[0]:
        # If the first dimensions don't match, we need to transpose
        if weights_1_np.shape[0] == weights_2_np.shape[1]:
            weights_2_np = weights_2_np.T
        elif weights_1_np.shape[1] == weights_2_np.shape[0]:
            weights_1_np = weights_1_np.T
        else:
            raise ValueError(f"Cannot align weight matrices with shapes {weights_1_np.shape} and {weights_2_np.shape}")

    # Ensure we're comparing the same number of features
    min_features = min(weights_1_np.shape[0], weights_2_np.shape[0])
    weights_1_np = weights_1_np[:min_features]
    weights_2_np = weights_2_np[:min_features]

    # Compute cost matrix (negative cosine similarity)
    cost_matrix = -np.dot(weights_1_np, weights_2_np.T)

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute similarities
    similarities = -cost_matrix[row_ind, col_ind]
    avg_score = np.mean(similarities)

    return cost_matrix, row_ind, col_ind, avg_score, similarities


def load_sae_model(checkpoint_dir: str, device: str) -> Tuple[Union[BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE], dict]:
    """
    Load an SAE model and its configuration from a checkpoint directory.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        device: Device to load the model onto

    Returns:
        Tuple of (model, config)

    Raises:
        FileNotFoundError: If checkpoint files are missing
        ValueError: If config is invalid
        RuntimeError: If model loading fails
    """
    model_path = os.path.join(checkpoint_dir, "sae.pt")
    config_path = os.path.join(checkpoint_dir, "config.json")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # Process config (convert string representations to proper types)
        for k, v in cfg.items():
            if isinstance(v, str) and v.startswith("["):
                try:
                    cfg[k] = json.loads(v)
                except json.JSONDecodeError:
                    pass
            elif isinstance(v, str) and v.startswith("torch."):
                try:
                    module_path = v.split(".")
                    if module_path[0] == "torch":
                        cfg[k] = getattr(torch, module_path[1])
                except (AttributeError, IndexError):
                    pass

        # Update device and finalize config
        cfg["device"] = device
        cfg = post_init_cfg(cfg)

        # Create and load SAE model
        sae: Union[BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE]
        if cfg["sae_type"] == "global-matryoshka-topk":
            sae = GlobalBatchTopKMatryoshkaSAE(cfg)
        else:
            sae = BatchTopKSAE(cfg)

        sae.load_state_dict(torch.load(model_path, map_location=device))
        sae.eval()
        print(f"SAE model loaded from {model_path}")

        return sae, cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {checkpoint_dir}: {str(e)}")


def plot_alignment_comparison(
    sae_1: torch.nn.Module,
    sae_2: torch.nn.Module,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    hungarian_batch_dim: int = 4096,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create a scatter plot comparing decoder and encoder alignments for a slice of features.

    Args:
        sae_1: First SAE model
        sae_2: Second SAE model
        start_idx: Starting index for feature slice
        end_idx: Ending index for feature slice (if None, uses all features)
        hungarian_batch_dim: Dimension for Hungarian alignment
        title: Optional title for the plot

    Returns:
        Plotly figure object

    Raises:
        ValueError: If input parameters are invalid
    """
    if not isinstance(sae_1, torch.nn.Module) or not isinstance(sae_2, torch.nn.Module):
        raise ValueError("sae_1 and sae_2 must be torch.nn.Module instances")
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative")
    if end_idx is not None and end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    if hungarian_batch_dim <= 0:
        raise ValueError("hungarian_batch_dim must be positive")

    # Get normalized weights
    decoder_1 = get_normalized_weights(sae_1)
    decoder_2 = get_normalized_weights(sae_2)
    encoder_1 = get_normalized_weights(sae_1, use_decoder=False)
    encoder_2 = get_normalized_weights(sae_2, use_decoder=False)

    # Slice if end_idx is specified
    if end_idx is not None:
        decoder_1 = decoder_1[start_idx:end_idx]
        decoder_2 = decoder_2[start_idx:end_idx]
        encoder_1 = encoder_1[start_idx:end_idx]
        encoder_2 = encoder_2[start_idx:end_idx]

    # Run Hungarian alignment
    dec_cost_matrix, dec_row_ind, dec_col_ind, dec_avg_score, dec_similarities = run_hungarian_alignment(
        decoder_1, decoder_2
    )
    enc_cost_matrix, enc_row_ind, enc_col_ind, enc_avg_score, enc_similarities = run_hungarian_alignment(
        encoder_1, encoder_2
    )

    # Ensure we're comparing the same number of features
    min_len = min(len(dec_col_ind), len(enc_col_ind))
    dec_col_ind = dec_col_ind[:min_len]
    enc_col_ind = enc_col_ind[:min_len]

    # Check identical alignments
    identical = enc_col_ind == dec_col_ind

    # Create dataframe for the scatter plot
    df = pd.DataFrame(
        {
            "Decoder alignment": dec_similarities[:min_len],
            "Encoder alignment": enc_similarities[:min_len],
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
        range_x=[-0.05, 1.05],
        range_y=[-0.05, 1.05],
        size_max=8,
    )

    # Update the scatter points separately to make them more visible
    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))

    # Contour parameters
    contour_params = dict(showscale=False, ncontours=8, line_width=1, contours=dict(showlabels=False, coloring="lines"))

    # Add contour for "Equal" points
    if not df_equal.empty:
        fig.add_trace(
            go.Histogram2dContour(
                x=df_equal["Decoder alignment"],
                y=df_equal["Encoder alignment"],
                opacity=0.7,
                **contour_params,
            )
        )

    # Add contour for "Different" points
    if not df_different.empty:
        fig.add_trace(
            go.Histogram2dContour(
                x=df_different["Decoder alignment"],
                y=df_different["Encoder alignment"],
                opacity=0.7,
                **contour_params,
            )
        )

    # Update layout
    fig.update_layout(
        title=title or f"Alignment Comparison (features {start_idx}-{end_idx if end_idx is not None else 'end'})",
        xaxis_title="Decoder alignment",
        yaxis_title="Encoder alignment",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=1.00, bgcolor="rgba(255, 255, 255, 0.8)"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=800,
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def main() -> None:
    """Main function to run the analysis."""
    try:
        # change the below
        save_file_name = "alignment_comparison"
        plot_title = "SAE Alignment Comparison (Global vs Batch TopK)"
        model_1_checkpoint_dir = "checkpoints/global-matryoshka-topk_16384_32_0.0003_42_243"
        model_2_checkpoint_dir = "checkpoints/global-matryoshka-topk_16384_32_0.0003_95_243"

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        print(f"Loading model from {model_1_checkpoint_dir}")
        sae_1, cfg_1 = load_sae_model(model_1_checkpoint_dir, str(device))
        print(f"Loading model from {model_2_checkpoint_dir}")
        sae_2, cfg_2 = load_sae_model(model_2_checkpoint_dir, str(device))

        # Get the minimum number of features between the two models
        n_features_1 = sae_1.W_dec.shape[0]
        n_features_2 = sae_2.W_dec.shape[0]
        min_features = min(n_features_1, n_features_2)
        print(f"n_features_1: {n_features_1}")
        print(f"n_features_2: {n_features_2}")
        print(f"min_features: {min_features}")

        # Set indices to compare only the features that exist in both models
        start_idx = 0
        end_idx = min_features

        print(f"Comparing first {min_features} features (out of {n_features_1} and {n_features_2} features)")

        # Create comparison plot
        fig = plot_alignment_comparison(sae_1, sae_2, start_idx=start_idx, end_idx=end_idx, title=plot_title)

        # Create directories if they don't exist
        os.makedirs("plots_html", exist_ok=True)
        os.makedirs("plots_png", exist_ok=True)

        # Save as HTML (interactive)
        html_path = os.path.join("plots_html", f"{save_file_name}.html")
        print(f"Saving HTML to {html_path}")
        fig.write_html(html_path)

        # Save as PNG (static)
        png_path = os.path.join("plots_png", f"{save_file_name}.png")
        print(f"Saving PNG to {png_path}")
        fig.write_image(png_path)
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
