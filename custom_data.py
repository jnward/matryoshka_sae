import random
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class CustomDataLoader:
    """Simple data loader for custom data"""

    def __init__(self, data: torch.Tensor, cfg: Dict[str, Any]):
        """
        Initialize the data loader.

        Args:
            data: Input data tensor of shape (num_samples, activation_size)
            cfg: Configuration dictionary containing at least 'batch_size' and 'device'
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.dim() != 2:
            raise ValueError("data must be 2-dimensional (num_samples, activation_size)")
        if data.shape[0] == 0:
            raise ValueError("data must contain at least one sample")

        self.data = data
        self.cfg = cfg

        # Validate required config parameters
        required_params = ["batch_size", "device"]
        for param in required_params:
            if param not in cfg:
                raise ValueError(f"Missing required config parameter: {param}")

        self.batch_size = cfg["batch_size"]
        self.device = cfg["device"]
        self.dataloader = self._create_dataloader()
        self.iterator = iter(self.dataloader)

    def _create_dataloader(self) -> DataLoader:
        """Create a PyTorch DataLoader from the data"""
        dataset = TensorDataset(self.data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def next_batch(self) -> torch.Tensor:
        """Get the next batch of data"""
        try:
            batch = next(self.iterator)
        except StopIteration:
            if len(self.dataloader) == 0:
                raise RuntimeError("Dataset is empty")
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch[0].to(self.device)


def generate_synthetic_data(
    n_samples: int,
    activation_size: int,
    n_signals: int,
    signal_strength: float,
    noise_level: float,
    seed: int,
    # New parameters in [0,1] to control the data properties:
    non_euclidean: float,  # 0: Euclidean; 1: fully warped
    superposition: float,  # 0: nearly one signal per sample; 1: full superposition
    non_orthogonal: float,  # 0: fully orthogonal signals; 1: as generated (non-orthogonal)
    hierarchical: float,  # 0: independent signals; 1: signals grouped in clusters
) -> torch.Tensor:
    """
    Generate synthetic data with control over several characteristics.

    Args:
        n_samples: Number of samples to generate.
        activation_size: Size of each activation vector.
        n_signals: Number of signal components.
        signal_strength: Strength of signal components.
        noise_level: Noise level added to the data.
        seed: Random seed for reproducibility.
        non_euclidean: Fraction (0 to 1) controlling the degree of non-linear warping.
        superposition: Fraction (0 to 1) controlling how many signals are active per sample.
        non_orthogonal: Fraction (0 to 1) controlling the deviation from an orthogonal basis.
                        0 means the signals are forced to be fully orthogonal (if possible),
                        1 uses the generated signals as-is.
        hierarchical: Fraction (0 to 1) controlling the degree of hierarchical structure.
                        0 yields independent signals; 1 forces signals to come from a few clusters.

    Returns:
        A data tensor of shape (n_samples, activation_size)
    """
    # Validate basic parameters
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if activation_size <= 0:
        raise ValueError("activation_size must be positive")
    if n_signals <= 0:
        raise ValueError("n_signals must be positive")
    if signal_strength < 0:
        raise ValueError("signal_strength must be non-negative")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    for param_name, param_value in [
        ("non_euclidean", non_euclidean),
        ("superposition", superposition),
        ("non_orthogonal", non_orthogonal),
        ("hierarchical", hierarchical),
    ]:
        if not (0.0 <= param_value <= 1.0):
            raise ValueError(f"{param_name} must be between 0 and 1")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Step 1: Generate base random signals and normalize
    random_signals = torch.randn(n_signals, activation_size)
    random_signals = random_signals / random_signals.norm(dim=1, keepdim=True)

    # Step 2: Introduce hierarchical structure if requested
    if hierarchical > 0:
        n_clusters = max(1, int(n_signals * hierarchical))
        # Generate cluster centers (normalized)
        cluster_centers = torch.randn(n_clusters, activation_size)
        cluster_centers = cluster_centers / cluster_centers.norm(dim=1, keepdim=True)
        hierarchical_signals = []
        # For each signal, assign a cluster and add a small noise offset
        for _ in range(n_signals):
            cluster_idx = random.randint(0, n_clusters - 1)
            # You can adjust noise_scale to control how tightly each signal adheres to its cluster center
            noise_scale = 0.1
            sig = cluster_centers[cluster_idx] + torch.randn(activation_size) * noise_scale
            norm = sig.norm()
            sig = sig if norm == 0 else sig / norm
            hierarchical_signals.append(sig)
        stacked_signals = torch.stack(hierarchical_signals)
        # Interpolate between independent signals and the hierarchical version
        signals = (1 - hierarchical) * random_signals + hierarchical * stacked_signals
        signals = signals / signals.norm(dim=1, keepdim=True)
    else:
        signals = random_signals

    # Step 3: Control non-orthogonality.
    # If non_orthogonal < 1, we interpolate signals with an orthogonalized version.
    if non_orthogonal < 1.0:
        if n_signals <= activation_size:
            # Compute a QR decomposition (orthogonalization)
            # Note: torch.linalg.qr is preferred if available.
            q, _ = torch.linalg.qr(signals)
            orthogonal_signals = q
        else:
            # If you have more signals than the dimension, full orthogonalization is impossible.
            orthogonal_signals = signals
        # Interpolate between the original and the orthogonal signals.
        signals = (non_orthogonal * signals) + ((1 - non_orthogonal) * orthogonal_signals)
        signals = signals / signals.norm(dim=1, keepdim=True)

    # Step 4: Create coefficient matrix.
    # Each sample gets a coefficient per signal, modulated by signal_strength.
    coeffs = torch.randn(n_samples, n_signals) * signal_strength

    # Control superposition: if superposition < 1, allow only a subset of signals
    if superposition < 1.0:
        # Create a mask with probability 'superposition' for each signal in each sample
        mask = (torch.rand(n_samples, n_signals) < superposition).float()
        # If a sample has no active signal, force one to be active
        for i in range(n_samples):
            if mask[i].sum() == 0:
                idx = random.randint(0, n_signals - 1)
                mask[i, idx] = 1.0
        coeffs = coeffs * mask

    # Step 5: Compute the data from the linear combination of signals.
    data = torch.matmul(coeffs, signals)

    # Step 6: Add noise component.
    data += torch.randn(n_samples, activation_size) * noise_level

    # Step 7: Apply a non-Euclidean transform if requested.
    # Here we warp the data by mixing in a sine nonlinearity.
    if non_euclidean > 0:
        data = (1 - non_euclidean) * data + non_euclidean * torch.sin(data)

    return data


def load_synthetic_data(cfg: Dict[str, Any]) -> tuple[CustomDataLoader, int]:
    """
    Generate synthetic data and load it using CustomDataLoader.

    Expected cfg keys include:
        - "num_tokens": Number of samples.
        - "act_size": Activation size (feature dimension).
        - "n_signals": Number of signal components.
        - "signal_strength": The strength to scale the signal contribution.
        - "noise_level": The noise level to add.
        - "seed": Random seed.
        - "batch_size": Batch size for the data loader.
        - "device": Device on which the data should be placed.
        - (Optionally) "non_euclidean", "superposition", "non_orthogonal", "hierarchical" in [0,1]

    Returns:
        Tuple of (CustomDataLoader, activation_size)
    """
    # Validate required parameters.
    required_params = [
        "num_tokens",
        "act_size",
        "n_signals",
        "signal_strength",
        "noise_level",
        "seed",
        "batch_size",
        "device",
    ]
    for param in required_params:
        if param not in cfg:
            raise ValueError(f"Missing required config parameter: {param}")

    # Extract or default additional parameters controlling the new effects.
    non_euclidean = cfg.get("non_euclidean", 0.0)
    superposition = cfg.get("superposition", 1.0)
    non_orthogonal = cfg.get("non_orthogonal", 1.0)
    hierarchical = cfg.get("hierarchical", 0.0)

    data = generate_synthetic_data(
        n_samples=cfg["num_tokens"],
        activation_size=cfg["act_size"],
        n_signals=cfg["n_signals"],
        signal_strength=cfg["signal_strength"],
        noise_level=cfg["noise_level"],
        seed=cfg["seed"],
        non_euclidean=non_euclidean,
        superposition=superposition,
        non_orthogonal=non_orthogonal,
        hierarchical=hierarchical,
    )
    data_loader = CustomDataLoader(data, cfg)
    return data_loader, cfg["act_size"]
