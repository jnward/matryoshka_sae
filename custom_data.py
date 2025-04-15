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
    n_samples: int = 10000,
    activation_size: int = 512,
    n_signals: int = 10,
    signal_strength: float = 1.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate synthetic data with signal and noise components.

    Args:
        n_samples: Number of samples to generate
        activation_size: Size of each activation vector
        n_signals: Number of signal components
        signal_strength: Strength of signal components
        noise_level: Level of noise in the data
        seed: Random seed for reproducibility

    Returns:
        Generated data tensor of shape (n_samples, activation_size)
    """
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate random signal directions
    signals = torch.randn(n_signals, activation_size)
    signals = signals / torch.norm(signals, dim=1, keepdim=True)

    # Generate random signal coefficients
    coeffs = torch.randn(n_samples, n_signals) * signal_strength

    # Generate data
    data = torch.matmul(coeffs, signals)  # Signal component
    data += torch.randn(n_samples, activation_size) * noise_level  # Noise component

    return data


def load_synthetic_data(cfg: Dict[str, Any]) -> tuple[CustomDataLoader, int]:
    """
    Generate synthetic data and load it using CustomDataLoader.

    Args:
        cfg: Configuration dictionary containing data generation parameters

    Returns:
        Tuple of (data_loader, activation_size)
    """
    required_params = ["n_samples", "activation_size", "n_signals", "signal_strength", "noise_level", "seed"]
    for param in required_params:
        if param not in cfg:
            raise ValueError(f"Missing required config parameter: {param}")

    data = generate_synthetic_data(
        n_samples=cfg["n_samples"],
        activation_size=cfg["activation_size"],
        n_signals=cfg["n_signals"],
        signal_strength=cfg["signal_strength"],
        noise_level=cfg["noise_level"],
        seed=cfg["seed"],
    )
    data_loader = CustomDataLoader(data, cfg)
    return data_loader, cfg["activation_size"]


if __name__ == "__main__":
    # Example usage
    cfg = {
        "n_samples": 10000,
        "activation_size": 512,
        "n_signals": 10,
        "signal_strength": 1.0,
        "noise_level": 0.1,
        "seed": 42,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    data_loader, act_size = load_synthetic_data(cfg)

    # Test the data loader
    batch = data_loader.next_batch()
    print(f"Batch shape: {batch.shape}")
    print(f"Batch mean: {batch.mean():.4f}")
    print(f"Batch std: {batch.std():.4f}")
