import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class CustomDataLoader:
    """Simple data loader for custom data"""

    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg
        self.batch_size = cfg["batch_size"]
        self.device = cfg["device"]
        self.dataloader = self._create_dataloader()
        self.iterator = iter(self.dataloader)

    def _create_dataloader(self):
        """Create a PyTorch DataLoader from the data"""
        dataset = TensorDataset(self.data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def next_batch(self):
        """Get the next batch of data"""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch[0].to(self.device)


def generate_synthetic_data(
    n_samples=10000,
    activation_size=512,
    n_signals=10,
    signal_strength=1.0,
    noise_level=0.1,
    seed=42,
):
    """Generate synthetic data with signal and noise components"""
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


def load_synthetic_data(cfg):
    """Generate synthetic data and load it using CustomDataLoader"""
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
