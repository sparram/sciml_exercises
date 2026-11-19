from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Normalizer:
    """Almacena media y desviación estándar para normalizar y desnormalizar tensores."""
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * (self.std + self.eps) + self.mean


def build_history_input(
    vel_history: torch.Tensor,
    params: torch.Tensor,
    vel_normalizer: Normalizer,
) -> torch.Tensor:
    """
    Construye el tensor de entrada para un instante dado.

    Args:
        vel_history: [2, num_delays + 1, H, W] con historia de (u, v).
        params: [3, H, W] con parámetros físicos (A, eps, omega).
        vel_normalizer: Normalizer de velocidades.

    Returns:
        Tensor de forma [2 * (num_delays + 1) + 3, H, W] (con num_delays=10 -> [25, H, W]).
    """
    vel_norm = vel_normalizer.normalize(vel_history)
    _, _, height, width = vel_norm.shape
    vel_flat = vel_norm.permute(1, 0, 2, 3).reshape(-1, height, width)
    return torch.cat([vel_flat, params], dim=0)


def create_dataloader(
    data: torch.Tensor,
    targets: torch.Tensor,
    num_delays: int = 10,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tuple[DataLoader, Normalizer, Normalizer]:
    """Crea el DataLoader con ventanas de retardo y devuelve los normalizadores."""
    vel_normalizer = Normalizer(mean=data[0:2].mean(), std=data[0:2].std())
    target_normalizer = Normalizer(mean=targets.mean(), std=targets.std())

    nt = data.shape[1]
    x_samples = []
    y_samples = []

    for k in range(num_delays, nt - 1):
        vel_history = data[0:2, k - num_delays : k + 1, :, :]
        params = data[2:5, k, :, :]
        x_in = build_history_input(vel_history, params, vel_normalizer)

        ds_target_norm = target_normalizer.normalize(targets[:, k, :, :])

        x_samples.append(x_in)
        y_samples.append(ds_target_norm)

    x_tensor = torch.stack(x_samples, dim=0)
    y_tensor = torch.stack(y_samples, dim=0)

    dataset = TensorDataset(x_tensor, y_tensor)
    effective_batch_size = len(dataset) if batch_size <= 0 else min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=shuffle)

    return loader, vel_normalizer, target_normalizer