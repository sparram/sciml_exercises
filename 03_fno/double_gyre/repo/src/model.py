import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neuralop.models import FNO
import tqdm


def build_fno_model(num_delays: int, cfg: dict) -> FNO:
    """Instancia el modelo Fourier Neural Operator según la configuración."""
    m_cfg = cfg["model"]
    in_channels = 2 * (num_delays + 1) + 3  # (u, v) por delay + (A, eps, omega)

    model = FNO(
        n_modes=tuple(m_cfg["n_modes"]),
        in_channels=in_channels,
        out_channels=2,  # du/dt, dv/dt
        hidden_channels=m_cfg["hidden_channels"],
        n_layers=m_cfg["n_layers"],
    )
    return model.to(m_cfg["device"])


def train_fno(model: FNO, dataloader: DataLoader, cfg: dict) -> list[float]:
    """Entrena el modelo FNO y guarda los mejores pesos en disco."""
    m_cfg = cfg["model"]
    device = m_cfg["device"]
    epochs = m_cfg["epochs"]

    optimizer = torch.optim.Adam(model.parameters(), lr=float(m_cfg["learning_rate"]))
    criterion = nn.MSELoss()

    best_loss = float("inf")
    loss_history = []

    pbar = tqdm.tqdm(range(epochs), desc="Training FNO")
    for _ in pbar:
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

        epoch_loss = running_loss / max(total_samples, 1)
        loss_history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), m_cfg["weights_path"])

        pbar.set_postfix(loss=f"{epoch_loss:.6f}", best=f"{best_loss:.6f}")

    return loss_history