import time
import torch
from neuralop.models import FNO
from src.dataset import Normalizer, build_history_input


def run_closed_loop_rollout(
    model: FNO,
    data: torch.Tensor,
    dt: float,
    num_delays: int,
    vel_normalizer: Normalizer,
    target_normalizer: Normalizer,
    device: str = "cpu",
) -> dict:
    """Ejecuta rollout autoregresivo con Euler explícito y devuelve trayectorias + métricas."""
    model.eval()
    nt = data.shape[1]
    start_idx = num_delays
    end_idx = nt - 1

    u_rollout = data[0].clone().to(device)
    v_rollout = data[1].clone().to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(start_idx, end_idx):
            vel_history = torch.stack(
                [
                    u_rollout[k - num_delays : k + 1],
                    v_rollout[k - num_delays : k + 1],
                ],
                dim=0,
            )
            params = data[2:5, k, :, :].to(device)

            x_in = build_history_input(vel_history, params, vel_normalizer).unsqueeze(0).to(device)
            pred_norm = model(x_in).squeeze(0)
            ds_pred = target_normalizer.denormalize(pred_norm)

            u_rollout[k + 1] = u_rollout[k] + dt * ds_pred[0]
            v_rollout[k + 1] = v_rollout[k] + dt * ds_pred[1]

    elapsed_sec = time.perf_counter() - t0

    # Error relativo L2 en cada instante k a partir de start_idx
    u_real = data[0].to(device)
    v_real = data[1].to(device)

    rel_l2_errors = []
    for k in range(start_idx, nt):
        diff_norm = torch.linalg.norm(
            torch.stack([u_rollout[k] - u_real[k], v_rollout[k] - v_real[k]])
        )
        ref_norm = torch.linalg.norm(torch.stack([u_real[k], v_real[k]])) + 1e-12
        rel_l2_errors.append((diff_norm / ref_norm).item())

    return {
        "u_rollout": u_rollout.cpu(),
        "v_rollout": v_rollout.cpu(),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "rollout_time_sec": elapsed_sec,
        "rel_l2_errors": rel_l2_errors,
    }