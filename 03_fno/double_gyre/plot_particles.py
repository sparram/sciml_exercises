from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import yaml

from src.physics import generate_double_gyre_data
from src.dataset import Normalizer
from src.model import build_fno_model
from src.rollout import run_closed_loop_rollout


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_field_bilinear(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_field: np.ndarray,
    v_field: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolación bilineal de (u, v) en posiciones de partículas (px, py) con malla (ny, nx)."""
    ny, nx = u_field.shape
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    px_c = np.clip(px, x_min, x_max)
    py_c = np.clip(py, y_min, y_max)

    tx = np.clip(((px_c - x_min) / (x_max - x_min)) * (nx - 1), 0.0, nx - 1.000001)
    ty = np.clip(((py_c - y_min) / (y_max - y_min)) * (ny - 1), 0.0, ny - 1.000001)

    ix = np.floor(tx).astype(np.int64)
    iy = np.floor(ty).astype(np.int64)
    fx = tx - ix
    fy = ty - iy

    u00 = u_field[iy, ix]
    u10 = u_field[iy, ix + 1]
    u01 = u_field[iy + 1, ix]
    u11 = u_field[iy + 1, ix + 1]

    v00 = v_field[iy, ix]
    v10 = v_field[iy, ix + 1]
    v01 = v_field[iy + 1, ix]
    v11 = v_field[iy + 1, ix + 1]

    u_p = (
        u00 * (1 - fx) * (1 - fy)
        + u10 * fx * (1 - fy)
        + u01 * (1 - fx) * fy
        + u11 * fx * fy
    )
    v_p = (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )
    return u_p, v_p
    
def advect_particles_rk4(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    times: np.ndarray,
    u_all: np.ndarray,
    v_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    n_x: int = 28,
    n_y: int = 14,
    substeps: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advecta una rejilla de partículas desde start_idx hasta end_idx usando RK4.
    Retorna:
      traj_x, traj_y: forma (n_steps, n_particles)
      color_init: coordenada x inicial para colorear según el giro de origen
    """
    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]

    margin_x = 0.04 * (x_max - x_min)
    margin_y = 0.04 * (y_max - y_min)
    gx, gy = np.meshgrid(
        np.linspace(x_min + margin_x, x_max - margin_x, n_x),
        np.linspace(y_min + margin_y, y_max - margin_y, n_y),
        indexing="xy",
    )
    px = gx.ravel().copy()
    py = gy.ravel().copy()
    color_init = px.copy()

    n_steps = end_idx - start_idx + 1
    traj_x = np.zeros((n_steps, len(px)), dtype=np.float64)
    traj_y = np.zeros((n_steps, len(py)), dtype=np.float64)
    traj_x[0] = px
    traj_y[0] = py

    for k in range(n_steps - 1):
        t_idx = start_idx + k
        dt_step = float(times[t_idx + 1] - times[t_idx])
        h = dt_step / substeps

        u0, v0 = u_all[t_idx], v_all[t_idx]
        u1, v1 = u_all[t_idx + 1], v_all[t_idx + 1]

        for s in range(substeps):
            alpha0 = s / substeps
            alpha_mid = (s + 0.5) / substeps
            alpha1 = (s + 1.0) / substeps

            u_a = (1 - alpha0) * u0 + alpha0 * u1
            v_a = (1 - alpha0) * v0 + alpha0 * v1
            u_m = (1 - alpha_mid) * u0 + alpha_mid * u1
            v_m = (1 - alpha_mid) * v0 + alpha_mid * v1
            u_b = (1 - alpha1) * u0 + alpha1 * u1
            v_b = (1 - alpha1) * v0 + alpha1 * v1

            k1x, k1y = sample_field_bilinear(x_grid, y_grid, u_a, v_a, px, py)
            k2x, k2y = sample_field_bilinear(
                x_grid, y_grid, u_m, v_m, px + 0.5 * h * k1x, py + 0.5 * h * k1y
            )
            k3x, k3y = sample_field_bilinear(
                x_grid, y_grid, u_m, v_m, px + 0.5 * h * k2x, py + 0.5 * h * k2y
            )
            k4x, k4y = sample_field_bilinear(
                x_grid, y_grid, u_b, v_b, px + h * k3x, py + h * k3y
            )

            px = np.clip(px + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x), x_min, x_max)
            py = np.clip(py + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y), y_min, y_max)

        traj_x[k + 1] = px
        traj_y[k + 1] = py

    return traj_x, traj_y, color_init


def save_particles_gif(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    times: np.ndarray,
    start_idx: int,
    traj_x_real: np.ndarray,
    traj_y_real: np.ndarray,
    traj_x_pred: np.ndarray,
    traj_y_pred: np.ndarray,
    color_init: np.ndarray,
    output_path: str = "outputs/double_gyre_particles.gif",
    fps: int = 15,
    trail_len: int = 12,
) -> None:
    """Genera un GIF comparativo de advección de partículas con estela corta."""
    n_frames = traj_x_real.shape[0]
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax_real, ax_pred = axes

    for ax in (ax_real, ax_pred):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    scat_real = ax_real.scatter(
        traj_x_real[0], traj_y_real[0], c=color_init, cmap="coolwarm", s=16, alpha=0.9
    )
    scat_pred = ax_pred.scatter(
        traj_x_pred[0], traj_y_pred[0], c=color_init, cmap="coolwarm", s=16, alpha=0.9
    )

    lines_real = [
        ax_real.plot([], [], linewidth=1.0, alpha=0.45, color="steelblue")[0]
        for _ in range(traj_x_real.shape[1])
    ]
    lines_pred = [
        ax_pred.plot([], [], linewidth=1.0, alpha=0.45, color="steelblue")[0]
        for _ in range(traj_x_pred.shape[1])
    ]

    title_real = ax_real.set_title(f"Real Flow | t = {times[start_idx]:.2f}s")
    title_pred = ax_pred.set_title(f"FNO Predicted Flow | t = {times[start_idx]:.2f}s")

    plt.tight_layout()

    def update(frame: int):
        t_val = times[start_idx + frame]
        title_real.set_text(f"Real Flow | t = {t_val:.2f}s")
        title_pred.set_text(f"FNO Predicted Flow | t = {t_val:.2f}s")

        scat_real.set_offsets(np.column_stack([traj_x_real[frame], traj_y_real[frame]]))
        scat_pred.set_offsets(np.column_stack([traj_x_pred[frame], traj_y_pred[frame]]))

        w_start = max(0, frame - trail_len)
        for i in range(traj_x_real.shape[1]):
            lines_real[i].set_data(traj_x_real[w_start : frame + 1, i], traj_y_real[w_start : frame + 1, i])
            lines_pred[i].set_data(traj_x_pred[w_start : frame + 1, i], traj_y_pred[w_start : frame + 1, i])

        return [scat_real, scat_pred, title_real, title_pred, *lines_real, *lines_pred]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF de partículas guardado en: {output_path}")


def main() -> None:
    cfg = load_config()
    num_delays = cfg["dataset"]["num_delays"]
    device = cfg["model"]["device"]
    weights_path = cfg["model"]["weights_path"]
    fps = int(cfg.get("viz", {}).get("fps", 15))

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    sim = generate_double_gyre_data(cfg)
    data = sim["data"]
    targets = sim["targets"]

    vel_norm = Normalizer(mean=data[0:2].mean(), std=data[0:2].std())
    target_norm = Normalizer(mean=targets.mean(), std=targets.std())

    model = build_fno_model(num_delays=num_delays, cfg=cfg)
    state_dict = torch.load(weights_path, weights_only=False, map_location=device)
    model.load_state_dict(state_dict)

    rollout_res = run_closed_loop_rollout(
        model=model,
        data=data,
        dt=sim["dt"],
        num_delays=num_delays,
        vel_normalizer=vel_norm,
        target_normalizer=target_norm,
        device=device,
    )

    start_idx = rollout_res["start_idx"]
    end_idx = rollout_res["end_idx"]

    # data[0] es u real en todo el tiempo, data[1] es v real en todo el tiempo
    # Ojo: verifica si u_rollout / v_rollout tienen longitud completa o solo desde start_idx
    u_real_all = data[0].numpy()
    v_real_all = data[1].numpy()

    u_pred_raw = rollout_res["u_rollout"].numpy()
    v_pred_raw = rollout_res["v_rollout"].numpy()

    if u_pred_raw.shape[0] == u_real_all.shape[0]:
        u_pred_all = u_pred_raw
        v_pred_all = v_pred_raw
    else:
        u_pred_all = u_real_all.copy()
        v_pred_all = v_real_all.copy()
        u_pred_all[start_idx : end_idx + 1] = u_pred_raw[: end_idx - start_idx + 1]
        v_pred_all[start_idx : end_idx + 1] = v_pred_raw[: end_idx - start_idx + 1]

    tx_real, ty_real, c_init = advect_particles_rk4(
        sim["x"], sim["y"], sim["times"], u_real_all, v_real_all, start_idx, end_idx
    )
    tx_pred, ty_pred, _ = advect_particles_rk4(
        sim["x"], sim["y"], sim["times"], u_pred_all, v_pred_all, start_idx, end_idx
    )

    save_particles_gif(
        x_grid=sim["x"],
        y_grid=sim["y"],
        times=sim["times"],
        start_idx=start_idx,
        traj_x_real=tx_real,
        traj_y_real=ty_real,
        traj_x_pred=tx_pred,
        traj_y_pred=ty_pred,
        color_init=c_init,
        output_path=str(out_dir / "double_gyre_particles.gif"),
        fps=fps,
    )


if __name__ == "__main__":
    main()