import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import clear_output


def compute_animated_lic(
    u: np.ndarray,
    v: np.ndarray,
    noise: np.ndarray,
    phase: float,
    length: int = 16,
    step_size: float = 0.008,
) -> np.ndarray:
    """Calcula textura LIC con onda viajera para animar la dirección del flujo."""
    ny, nx = noise.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx].astype(np.float64)

    mag = np.hypot(u, v) + 1e-12
    u_norm = u / mag
    v_norm = v / mag

    u_px = u_norm * (nx / 2.0) * step_size
    v_px = v_norm * (ny / 1.0) * step_size

    w0 = 0.5 * (1.0 + np.cos(phase))
    acc = noise * w0
    weight_sum = w0

    for direction in (1.0, -1.0):
        cx, cy = x_grid.copy(), y_grid.copy()
        for step in range(1, length + 1):
            ix = np.clip(np.round(cx).astype(int), 0, nx - 1)
            iy = np.clip(np.round(cy).astype(int), 0, ny - 1)

            mx = np.clip(np.round(cx + 0.5 * direction * u_px[iy, ix]).astype(int), 0, nx - 1)
            my = np.clip(np.round(cy + 0.5 * direction * v_px[iy, ix]).astype(int), 0, ny - 1)

            cx += direction * u_px[my, mx]
            cy += direction * v_px[my, mx]

            sx = np.clip(np.round(cx).astype(int), 0, nx - 1)
            sy = np.clip(np.round(cy).astype(int), 0, ny - 1)

            s = direction * step * 0.45
            hanning = 0.5 * (1.0 + np.cos(np.pi * step / (length + 1)))
            weight = hanning * 0.5 * (1.0 + np.cos(s - phase))

            acc += noise[sy, sx] * weight
            weight_sum += weight

    lic = acc / (weight_sum + 1e-12)
    lic = (lic - lic.min()) / (lic.max() - lic.min() + 1e-12)

    mag_norm = mag / (mag.max() + 1e-12)
    return lic * (0.35 + 0.65 * np.sqrt(mag_norm))


def animate_velocity_comparison(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    times: np.ndarray,
    u_real_all: np.ndarray,
    u_pred_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    in_notebook: bool = True,
) -> None:
    """Muestra animación comparativa de la componente u (Real vs. FNO Lazo Cerrado)."""
    t0 = time.time()
    for i in range(start_idx, end_idx):
        t_eval = times[i]
        if in_notebook:
            clear_output(wait=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].pcolormesh(grid_x, grid_y, u_real_all[i], shading="auto", cmap="coolwarm")
        axes[0].set_title(f"REAL (Simulación) | t = {t_eval:.2f}s", fontweight="bold")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(grid_x, grid_y, u_pred_all[i], shading="auto", cmap="coolwarm")
        axes[1].set_title(f"FNO (Lazo Cerrado) | t = {t_eval:.2f}s", fontweight="bold")
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()
        if in_notebook:
            plt.pause(0.01)

    print(f"Animación u completada en {time.time() - t0:.2f}s.")


def save_lic_gif(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    times: np.ndarray,
    u_real_all: np.ndarray,
    v_real_all: np.ndarray,
    u_pred_all: np.ndarray,
    v_pred_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    cfg: dict,
    output_path: str = "double_gyre_lic.gif",
    fps: int = 15,
) -> None:
    """Genera y guarda un GIF comparativo con LIC animado (Real vs. FNO Lazo Cerrado)."""
    v_cfg = cfg["viz"]
    ny, nx = u_real_all.shape[1], u_real_all.shape[2]

    rng = np.random.default_rng(v_cfg["noise_seed"])
    base_noise = rng.random((ny, nx))

    indices = list(range(start_idx, end_idx))
    frames_real = []
    frames_pred = []

    for i in indices:
        phase = -float(times[i]) * float(v_cfg["phase_speed"])
        frames_real.append(
            compute_animated_lic(
                u_real_all[i], v_real_all[i], base_noise, phase,
                length=v_cfg["lic_length"], step_size=v_cfg["lic_step_size"],
            )
        )
        frames_pred.append(
            compute_animated_lic(
                u_pred_all[i], v_pred_all[i], base_noise, phase,
                length=v_cfg["lic_length"], step_size=v_cfg["lic_step_size"],
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].pcolormesh(grid_x, grid_y, frames_real[0], shading="auto", cmap="gray", vmin=0, vmax=1)
    title0 = axes[0].set_title(f"REAL | t = {times[indices[0]]:.2f}s", fontweight="bold")

    im1 = axes[1].pcolormesh(grid_x, grid_y, frames_pred[0], shading="auto", cmap="gray", vmin=0, vmax=1)
    title1 = axes[1].set_title(f"FNO (Lazo Cerrado) | t = {times[indices[0]]:.2f}s", fontweight="bold")

    plt.tight_layout()

    def update(frame_idx: int):
        im0.set_array(frames_real[frame_idx].ravel())
        im1.set_array(frames_pred[frame_idx].ravel())
        t_val = times[indices[frame_idx]]
        title0.set_text(f"REAL | t = {t_val:.2f}s")
        title1.set_text(f"FNO (Lazo Cerrado) | t = {t_val:.2f}s")
        return im0, im1, title0, title1

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF guardado en: {output_path}")