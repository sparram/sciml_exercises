import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom


def get_paraview_colormap(name: str = "cool_to_warm") -> LinearSegmentedColormap:
    """Devuelve un colormap inspirado en ParaView (Cool to Warm por defecto)."""
    if name == "cool_to_warm":
        colors = [
            (0.231, 0.298, 0.753),  # Azul frío (ParaView Cool to Warm)
            (0.865, 0.865, 0.865),  # Gris neutro intermedio
            (0.706, 0.016, 0.150),  # Rojo cálido
        ]
        return LinearSegmentedColormap.from_list("paraview_cool_to_warm", colors, N=256)
    raise ValueError(f"Colormap de ParaView no soportado: {name}")


def upsample_field_2d(field: np.ndarray, target_ny: int, target_nx: int, order: int = 3) -> np.ndarray:
    """Interpola un campo 2D [Ny, Nx] a una malla más fina [target_ny, target_nx]."""
    ny, nx = field.shape
    if (ny, nx) == (target_ny, target_nx):
        return field
    zoom_factors = (target_ny / ny, target_nx / nx)
    return zoom(field, zoom=zoom_factors, order=order)


def compute_animated_lic(
    u: np.ndarray,
    v: np.ndarray,
    noise: np.ndarray,
    phase: float,
    x_span: float = 2.0,
    y_span: float = 1.0,
    length: int = 16,
    step_size: float = 0.008,
) -> np.ndarray:
    """Calcula textura LIC con onda viajera para animar la dirección del flujo."""
    ny, nx = noise.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx].astype(np.float64)

    mag = np.hypot(u, v) + 1e-12
    u_norm = u / mag
    v_norm = v / mag

    u_px = u_norm * (nx / x_span) * step_size
    v_px = v_norm * (ny / y_span) * step_size

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


def save_lic_gif(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    times: np.ndarray,
    u_real_all: np.ndarray,
    v_real_all: np.ndarray,
    u_pred_all: np.ndarray,
    v_pred_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    cfg: dict,
    output_path: str = "outputs/double_gyre_lic.gif",
    fps: int = 15,
    viz_res_x: int | None = None,
    viz_res_y: int | None = None,
    use_paraview_cmap: bool = True,
) -> None:
    """Genera y guarda un GIF comparativo con LIC animado en malla fina y colormap tipo ParaView."""
    v_cfg = cfg["viz"]
    p_cfg = cfg["physics"]

    x_span = float(x_coords[-1] - x_coords[0])
    y_span = float(y_coords[-1] - y_coords[0])

    target_nx = viz_res_x if viz_res_x is not None else int(p_cfg["res_x"])
    target_ny = viz_res_y if viz_res_y is not None else int(p_cfg["res_y"])

    fine_x = np.linspace(float(x_coords[0]), float(x_coords[-1]), target_nx)
    fine_y = np.linspace(float(y_coords[0]), float(y_coords[-1]), target_ny)
    fine_gx, fine_gy = np.meshgrid(fine_x, fine_y)

    rng = np.random.default_rng(v_cfg["noise_seed"])
    base_noise = rng.random((target_ny, target_nx))

    cmap = get_paraview_colormap("cool_to_warm") if use_paraview_cmap else "gray"

    indices = list(range(start_idx, end_idx))
    frames_real = []
    frames_pred = []

    for i in indices:
        u_r = upsample_field_2d(u_real_all[i], target_ny, target_nx, order=3)
        v_r = upsample_field_2d(v_real_all[i], target_ny, target_nx, order=3)
        u_p = upsample_field_2d(u_pred_all[i], target_ny, target_nx, order=3)
        v_p = upsample_field_2d(v_pred_all[i], target_ny, target_nx, order=3)

        phase = -float(times[i]) * float(v_cfg["phase_speed"])
        frames_real.append(
            compute_animated_lic(
                u_r, v_r, base_noise, phase,
                x_span=x_span, y_span=y_span,
                length=v_cfg["lic_length"], step_size=v_cfg["lic_step_size"],
            )
        )
        frames_pred.append(
            compute_animated_lic(
                u_p, v_p, base_noise, phase,
                x_span=x_span, y_span=y_span,
                length=v_cfg["lic_length"], step_size=v_cfg["lic_step_size"],
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].pcolormesh(fine_gx, fine_gy, frames_real[0], shading="auto", cmap=cmap, vmin=0, vmax=1)
    title0 = axes[0].set_title(f"Real Fluid | t = {times[indices[0]]:.2f}s", fontweight="bold")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(fine_gx, fine_gy, frames_pred[0], shading="auto", cmap=cmap, vmin=0, vmax=1)
    title1 = axes[1].set_title(f"FNO Prediction | t = {times[indices[0]]:.2f}s", fontweight="bold")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()

    def update(frame_idx: int):
        im0.set_array(frames_real[frame_idx].ravel())
        im1.set_array(frames_pred[frame_idx].ravel())
        t_val = times[indices[frame_idx]]
        title0.set_text(f"Real Fluid | t = {t_val:.2f}s")
        title1.set_text(f"FNO Prediction | t = {t_val:.2f}s")
        return im0, im1, title0, title1

    anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF guardado en: {output_path}")