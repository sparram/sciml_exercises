import numpy as np
import torch


class DoubleGyre2D:
    """Campo vectorial 2D del sistema Double Gyre."""

    def __init__(self, a: float = 0.1, eps: float = 0.35, omega: float = np.pi / 5):
        self.A = a
        self.EPS = eps
        self.OMEGA = omega

    def sample(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Evalúa el campo de velocidad (u, v) en el instante t."""
        sin_wt = np.sin(self.OMEGA * t)
        f = self.EPS * sin_wt * x * x + (1.0 - 2.0 * self.EPS * sin_wt) * x
        df_dx = 2.0 * self.EPS * sin_wt * x - 2.0 * self.EPS * sin_wt + 1.0

        u = -self.A * np.pi * np.sin(np.pi * f) * np.cos(np.pi * y)
        v = self.A * np.pi * df_dx * np.cos(np.pi * f) * np.sin(np.pi * y)
        return u, v

    def sample_dt(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Evalúa las derivadas temporales (du/dt, dv/dt) en el instante t."""
        sin_wt = np.sin(self.OMEGA * t)
        cos_wt = np.cos(self.OMEGA * t)

        f = self.EPS * sin_wt * x * x + (1.0 - 2.0 * self.EPS * sin_wt) * x
        df_dx = 2.0 * self.EPS * sin_wt * x - 2.0 * self.EPS * sin_wt + 1.0
        df_dt = self.EPS * self.OMEGA * cos_wt * x * x - 2.0 * self.EPS * self.OMEGA * cos_wt * x

        du_dt = -self.A * np.pi * np.pi * df_dt * np.cos(np.pi * f) * np.cos(np.pi * y)
        dv_dt = (
            self.A * np.pi * (2.0 * self.EPS * self.OMEGA * cos_wt * x - 2.0 * self.EPS * self.OMEGA * cos_wt)
            * np.cos(np.pi * f)
            * np.sin(np.pi * y)
            - self.A * np.pi * np.pi * df_dx * df_dt * np.sin(np.pi * f) * np.sin(np.pi * y)
        )
        return du_dt, dv_dt


def generate_double_gyre_data(cfg: dict) -> dict:
    """Genera la malla espacial/temporal, tensores de entrada (u, v, A, eps, omega) y targets (du/dt, dv/dt)."""
    p_cfg = cfg["physics"]
    omega = np.pi * p_cfg["omega_factor"]

    dg = DoubleGyre2D(a=p_cfg["a"], eps=p_cfg["eps"], omega=omega)

    x = np.linspace(0.0, p_cfg["x_max"], p_cfg["res_x"])
    y = np.linspace(0.0, p_cfg["y_max"], p_cfg["res_y"])
    grid_x, grid_y = np.meshgrid(x, y)

    times = np.arange(0.0, p_cfg["t_max"], p_cfg["dt"])
    nt = len(times)
    ny, nx = grid_y.shape

    u_arr = torch.zeros((nt, ny, nx), dtype=torch.float32)
    v_arr = torch.zeros((nt, ny, nx), dtype=torch.float32)
    du_arr = torch.zeros_like(u_arr)
    dv_arr = torch.zeros_like(v_arr)

    for k, t_eval in enumerate(times):
        u, v = dg.sample(grid_x, grid_y, float(t_eval))
        du, dv = dg.sample_dt(grid_x, grid_y, float(t_eval))

        u_arr[k] = torch.from_numpy(u).float()
        v_arr[k] = torch.from_numpy(v).float()
        du_arr[k] = torch.from_numpy(du).float()
        dv_arr[k] = torch.from_numpy(dv).float()

    a_ch = torch.full_like(u_arr, float(p_cfg["a"]))
    eps_ch = torch.full_like(u_arr, float(p_cfg["eps"]))
    omega_ch = torch.full_like(u_arr, float(omega))

    # data: [5, Nt, H, W] -> canales (u, v, A, eps, omega)
    # targets: [2, Nt, H, W] -> canales (du/dt, dv/dt)
    data = torch.stack([u_arr, v_arr, a_ch, eps_ch, omega_ch], dim=0)
    targets = torch.stack([du_arr, dv_arr], dim=0)

    return {
        "x": x,
        "y": y,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "times": times,
        "dt": float(p_cfg["dt"]),
        "data": data,
        "targets": targets,
    }