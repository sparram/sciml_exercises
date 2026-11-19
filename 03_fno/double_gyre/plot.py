from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml

from src.physics import generate_double_gyre_data
from src.dataset import Normalizer
from src.model import build_fno_model
from src.rollout import run_closed_loop_rollout
from src.viz import save_lic_gif


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_rollout_error(
    times: np.ndarray,
    rel_l2_errors: list[float],
    start_idx: int,
    save_path: str = "outputs/rollout_error.png",
) -> None:
    """Genera y guarda la evolución temporal del error relativo L2 en el rollout."""
    t_eval = times[start_idx:]
    err_arr = np.asarray(rel_l2_errors, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_eval, err_arr, linewidth=2.0, label="Closed-loop FNO (Relative $L_2$ error)")
    ax.set_xlabel("Time $t$ [s]")
    ax.set_ylabel("Relative $L_2$ Error")
    ax.set_title("Closed-Loop Rollout Error Growth (Double Gyre)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Gráfica de error guardada en: {save_path}")


def main() -> None:
    cfg = load_config()
    num_delays = cfg["dataset"]["num_delays"]
    device = cfg["model"]["device"]
    weights_path = cfg["model"]["weights_path"]
    v_cfg = cfg.get("viz", {})

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Generar datos físicos de referencia
    sim = generate_double_gyre_data(cfg)
    data = sim["data"]
    targets = sim["targets"]

    # Estadísticas de normalización (idealmente guardadas junto al checkpoint)
    vel_norm = Normalizer(mean=data[0:2].mean(), std=data[0:2].std())
    target_norm = Normalizer(mean=targets.mean(), std=targets.std())

    # 2. Cargar modelo entrenado
    model = build_fno_model(num_delays=num_delays, cfg=cfg)
    state_dict = torch.load(weights_path, weights_only=False, map_location=device)
    model.load_state_dict(state_dict)

    # 3. Correr rollout en lazo cerrado
    rollout_res = run_closed_loop_rollout(
        model=model,
        data=data,
        dt=sim["dt"],
        num_delays=num_delays,
        vel_normalizer=vel_norm,
        target_normalizer=target_norm,
        device=device,
    )

    print(
        f"Rollout completado en {rollout_res['rollout_time_sec']:.2f}s | "
        f"Error relativo L2 final: {rollout_res['rel_l2_errors'][-1]:.4f}"
    )

    # 4. Guardar gráfica de error en el tiempo y GIF con LIC
    plot_rollout_error(
        times=sim["times"],
        rel_l2_errors=rollout_res["rel_l2_errors"],
        start_idx=rollout_res["start_idx"],
        save_path=str(out_dir / "rollout_error.png"),
    )

    save_lic_gif(
        x_coords=sim["x"],
        y_coords=sim["y"],
        times=sim["times"],
        u_real_all=data[0].numpy(),
        v_real_all=data[1].numpy(),
        u_pred_all=rollout_res["u_rollout"].numpy(),
        v_pred_all=rollout_res["v_rollout"].numpy(),
        start_idx=rollout_res["start_idx"],
        end_idx=rollout_res["end_idx"],
        cfg=cfg,
        output_path=str(out_dir / "double_gyre_lic.gif"),
        fps=int(v_cfg.get("fps", 15)),
        viz_res_x=v_cfg.get("viz_res_x", 240),
        viz_res_y=v_cfg.get("viz_res_y", 120),
        use_paraview_cmap=bool(v_cfg.get("use_paraview_cmap", True)),
    )


if __name__ == "__main__":
    main()