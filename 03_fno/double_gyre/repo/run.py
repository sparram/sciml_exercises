from pathlib import Path
import yaml

from src.physics import generate_double_gyre_data
from src.dataset import create_dataloader
from src.model import build_fno_model, train_fno
from src.rollout import run_closed_loop_rollout
from src.viz import save_lic_gif


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    num_delays = cfg["dataset"]["num_delays"]
    device = cfg["model"]["device"]

    # 1. Simulación física
    sim = generate_double_gyre_data(cfg)

    # 2. Dataset con historia (time-delay)
    loader, vel_norm, target_norm = create_dataloader(
        data=sim["data"],
        targets=sim["targets"],
        num_delays=num_delays,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle"],
    )

    # 3. Modelo y entrenamiento
    model = build_fno_model(num_delays=num_delays, cfg=cfg)
    train_fno(model=model, dataloader=loader, cfg=cfg)

    # 4. Rollout en lazo cerrado
    rollout_res = run_closed_loop_rollout(
        model=model,
        data=sim["data"],
        dt=sim["dt"],
        num_delays=num_delays,
        vel_normalizer=vel_norm,
        target_normalizer=target_norm,
        device=device,
    )

    final_err = rollout_res["rel_l2_errors"][-1]
    print(
        f"Rollout finalizado en {rollout_res['rollout_time_sec']:.2f}s | "
        f"Error relativo L2 final: {final_err:.4f}"
    )

    # 5. Guardar GIF LIC para README / LinkedIn
    Path("outputs").mkdir(exist_ok=True)
    save_lic_gif(
        grid_x=sim["grid_x"],
        grid_y=sim["grid_y"],
        times=sim["times"],
        u_real_all=sim["data"][0].numpy(),
        v_real_all=sim["data"][1].numpy(),
        u_pred_all=rollout_res["u_rollout"].numpy(),
        v_pred_all=rollout_res["v_rollout"].numpy(),
        start_idx=rollout_res["start_idx"],
        end_idx=rollout_res["end_idx"],
        cfg=cfg,
        output_path="outputs/double_gyre_lic.gif",
        fps=15,
    )


if __name__ == "__main__":
    main()