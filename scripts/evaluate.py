"""
Robustness evaluation: PID vs three PPO agents (no randomization, mass/length randomization,
full randomization incl. sensor noise and pushes).

Each controller is tested while one disturbance is varied at a time:
  pole mass, pole length, sensor noise, random pushes, plus one combined "hard" scenario.

Metrics per setting (averaged over N seeded episodes, same seeds for every controller):
  survival      - steps balanced, out of a 500-step maximum
  success_rate  - % of episodes that balanced the full 500 steps
  angle_err     - mean |pole angle| in degrees (lower = steadier)
  switch_rate   - how often the push direction flips per step (lower = smoother control, less actuator wear)

Run: python scripts/evaluate.py   ->  results/robustness.csv, results/robustness.png, results/summary.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.envs import make_env
from src.pid import PIDController
from src.utils import ROOT, load_config


def latest_run(prefix: str) -> Path:
    runs = sorted((ROOT / "models").glob(f"{prefix}_*"))
    if not runs:
        raise FileNotFoundError(f"No models/{prefix}_* found. Run scripts/train.py first.")
    return runs[-1] / "final_model.zip"


class PPOController:
    def __init__(self, path):
        self.model = PPO.load(path, device="cpu")

    def reset(self):
        pass

    def __call__(self, obs):
        return int(self.model.predict(obs, deterministic=True)[0])


def run_episodes(ctrl, n, seed0=10_000, **env_kwargs):
    env = make_env(**env_kwargs)
    surv, ang, sw = [], [], []
    for ep in range(n):
        obs, _ = env.reset(seed=seed0 + ep)
        ctrl.reset()
        done, steps, angles, switches, last = False, 0, [], 0, None
        while not done:
            a = ctrl(obs)
            switches += int(last is not None and a != last)
            last = a
            obs, _, term, trunc, _ = env.step(a)
            angles.append(abs(env.unwrapped.state[2]))   # true angle, not the noisy sensor reading
            steps += 1
            done = term or trunc
        surv.append(steps)
        ang.append(np.degrees(np.mean(angles)))
        sw.append(switches / steps)
    env.close()
    return dict(survival=np.mean(surv), success_rate=100 * np.mean(np.array(surv) >= 500),
                angle_err=np.mean(ang), switch_rate=np.mean(sw))


def main():
    cfg = load_config()
    e = cfg["eval"]
    n = e["episodes"]
    controllers = {
        "PID": PIDController(**cfg["pid"]),
        "PPO (no rand.)": PPOController(latest_run("ppo_nominal")),
        "PPO (rand. mass/length)": PPOController(latest_run("ppo_drphys")),
        "PPO (full rand.)": PPOController(latest_run("ppo_drfull")),
    }
    sweeps = {
        "Pole mass (x default)": ("mass_scale", e["mass_scales"]),
        "Pole length (x default)": ("length_scale", e["length_scales"]),
        "Sensor noise (std)": ("noise_std", e["noise_stds"]),
        "Push strength (rad/s)": ("push_strength", e["push_strengths"]),
    }
    hard = dict(mass_scale=3.0, length_scale=1.5, noise_std=0.1, push_strength=1.0)

    rows = []
    for name, ctrl in controllers.items():
        for sweep, (param, values) in sweeps.items():
            for v in values:
                m = run_episodes(ctrl, n, push_prob=e["push_prob"], **{param: v})
                rows.append(dict(controller=name, sweep=sweep, value=v, **m))
        m = run_episodes(ctrl, n, push_prob=e["push_prob"], **hard)
        rows.append(dict(controller=name, sweep="Combined hard", value=0, **m))
        print(f"evaluated {name}")

    df = pd.DataFrame(rows).round(3)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)

    # ---- plot: survival under each disturbance ----
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2), sharey=True)
    styles = {"PID": ("#6b7280", "s", "--"), "PPO (no rand.)": ("#93c5fd", "o", "-"),
              "PPO (rand. mass/length)": ("#2563eb", "D", "-"),
              "PPO (full rand.)": ("#dc2626", "^", "-")}
    for ax, sweep in zip(axes, sweeps):
        for name, (c, mk, ls) in styles.items():
            d = df[(df.controller == name) & (df.sweep == sweep)]
            ax.plot(d.value, d.survival, marker=mk, ls=ls, color=c, label=name, lw=2)
        ax.set_title(sweep)
        ax.set_xlabel(sweep.split(" (")[1].rstrip(")") if "(" in sweep else "")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 520)
    hd = df[df.sweep == "Combined hard"].set_index("controller")
    names = list(styles)
    axes[4].bar(range(len(names)), [hd.loc[c, "survival"] for c in names], color=[styles[c][0] for c in names])
    axes[4].set_xticks(range(len(names)), [c.replace(" (", "\n(") for c in names], fontsize=8)
    axes[4].set_title("Combined hard scenario")
    for i, c in enumerate(names):
        axes[4].text(i, hd.loc[c, "survival"] + 8, f"{hd.loc[c, 'survival']:.0f}", ha="center", fontsize=9)
    axes[4].grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Steps balanced (max 500)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Robustness: PID vs PPO controllers on CartPole (30 episodes per point)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=130)

    # ---- markdown summary ----
    nominal = df[(df.sweep == "Pole mass (x default)") & (df.value == 1.0)].set_index("controller")
    hard_df = df[df.sweep == "Combined hard"].set_index("controller")
    avg = df[df.sweep != "Combined hard"].groupby("controller")[["survival", "success_rate"]].mean()
    lines = ["# Results", "",
             f"{n} seeded episodes per setting; same seeds for every controller.", "",
             "## Nominal conditions", "",
             nominal[["survival", "success_rate", "angle_err", "switch_rate"]].to_markdown(), "",
             "## Combined hard scenario (3x mass, 1.5x length, noise 0.1, pushes 1.0)", "",
             hard_df[["survival", "success_rate", "angle_err", "switch_rate"]].to_markdown(), "",
             "## Average over all single-disturbance settings", "",
             avg.round(1).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
