"""
Train a PPO agent on CartPole with domain randomization.

MLOps practices used:
  - every setting comes from config.yaml (a copy is saved with each run)
  - fixed seeds for reproducibility
  - TensorBoard logging   ->  tensorboard --logdir runs/
  - each run gets a versioned folder: models/<run_name>/ ; models/latest points to it
  - periodic validation on held-out episodes from the training distribution (logged to TensorBoard);
    evaluation uses final_model.zip, because on CartPole many checkpoints tie at the 500 cap

Run:  python scripts/train.py            (about 3-6 minutes on a laptop CPU)
      python scripts/train.py --mode nominal   (no randomization, for comparison)
      python scripts/train.py --mode drphys    (randomize mass/length only)
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env

from src.envs import make_env
from src.network import ControlMLP
from src.utils import ROOT, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nominal", "drphys", "drfull"], default="drfull",
                    help="nominal: default physics only | drphys: random mass/length | "
                         "drfull: random mass/length + sensor noise + pushes")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    t = cfg["train"]
    dr = args.mode != "nominal"
    run_name = f"ppo_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "config.yaml", run_dir / "config.yaml")

    env_kwargs = dict(env_id=cfg["env_id"], randomize=dr,
                      mass_range=t["mass_scale_range"], length_range=t["length_scale_range"])
    if args.mode == "drfull":
        env_kwargs.update(noise_range=t["noise_std_range"], push_range=t["push_strength_range"],
                          push_prob=t["push_prob"])
    train_env = make_vec_env(make_env, n_envs=t["n_envs"], seed=cfg["seed"], env_kwargs=env_kwargs)
    eval_env = make_vec_env(make_env, n_envs=1, seed=cfg["seed"] + 1000, env_kwargs=env_kwargs)

    model = PPO(
        "MlpPolicy", train_env, seed=cfg["seed"], verbose=0,
        tensorboard_log=str(ROOT / "runs"),
        policy_kwargs=dict(
            features_extractor_class=ControlMLP,
            features_extractor_kwargs=dict(hidden_size=cfg["network"]["hidden_size"]),
            net_arch=dict(pi=[], vf=[]),   # heads are single linear layers on top of ControlMLP
        ),
        **cfg["ppo"],
    )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir),
        eval_freq=max(5000 // t["n_envs"], 1), n_eval_episodes=10, deterministic=True,
        callback_on_new_best=None, verbose=0,
    )
    steps = args.timesteps or t["total_timesteps"]
    print(f"Training {run_name} for {steps:,} steps (mode={args.mode}) ...")
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=eval_cb, tb_log_name=run_name, progress_bar=False)
    model.save(run_dir / "final_model")
    print(f"Done in {time.time() - t0:.0f}s. Best eval reward: {eval_cb.best_mean_reward:.1f}")

    latest = ROOT / "models" / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    print(f"Saved to {run_dir}  (models/latest -> {run_name})")


if __name__ == "__main__":
    main()
