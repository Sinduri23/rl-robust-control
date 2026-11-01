"""
Record a side-by-side GIF: PID vs PPO (full randomization) under the same disturbances.

Run: python scripts/make_gif.py   ->  results/demo.gif
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from scripts.evaluate import PPOController, latest_run
from src.envs import make_env
from src.pid import PIDController
from src.utils import ROOT, load_config

SCENARIO = dict(noise_std=0.15, push_strength=0.5, push_prob=0.05)  # avg over 50 eps: PID 174, PPO 218 steps
SEED, MAX_FRAMES = 7, 500


def rollout(ctrl):
    env = make_env(render_mode="rgb_array", **SCENARIO)
    obs, _ = env.reset(seed=SEED)
    ctrl.reset()
    frames, done = [], False
    while not done and len(frames) < MAX_FRAMES:
        frames.append(env.render())
        obs, _, term, trunc, _ = env.step(ctrl(obs))
        done = term or trunc
    frames.append(env.render())
    env.close()
    return frames


def label(frame, text, step, fell):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.text((10, 8), text, fill=(0, 0, 0))
    d.text((10, 24), f"step {step}" + ("  - FELL" if fell else ""), fill=(200, 0, 0) if fell else (0, 0, 0))
    return np.array(img)


def main():
    cfg = load_config()
    a = rollout(PIDController(**cfg["pid"]))
    b = rollout(PPOController(latest_run("ppo_drfull")))
    n = max(len(a), len(b))
    frames = []
    for i in range(0, n, 2):  # every 2nd frame keeps the file small
        fa = label(a[min(i, len(a) - 1)], "PID", min(i, len(a) - 1), i >= len(a) - 1 and len(a) < MAX_FRAMES)
        fb = label(b[min(i, len(b) - 1)], "PPO (full randomization)", min(i, len(b) - 1),
                   i >= len(b) - 1 and len(b) < MAX_FRAMES)
        frames.append(np.concatenate([fa, fb], axis=1)[::2, ::2])
    out = ROOT / "results" / "demo.gif"
    imageio.mimsave(out, frames, duration=0.04, loop=0)
    print(f"PID lasted {len(a) - 1} steps, PPO lasted {len(b) - 1} steps -> {out}")


if __name__ == "__main__":
    main()
