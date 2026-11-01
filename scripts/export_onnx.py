"""
Export the trained policy to ONNX so it can run outside Python (C++, ROS2, embedded targets).

Only the actor (observation -> action logits) is exported. The checks below confirm the
ONNX model gives the same actions as the original PyTorch policy.

Run: python scripts/export_onnx.py  ->  models/policy.onnx
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from scripts.evaluate import latest_run
from src.network import DeployablePolicy
from src.utils import ROOT


def main():
    model = PPO.load(latest_run("ppo_drfull"), device="cpu")
    policy = DeployablePolicy(model.policy).eval()
    out = ROOT / "models" / "policy.onnx"

    dummy = torch.zeros(1, 4, dtype=torch.float32)
    torch.onnx.export(policy, dummy, out, input_names=["obs"], output_names=["logits"],
                      dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)

    # Parity check on 10,000 random states
    obs = np.random.default_rng(0).uniform(-1, 1, size=(10_000, 4)).astype(np.float32)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"obs": obs})[0]
    with torch.no_grad():
        torch_logits = policy(torch.from_numpy(obs)).numpy()
    sb3_actions = model.predict(obs, deterministic=True)[0]
    print(f"max |logit diff|           : {np.abs(onnx_logits - torch_logits).max():.2e}")
    print(f"action agreement vs SB3    : {(onnx_logits.argmax(1) == sb3_actions).mean():.2%}")
    print(f"exported {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
