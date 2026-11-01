"""ROS-independent logic, so it can be unit-tested without a ROS2 install."""
import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def act(self, obs) -> int:
        x = np.asarray(obs, dtype=np.float32).reshape(1, 4)
        return int(self.sess.run(None, {"obs": x})[0].argmax())


class CartPoleSim:
    """Same physics as Gymnasium CartPole-v1 (used by the sim node in place of real hardware)."""

    def __init__(self, mass_scale=1.0, seed=0):
        self.g, self.mc, self.mp, self.l, self.f, self.tau = 9.8, 1.0, 0.1 * mass_scale, 0.5, 10.0, 0.02
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        return self.state.copy()

    def step(self, action: int) -> bool:
        x, xd, th, thd = self.state
        tot, pml = self.mp + self.mc, self.mp * self.l
        force = self.f if action == 1 else -self.f
        c, s = np.cos(th), np.sin(th)
        temp = (force + pml * thd ** 2 * s) / tot
        thacc = (self.g * s - c * temp) / (self.l * (4 / 3 - self.mp * c ** 2 / tot))
        xacc = temp - pml * thacc * c / tot
        self.state = np.array([x + self.tau * xd, xd + self.tau * xacc, th + self.tau * thd, thd + self.tau * thacc])
        return abs(self.state[0]) > 2.4 or abs(self.state[2]) > np.radians(12)
