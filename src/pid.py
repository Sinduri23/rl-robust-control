"""
Classical control baseline: a PID controller on the pole angle, plus a PD term on
cart position so the cart doesn't drift off the track.

CartPole only allows two actions (full push left / full push right), so the PID output
u is turned into an action by its sign. This is called bang-bang control.
"""
import numpy as np


class PIDController:
    def __init__(self, kp=10.0, ki=0.0, kd=2.0, kx=0.5, kxd=1.0, dt=0.02):
        self.kp, self.ki, self.kd, self.kx, self.kxd, self.dt = kp, ki, kd, kx, kxd, dt
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def __call__(self, obs: np.ndarray) -> int:
        x, x_dot, theta, theta_dot = obs
        self.integral += theta * self.dt
        # Pole leaning right (theta > 0) -> push right to get the cart back under it.
        # Position terms are positive too: to bring the cart back left, the pole must
        # first be tipped left, which needs a small push right (non-minimum-phase system).
        u = (self.kp * theta + self.ki * self.integral + self.kd * theta_dot
             + self.kx * x + self.kxd * x_dot)
        return int(u > 0)
