"""
Environment + disturbance wrappers.

CartPole: a cart moves left/right to keep a pole upright.
Observation: [cart position x, cart velocity x_dot, pole angle theta, pole angular velocity theta_dot]
Action: 0 = push left, 1 = push right (fixed 10 N force)

The wrappers below make the simulator behave more like a real machine:
  - PhysicsWrapper     : change pole mass / length (a different or loaded vehicle)
  - SensorNoiseWrapper : add Gaussian noise to observations (imperfect sensors)
  - PushWrapper        : random kicks to the pole (wind, bumps, waves)
"""
import gymnasium as gym
import numpy as np

# Default CartPole physics, read once so wrappers can scale them
_BASE = gym.make("CartPole-v1").unwrapped
BASE_MASSPOLE, BASE_LENGTH = _BASE.masspole, _BASE.length


def set_physics(env: gym.Env, mass_scale: float = 1.0, length_scale: float = 1.0) -> None:
    """Change pole mass/length. CartPole caches derived values, so update those too."""
    u = env.unwrapped
    u.masspole = BASE_MASSPOLE * mass_scale
    u.length = BASE_LENGTH * length_scale
    u.total_mass = u.masspole + u.masscart
    u.polemass_length = u.masspole * u.length


class PhysicsWrapper(gym.Wrapper):
    """Fixed physics, or a new random mass/length every episode (domain randomization)."""

    def __init__(self, env, mass_scale=1.0, length_scale=1.0, randomize=False,
                 mass_range=(0.5, 3.0), length_range=(0.7, 1.5)):
        super().__init__(env)
        self.mass_scale, self.length_scale = mass_scale, length_scale
        self.randomize, self.mass_range, self.length_range = randomize, mass_range, length_range

    def reset(self, **kwargs):
        if self.randomize:
            self.mass_scale = self.np_random.uniform(*self.mass_range)
            self.length_scale = self.np_random.uniform(*self.length_range)
        set_physics(self.env, self.mass_scale, self.length_scale)
        return self.env.reset(**kwargs)


class SensorNoiseWrapper(gym.ObservationWrapper):
    """Fixed noise level, or a random level per episode if std_range is given."""

    def __init__(self, env, std=0.0, std_range=None):
        super().__init__(env)
        self.std, self.std_range = std, std_range

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.std_range is not None:
            self.std = self.np_random.uniform(*self.std_range)
        return self.observation(obs), info

    def observation(self, obs):
        if self.std <= 0:
            return obs
        return (obs + self.np_random.normal(0, self.std, size=obs.shape)).astype(np.float32)


class PushWrapper(gym.Wrapper):
    """With probability `prob` each step, kick the pole's angular velocity by +/- `strength` rad/s."""

    def __init__(self, env, strength=0.0, prob=0.05, strength_range=None):
        super().__init__(env)
        self.strength, self.prob, self.strength_range = strength, prob, strength_range

    def reset(self, **kwargs):
        if self.strength_range is not None:
            self.strength = self.np_random.uniform(*self.strength_range)
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.strength > 0 and self.np_random.random() < self.prob:
            u = self.env.unwrapped
            x, x_dot, th, th_dot = u.state
            u.state = np.array([x, x_dot, th, th_dot + self.np_random.choice([-1, 1]) * self.strength])
        return self.env.step(action)


def make_env(env_id="CartPole-v1", mass_scale=1.0, length_scale=1.0, noise_std=0.0,
             push_strength=0.0, push_prob=0.05, randomize=False,
             mass_range=(0.5, 3.0), length_range=(0.7, 1.5),
             noise_range=None, push_range=None, render_mode=None):
    """randomize=True samples mass/length per episode; noise_range / push_range (if given)
    also sample sensor noise and push strength per episode (full domain randomization)."""
    env = gym.make(env_id, render_mode=render_mode)
    env = PhysicsWrapper(env, mass_scale, length_scale, randomize, tuple(mass_range), tuple(length_range))
    env = PushWrapper(env, push_strength, push_prob, tuple(push_range) if push_range else None)
    env = SensorNoiseWrapper(env, noise_std, tuple(noise_range) if noise_range else None)
    return env
