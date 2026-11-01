"""
Policy network written directly in PyTorch and plugged into Stable-Baselines3.

SB3's PPO uses an actor-critic: a shared feature network, then
  - actor head  -> action logits (what to do)
  - critic head -> value estimate (how good the current state is)
This file defines the shared feature network. SB3 adds the two linear heads.
"""
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ControlMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, hidden_size: int = 64):
        super().__init__(observation_space, features_dim=hidden_size)
        n_in = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden_size),
            nn.Tanh(),            # tanh: smooth, bounded; standard choice for small control policies
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DeployablePolicy(nn.Module):
    """Just the actor path (features -> action logits), for ONNX export / C++ / ROS2.
    The critic is only needed during training, so it is dropped."""

    def __init__(self, sb3_policy):
        super().__init__()
        self.features = sb3_policy.features_extractor
        self.mlp_actor = sb3_policy.mlp_extractor.policy_net
        self.action_net = sb3_policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.mlp_actor(self.features(obs)))
