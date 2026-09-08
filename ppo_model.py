import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()

        # --- Actor Network ---
        # Outputs the mean of the continuous action distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bounds raw outputs smoothly to [-1, 1]
        )

        # Action standard deviation parameter.
        # FIX: torch.zeros(1, action_dim) * -0.5 == 0 * -0.5 == 0 (zeros times
        # a scalar are still zeros). That made log_std start at 0.0, so
        # action_std = exp(0.0) = 1.0 instead of the intended ~0.6 — a huge
        # initial std relative to the tanh-bounded [-1, 1] action range,
        # which caused sampled actions to be clamped near the extremes
        # (max/min angular and linear velocity) almost every step, i.e.
        # near-random full-speed spinning. Use torch.full to actually set
        # the initial value.
        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))

        # --- Critic Network ---
        # Predicts the baseline Value state-advantage evaluation matrix
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        action_mean = self.actor(state)

        # Clamp log_std to maintain healthy numeric boundaries
        log_std = torch.clamp(self.log_std, -2.0, 0.5)
        action_std = torch.exp(log_std)

        dist = Normal(action_mean, action_std)
        return dist, value

    def evaluate(self, state, action):
        dist, value = self.forward(state)
        action_logprobs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1, keepdim=True)
        return action_logprobs, value, dist_entropy