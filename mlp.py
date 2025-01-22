import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, state_size, action_size, device, num_hidden=32):
        super().__init__()

        self.to(device)

        self.fc = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(num_hidden, 1)

        self.policy_head = nn.Linear(num_hidden, action_size)

    def get_policy(self, x):
        x = self.fc(x)
        raw_logit = self.policy_head(x)
        probs = F.softmax(raw_logit, dim=-1)
        return raw_logit, probs

    def get_value(self, x):
        x = self.fc(x)
        x = self.value_head(x)
        return x
