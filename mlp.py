import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, state_size, action_size, device, num_hidden=32):
        super().__init__()
        self.device = device
        self.to(device)

        self.fc = nn.Sequential(
            nn.Linear(state_size, num_hidden),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(num_hidden, 1)
        self.policy_head = nn.Linear(num_hidden, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def get_policy(self, x):
        x = self.fc(x)
        raw_logit = self.policy_head(x)
        probs = F.softmax(raw_logit, dim=-1)
        return raw_logit, probs

    def get_value(self, x):
        x = self.fc(x)
        x = self.value_head(x)
        return x

    def save(self, path):
        """
        Save the model's state dictionary and optimizer state to a file.

        Args:
            path (str): Path to the file where the model and optimizer will be saved.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Model and optimizer saved to {path}")

    def load(self, path):
        """
        Load the model's state dictionary and optimizer state from a file.

        Args:
            path (str): Path to the file from which the model and optimizer will be loaded.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.to(self.device)
        print(f"Model and optimizer loaded from {path}")
