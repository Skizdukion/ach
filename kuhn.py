import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# Define the Kuhn Poker environment
class KuhnPokerEnv:
    def __init__(self, embedding_dim=2, hidden_dim=4, verbose=False):
        self.deck = [0, 1, 2]  # Card values
        self.action_value_map = {"p": 0, "b": 1}
        self.value_action_map = {0: "p", 1: "b"}
        self.num_actions = len(self.action_value_map)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.num_actions, self.embedding_dim)
        self.gru = nn.GRU(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim, batch_first=True
        )
        self.encode_state_size = hidden_dim + 1
        self.verbose = verbose

    def reset(self):
        self.player_cards = random.sample(self.deck, 2)
        self.history = ""
        self.current_player = 0
        return self.get_state()

    def get_state(self):
        res = (self.player_cards[self.current_player], self.history)
        return res

    def encode_states(self, states):
        """
        Encode a batch of states into a tensor.
        :param states: List of states, where each state is a tuple (player_card, history).
        :return: Tensor of encoded states, shape: (batch_size, encoded_state_size).
        """
        batch_card_tensors = []
        batch_history_summaries = []

        for state in states:
            player_card, history = state

            # Encode player card as a scalar tensor
            card_tensor = torch.tensor([player_card], dtype=torch.float32)
            batch_card_tensors.append(card_tensor)

            # Encode history
            if len(history) > 0:
                # Convert actions to indices
                history_indices = torch.tensor(
                    [self.action_value_map[action] for action in history],
                    dtype=torch.long,
                ).unsqueeze(
                    0
                )  # Shape: (1, len(history))
                # Embed the action indices
                history_embeddings = self.embedding(
                    history_indices
                )  # Shape: (1, len(history), embedding_dim)
                _, hidden_state = self.gru(
                    history_embeddings
                )  # hidden_state: (1, hidden_dim)
                history_summary = hidden_state.squeeze(0)  # Shape: (hidden_dim,)
            else:
                history_summary = torch.zeros(self.hidden_dim, dtype=torch.float32)

            batch_history_summaries.append(history_summary)

        # Stack all card tensors and history summaries
        batch_card_tensors = torch.stack(batch_card_tensors)  # Shape: (batch_size, 1)
        batch_history_summaries = torch.stack(
            batch_history_summaries
        )  # Shape: (batch_size, hidden_dim)

        # Concatenate card tensors and history summaries
        encoded_states = torch.cat(
            [batch_card_tensors, batch_history_summaries], dim=1
        )  # Shape: (batch_size, encoded_state_size)

        return encoded_states

    def encoded_single_state(self, state):
        player_card, history = state

        card_tensor = torch.tensor([player_card], dtype=torch.float32)

        if len(history) > 0:
            # Convert actions to indices
            history_indices = torch.tensor(
                [self.action_value_map[action] for action in history], dtype=torch.long
            )
            # Embed the action indices
            history_embeddings = self.embedding(
                history_indices
            )  # Shape: (1, len(history), embedding_dim)
            _, hidden_state = self.gru(history_embeddings)
            history_summary = hidden_state.squeeze(0)  # Shape: (hidden_dim,)
        else:
            history_summary = torch.zeros(self.hidden_dim, dtype=torch.float32)

        state_tensor = torch.cat([card_tensor, history_summary])
        return state_tensor

    def step(self, action):
        action = self.value_action_map[action]

        if action is None:
            raise Exception("Invalid action")

        self.history += action  # Append 'p' for pass, 'b' for bet

        # Check if the game is over
        if len(self.history) >= 2:
            # If the history is "pp", "pb", "bp", or "bb", the game is over
            if self.history in ["pp", "bp", "bb"]:
                reward = self.calculate_reward()
                return self.get_state(), reward, True
            # If the history is "pbb" or "pbp", the game is also over
            elif self.history in ["pbb", "pbp"]:
                reward = self.calculate_reward()
                return self.get_state(), reward, True

        # Switch players if the game is not over
        self.current_player = 1 - self.current_player
        return self.get_state(), 0, False

    def calculate_reward(self):
        if self.verbose:
            print(f"History: {self.history}")
            print(f"Player Cards: {self.player_cards}")

        if self.history == "pp":
            # Both players pass, the higher card wins 1
            reward = 1 if self.player_cards[0] > self.player_cards[1] else -1
        elif self.history == "pbp":
            # Player 1 passes, Player 2 bets, Player 1 folds, Player 2 wins 1
            reward = -1
        elif self.history == "bp":
            # Player 1 bets, Player 2 passes, Player 1 wins 1
            reward = 1
        elif self.history == "bb":
            # Both players bet, the higher card wins 2
            reward = 2 if self.player_cards[0] > self.player_cards[1] else -2
        elif self.history == "pbb":
            # Player 1 passes, Player 2 bets, Player 1 calls, the higher card wins 2
            reward = 2 if self.player_cards[0] > self.player_cards[1] else -2
        else:
            reward = 0  # Handle unexpected cases

        if self.verbose:
            print(f"Calculated Reward: {reward}")

        return reward
