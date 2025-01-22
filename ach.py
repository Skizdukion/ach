from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torch
from kuhn import KuhnPokerEnv
from mlp import Mlp
from collections import defaultdict, deque
from torch.distributions import Categorical
import random
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrajectorySegment:
    states: torch.Tensor  # (M, R, S_dim)
    actions: torch.Tensor  # (M, R)
    logprobs: torch.Tensor  # (M, R)
    values: torch.Tensor  # (M, R)
    rewards: torch.Tensor  # (M, R)
    dones: torch.Tensor  # (M, R)

    next_start_state: torch.Tensor  # (M, S_dim)

    def __len__(self):
        return self.states.shape[0]


@torch.no_grad()
def basic_advantages_and_returns(
    segment: TrajectorySegment, next_return: torch.Tensor, gamma=0.9
):
    returns = torch.zeros_like(segment.rewards).detach()

    roll_out_size = segment.rewards.shape[1]

    for t in reversed(range(roll_out_size)):
        is_terminal = 1 - segment.dones[:, t]
        returns[:, t] = segment.rewards[:, t] + gamma * next_return * is_terminal
        next_return = returns[:, t]

    advantages = returns - segment.values
    return advantages, returns


@torch.no_grad()
def gae_advantages_and_returns(
    segment: TrajectorySegment, next_return: torch.Tensor, gamma=0.9, lam=0.95
):
    roll_out_size = segment.rewards.shape[1]

    advantages = torch.zeros_like(segment.rewards)
    returns = torch.zeros_like(segment.rewards)

    last_advantage = 0
    last_return = next_return

    for t in reversed(range(roll_out_size)):
        is_terminal = 1 - segment.dones[:, t]

        delta = (
            segment.rewards[:, t]
            + gamma * last_return * is_terminal
            - segment.values[:, t]
        )
        advantages[:, t] = last_advantage = (
            delta + gamma * lam * last_advantage * is_terminal
        )
        returns[:, t] = segment.values[:, t] + advantages[:, t]

        last_return = segment.values[:, t] if segment.dones[:, t] else returns[:, t]

    return advantages, returns


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def add_batch(
        self, states, actions, logprobs, values, rewards, dones, advantages, returns
    ):
        """Stores multiple experiences at once."""
        for i in range(len(states)):
            self.buffer.append(
                (
                    states[i],
                    actions[i],
                    logprobs[i],
                    values[i],
                    rewards[i],
                    dones[i],
                    advantages[i],
                    returns[i],
                )
            )

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class ACH:
    def __init__(
        self,
        env: KuhnPokerEnv,
        mlp: Mlp,
        rollout_size=512,
        parralel_size=1,
        learn_each_train=10,
        epsilon=0.2,
        l_thresh=5.0,
        entropy_beta=1e-2,
        alpha=0.5,
        # gae_enabled=True,
    ):
        self.env = env
        self.learners = [mlp]
        self.rollout_size = rollout_size
        self.parralel_size = parralel_size
        self.replay_buffer = ReplayBuffer()
        self.learn_each_train = learn_each_train
        self.epsilon = epsilon
        self.l_thresh = l_thresh
        self.entropy_beta = entropy_beta
        self.alpha = alpha

    def train(self, train_length):
        for train_i in range(1, train_length + 1):
            segment = self.self_play()
            next_returns = (
                self.learners[-1].get_value(segment.next_start_state).view(-1)
            )

            advantages, returns = gae_advantages_and_returns(segment, next_returns)

            self.replay_buffer.add_batch(
                segment.states.view(-1, *segment.states.shape[2:]),
                segment.actions.view(-1),
                segment.logprobs.view(-1),
                segment.values.view(-1),
                segment.rewards.view(-1),
                segment.dones.view(-1),
                advantages.view(-1),
                returns.view(-1),
            )

            for _ in range(self.learn_each_train):
                mini_batch = self.replay_buffer.sample(64)

                # Reset Lsum for each mini-batch
                losses = []

                for transition in mini_batch:
                    s, a, log_p_old, _, _, _, A_sa, G = transition

                    p_old = torch.exp(log_p_old)
                    y, probs = self.learners[-1].get_policy(s)
                    a = a.long()
                    p = probs.gather(0, a)
                    y_a = y.gather(0, a)
                    y_mean = y.mean(dim=-1)

                    ratio = p / p_old

                    if A_sa >= 0:
                        c = (ratio <= 1 + self.epsilon) & (y_a - y_mean < self.l_thresh)
                    else:
                        c = (ratio >= 1 - self.epsilon) & (
                            y_a - y_mean > -self.l_thresh
                        )

                    c = c.float()

                    loss_policy = -c * (y_a / p_old) * A_sa
                    loss_value = (
                        0.5 * self.alpha * (self.learners[-1].get_value(s) - G) ** 2
                    )
                    loss_entropy = self.entropy_beta * (p * torch.log(p + 1e-10)).sum(
                        dim=-1
                    )

                    # Accumulate losses for the mini-batch
                    losses.append(
                        loss_policy.mean() + loss_value.mean() + loss_entropy.mean()
                    )

                # Backpropagate the accumulated loss for the mini-batch
                self.learners[-1].optimizer.zero_grad()
                torch.stack(losses).sum().backward(
                    retain_graph=True
                )  # Compute gradients properly
                self.learners[-1].optimizer.step()

                del losses, loss_policy, loss_value, loss_entropy

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            if train_i % 5 == 0:
                print(
                    f"Trainning checkpoint {train_i}, Exploitability: {self.calculate_exploitability()}"
                )

    def self_play(self):
        s_states = torch.zeros(
            (self.parralel_size, self.rollout_size, self.env.encode_state_size),
            dtype=torch.float32,
            device=DEVICE,
        )
        s_actions = np.zeros((self.parralel_size, self.rollout_size))
        s_logprobs = np.zeros((self.parralel_size, self.rollout_size))
        s_values = np.zeros((self.parralel_size, self.rollout_size))
        s_rewards = np.zeros((self.parralel_size, self.rollout_size))
        s_dones = np.zeros((self.parralel_size, self.rollout_size))
        s_next_states = torch.zeros(
            (
                self.parralel_size,
                self.env.encode_state_size,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        with ThreadPoolExecutor(max_workers=self.parralel_size) as executor:
            futures = []
            for parralel_id in range(self.parralel_size):
                futures.append(
                    executor.submit(
                        self.simulate_play,
                        s_states,
                        s_actions,
                        s_logprobs,
                        s_values,
                        s_rewards,
                        s_dones,
                        parralel_id,
                    )
                )

            for i in range(len(futures)):
                next_states = futures[i].result()
                s_next_states[i] = next_states

        return TrajectorySegment(
            s_states,
            torch.from_numpy(s_actions).to(DEVICE).float(),
            torch.from_numpy(s_logprobs).to(DEVICE).float(),
            torch.from_numpy(s_values).to(DEVICE).float(),
            torch.from_numpy(s_rewards).to(DEVICE).float(),
            torch.from_numpy(s_dones).to(DEVICE).float(),
            next_start_state=s_next_states,
        )

    def simulate_play(
        self, s_states, s_actions, s_logprobs, s_values, s_rewards, s_dones, parralel_id
    ):
        cur_state = self.env.reset()
        cur_player_index = 0
        learner_player_index = 0
        total_player = 2
        step = 0
        is_new_game = True
        opp = self.random_opp()

        while step < self.rollout_size:
            if cur_player_index % total_player == learner_player_index:
                step, next_state, is_new_game = self.learner_turn(
                    cur_state,
                    step,
                    s_states[parralel_id],
                    s_actions[parralel_id],
                    s_logprobs[parralel_id],
                    s_values[parralel_id],
                    s_rewards[parralel_id],
                    s_dones[parralel_id],
                )
            else:
                step, next_state, is_new_game = self.opp_turn(
                    opp,
                    cur_state,
                    step,
                    s_rewards[parralel_id],
                    s_dones[parralel_id],
                    is_new_game,
                )

            cur_player_index += 1
            if is_new_game:
                opp = self.random_opp()
                cur_player_index = random.randint(0, 1)

            cur_state = next_state

        return self.env.encoded_single_state(cur_state)

    def random_opp(self):
        if random.random() < 0.5:
            return self.learners[len(self.learners) - 1]
        else:
            return random.choice(self.learners)

    @torch.no_grad()
    def opp_turn(
        self,
        opp: Mlp,
        state,
        step,
        s_rewards,
        s_dones,
        is_new_game,
    ):
        encode_state = self.env.encoded_single_state(state)
        _, probs = opp.get_policy(encode_state)
        dist = Categorical(probs)
        action = dist.sample().item()

        next_state, reward, done = self.env.step(action)

        # Bot go first so ignore adding to history
        if is_new_game:
            return step, next_state, False

        s_dones[step] = done

        if done:
            s_rewards[step] = reward
            next_state = self.env.reset()
        else:
            s_rewards[step] = 0

        return step + 1, next_state, done

    def learner_turn(
        self,
        state,
        step,
        s_states,
        s_actions,
        s_logprobs,
        s_values,
        s_rewards,
        s_dones,
    ):
        learner = self.learners[-1]
        encode_state = self.env.encoded_single_state(state)
        _, probs = learner.get_policy(encode_state)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = learner.get_value(encode_state)

        next_state, reward, done = self.env.step(action.item())

        s_states[step] = encode_state
        s_actions[step] = action
        s_logprobs[step] = logprob
        s_values[step] = value

        # Finish at player turn, start new game completely
        if done:
            s_rewards[step] = (
                -reward
            )  # env resonpse reward at the next turn rather than this turn
            s_dones[step] = True
            step += 1
            next_state = self.env.reset()

        return step, next_state, done

    def calculate_exploitability(self):
        """
        Correct exploitability calculation for Kuhn Poker.
        """
        # Step 1: Compute the best response value for each player
        br_value_player_0 = self.compute_best_response_value(player=0)
        br_value_player_1 = self.compute_best_response_value(player=1)

        # Step 2: Exploitability is the sum of the best response values
        exploitability = br_value_player_0 + br_value_player_1
        return exploitability

    @torch.no_grad()
    def compute_best_response_value(self, player):
        """
        Compute the best response value for a given player against the current policy.
        """
        total_value = 0.0
        num_games = 1000  # Number of games to sample

        for _ in range(num_games):
            state = self.env.reset()
            done = False
            while not done:
                if self.env.current_player == player:
                    # Best response: choose the action that maximizes expected value
                    _, probs = self.learners[-1].get_policy(
                        self.env.encoded_single_state(state)
                    )
                    action = torch.argmax(probs).item()  # Greedy best response
                else:
                    # Opponent plays according to the current policy
                    _, probs = self.learners[-1].get_policy(
                        self.env.encoded_single_state(state)
                    )
                    action = Categorical(probs).sample().item()

                next_state, reward, done = self.env.step(action)

                if done:
                    # Next turn is player, so current reward is opp
                    if self.env.current_player == player:
                        total_value -= reward
                    else:
                        total_value += reward

                state = next_state

        # Normalize by the number of games
        return total_value / num_games


if __name__ == "__main__":
    env = KuhnPokerEnv(verbose=False)
    mlp = Mlp(env.encode_state_size, env.num_actions, DEVICE)
    # mlp.load("ach_checkpoint.pth")
    ach = ACH(env, mlp)
    ach.train(100)
    # mlp.save("ach_checkpoint.pth")
    # for i in range(5):
    #     print("Exploitability: " + str(ach.calculate_exploitability()))
