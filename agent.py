import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Normal


class PPOAgent:
    def __init__(
        self, policy, lr=3e-4, gamma=0.98, epsilon=0.2, batch_size=64, epochs=10
    ):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.memory = []

    def __call__(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        mean = self.policy(state)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)  # Fixed std
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.memory.append((state, action, log_prob))
        return action.cpu().numpy()

    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def optimize(self, rewards):
        states, actions, old_log_probs = zip(*self.memory)
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs).detach()
        returns = self.compute_returns(rewards)

        for _ in range(self.epochs):
            for idx in range(0, len(rewards), self.batch_size):
                sampled_idx = slice(idx, idx + self.batch_size)
                sampled_states = states[sampled_idx]
                sampled_actions = actions[sampled_idx]
                sampled_old_log_probs = old_log_probs[sampled_idx]
                sampled_returns = returns[sampled_idx]

                mean = self.policy(sampled_states)
                dist = Normal(mean, torch.ones_like(mean) * 0.1)
                new_log_probs = dist.log_prob(sampled_actions)
                ratio = (new_log_probs - sampled_old_log_probs).exp()

                advantage = sampled_returns - sampled_returns.mean()
                advantage = advantage.view(-1, 1)
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []  # Clear memory after optimization

    def train(
        self,
        env_name,
        seed=None,
        batch_size=5000,
        iterations=100,
        max_episode_length=250,
        verbose=True,
    ):
        env = gym.make(env_name)
        if seed:
            env.seed(seed)
            torch.manual_seed(seed)

        observation = env.reset()
        episode_rewards = []

        for iteration in range(iterations):
            total_rewards = []
            for step in range(batch_size):
                action = self(observation)
                observation, reward, done, _ = env.step(action)
                episode_rewards.append(reward)

                if done or len(episode_rewards) >= max_episode_length:
                    total_rewards.extend(episode_rewards)
                    self.optimize(episode_rewards)
                    observation = env.reset()
                    episode_rewards = []

            if verbose:
                print(
                    f"Iteration {iteration}, Average Reward: {np.mean(total_rewards)}"
                )

        env.close()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        checkpoint = torch.load(path)

        # If saved with "policy" and "logstd" keys
        if "policy" in checkpoint and "logstd" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"])
            self.logstd = checkpoint["logstd"].to(self.device)
        else:
            # Load directly if it’s just the raw state_dict
            self.policy.load_state_dict(checkpoint)
