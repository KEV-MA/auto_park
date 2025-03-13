import torch
import torch.nn.functional as F
import gym
from numpy.random import choice
from copy import deepcopy
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters

class PPOAgent:
    """Continuous PPO agent."""

    def __init__(self, policy, discount=0.98, clip_epsilon=0.2, cg_iteration=10,
                 cg_dampening=0.001, cg_tolerance=1e-10, cg_state_percent=0.1):
        self.policy = policy
        self.discount = discount
        self.clip_epsilon = clip_epsilon
        self.cg_iteration = cg_iteration
        self.cg_dampening = cg_dampening
        self.cg_tolerance = cg_tolerance
        self.cg_state_percent = cg_state_percent
        self.distribution = torch.distributions.normal.Normal

        # Cuda check
        self.device = (torch.device('cuda') if torch.cuda.is_available()
                       else torch.device('cpu'))
        policy.to(self.device)

        # Infer action dimensions from policy
        policy_modules = [module for module in policy.modules() if not
        isinstance(module, torch.nn.Sequential)]
        action_dims = policy_modules[-1].out_features

        # Set logstd
        self.logstd = torch.ones(action_dims, requires_grad=True,
                                 device=self.device)
        with torch.no_grad():
            self.logstd /= self.logstd.exp()

        self.buffers = {'log_probs': [], 'actions': [],
                        'completed_rewards': [], 'states': []}

    def __call__(self, state):
        """
        Peforms forward pass on the NN and parameterized distribution.
        Parameters
        ----------
        state : torch.Tensor
            Tensor passed into NN and distribution.
        Returns
        -------
            Action choice for each action dimension.
        """
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        # Parameterize distribution with policy, sample action
        normal_dist = self.distribution(self.policy(state), self.logstd.exp())
        action = normal_dist.sample()
        # Save information
        self.buffers['actions'].append(action)
        self.buffers['log_probs'].append(normal_dist.log_prob(action))
        self.buffers['states'].append(state)
        return action.cpu().numpy()

    def surrogate_objective(self, new_policy, new_std, states, actions,
                            log_probs, advantages):
        new_dist = self.distribution(new_policy(states), new_std.exp())
        new_prob = new_dist.log_prob(actions)

        # Detach old log_probs, only compute grad w.r.t. new policy
        ratio = new_prob.exp() / log_probs.detach().exp()

        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        return torch.min(ratio * advantages.view(-1, 1), clipped_ratio * advantages.view(-1, 1)).mean()

    def optimize(self):
        # Return if no completed episodes
        if len(self.buffers['completed_rewards']) == 0:
            return

        # Convert all buffers to tensors
        num_batch_steps = len(self.buffers['completed_rewards'])
        rewards = torch.tensor(self.buffers['completed_rewards'])
        actions = torch.stack(self.buffers['actions'][:num_batch_steps])
        states = torch.stack(self.buffers['states'][:num_batch_steps])
        log_probs = torch.stack(self.buffers['log_probs'][:num_batch_steps])
        rewards, actions, states, log_probs = (rewards.to(self.device),
                                               actions.to(self.device),
                                               states.to(self.device),
                                               log_probs.to(self.device))

        # Normalize rewards over episodes
        rewards = (rewards - rewards.mean()) / rewards.std()

        # Compute advantages
        advantages = rewards - rewards.mean()

        # Compute surrogate objective
        loss = -self.surrogate_objective(self.policy, self.logstd, states, actions,
                                         log_probs, advantages)
        loss.backward()

        # Update policy
        optimizer = torch.optim.Adam(self.policy.parameters())
        optimizer.step()
        optimizer.zero_grad()

        # Update buffers removing processed steps
        for key, storage in self.buffers.items():
            del storage[:num_batch_steps]

    def train(self, env_name, seed=None, batch_size=12000, iterations=100,
              max_episode_length=None, verbose=False):

        # Initialize env
        env = gym.make(env_name)
        if seed is not None:
            torch.manual_seed(seed)
            env.seed(seed)
        if max_episode_length is None:
            max_episode_length = float('inf')
        # Recording
        recording = {'episode_reward': [[]],
                     'episode_length': [0],
                     'num_episodes_in_iteration': []}

        # Begin training
        observation = env.reset()
        for iteration in range(iterations):
            # Set initial value to 0
            recording['num_episodes_in_iteration'].append(0)

            for step in range(batch_size):
                # Take step with agent
                observation, reward, done, _ = env.step(self(observation))

                # Recording, increment episode values
                recording['episode_length'][-1] += 1
                recording['episode_reward'][-1].append(reward)

                # End of episode
                if (done or
                        recording['episode_length'][-1] >= max_episode_length):
                    # Calculate discounted reward
                    discounted_reward = recording['episode_reward'][-1].copy()
                    for index in range(len(discounted_reward) - 2, -1, -1):
                        discounted_reward[index] += self.discount * \
                                                    discounted_reward[index + 1]
                    self.buffers['completed_rewards'].extend(discounted_reward)

                    # Set final recording of episode reward to total
                    recording['episode_reward'][-1] = \
                        sum(recording['episode_reward'][-1])
                    # Recording
                    recording['episode_length'].append(0)
                    recording['episode_reward'].append([])
                    recording['num_episodes_in_iteration'][-1] += 1

                    # Reset environment
                    observation = env.reset()

            # Print information if verbose
            if verbose:
                num_episode = recording['num_episodes_in_iteration'][-1]
                avg = (round(sum(recording['episode_reward'][-num_episode:-1])
                             / (num_episode - 1), 3))
                print(f'Average Reward over Iteration {iteration}: {avg}')
            # Optimize after batch
            self.optimize()

        env.close()
        # Return recording information
        return recording

    def save_model(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'logstd': self.logstd
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        self.logstd = checkpoint['logstd'].to(self.device)