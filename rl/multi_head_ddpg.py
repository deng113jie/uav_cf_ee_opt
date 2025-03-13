import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from matlab_env import MatlabEnv
import pickle
import os
from utils import RewardNormalizer, init_weights


# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


# Actor-Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, ap_size=5):
        super(Actor, self).__init__()
        self.ap_size = ap_size
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for _ in range(ap_size)
        ])

        self.shared.apply(init_weights)
        self.heads.apply(init_weights)

    def forward(self, state):
        x = self.shared(state)
        return torch.cat([head(x) for head in self.heads], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.net.apply(init_weights)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))


# Complete DDPG Agent
class DDPG:
    def __init__(self, state_dim, action_dim, ap_size=5):
        # Networks
        self.actor = Actor(state_dim, ap_size)
        self.actor_target = Actor(state_dim,  ap_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Replay buffer and hyperparameters
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.tau = 0.005
        self.gamma = 0.99
        self.noise = OUNoise(action_dim)
        self.ap_size = ap_size

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).numpy().reshape(1,-1).squeeze()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, 0, 1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0,0

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        # rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))  # for contribution reward
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            # target_q = torch.clamp(target_q, -1e6, 1e6)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        self.critic_optimizer.zero_grad()
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        # Update critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor loss
        self.actor_optimizer.zero_grad()
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean() #.detach()

        # Update actor

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        # Soft update targets
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.data.numpy(), actor_loss.data.numpy()

    def compute_ap_contributions(self, state, action, global_reward):
        """Optional: Gradient-based attribution"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).requires_grad_()

        q_value = self.critic(state_tensor, action_tensor)
        q_value.backward()

        contributions = torch.abs(action_tensor.grad).squeeze().numpy()
        contributions = contributions / (np.sum(contributions) + 1e-8)
        return contributions * global_reward


# Training Loop with Curriculum Learning
def train_ddpg(curriculum=True):
    env = MatlabEnv(k=50, grid_size=2000, ap_size=5)
    state_dim = env.observation_space_size
    action_dim = env.action_space_size
    ap_s = action_dim
    pkl_path = 'multi_actor_ddpg_50k5l.pkl'
    if os.path.exists(pkl_path):
        print('loading from previous model...')
        agent = pickle.load(open(pkl_path, 'br'))
    else:
        agent = DDPG(state_dim, action_dim, ap_s)

    episodes = 1000
    steps_p_epi = 100
    max_rwd = 0
    if curriculum:
        for phase in range(ap_s):
            print(f"Curriculum Phase {phase + 1}/{ap_s}")
            for episode in range(episodes):
                state = env.reset()
                agent.noise.reset()
                total_reward = 0
                loss_cal = np.zeros((steps_p_epi, 2))
                for _1 in range(steps_p_epi):  # 100 steps per episode
                    # Freeze non-target APs during curriculum phase
                    action = agent.act(state)
                    if phase < ap_s - 1:
                        mask = np.ones(ap_s)
                        mask[:phase + 1] = 0  # Only allow current phase AP to move
                        action = action * mask + 0.5 * (1 - mask)
                    # ap_l = np.zeros(len(action))
                    # for i in range(len(action)):  # move the ap location
                    #     ap_l[i] = env.ap_locations[i] * (1 + action[i] - 0.5)
                    #     ap_l[i] = ap_l[i] if 0 < ap_l[i] < env.grid_size else 1
                    ap_l = np.array([i*env.grid_size for i in action])

                    next_state, reward, done, _ = env.step(ap_l)
                    # Optional: Use gradient attribution
                    contributions = agent.compute_ap_contributions(state, action, reward)
                    # adjusted_reward = np.sum(contributions)  # Or use per-AP rewards
                    # reward = (reward - 8) / 5

                    agent.remember(state, action, contributions, next_state, done)
                    critic_loss, actor_loss = agent.learn()
                    loss_cal[_1, :] = [critic_loss, actor_loss]
                    state = next_state
                    total_reward += reward

                print(f"Phase {phase + 1} Episode {episode + 1}: Avg Reward {total_reward / steps_p_epi:.2f}"
                      f" Loss Critic {np.average(loss_cal[:, 0])} Actor {np.average(loss_cal[:, 1])}")
                if max_rwd < total_reward / steps_p_epi:
                    max_rwd = total_reward / steps_p_epi
                    pickle.dump(agent, open(pkl_path,'bw'))
    else:
        for episode in range(episodes):
            state = env.reset()
            agent.noise.reset()
            total_reward = 0
            loss_cal = np.zeros((steps_p_epi, 2))
            for _1 in range(steps_p_epi):  # 100 steps per episode
                # Freeze non-target APs during curriculum phase
                action = agent.act(state)
                # ap_l = np.zeros(len(action))
                # for i in range(len(action)):  # move the ap location
                #     ap_l[i] = env.ap_locations[i] * (1 + action[i] - 0.5)
                #     ap_l[i] = ap_l[i] if 0 < ap_l[i] < env.grid_size else 1
                ap_l = np.array([i * env.grid_size for i in action])
                next_state, reward, done, _ = env.step(ap_l)
                # Optional: Use gradient attribution
                # contributions = agent.compute_ap_contributions(state, action, reward)
                # adjusted_reward = np.sum(contributions)  # Or use per-AP rewards

                reward = (reward - 8) / 5
                agent.remember(state, action, reward, next_state, done)
                critic_loss, actor_loss = agent.learn()
                loss_cal[_1, :] = [critic_loss, actor_loss]
                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}: Avg Reward {total_reward / steps_p_epi:.2f}"
                  f" Loss Critic {np.average(loss_cal[:, 0])} Actor {np.average(loss_cal[:, 1])}")
            if max_rwd < total_reward / steps_p_epi:
                max_rwd = total_reward / steps_p_epi
                pickle.dump(agent, open(pkl_path, 'bw'))


if __name__ == "__main__":
    train_ddpg(curriculum=True)
