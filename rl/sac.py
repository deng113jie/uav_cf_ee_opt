import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from matlab_env import MatlabEnv
import pickle
import os
from utils import RewardNormalizer, init_weights, test

torch.set_default_device('cuda:0')
# Gaussian Policy with Multi-Head Output
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        self.std = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.net.apply(init_weights)
        self.std.apply(init_weights)

    def forward(self, state):
        mu = self.net(state)
        mu = torch.nan_to_num(mu, nan=0.1, posinf=0.0, neginf=0.0)
        log_std = self.std(state).clamp(-20, 2)
        log_std = torch.nan_to_num(log_std, nan=0.1, posinf=0.0, neginf=0.0)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.nn.Sigmoid()(z)  # output 0 to 1
        # log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6 + torch.log(torch.tensor(1))  )
        log_prob = normal.log_prob(z) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        log_prob = torch.clamp(log_prob, -50, 50)  # Clamp to prevent NaNs
        return action, log_prob


# Twin Critics
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q1.apply(init_weights)
        self.q2.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.memory = deque(maxlen=1000000)
        self.batch_size = 128
        self.gamma = 0.9  # lower more immediate reward
        self.tau = 0.01
        # Entropy temperature (alpha)
        self.alpha = .8
        self.target_entropy = -action_dim  # action range 0 to 1
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-5)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.tensor(state).float().unsqueeze(0)  # default device
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().squeeze(0)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0,0,0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states)).float()
        actions = torch.tensor(np.array(actions)).float()
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        rewards = torch.tensor(np.array(rewards)).float().unsqueeze(1)
        next_states = torch.tensor(np.array(next_states)).float()
        dones = torch.tensor(np.array(dones).astype(int)).unsqueeze(1)

        # Compute target Q-value
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            # min_next_q = torch.clamp(min_next_q, -1e3, 1e3)
            target_q = rewards + (1 - dones) * self.gamma * min_next_q

        # Update Critics
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update Alpha (Entropy Adjustment)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().clamp(min=1e-5, max=1.0)

        # Soft update target critics
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy(), alpha_loss.cpu().data.numpy()


def train():
    env = MatlabEnv(k=50, grid_size=2000, ap_size=50)
    state_dim = env.observation_space_size
    action_dim = env.action_space_size  # Note: action_dim equals the number of APs
    pkl_file = 'sac_50k50l.pth'
    if os.path.exists(pkl_file):
        print('loading from previous model')
        agent = torch.load(pkl_file)
    else:
        agent = SAC(state_dim, action_dim)

    episodes = 1000
    steps_per_episode = 100
    higtest_rwd = 0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        loss_cal = np.zeros((steps_per_episode, 3))
        for stp in range(steps_per_episode):
            action = agent.act(state)
            ap_positions = np.array([i*env.grid_size for i in action])
            next_state, reward, done, nth = env.step(ap_positions)
            reward = (reward - 8)/5
            agent.remember(state, action, reward, next_state, done)
            critic_loss, actor_loss, alpha_loss = agent.learn()
            loss_cal[stp, :] = [critic_loss, actor_loss, alpha_loss]
            state = next_state
            total_reward += reward
        rwd = test(agent, env, runs=30)
        if rwd > higtest_rwd:
            torch.save(agent, pkl_file)
            higtest_rwd = rwd
        print(f"Episode {episode + 1}: Avg Reward {total_reward / steps_per_episode:.2f}"
              f" Loss Critic {np.average(loss_cal[:, 0])} Actor {np.average(loss_cal[:, 1])} Alpha {np.average(loss_cal[:, 2])}"
              f" Highest test rwd {higtest_rwd}")



if __name__ == "__main__":
    train()
