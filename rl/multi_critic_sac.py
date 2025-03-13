import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from matlab_env import MatlabEnv
import pickle
import os

# Gaussian Policy with Multi-Head Output
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # Output: mu and log_std
            ) for _ in range(action_dim)
        ])

    def forward(self, state):
        x = self.net(state)
        mu_std = [head(x) for head in self.heads]
        mu_std = torch.stack(mu_std, dim=1)  # Shape: (batch, action_dim, 2)
        mu = mu_std[:, :, 0]
        mu = torch.nan_to_num(mu, nan=0.1, posinf=0.0, neginf=0.0)
        log_std = mu_std[:, :, 1].clamp(-20, 2)  # Constraint log_std
        log_std = torch.nan_to_num(log_std, nan=0.1, posinf=0.0, neginf=0.0)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)  # Ensure action is bounded (-1,1)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


# Twin Critics with Multi-Head Rewards
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + 1, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(action_dim)
        ])
        self.q2_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + 1, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(action_dim)
        ])

    def forward(self, state, action):
        q1_values = [head(torch.cat([state, action[:, i:i + 1]], dim=-1)) for i, head in enumerate(self.q1_heads)]
        q2_values = [head(torch.cat([state, action[:, i:i + 1]], dim=-1)) for i, head in enumerate(self.q2_heads)]
        return torch.cat(q1_values, dim=1), torch.cat(q2_values, dim=1)


# SAC Agent with Per-Head Learning
class SAC:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

        # Entropy temperature (alpha)
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Compute target Q-values per head
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            min_next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones) * self.gamma * min_next_q

        # Update Critics per head
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor per head
        new_actions, log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha (Entropy Adjustment)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update target critics
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Training Loop
if __name__ == "__main__":
    env = MatlabEnv(k=50, grid_size=2000, ap_size=5)
    state_dim = env.observation_space_size
    action_dim = env.action_space_size
    pkl_file = 'multi_critic.pkl'
    if os.path.exists(pkl_file):
        agent = pickle.load(open(pkl_file, 'rb'))
    else:
        agent = SAC(state_dim, action_dim)

    episodes = 1000
    steps_per_episode = 200

    ap_size = action_dim  # Total number of APs

    # Curriculum: gradually free more APs over phases
    for phase in range(ap_size):
        print(f"Curriculum Phase {phase + 1}/{ap_size}")
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for _ in range(steps_per_episode):
                action = agent.act(state)
                # Freeze non-target APs during this curriculum phase
                if phase < ap_size - 1:
                    mask = np.ones(ap_size)
                    mask[:phase + 1] = 0  # Freeze the first (phase+1) APs
                    # Replace frozen AP actions with baseline value (e.g., 0.5)
                    action = action * mask + 0.5 * (1 - mask)
                # Calculate the new AP positions based on actions
                ap_positions = np.zeros(ap_size)
                for i in range(ap_size):
                    ap_positions[i] = env.ap_locations[i] * (1 + action[i] - 0.5)
                next_state, reward, done, _ = env.step(ap_positions)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
                total_reward += reward

            print(f"Phase {phase + 1} Episode {episode + 1}: Avg Reward {total_reward / steps_per_episode:.2f}")
            pickle.dump(agent, open(pkl_file, 'wb'))
