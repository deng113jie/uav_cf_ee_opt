import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from matlab_env import MatlabEnv
import pickle
import os

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


# Actor-Critic Networks for each AP agent
class APActor(nn.Module):
    def __init__(self, state_dim):
        super(APActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.net(state)


class APCritic(nn.Module):
    def __init__(self, global_state_dim, total_action_dim):
        super(APCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim + total_action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_state, all_actions):
        return self.net(torch.cat([global_state, all_actions], dim=1))


# Individual AP Agent
class APAgent:
    def __init__(self, agent_id, global_state_dim, local_state_dim, action_dim):
        self.id = agent_id
        self.action_dim = action_dim

        # Networks
        self.actor = APActor(local_state_dim)
        self.actor_target = APActor(local_state_dim)
        self.critic = APCritic(global_state_dim,  action_dim)
        self.critic_target = APCritic(global_state_dim,  action_dim)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Noise
        self.noise = OUNoise(1)

    def act(self, state, noise=True):
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state))
        if noise:
            action += torch.FloatTensor(self.noise.sample())
        return np.clip(action.numpy(), 0, 1)


# Multi-Agent Coordinator
class MADDPG:
    def __init__(self, env):
        self.env = env
        self.ap_size = env.l * env.n_coordinates
        # global for critic, local for action
        # global: ue locations, local:
        self.global_state_dim = env.observation_space_size
        self.local_state_dim = env.observation_space_size     # Assuming divisible
        self.action_dim = self.ap_size  # Each AP controls its position movement

        # Create agents
        self.agents = [
            APAgent(i, self.global_state_dim, self.local_state_dim, self.action_dim)
            for i in range(self.ap_size)
        ]

        # Shared replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 256
        self.gamma = 0.95
        self.tau = 0.01

    def store_transition(self, states, actions, reward, next_states, done):
        self.memory.append((
            states.flatten(),  # Global state
            np.concatenate(actions),  # All actions
            reward,
            next_states.flatten(),  # Next global state
            done
        ))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        # print('learning...')
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        global_states, all_actions, rewards, next_global_states, dones = zip(*batch)

        # Convert to tensors
        global_states = torch.FloatTensor(np.array(global_states))
        all_actions = torch.FloatTensor(np.array(all_actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_global_states = torch.FloatTensor(np.array(next_global_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Train each agent
        for agent in self.agents:
            # --------- Critic Update ---------
            with torch.no_grad():
                # Target actions from all agents
                # next_local_states = next_global_states.reshape(-1, self.ap_size, self.local_state_dim)
                next_local_states = next_global_states
                next_actions = []
                for i, a in enumerate(self.agents):
                    # next_local_state = next_local_states[:, i, :]
                    next_actions.append(a.actor_target(next_local_states))
                next_actions = torch.cat(next_actions, dim=1)

                # Target Q-value
                target_q = agent.critic_target(next_global_states, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q

            # Current Q-value
            current_q = agent.critic(global_states, all_actions)
            critic_loss = nn.MSELoss()(current_q, target_q)

            # Update critic
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # --------- Actor Update ---------
            # Get current actions from all agents
            # current_local_states = global_states.reshape(-1, self.ap_size, self.local_state_dim)
            current_local_states = global_states
            new_actions = []
            for i, a in enumerate(self.agents):
                # local_state = current_local_states[:, i, :]
                if a == agent:
                    # Use current agent's actor
                    new_action = agent.actor(current_local_states)
                else:
                    # Use target actors for other agents
                    with torch.no_grad():
                        new_action = a.actor_target(current_local_states)
                new_actions.append(new_action)
            new_actions = torch.cat(new_actions, dim=1)

            # Actor loss
            actor_loss = -agent.critic(global_states, new_actions).mean()

            # Update actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # Soft update targets
            for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Training Loop
def train_maddpg():
    env = MatlabEnv(k=50, grid_size=2000, ap_size=5)
    maddpg = MADDPG(env)
    maddpg_agt_f = 'multi_actor_ddpg.pkl'
    if os.path.exists(maddpg_agt_f):
        maddpg.agents = pickle.load(open(maddpg_agt_f,'rb'))
    episodes = 1000
    best_reward = 0
    n_steps = 100

    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0
        noise_scaling = max(0.1, 1.0 - ep / 500)  # Decay exploration

        for step in range(n_steps):  # 100 steps per episode
            # Get actions from all agents
            actions = []
            # local_states = state.reshape(maddpg.ap_size, -1)  # Split global state

            for i, agent in enumerate(maddpg.agents):
                action = agent.act(state, noise=noise_scaling)
                actions.append(action)

            # Environment step
            next_state, reward, done, _ = env.step(np.concatenate(actions))
            episode_reward += reward

            # Store experience
            maddpg.store_transition(state, actions, reward, next_state, done)

            # Learn
            maddpg.learn()

            state = next_state

        # Performance tracking
        avg_reward = episode_reward / n_steps
        if avg_reward > best_reward:
            best_reward = avg_reward
            pickle.dump(maddpg.agents, open(maddpg_agt_f,'wb'))

        print(f"Episode {ep + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Best: {best_reward:.2f}")


if __name__ == "__main__":
    train_maddpg()
