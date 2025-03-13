from sac import SAC
import numpy as np
from matlab_env import MatlabEnv
import os
import pickle
import torch
from utils import test


class MultiAgentSAC:
    def __init__(self, state_dim, num_agents):
        self.num_agents = num_agents
        self.agents = [SAC(state_dim, action_dim=1) for _ in range(num_agents)]
        # [agent.cuda() for agent in self.agents]

    def act(self, state):
        actions = []
        for agent in self.agents:
            action = agent.act(state)  # Each agent returns a scalar action
            actions.append(action)
        return np.array(actions).squeeze()  # Combine into action vector

    def remember(self, state, action, reward, next_state, done):
        # Split joint action into individual actions
        for i, agent in enumerate(self.agents):
            agent_action = action[i] if len(action) > i else 0
            agent.remember(state, agent_action, reward, next_state, done)

    def learn(self):
        critic_losses = []
        actor_losses = []
        alpha_losses = []

        for agent in self.agents:
            cl, al, al_ = agent.learn()
            critic_losses.append(cl)
            actor_losses.append(al)
            alpha_losses.append(al_)

        return (
            np.mean(critic_losses),
            np.mean(actor_losses),
            np.mean(alpha_losses)
        )


if __name__=="__main__":

    env = MatlabEnv(k=50, grid_size=2000, ap_size=5)
    state_dim = env.observation_space_size
    action_dim = env.action_space_size  # Note: action_dim equals the number of APs
    pkl_file = 'multi_agent_sac50k5l.pth'
    if os.path.exists(pkl_file):
        print('load from previous model..')
        agent = torch.load(pkl_file)
    else:
        agent = MultiAgentSAC(state_dim, action_dim)
    episodes = 1000
    steps_per_episode = 100
    ap_size = action_dim  # Total number of APs
    higtest_rwd = 0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        loss_cal = np.zeros((steps_per_episode, 3))
        for stp in range(steps_per_episode):
            action = agent.act(state)
            ap_l = np.array([i * env.grid_size for i in action])
            next_state, reward, done, _1 = env.step(ap_l)
            reward = (reward - 8) / 5
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
