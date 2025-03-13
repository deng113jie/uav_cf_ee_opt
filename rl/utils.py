from torch import nn
import numpy as np
import torch


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class RewardNormalizer:  # no good! critic super large
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        self.epsilon = epsilon

    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        # Update standard deviation using Welford's algorithm
        self.std = np.sqrt(((self.count - 1) * (self.std ** 2) + delta * delta2) / self.count)

    def normalize(self, reward):
        return (reward - self.mean) / (self.std + self.epsilon)

def test(agent, env, runs=10):
    rwd = []
    for i in range(runs):
        state = env.reset()
        action = agent.act(state)
        ap_l = np.array([i * env.grid_size for i in action])
        next_state, reward, done, _1 = env.step(ap_l)
        rwd.append(reward)
    return np.mean(rwd)