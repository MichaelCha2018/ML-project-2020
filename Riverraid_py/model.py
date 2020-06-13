import torch.nn as nn
from collections import deque
import random

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, 18))
    # The action space of Riverraid is 18

    def forward(self, obs):
        obs = self.conv(obs)
        obs = obs.view(obs.shape[0], obs.shape[1] * obs.shape[2] * obs.shape[3])
        actions = self.fc(obs)
        return actions


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)