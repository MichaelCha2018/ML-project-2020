import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import deque
import random

MEMORY_CAPACITY = 10000000
MEMORY_THRESHOLD = 10000
EPSILON = 0.9
BATCH_SIZE = 32
LR = 0.01
GAMMA = 0.99
TARGET_REPLACE_ITER = 100


class Net(nn.Module):
    def __init__(self, N_ACTIONS):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(512, N_ACTIONS, bias=True)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.out(x)
        return x

class DQN(object):
    def __init__(self, N_ACTIONS):
        self.n_actions = N_ACTIONS
        self.network = Net(N_ACTIONS)
        self.memory = deque()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:
            return np.random.randint(0, self.n_actions)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.network.forward(state)
        return torch.max(actions_value, 1)[1].data.numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()
        if len(self.memory) < MEMORY_THRESHOLD:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])

        eval_q = self.network.forward(state).gather(1, action)
        next_q = self.network(next_state).detach()
        target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
        loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()