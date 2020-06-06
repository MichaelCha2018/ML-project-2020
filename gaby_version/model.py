import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from collections import deque
import random
from utils import FrameStack, WarpFrame, Uint2Float
import time



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class DQNNet(nn.Module):
    def __init__(self, n_actions):
        super(DQNNet, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, obs):
        #obs = TorchFrame(obs).to(device)
        obs = Uint2Float(obs)
        obs = obs.view(-1, 4, 84, 84)
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
        #state, action, reward, next_state, done = map(list, zip(*random.sample(self.buffer, batch_size)))
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent():
    def __init__(self,
                 env,
                 lr=0.00025,
                 alpha=0.95,
                 gamma=0.99,
                 rep_buf_size=125000,
                 rep_buf_ini=12500,
                 batch_size=32,
                 target_update=5000,
                 skip_frame = 4):
        self.env = env
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.rep_buf_size = rep_buf_size
        self.rep_buf_ini = rep_buf_ini
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.policy_model = DQNNet(self.n_actions).to(device)
        self.target_model = DQNNet(self.n_actions).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.policy_model.parameters(), lr=lr, alpha=alpha)
        self.replay_buffer = ReplayBuffer(rep_buf_size)
        self.skip_frame = skip_frame
        self.init_replay()

    def init_replay(self):
        while len(self.replay_buffer) < self.rep_buf_ini:
            observation = self.env.reset()
            observation = WarpFrame(observation)
            observation = np.stack([observation] * 4, axis=0)
            done = False

            while not done:
                action = self.env.action_space.sample()

                next_observation, reward, done, info = self.env.step(action)
                next_observation = FrameStack(next_observation, observation)

                self.replay_buffer.push(observation, action, reward, next_observation, done)

                observation = next_observation

        print('Experience Replay buffer initialized')

    def choose_action(self, obs, epsilon):
        if random.random() > epsilon:
            obs = torch.from_numpy(np.array(obs)).float().to(device)
            q_value = self.policy_model(obs)
            action = q_value.argmax(1).data.cpu().numpy().astype(int)[0]
        else:
            action = self.env.action_space.sample()
        return action

    def huber_loss(self, input, target, beta=1, size_average=True):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

    def learn(self, num_frames):
        """
        Update the policy
        """
        if len(self.replay_buffer) > self.batch_size and num_frames % self.skip_frame == 0:
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)
            observations = torch.from_numpy(np.array(observations)).float().to(device)

            actions = torch.from_numpy(np.array(actions).astype(int)).float().to(device)
            actions = actions.view(actions.shape[0], 1)

            rewards = torch.from_numpy(np.array(rewards)).float().to(device)
            rewards = rewards.view(rewards.shape[0], 1)

            dones = torch.from_numpy(np.array(dones).astype(int)).float().to(device)
            dones = dones.view(dones.shape[0], 1)

            next_observations = torch.from_numpy(np.array(next_observations)).float().to(device)

            q_values = self.policy_model(observations)
            next_q_values = self.target_model(next_observations)

            q_value = q_values.gather(1, actions.long())
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

            loss = self.huber_loss(q_value, expected_q_value)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        if num_frames % self.target_update == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

